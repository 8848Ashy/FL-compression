import math
import torch
from utils.communication import normalized_bits_per_dim, srk_encoded_dimension
from utils.state_dict import flatten_state_dict, unflatten_state_dict, state_dict_add
from compression.quantization import stochastic_k_level_quantize


def _fwht_fast(tensor):
    """快速沃尔什-哈达玛变换 (Fast Walsh-Hadamard Transform, FWHT)。
    用于实现随机正交旋转，将向量的离群值（Outliers）均匀摊平到各个维度。
    """
    n = tensor.shape[0]; result = tensor.clone(); h = 1
    while h < n:
        blocks = result.view(-1, 2 * h); left, right = blocks[:, :h].clone(), blocks[:, h:].clone()
        blocks[:, :h], blocks[:, h:] = left + right, left - right; h *= 2
    return result.view(n) / math.sqrt(n)


def federated_round_srk_update(global_model, client_deltas, k_levels, rotation_seed=None,
                               quant_seeds=None, relerr_out=None,
                               rotation_seeds=None):
    """Compress client updates with randomized Hadamard rotations.

    ``rotation_seeds=None`` preserves the original shared-mask behavior.
    Supplying one seed per client gives each update an independent mask, as in
    DRIVE's distributed setting; each is inverse-rotated before averaging.
    """
    
    # 1. 展平模型更新参数，计算原维度 d 和扩充后的对齐维度 D
    flat0, shapes = flatten_state_dict(client_deltas[0])
    d = flat0.shape[0]
    D = srk_encoded_dimension(d)
    device = flat0.device
    
    if rotation_seeds is not None and len(rotation_seeds) != len(client_deltas):
        raise ValueError("rotation_seeds must contain one seed per client update")

    def make_signs(seed):
        generator = torch.Generator()
        generator.manual_seed(seed if seed is not None else torch.seed())
        return ((torch.rand(D, generator=generator) < .5).float() * 2 - 1).to(device)

    shared_signs = make_signs(rotation_seed) if rotation_seeds is None else None
    
    # 初始化聚合累加器
    aggregate = torch.zeros(D, device=device)
    
    # 3. 遍历每个客户端的本地更新
    for index, delta in enumerate(client_deltas):
        signs = shared_signs if rotation_seeds is None else make_signs(rotation_seeds[index])
        flat, _ = flatten_state_dict(delta)
        padded = torch.zeros(D, device=device)
        padded[:d] = flat  # 将梯度补零对齐到维度 D
        
        # 步骤 A: 随机旋转 (Random Rotation) -> 乘以 signs 并做 FWHT 变换
        rotated = _fwht_fast(padded * signs)
        
        # 设置量化随机种子
        quant_generator = None
        if quant_seeds is not None:
            quant_generator = torch.Generator(device=device)
            quant_generator.manual_seed(quant_seeds[index])
            
        # 步骤 B: k 级量化 (Quantization) -> 这里内部包含了 min/max 极值缩放
        quantized, _ = stochastic_k_level_quantize(rotated, k_levels, generator=quant_generator)
        
        # Independent masks require decoding each update in its own basis.
        if rotation_seeds is None:
            aggregate += quantized
        else:
            aggregate += _fwht_fast(quantized) * signs
        
        # 步骤 C: 计算并记录相对误差 (NMSE)
        if relerr_out is not None:
            reconstructed = _fwht_fast(quantized) * signs
            relerr_out.append(float(((reconstructed[:d] - flat) ** 2).sum() / (flat ** 2).sum().clamp_min(1e-12)))
            
    # 4. 服务器端聚合。独立掩码模式已逐客户端逆变换。
    aggregate /= len(client_deltas)

    # 逆旋转恢复到原空间（共享掩码时可在均值上统一逆变换）。
    recovered = aggregate if rotation_seeds is not None else _fwht_fast(aggregate) * shared_signs

    # !!!原始只传0/1或者有传幅值
    recovered[:d] = torch.where(recovered[:d] >= 0, torch.ones_like(recovered[:d]), -torch.ones_like(recovered[:d]))
  
    # 5. 将更新量加回到全局模型中
    global_model.load_state_dict(state_dict_add(global_model.state_dict(), unflatten_state_dict(recovered[:d], shapes)))
    
    return normalized_bits_per_dim(D, d, int(math.ceil(math.log2(k_levels))))
