import math
import torch


# ============================================================
# 量化模块：把客户端的模型更新（浮点向量）压成 b 比特/坐标
#
# 两大类用法（对应实验里的 scale_mode）：
#   - doscale   （动态缩放）：每轮根据当前向量估一个 scale，
#               量化后 server 靠这个 scale 恢复数值，
#               代价是每轮每向量多传 32 bit（见 encoders.py 的 bpd 计算）。
#   - noscale   （静态边界）：用事先约定的固定区间 [-clip, clip]，
#               不传任何 scale 信息，client/server 各自知道边界即可。
#
# 所有"返回 dict"的函数统一输出：
#   values : 量化后再解码回来的浮点向量（和输入同形状）
#   bpc    : 每个坐标用的比特数 (bits per coordinate)
#   scale  : 本次量化使用的 scale / 边界值
# ============================================================


def stochastic_k_level_quantize(X_tensor, k_levels, generator=None):
    """通用随机 k 级量化：量化区间取输入向量的 [min, max]。

    返回 (解码后的浮点向量, 每个元素选中的级号)。
    随机舍入保证无偏：E[解码值] = 原值（这是收敛性分析的关键）。
    当前主流程没有直接用它，属于工具函数。
    """
    if not X_tensor.is_floating_point() or not X_tensor.numel() or not torch.isfinite(X_tensor).all():
        raise ValueError('Expected nonempty finite floating-point coefficients')
    if not isinstance(k_levels, int) or k_levels < 2:
        raise ValueError('k_levels must be an integer >= 2')
    X_max, X_min = torch.max(X_tensor), torch.min(X_tensor)
    if X_max == X_min:
        # 向量是常数：任何级都一样，直接返回
        indices = torch.zeros_like(X_tensor, dtype=torch.long)
        return torch.full_like(X_tensor, X_min), indices
    # Double intermediates also handle subnormal ranges and wide finite inputs.
    scale = X_max.double() - X_min.double()
    # 把每个元素线性映射到 [0, k_levels-1] 的"连续级号"（带小数部分）
    normalized = (X_tensor.double() - X_min.double()) / scale * (k_levels - 1)
    r = torch.floor(normalized).long(); r = torch.clamp(r, 0, k_levels - 2)
    prob = normalized - r  # 小数部分 = 向上取整一级的概率
    if generator is None:
        is_upper = (torch.rand_like(X_tensor) < prob).float()
    else:
        # generator must live on the same device as X_tensor
        is_upper = (torch.rand(X_tensor.shape, generator=generator, dtype=X_tensor.dtype, device=X_tensor.device) < prob).float()
    quantized_r = r.float() + is_upper
    # 解码：把级号映射回 [X_min, X_max] 区间内的具体数值
    decoded = (X_min.double() + quantized_r.double() / (k_levels - 1) * scale).to(X_tensor.dtype)
    return decoded, quantized_r.long()


def uniform_quantize(x, bits):
    """确定性均匀量化（最近邻舍入），区间取 [min, max]。只做最近级映射，无随机性。"""
    levels = 2 ** bits; xmin, xmax = x.min(), x.max()
    if (xmax - xmin).abs() < 1e-12: return x.clone()
    step = (xmax - xmin) / (levels - 1)  # 相邻两级的间距
    indices = torch.round((x - xmin) / step).clamp(0, levels - 1)
    return xmin + indices * step


def lloyd_max_codebook(values, k_levels, iterations=20):
    """Lloyd-Max 算法：给定一批数值，迭代求最优量化中心（使 MSE 最小）。

    交替两步直到收敛：
      1. 用相邻中心的中点做边界，把数值分桶；
      2. 每个桶的中心更新为桶内数值的均值。
    返回 k_levels 个最优中心。
    """
    xmin, xmax = values.min(), values.max()
    centers = torch.linspace(xmin, xmax, k_levels, device=values.device, dtype=values.dtype)
    for _ in range(iterations):
        boundaries = (centers[:-1] + centers[1:]) / 2  # 相邻中心的中间点 = 决策边界
        indices = torch.bucketize(values, boundaries)
        updated = torch.stack([values[indices == i].mean() if torch.any(indices == i) else centers[i] for i in range(k_levels)])
        if torch.max(torch.abs(updated - centers)) < 1e-8: break
        centers = updated
    return centers


def quantize_with_codebook(values, centers):
    """按码本量化：每个值映射到最近的中心，返回 (解码值, 中心编号)。"""
    indices = torch.argmin(torch.abs(values[:, None] - centers[None, :]), dim=1)
    return centers[indices], indices


def lloyd_max_quantize(values, k_levels, iterations=20):
    """Lloyd-Max 量化一步到位：先求最优码本，再按码本量化。"""
    centers = lloyd_max_codebook(values, k_levels, iterations)
    return quantize_with_codebook(values, centers), centers


def quantize_doscale(x, bits=1, scale_type="l1_mean", stochastic=True, generator=None,
                     clip_quantile=0.0):
    """动态缩放量化（doscale 模式的主入口，统一网格版）。

    统一口径（老师要求：同一方法内部，1/2/3-bit 是同一个算法、只改比特数）：
    对任意 bits，动态范围取 max|x|（每轮现算，随向量传一个 32bit 浮点），
    再用 _quantize_symmetric 在 [-bound, bound] 上均匀网格量化。
      - stochastic=True  : 随机舍入（无偏），用于 hadamard / kashin 一脉
      - stochastic=False : 确定性最近格舍入（signSGD 一脉，多 bit 为其自然扩展）
      - bits==1 时网格退化为 {±bound}：stochastic=False 即 signSGD（sign×尺度，
        尺度对应原文步长 delta）；stochastic=True 即为无偏符号量化。
      - clip_quantile=q in (0,1): 边界改取 |x| 的 q 分位数并截断（温和截断，
        出处 TernGrad 2.5sigma / ACIQ 最优截断；治重尾梯度下 max 被离群值绑架）。
    `scale_type` 仅为接口兼容保留，本路径不再使用（旧 mean 版已移除）。
    """
    if x.ndim == 0 or not x.numel() or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("x must be a nonempty finite floating tensor")
    if bits < 1:
        raise ValueError("bits must be >= 1")
    if not 0.0 <= clip_quantile < 1.0:
        raise ValueError("clip_quantile must be in [0, 1)")
    if x.abs().max().item() == 0:
        return {"values": torch.zeros_like(x), "bpc": float(bits), "scale": 0.0}
    if clip_quantile > 0:
        bound = x.abs().quantile(clip_quantile).clamp_min(torch.finfo(x.dtype).tiny)
        x = x.clamp(-bound, bound)
    else:
        bound = x.abs().max().clamp_min(torch.finfo(x.dtype).tiny)
    decoded = _quantize_symmetric(x, bound, bits, stochastic, generator)
    return {"values": decoded, "bpc": float(bits), "scale": float(bound.item())}


def quantize_static_noscale(x, bits=1, clip_val=0.05, stochastic=True, generator=None):
    """静态边界量化（noscale 模式的主入口）：用固定区间 [-clip_val, clip_val]，不传输任何 scale。

    client 和 server 事先约定好 clip_val（由 encoders.py 的 _effective_clip
    按 transform 和维度 d/D 计算），所以除了量化后的比特外零额外通信。

    两种比特数的处理方式（和 doscale 一样有本质区别）：
      bits == 1 : 确定性符号量化：解码值 = ±clip_val（每个坐标幅度都相同，
                  即 SignSGD —— 方向完全保留，幅度信息丢弃）
      bits >= 2 : 先把 x 截断到 ±clip_val（超出的离群值被削平），
                  再在 [-clip_val, clip_val] 上做多比特均匀量化

    注意：bound 是固定的，训练后期更新幅度变小后，量化噪声相对变大——
    这就是上周报告里 noscale 后期震荡/效果变差的根源。
    """
    if x.ndim == 0 or not x.numel() or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("x must be a nonempty finite floating tensor")
    if bits < 1 or not math.isfinite(float(clip_val)) or clip_val <= 0:
        raise ValueError("bits must be >= 1 and clip_val must be positive")
    bound = torch.as_tensor(clip_val, dtype=x.dtype, device=x.device)
    clipped = x.clamp(-bound, bound)  # 截断：|x| > bound 的离群值压到 ±bound
    if bits == 1:
        decoded = torch.where(clipped >= 0, bound, -bound)
    else:
        decoded = _quantize_symmetric(clipped, bound, bits, stochastic, generator)
    return {"values": decoded, "bpc": float(bits), "scale": float(clip_val)}


def _quantize_symmetric(x, bound, bits, stochastic, generator):
    """多比特对称均匀量化的公共核心：在 [-bound, bound] 上铺 2^bits 个等距台阶。

    网格：-bound, -bound+step, ..., +bound，共 2^bits 个点，step = 2*bound/(2^bits - 1)
    （两个端点都包含在内，所以分母是 levels-1）。

    stochastic=True （随机舍入，当前实验默认）：
        值落在相邻两级之间时，按距离比例随机选上级/下级，
        使 E[解码值] = 原值（无偏）—— server 端平均后误差相互抵消的关键。
        generator 提供可复现的随机数（联邦实验按轮次固定种子）。
    stochastic=False（最近邻舍入）：
        直接四舍五入到最近一级，实现简单但误差有偏。
    """
    levels = 2 ** bits
    # Use equally spaced levels including both endpoints.
    step = 2 * bound / (levels - 1)
    position = ((x + bound) / step).clamp(0, levels - 1)  # 每个值在网格上的"连续位置"（带小数）
    if stochastic:
        lower = position.floor()           # 下一级
        probability = position - lower     # 到上级的距离比例 = 选上级的概率
        draw = torch.rand(position.shape, device=x.device, dtype=x.dtype, generator=generator)
        indices = lower + (draw < probability).to(x.dtype)  # 按概率随机跳一级
    else:
        indices = position.round()
    return -bound + indices * step  # 把级号映射回具体数值
