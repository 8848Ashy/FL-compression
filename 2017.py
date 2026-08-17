import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import math
import sys
from pathlib import Path
import matplotlib.pyplot as plt  # 导入画图库
from datetime import datetime

OUTPUT_DIR = Path(__file__).resolve().parent / "figures_2017"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子保证对比的绝对公平
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 实验开关配置
# ==========================================
# RUN_FULL_EXPERIMENT = False：主程序只运行四个等价性验证后正常退出（轻量验证模式）；
# RUN_FULL_EXPERIMENT = True ：四个验证都通过后，继续运行完整实验。
RUN_FULL_EXPERIMENT = True

# 聚焦三算法（Original / SRK / Kashin）对比实验开关与参数
# 当 FOCUS_ON_KASHIN = True 时：
#   - 只运行 Original / SRK / Kashin 三种 Δw 方法，不运行 SK / SVK；
#   - 第一阶段固定通信预算 FIXED_BITS（FIXED_K_LEVELS = 2 ** FIXED_BITS）；
#   - 第二阶段按 TRADEOFF_BITS 扫描，每点跑 TRADEOFF_ROUNDS 轮；
#   - 生成两张新图（收敛曲线 + trade-off 曲线）；
#   - 旧五/四算法流程保留在主程序的 else 分支（FOCUS_ON_KASHIN=False 时运行）。
FOCUS_ON_KASHIN = True
FIXED_BITS = 2
FIXED_K_LEVELS = 2 ** FIXED_BITS
NUM_ROUNDS_FOCUS = 8
TRADEOFF_BITS = [1, 2, 3]
TRADEOFF_ROUNDS = 8

# 调试开关：在量化前打印各方法向量的动态范围（xmin / xmax / range / max_abs）
# True = 打印；False = 不打印。仅增加日志输出，不改变任何算法逻辑。
DEBUG_RANGE = True

# 分布诊断开关（Kashin + Lloyd-Max 研究第一步：只分析变换后系数的分布）。
# True 时主程序只运行 analyze_transform_coefficient_distribution()：
#   - 不实现 Lloyd-Max、不修改量化函数、不改任何聚合算法；
#   - 不运行完整联邦训练、不绘制准确率 / trade-off 图；
#   - 跑完自动把本开关恢复为 False 后退出。
RUN_DISTRIBUTION_TEST = False

# Lloyd-Max 独立测试开关；测试完成后保持 False，不自动进入。
RUN_LLOYD_MAX_TEST = False

# 分布诊断使用相对冗余度选择 Kashin 的 D，故不再与 SRK 的 65536 绑定。
# 例如 1.10 表示 D≈1.10d；D 可以不是 2 的幂。
KASHIN_REDUNDANCY_SETTINGS = [1.10, 1.25, 1.50, 2.00]

# ==========================================
# 1. 定义网络模型：简单的多层感知机 (MLP)
# ==========================================
class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)

# ==========================================
# 2. 准备数据：分发数据至 10 个客户端
# ==========================================
print("【数据准备】正在加载 MNIST 数据集并分发至 10 个客户端...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

num_clients = 10
client_loaders = []
images_per_client = 600 # 每个客户端分 600 张图

for i in range(num_clients):
    start_idx = i * images_per_client
    end_idx = start_idx + images_per_client
    subset = Subset(train_dataset, list(range(start_idx, end_idx)))
    loader = DataLoader(subset, batch_size=32, shuffle=True)
    client_loaders.append(loader)

test_subset = Subset(test_dataset, list(range(1000)))
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
print("【数据准备】数据准备完毕！")

# ==========================================
# 辅助函数：状态字典（state_dict）的展平与还原
# ==========================================
def flatten_state_dict(state_dict):
    tensors = []
    shapes = {}
    for key, val in state_dict.items():
        tensors.append(val.flatten())
        shapes[key] = val.shape
    flat_tensor = torch.cat(tensors)
    return flat_tensor, shapes

def unflatten_state_dict(flat_tensor, shapes):
    new_dict = {}
    idx = 0
    for key, shape in shapes.items():
        numel = int(np.prod(shape))
        new_dict[key] = flat_tensor[idx:idx+numel].view(shape)
        idx += numel
    return new_dict

# ==========================================
# 辅助函数：快速哈达玛变换 (FWHT)
# ==========================================
def fast_walsh_hadamard_transform(tensor):
    x = tensor.clone()
    n = x.shape[0]
    m = int(np.log2(n))
    for i in range(m):
        step = 2 ** i
        for j in range(0, n, 2 * step):
            for k in range(step):
                u = x[j + k].clone()
                v = x[j + k + step].clone()
                x[j + k] = u + v
                x[j + k + step] = u - v
    return x / np.sqrt(n)

# ==========================================
# 核心算子：随机 k 级量化函数
# ==========================================
def stochastic_k_level_quantize(X_tensor, k_levels):
    X_max = torch.max(X_tensor)
    X_min = torch.min(X_tensor)
    s_i = X_max - X_min
    
    if s_i < 1e-8:
        return X_tensor, torch.zeros_like(X_tensor, dtype=torch.long)
        
    normalized = (X_tensor - X_min) / s_i * (k_levels - 1)
    r = torch.floor(normalized).long()
    r = torch.clamp(r, 0, k_levels - 2)
    
    prob = normalized - r
    rand_val = torch.rand_like(X_tensor)
    is_upper = (rand_val < prob).float()
    
    quantized_r = r.float() + is_upper
    Y_tensor = X_min + (quantized_r * s_i) / (k_levels - 1)
    
    return Y_tensor, quantized_r.long()


def lloyd_max_quantize(X_tensor, k_levels, iterations=20, tolerance=1e-6):
    """Lloyd-Max 非均匀标量量化，不改变输入张量。

    返回：量化结果、量化中心、区间边界。中心在样本密集区域会更集中。
    """
    if X_tensor.ndim != 1:
        raise ValueError("lloyd_max_quantize 只接受一维张量")
    if k_levels < 2:
        raise ValueError("k_levels 必须不小于 2")

    x = X_tensor.detach()
    xmin, xmax = x.min(), x.max()
    if (xmax - xmin).abs() < 1e-12:
        centers = x.repeat(k_levels)
        boundaries = torch.full((k_levels - 1,), xmin, dtype=x.dtype, device=x.device)
        return x.clone(), centers, boundaries

    centers = torch.linspace(xmin, xmax, k_levels, dtype=x.dtype, device=x.device)
    for _ in range(iterations):
        boundaries = (centers[:-1] + centers[1:]) / 2.0
        indices = torch.bucketize(x, boundaries)
        new_centers = centers.clone()
        for level in range(k_levels):
            selected = x[indices == level]
            if selected.numel() > 0:
                new_centers[level] = selected.mean()
        new_centers, _ = torch.sort(new_centers)
        if torch.max(torch.abs(new_centers - centers)) <= tolerance * (xmax - xmin):
            centers = new_centers
            break
        centers = new_centers

    boundaries = (centers[:-1] + centers[1:]) / 2.0
    indices = torch.bucketize(x, boundaries)
    return centers[indices], centers, boundaries


def lloyd_max_codebook(X_tensor, k_levels, iterations=20):
    """只拟合共享 Lloyd-Max 码本。客户端和服务器可复用同一 centers。"""
    _, centers, _ = lloyd_max_quantize(X_tensor, k_levels, iterations)
    return centers


def quantize_with_codebook(X_tensor, centers):
    """使用共享码本量化，返回量化值和索引。"""
    boundaries = (centers[:-1] + centers[1:]) / 2.0
    indices = torch.bucketize(X_tensor, boundaries)
    return centers[indices], indices

# ==========================================
# 调试工具：量化前的动态范围累计与汇总（仅日志，不改变任何算法逻辑）
# ==========================================
_RANGE_ACC = {}   # tag -> list[(range, max_abs)]，量化前向量的动态范围累计

def _accumulate_range(tag, tensor):
    """收集量化前向量的动态范围 (xmax-xmin) 与 max_abs，供实验结束统一汇总打印。
    仅供调试比较 SRK(Hadamard Rotation) 与 Kashin Transform 在量化前的动态范围。"""
    xmin = tensor.min().item()
    xmax = tensor.max().item()
    _RANGE_ACC.setdefault(tag, []).append((xmax - xmin, tensor.abs().max().item()))

def _print_range_summary():
    """实验结束时打印一张汇总表：各方法量化前 range (xmax-xmin) 与 max_abs 的统计。"""
    if not _RANGE_ACC:
        print("[debug] 无量化前范围数据（DEBUG_RANGE 未开启？）")
        return
    print("\n" + "="*100)
    print("量化前动态范围 (xmax-xmin) 汇总 (mean over all clients & rounds)")
    print("="*100)
    print(f"  {'method':20s} | {'count':>5s} | {'mean_range':>12s} | {'std_range':>10s} | "
          f"{'min_range':>10s} | {'max_range':>10s} | {'mean_max_abs':>12s}")
    print("-"*100)
    for tag in ("Original delta_w", "SRK transformed", "Kashin coefficient"):
        if tag not in _RANGE_ACC:
            continue
        arr = np.array(_RANGE_ACC[tag])
        rng = arr[:, 0]
        mabs = arr[:, 1]
        print(f"  {tag:20s} | {len(rng):5d} | {rng.mean():12.6e} | {rng.std():10.6e} | "
              f"{rng.min():10.6e} | {rng.max():10.6e} | {mabs.mean():12.6e}")
    print("="*100)

# ==========================================
# 3. 客户端本地训练函数
# ==========================================
def local_train(global_model, dataloader, epochs=2, lr=0.05):
    local_model = copy.deepcopy(global_model)
    optimizer = optim.SGD(local_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    local_model.train()
    for epoch in range(epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    return local_model.state_dict()

# ==========================================
# 3.5  state_dict 加减与"本地训练返回更新量 Δw"辅助函数
# ==========================================
def state_dict_subtract(local_state, global_state):
    """返回 delta_state：对每个 key，delta[key] = local_state[key] - global_state[key]（逐元素）。
    不原地修改任何输入。"""
    delta_state = {}
    for key in local_state.keys():
        delta_state[key] = local_state[key] - global_state[key]
    return delta_state

def state_dict_add(global_state, delta_state):
    """返回 updated_state：对每个 key，updated[key] = global_state[key] + delta_state[key]（逐元素）。
    不原地修改任何输入。"""
    updated_state = {}
    for key in global_state.keys():
        updated_state[key] = global_state[key] + delta_state[key]
    return updated_state

def local_train_delta(global_model, dataloader, epochs=2, lr=0.05):
    """在 global_model 上本地训练，但返回"模型更新量" Δw = w_local - w_global，而不是完整 local_state。
    先深拷贝保存 global_model 当前 state_dict，再调用现有 local_train 得到 local_state，
    最后返回 state_dict_subtract(local_state, global_state_copy)。"""
    global_state_copy = copy.deepcopy(global_model.state_dict())
    local_state = local_train(global_model, dataloader, epochs=epochs, lr=lr)
    return state_dict_subtract(local_state, global_state_copy)

# ==========================================
# 4. 评估函数
# ==========================================
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ==========================================
# 5. 三大算法联邦聚合运行器
# ==========================================
def federated_round_sk(global_model, client_weights_list, k_levels):
    aggregated_flat = None
    template_keys = client_weights_list[0].keys()
    
    for idx, local_dict in enumerate(client_weights_list):
        flat, shapes = flatten_state_dict(local_dict)
        quantized_flat, _ = stochastic_k_level_quantize(flat, k_levels)
        
        if aggregated_flat is None:
            aggregated_flat = quantized_flat
        else:
            aggregated_flat += quantized_flat
            
    aggregated_flat /= len(client_weights_list)
    new_state_dict = unflatten_state_dict(aggregated_flat, shapes)
    global_model.load_state_dict(new_state_dict)

    # 统一通信口径：每个客户端、每个原始参数维度的通信 bit 数
    flat, _ = flatten_state_dict(client_weights_list[0])
    d_size = flat.shape[0]
    # 量化索引需 ceil(log2(k)) bit/维 + 每客户端发送 (X_min, X_max) 两个 float32 (64 bit) 元数据并摊销到各维
    bits_per_dim = int(np.ceil(np.log2(k_levels))) + 64.0 / d_size
    return bits_per_dim

def federated_round_srk(global_model, client_weights_list, k_levels):
    aggregated_rotated = None
    template_keys = client_weights_list[0].keys()
    
    flat_test, shapes = flatten_state_dict(client_weights_list[0])
    d_orig = flat_test.shape[0]
    d_pow2 = int(2 ** np.ceil(np.log2(d_orig)))
    
    D = (torch.rand(d_pow2) < 0.5).float() * 2.0 - 1.0
    
    for idx, local_dict in enumerate(client_weights_list):
        flat, _ = flatten_state_dict(local_dict)
        padded = torch.zeros(d_pow2)
        padded[:d_orig] = flat
        
        rotated = fast_walsh_hadamard_transform(padded * D)
        quantized_rotated, _ = stochastic_k_level_quantize(rotated, k_levels)
        
        if aggregated_rotated is None:
            aggregated_rotated = quantized_rotated
        else:
            aggregated_rotated += quantized_rotated
            
    aggregated_rotated /= len(client_weights_list)
    reconstructed_padded = fast_walsh_hadamard_transform(aggregated_rotated) * D
    reconstructed_flat = reconstructed_padded[:d_orig]
    new_state_dict = unflatten_state_dict(reconstructed_flat, shapes)
    global_model.load_state_dict(new_state_dict)

    # 统一通信口径：每个客户端、每个原始参数维度的通信 bit 数
    # 哈达玛变换把信息铺展到 d_pow2 维，量化后 d_pow2 个分量全部需要传输，
    # 再统一折算回原始 d_orig 维/客户端；另 +64 bit 元数据 (X_min, X_max)
    # 注意：随机符号矩阵 D 为各客户端共享（同一随机种子生成），其开销视为共享协议开销、摊销后不计入。
    bits_per_dim = (d_pow2 * int(np.ceil(np.log2(k_levels))) + 64.0) / d_orig
    return bits_per_dim

def federated_round_svk(global_model, client_weights_list, k_levels):
    aggregated_flat = None
    template_keys = client_weights_list[0].keys()
    total_entropy_bits = 0
    
    for idx, local_dict in enumerate(client_weights_list):
        flat, shapes = flatten_state_dict(local_dict)
        quantized_flat, bin_indices = stochastic_k_level_quantize(flat, k_levels)
        
        if aggregated_flat is None:
            aggregated_flat = quantized_flat
        else:
            aggregated_flat += quantized_flat
            
        _, counts = torch.unique(bin_indices, return_counts=True)
        probs = counts.float() / flat.shape[0]
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10)).item()
        
        d_size = flat.shape[0]
        metadata_bits = k_levels * np.log2((d_size + k_levels) / k_levels)
        total_entropy_bits += d_size * entropy + metadata_bits + 64
        
    aggregated_flat /= len(client_weights_list)
    new_state_dict = unflatten_state_dict(aggregated_flat, shapes)
    global_model.load_state_dict(new_state_dict)

    # 统一通信口径：每个客户端、每个原始参数维度的通信 bit 数
    # 每个客户端 = d_size * entropy(经验熵) + metadata_bits(计数编码) + 64(元数据)，
    # 再除以客户端数与 d_size 得到"每客户端每原始维度"的平均 bit 数。
    avg_bits_per_dim = total_entropy_bits / (len(client_weights_list) * d_size)
    return avg_bits_per_dim

# ==========================================
# 6. 原始联邦聚合函数（不进行量化压缩）
# ==========================================
def federated_round_original(global_model, client_weights_list):
    aggregated_weights = None

    for local_dict in client_weights_list:
        if aggregated_weights is None:
            aggregated_weights = copy.deepcopy(local_dict)
        else:
            for key in aggregated_weights.keys():
                aggregated_weights[key] += local_dict[key]

    for key in aggregated_weights.keys():
        aggregated_weights[key] /= len(client_weights_list)

    global_model.load_state_dict(aggregated_weights)

    # 统一通信口径：每个客户端、每个原始参数维度的通信 bit 数
    # 原始联邦直接传输 float32 权重，即每维度 32 bit/客户端，无任何压缩。
    return 32.0

# ==========================================
# 6.1  无压缩"模型更新量 Δw"联邦聚合函数
# ==========================================
def federated_round_original_update(global_model, client_deltas):
    """无压缩更新量聚合：对所有客户端的 delta_state_dict 按参数逐元素平均后加回全局模型。
    1) 对所有客户端的 Δw 逐元素平均；
    2) 读取 global_model 当前 state_dict；
    3) 用 state_dict_add(global_state, average_delta) 得到新参数；
    4) load_state_dict 更新 global_model；
    5) 通信口径与原始联邦一致：每维度 32 bit/客户端（float32 原样上传 Δw），无压缩。
    """
    aggregated_delta = None
    for idx, delta_dict in enumerate(client_deltas):
        # 调试：累计原始 delta_w 量化前的动态范围（仅日志，实验结束统一汇总）
        if DEBUG_RANGE:
            dbg_flat, _ = flatten_state_dict(delta_dict)
            _accumulate_range("Original delta_w", dbg_flat)
        if aggregated_delta is None:
            aggregated_delta = copy.deepcopy(delta_dict)
        else:
            for key in aggregated_delta.keys():
                aggregated_delta[key] += delta_dict[key]

    for key in aggregated_delta.keys():
        aggregated_delta[key] /= len(client_deltas)

    global_state = global_model.state_dict()
    updated_state = state_dict_add(global_state, aggregated_delta)
    global_model.load_state_dict(updated_state)
    return 32.0

# ==========================================
# 6.1.1  SRK 的"模型更新量 Δw 压缩上传"联邦聚合函数
# ==========================================
def federated_round_srk_update(global_model, client_deltas, k_levels, rotation_seed=None):
    """SRK 旋转量化压缩的"模型更新量 Δw 上传"版本（第一阶段 SRK 现使用此版本）。

    与旧 federated_round_srk 的差异（旧函数保留，用于对照）：
      - 压缩对象已从完整模型参数改为更新量 Δw：每个客户端上传 local_train_delta
        得到的 delta_state_dict，服务器平均"压缩更新量"后加回全局模型；
      - 随机 ±1 符号向量 D 为一整轮共享：所有客户端用同一个 D 压缩，服务器逆变换
        也用同一个 D。若 rotation_seed 非空，用独立 torch.Generator 复现 D，
        不扰动全局 RNG；为 None 时保持旧版随机行为（消耗全局 RNG）；
      - 正/逆哈达玛变换使用 _fwht_fast，它与旧版 fast_walsh_hadamard_transform
        数学等价（归一化同为 1/sqrt(n)），仅用于加速；
      - 返回的是"每客户端、每原始参数维度"的上行通信 bit。

    流程：
      1) client_deltas 中每个元素是长度为 d 的 delta_state_dict；
      2) 展平每个 delta；
      3) 计算 d_orig 与 d_pow2；
      4) 为一整轮生成同一个随机 ±1 符号向量 D；
      5) 每个客户端：delta_flat -> 零填充至 d_pow2 -> _fwht_fast(padded * D)
         -> stochastic_k_level_quantize；
      6) 服务器平均量化后的旋转系数；
      7) 用 _fwht_fast 逆变换聚合结果并乘 D，截取前 d_orig 个元素，得到平均压缩更新量 delta_bar；
      8) unflatten_state_dict(delta_bar, shapes)；
      9) 读取 global_model 当前 state_dict，用 state_dict_add(global_state, average_delta) 加回更新；
      10) load_state_dict 更新全局模型。
    """
    aggregated_rotated = None

    flat_test, shapes = flatten_state_dict(client_deltas[0])
    d_orig = flat_test.shape[0]
    d_pow2 = int(2 ** np.ceil(np.log2(d_orig)))

    # 为一整轮生成同一个随机 ±1 符号向量 D（所有客户端共享，服务器逆变换复用同一 D）
    if rotation_seed is not None:
        gen = torch.Generator()
        gen.manual_seed(rotation_seed)
        D = (torch.rand(d_pow2, generator=gen) < 0.5).float() * 2.0 - 1.0
    else:
        D = (torch.rand(d_pow2) < 0.5).float() * 2.0 - 1.0

    for idx, delta_dict in enumerate(client_deltas):
        flat, _ = flatten_state_dict(delta_dict)
        padded = torch.zeros(d_pow2)
        padded[:d_orig] = flat

        rotated = _fwht_fast(padded * D)
        # 调试：累计 Hadamard rotation 之后、随机量化之前的动态范围（仅日志，实验结束统一汇总）
        if DEBUG_RANGE:
            _accumulate_range("SRK transformed", rotated)
        quantized_rotated, _ = stochastic_k_level_quantize(rotated, k_levels)

        if aggregated_rotated is None:
            aggregated_rotated = quantized_rotated
        else:
            aggregated_rotated += quantized_rotated

    aggregated_rotated /= len(client_deltas)
    reconstructed_padded = _fwht_fast(aggregated_rotated) * D
    delta_bar = reconstructed_padded[:d_orig]

    average_delta = unflatten_state_dict(delta_bar, shapes)
    global_state = global_model.state_dict()
    updated_state = state_dict_add(global_state, average_delta)
    global_model.load_state_dict(updated_state)

    # 统一通信口径：每客户端、每原始参数维度的上行通信 bit
    # 哈达玛变换把信息铺展到 d_pow2 维，量化后 d_pow2 个分量全部需要传输，
    # 再统一折算回原始 d_orig 维/客户端；另 +64 bit 元数据 (X_min, X_max)。
    # 随机符号矩阵 D 为各客户端共享，其开销视为共享协议开销、摊销后不计入。
    bits_per_dim = (d_pow2 * int(np.ceil(np.log2(k_levels))) + 64.0) / d_orig
    return bits_per_dim

# ==========================================
# 6.2  轻量等价性验证：平均完整本地模型 vs 平均 Δw 后加回全局模型
# ==========================================
def verify_fedavg_update_equivalence():
    """验证在无压缩条件下，“直接平均完整本地模型参数”与“平均 Δw 后加回全局模型”是否等价。
    - 只用前 2 个客户端、同一个初始全局模型；
    - 两条路径本地训练前均显式重置随机种子（torch 与 numpy 同为 seed=0），
      保证本地训练顺序与随机性一致；
    - 验证前后保存/恢复调用方的随机状态，避免影响主实验流程的随机性；
    - 比较两条路径的全部参数，打印最大绝对差 max_abs_diff；
    - max_abs_diff < 1e-6 打印 [PASS] 并返回 True，否则打印 [FAIL] 返回 False。
    """
    print("\n" + "="*80)
    print("【等价性验证】直接平均完整本地模型 vs 平均 Δw 后加回全局模型（无压缩）")
    print("="*80)

    # 保存调用方（主流程）随机状态，验证结束恢复，避免污染主实验
    saved_torch_rng = torch.get_rng_state()
    saved_np_rng = np.random.get_state()

    epochs = 2
    initial_model = MNIST_MLP()

    # ---- 路径 A：直接平均两个完整 local_state ----
    torch.manual_seed(0)
    np.random.seed(0)
    local_states = [
        local_train(initial_model, client_loaders[0], epochs=epochs),
        local_train(initial_model, client_loaders[1], epochs=epochs),
    ]
    avg_full = copy.deepcopy(local_states[0])
    for key in avg_full.keys():
        avg_full[key] = (local_states[0][key] + local_states[1][key]) / 2.0

    # ---- 路径 B：同一初始全局模型，平均 Δw 后加回 ----
    model_b = copy.deepcopy(initial_model)
    torch.manual_seed(0)
    np.random.seed(0)
    deltas = [
        local_train_delta(model_b, client_loaders[0], epochs=epochs),
        local_train_delta(model_b, client_loaders[1], epochs=epochs),
    ]
    b_bits = federated_round_original_update(model_b, deltas)
    state_B = model_b.state_dict()

    # ---- 比较两条路径的所有参数 ----
    max_abs_diff = 0.0
    for key in avg_full.keys():
        diff = (avg_full[key] - state_B[key]).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)

    print(f"  federated_round_original_update 返回的通信量 = {b_bits} Bits/维/客户端")
    print(f"  全部参数最大绝对差 max_abs_diff = {max_abs_diff:.6e}")

    # 恢复调用方的随机状态
    torch.set_rng_state(saved_torch_rng)
    np.random.set_state(saved_np_rng)

    if max_abs_diff < 1e-6:
        print("  [PASS] 平均 Δw 后加回全局模型 与 直接平均完整本地模型 完全等价。")
        return True
    else:
        print("  [FAIL] 两条路径不等价。")
        return False

# ==========================================
# 6.2.1  轻量验证：第一阶段 Original 采用 Δw 管线后的等价性（更贴近主循环）
# ==========================================
def verify_first_round_original_update_path():
    """模拟第一阶段第一轮 Original (No Compression) 的 Δw 上传管线，验证其与
    原始"直接平均完整 local state_dict"流程的结果等价。

    只用前 2 个客户端、训练 1 个 epoch（比 verify_fedavg_update_equivalence 的
    2 个 epoch 更贴近第一阶段主循环的训练节奏）：
      - 路径 A：直接平均两个完整 local state_dict（老流程 federated_round_original 的做法）；
      - 路径 B：local_train_delta + federated_round_original_update（第一阶段主循环新做法）；
      - 两条路径训练前都重置到同一个独立随机种子（seed=0），保证训练顺序与随机性完全一致；
      - 验证前后保存/恢复调用方的 torch 与 numpy 随机状态，不扰动主流程的随机性；
      - 打印全部参数的最大绝对差；< 1e-6 打印 [PASS] 返回 True，否则 [FAIL] 返回 False。
    """
    print("\n" + "="*80)
    print("【轻量验证】第一阶段 Original Δw 管线 与 直接平均完整本地模型 等价（1轮/2客户端/1epoch）")
    print("="*80)

    saved_torch_rng = torch.get_rng_state()
    saved_np_rng = np.random.get_state()

    epochs = 1
    initial_model = MNIST_MLP()

    # ---- 路径 A：直接平均两个完整 local_state ----
    torch.manual_seed(0)
    np.random.seed(0)
    local_states = [
        local_train(initial_model, client_loaders[0], epochs=epochs),
        local_train(initial_model, client_loaders[1], epochs=epochs),
    ]
    avg_full = copy.deepcopy(local_states[0])
    for key in avg_full.keys():
        avg_full[key] = (local_states[0][key] + local_states[1][key]) / 2.0

    # ---- 路径 B：local_train_delta + federated_round_original_update（主循环新做法） ----
    model_b = copy.deepcopy(initial_model)
    torch.manual_seed(0)
    np.random.seed(0)
    deltas = [
        local_train_delta(model_b, client_loaders[0], epochs=epochs),
        local_train_delta(model_b, client_loaders[1], epochs=epochs),
    ]
    b_bits = federated_round_original_update(model_b, deltas)
    state_B = model_b.state_dict()

    # ---- 比较全部参数 ----
    max_abs_diff = 0.0
    for key in avg_full.keys():
        diff = (avg_full[key] - state_B[key]).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)

    print(f"  federated_round_original_update 返回的通信量 = {b_bits} Bits/维/客户端")
    print(f"  全部参数最大绝对差 max_abs_diff = {max_abs_diff:.6e}")

    torch.set_rng_state(saved_torch_rng)
    np.random.set_state(saved_np_rng)

    if max_abs_diff < 1e-6:
        print("  [PASS] 第一阶段 Original Δw 管线 与 直接平均完整本地模型 完全等价。")
        return True
    else:
        print("  [FAIL] 两条路径不等价。")
        return False

# ==========================================
# 6.2.2  轻量验证：SRK 的 Δw 压缩上传管线结构验证
# ==========================================
def verify_srk_update_pipeline():
    """验证 SRK 的"Δw 压缩上传"管线的结构正确性（不要求与无压缩 Original 参数相同，
    因为 SRK 含随机量化误差，只需管线本身构造正确）。

    仅用前 2 个客户端、1 epoch、一个干净初始模型：
      1) 用 local_train_delta 生成两个 delta；
      2) 调用 federated_round_srk_update(..., k_levels=4, rotation_seed=2026)；
      3) 验证并打印：
         - 输出模型所有参数有限（无 NaN/Inf）；
         - 恢复后的平均更新量维度 == 原始参数维度 d；
         - 通信量 == 公式 (d_pow2*ceil(log2(k)) + 64) / d_orig；
         - SRK 与 Kashin 在 k=4、D=d_pow2=65536 时的理论通信量差值 < 1e-12。
    验证前后保存/恢复调用方 torch 与 numpy 随机状态。
    """
    print("\n" + "="*80)
    print("【轻量验证】SRK Δw 压缩上传管线 结构验证（1轮/2客户端/1epoch, k=4, rotation_seed=2026）")
    print("="*80)

    saved_torch_rng = torch.get_rng_state()
    saved_np_rng = np.random.get_state()

    epochs = 1
    k_levels = 4
    rotation_seed = 2026
    initial_model = MNIST_MLP()
    model_b = copy.deepcopy(initial_model)

    # 干净初始模型上训练前 2 个客户端，各得到 Δw
    torch.manual_seed(0)
    np.random.seed(0)
    deltas = [
        local_train_delta(model_b, client_loaders[0], epochs=epochs),
        local_train_delta(model_b, client_loaders[1], epochs=epochs),
    ]

    model_dim_d = sum(p.numel() for p in model_b.parameters())
    d_orig = model_dim_d
    d_pow2 = int(2 ** np.ceil(np.log2(d_orig)))
    initial_state = copy.deepcopy(model_b.state_dict())

    bits = federated_round_srk_update(model_b, deltas, k_levels=k_levels, rotation_seed=rotation_seed)
    updated_state = model_b.state_dict()

    # ---- 1) 输出模型参数有限性（无 NaN/Inf） ----
    all_finite = True
    for key in updated_state:
        if not torch.isfinite(updated_state[key]).all():
            all_finite = False
            break

    # ---- 2) 恢复后的平均更新量维度 == 原始参数维度 d ----
    applied_delta = state_dict_subtract(updated_state, initial_state)
    flat_applied, _ = flatten_state_dict(applied_delta)
    d_check = flat_applied.shape[0]
    dim_ok = (d_check == d_orig)

    # ---- 3) 通信量 == 公式值 ----
    formula_bits = (d_pow2 * int(np.ceil(np.log2(k_levels))) + 64.0) / d_orig
    bits_ok = (abs(bits - formula_bits) < 1e-12)

    # ---- 4) SRK 与 Kashin 在 k=4、D=d_pow2=65536 时的理论通信量差值 ----
    D_kashin = 65536
    kashin_bits = (D_kashin * int(np.ceil(np.log2(k_levels))) + 64.0) / d_orig
    srk_kashin_diff = abs(bits - kashin_bits)

    print(f"  模型参数维度 d_orig          = {d_orig}")
    print(f"  零填充维度 d_pow2            = {d_pow2}")
    print(f"  1) 输出模型参数全有限        = {all_finite}")
    print(f"  2) 恢复更新量维度 {d_check} == d_orig {d_orig} : {dim_ok}")
    print(f"  3) 通信量 {bits:.6f} == 公式 {formula_bits:.6f} Bits/维/客户端 : {bits_ok}")
    print(f"  4) SRK/Kashin 理论通信量差值 = {srk_kashin_diff:.3e} (< 1e-12: {srk_kashin_diff < 1e-12})")
    print(f"     (SRK bits = {bits:.6f}, Kashin bits = {kashin_bits:.6f})")

    torch.set_rng_state(saved_torch_rng)
    np.random.set_state(saved_np_rng)

    if all_finite and dim_ok and bits_ok and (srk_kashin_diff < 1e-12):
        print("  [PASS] SRK Δw 压缩上传管线结构正确。")
        return True
    else:
        print("  [FAIL] SRK Δw 压缩上传管线结构有误。")
        return False

# ==========================================
# 6.3.2  轻量验证：Kashin 的 Δw 压缩上传管线结构验证
# ==========================================
def verify_kashin_update_pipeline():
    """验证 Kashin 的"Δw 压缩上传"管线的结构正确性（不要求与无压缩 Original 参数相同，
    因为 Kashin 含随机量化误差，只需管线本身构造正确）。

    仅用前 2 个客户端、1 epoch、一个干净初始模型：
      1) 创建共享 KashinFrame（d = 真实模型参数维度，D = 65536，seed = 2026）；
      2) 用 local_train_delta 得到两个客户端 delta；
      3) 调用 federated_round_kashin_update(..., k_levels=4, iterations=10)；
      4) 打印并验证：
         - 更新后模型所有参数有限（无 NaN/Inf）；
         - frame.frame_synthesis 输出维度 == 原始参数维度 d；
         - 通信量 == 公式 (D*ceil(log2(k)) + 64) / d；
         - Kashin 与 SRK 的理论通信量差值 < 1e-12；
         - Kashin 系数量化前最大绝对值的均值。
    验证前后保存/恢复调用方 torch 与 numpy 随机状态。
    """
    print("\n" + "="*80)
    print("【轻量验证】Kashin Δw 压缩上传管线 结构验证（1轮/2客户端/1epoch, k=4, iterations=10）")
    print("="*80)

    saved_torch_rng = torch.get_rng_state()
    saved_np_rng = np.random.get_state()

    epochs = 1
    k_levels = 4
    iterations = 10
    initial_model = MNIST_MLP()
    model_b = copy.deepcopy(initial_model)

    d = sum(p.numel() for p in model_b.parameters())
    D = 65536
    frame = KashinFrame(d, D, seed=2026)

    # 干净初始模型上训练前 2 个客户端，各得到 Δw
    torch.manual_seed(0)
    np.random.seed(0)
    deltas = [
        local_train_delta(model_b, client_loaders[0], epochs=epochs),
        local_train_delta(model_b, client_loaders[1], epochs=epochs),
    ]

    initial_state = copy.deepcopy(model_b.state_dict())

    # ---- Kashin 系数量化前最大绝对值的均值（每客户端系数最大 |a| 取平均） ----
    a_peaks = []
    for delta_dict in deltas:
        flat, _ = flatten_state_dict(delta_dict)
        a = kashin_solve(frame, flat, iterations)
        a_peaks.append(a.abs().max().item())
    mean_a_peak = sum(a_peaks) / len(a_peaks)

    bits = federated_round_kashin_update(model_b, deltas, k_levels=k_levels, frame=frame, iterations=iterations)
    updated_state = model_b.state_dict()

    # ---- 1) 更新后模型参数有限性（无 NaN/Inf） ----
    all_finite = True
    for key in updated_state:
        if not torch.isfinite(updated_state[key]).all():
            all_finite = False
            break

    # ---- 2) frame.frame_synthesis 输出维度 == 原始参数维度 d ----
    applied_delta = state_dict_subtract(updated_state, initial_state)
    flat_applied, _ = flatten_state_dict(applied_delta)
    d_check = flat_applied.shape[0]
    dim_ok = (d_check == d)

    # ---- 3) 通信量 == 公式值 ----
    formula_bits = (frame.D * int(np.ceil(np.log2(k_levels))) + 64.0) / d
    bits_ok = (abs(bits - formula_bits) < 1e-12)

    # ---- 4) Kashin 与 SRK 的理论通信量差值 ----
    d_pow2 = int(2 ** np.ceil(np.log2(d)))
    srk_bits = (d_pow2 * int(np.ceil(np.log2(k_levels))) + 64.0) / d
    kashin_srk_diff = abs(bits - srk_bits)

    print(f"  模型参数维度 d                = {d}")
    print(f"  框架系数维度 D                = {frame.D}")
    print(f"  D/d                           = {frame.D/d:.4f}")
    print(f"  1) 更新后模型参数全有限        = {all_finite}")
    print(f"  2) 重构更新量维度 {d_check} == d {d} : {dim_ok}")
    print(f"  3) 通信量 {bits:.6f} == 公式 {formula_bits:.6f} Bits/维/客户端 : {bits_ok}")
    print(f"  4) Kashin/SRK 理论通信量差值 = {kashin_srk_diff:.3e} (< 1e-12: {kashin_srk_diff < 1e-12})")
    print(f"     (Kashin bits = {bits:.6f}, SRK bits = {srk_bits:.6f})")
    print(f"  5) Kashin 系数量化前最大绝对值均值 = {mean_a_peak:.6f}")

    torch.set_rng_state(saved_torch_rng)
    np.random.set_state(saved_np_rng)

    if all_finite and dim_ok and bits_ok and (kashin_srk_diff < 1e-12):
        print("  [PASS] Kashin Δw 压缩上传管线结构正确。")
        return True
    else:
        print("  [FAIL] Kashin Δw 压缩上传管线结构有误。")
        return False

# ==========================================
# 6.3  Kashin 变换联邦聚合函数（第五种算法）
# ==========================================
def federated_round_kashin(global_model, client_weights_list, k_levels, frame, iterations=10):
    """Kashin 变换联邦聚合（第五种算法，聚合在 Kashin 系数域进行）。

    每个客户端：
      1) state_dict 展平为长度 d 的 flat；
      2) a = kashin_solve(frame, flat, iterations)：在共享冗余紧框架上解出长度 D 的 Kashin 系数；
      3) 对 a 用 stochastic_k_level_quantize 做随机 k 级量化；
      4) 服务器平均所有客户端量化后的系数（aggregated_a）；
      5) frame.frame_synthesis(aggregated_a) 重构回长度 d 的权重向量；
      6) unflatten_state_dict 恢复各层参数并写入 global_model。

    统一通信口径（每客户端、每原始参数维度的上行 bit 数）：
        bits_per_original_dim = (D * ceil(log2(k_levels)) + 64) / d
      - D 个量化索引，每索引需 ceil(log2(k_levels)) bit（与 SRK 中 d_pow2 同构）；
      - +64 bit 为该客户端量化所需的 (X_min, X_max) 两个 float32 元数据；
      - 框架的随机行索引 (indices) 与列符号 (column_signs) 由固定 seed 双方共享，
        属于共享协议开销、摊销后不计入每轮通信量。
    """
    aggregated_a = None
    flat, shapes = flatten_state_dict(client_weights_list[0])
    d_size = flat.shape[0]

    for idx, local_dict in enumerate(client_weights_list):
        flat_i, _ = flatten_state_dict(local_dict)
        a = kashin_solve(frame, flat_i, iterations)
        quantized_a, _ = stochastic_k_level_quantize(a, k_levels)

        if aggregated_a is None:
            aggregated_a = quantized_a
        else:
            aggregated_a += quantized_a

    aggregated_a /= len(client_weights_list)
    reconstructed_flat = frame.frame_synthesis(aggregated_a)
    new_state_dict = unflatten_state_dict(reconstructed_flat, shapes)
    global_model.load_state_dict(new_state_dict)

    bits_per_original_dim = (frame.D * int(np.ceil(np.log2(k_levels))) + 64.0) / d_size
    return bits_per_original_dim

# ==========================================
# 6.3.1  Kashin 的"模型更新量 Δw 压缩上传"联邦聚合函数（第一阶段 Kashin 使用）
# ==========================================
def federated_round_kashin_update(global_model, client_deltas, k_levels, frame, iterations=10):
    """Kashin 旋转压缩的"模型更新量 Δw 上传"版本（第一阶段 Kashin 现使用此版本）。

    与旧 federated_round_kashin 的差异（旧函数保留，用于对照）：
      - 压缩对象已从完整模型参数改为本地模型更新量 Δw：每个客户端上传
        local_train_delta 得到的 delta_state_dict，服务器平均"压缩更新量"后加回全局模型；
      - frame 是所有客户端与服务器共享的 KashinFrame（行索引与列符号由固定 seed
        双方复现），不能每轮重建；
      - kashin_solve 通过冗余紧框架和迭代截断，求出"能量被摊平"的均匀 Kashin 系数；
      - 返回的是"每客户端、每原始参数维度"的上行 bit 数。

    流程：
      1) client_deltas 中每个元素是 delta_state_dict；
      2) 展平每个 delta，原始维度为 d；
      3) a = kashin_solve(frame, delta_flat, iterations)：得到长度 D 的 Kashin 系数；
      4) 对 a 调用现有 stochastic_k_level_quantize 做随机 k 级量化；
      5) 服务器平均所有客户端的量化 Kashin 系数（aggregated_a）；
      6) average_delta_flat = frame.frame_synthesis(aggregated_a) 重构平均更新量；
      7) unflatten_state_dict 还原 average_delta_state；
      8) 读取 global_model 当前 state_dict；
      9) state_dict_add(global_state, average_delta_state) 加回更新量；
      10) load_state_dict 更新全局模型。

    通信量：bits_per_dim = (frame.D * ceil(log2(k_levels)) + 64.0) / d
      当 frame.D == d_pow2 时（此处 D=65536 恰等于 SRK 的 d_pow2=65536），Kashin 与
      SRK 在相同 k 下的通信量相同。
    """
    aggregated_a = None
    flat, shapes = flatten_state_dict(client_deltas[0])
    d_size = flat.shape[0]

    for idx, delta_dict in enumerate(client_deltas):
        flat_i, _ = flatten_state_dict(delta_dict)
        a = kashin_solve(frame, flat_i, iterations)
        # 调试：累计 Kashin 系数生成之后、随机量化之前的动态范围（仅日志，实验结束统一汇总）
        if DEBUG_RANGE:
            _accumulate_range("Kashin coefficient", a)
        quantized_a, _ = stochastic_k_level_quantize(a, k_levels)

        if aggregated_a is None:
            aggregated_a = quantized_a
        else:
            aggregated_a += quantized_a

    aggregated_a /= len(client_deltas)
    average_delta_flat = frame.frame_synthesis(aggregated_a)
    average_delta_state = unflatten_state_dict(average_delta_flat, shapes)

    global_state = global_model.state_dict()
    updated_state = state_dict_add(global_state, average_delta_state)
    global_model.load_state_dict(updated_state)

    # 统一通信口径：每客户端、每原始参数维度的上行 bit 数
    # D 个量化索引，每索引需 ceil(log2(k)) bit；+64 bit 元数据 (X_min, X_max)；
    # 框架行索引/列符号为共享协议开销，摊销后不计入。
    bits_per_dim = (frame.D * int(np.ceil(np.log2(k_levels))) + 64.0) / d_size
    return bits_per_dim


def federated_round_srk_lloyd_update(global_model, client_deltas, k_levels,
                                     centers, rotation_seed=None):
    """SRK + 共享归一化 Lloyd-Max：每客户端额外发送一个 scale。"""
    flat0, shapes = flatten_state_dict(client_deltas[0])
    d_orig = flat0.shape[0]
    d_pow2 = int(2 ** np.ceil(np.log2(d_orig)))
    gen = torch.Generator()
    gen.manual_seed(0 if rotation_seed is None else rotation_seed)
    signs = (torch.rand(d_pow2, generator=gen) < 0.5).float() * 2.0 - 1.0
    aggregated = None
    for delta in client_deltas:
        flat, _ = flatten_state_dict(delta)
        padded = torch.zeros(d_pow2)
        padded[:d_orig] = flat
        coeff = _fwht_fast(padded * signs)
        scale = coeff.abs().max().clamp_min(1e-12)
        quantized, _ = quantize_with_codebook(coeff / scale, centers)
        restored = quantized * scale
        aggregated = restored if aggregated is None else aggregated + restored
    aggregated /= len(client_deltas)
    delta_bar = (_fwht_fast(aggregated) * signs)[:d_orig]
    global_model.load_state_dict(state_dict_add(
        global_model.state_dict(), unflatten_state_dict(delta_bar, shapes)))
    return (d_pow2 * int(np.ceil(np.log2(k_levels))) + 32.0) / d_orig


def federated_round_kashin_lloyd_update(global_model, client_deltas, k_levels,
                                        frame, iterations=10, lloyd_iterations=20,
                                        shared_centers=None, include_codebook_bits=False):
    """Kashin + Lloyd-Max 的 Δw 聚合版本。

    shared_centers=None 时保留独立码本兼容模式；主实验应传入共享 centers。
    共享模式下所有客户端使用同一套量化中心。通信量可选择计入一次性码本：
        (D*ceil(log2(k)) + optional 32*k) / d
    这里不再额外发送 min/max；区间边界可由相邻中心计算。
    """
    aggregated_a = None
    flat, shapes = flatten_state_dict(client_deltas[0])
    d_size = flat.shape[0]

    for delta_dict in client_deltas:
        flat_i, _ = flatten_state_dict(delta_dict)
        a = kashin_solve(frame, flat_i, iterations)
        if DEBUG_RANGE:
            _accumulate_range("Kashin Lloyd coefficient", a)
        if shared_centers is None:
            quantized_a, _, _ = lloyd_max_quantize(a, k_levels, iterations=lloyd_iterations)
        else:
            scale = a.abs().max().clamp_min(1e-12)
            quantized_a, _ = quantize_with_codebook(a / scale, shared_centers)
            quantized_a = quantized_a * scale
        aggregated_a = quantized_a if aggregated_a is None else aggregated_a + quantized_a

    aggregated_a /= len(client_deltas)
    average_delta_flat = frame.frame_synthesis(aggregated_a)
    average_delta_state = unflatten_state_dict(average_delta_flat, shapes)
    updated_state = state_dict_add(global_model.state_dict(), average_delta_state)
    global_model.load_state_dict(updated_state)

    codebook_bits = 32.0 if shared_centers is not None else 32.0 * k_levels
    if not include_codebook_bits:
        codebook_bits = 32.0 if shared_centers is not None else 0.0
    return (frame.D * int(np.ceil(np.log2(k_levels))) + codebook_bits) / d_size


def verify_kashin_lloyd_update_pipeline():
    """用两个客户端做轻量验证，不运行完整实验。"""
    model = MNIST_MLP()
    frame = KashinFrame(sum(p.numel() for p in model.parameters()), D=65536, seed=2026)
    deltas = [local_train_delta(model, client_loaders[i], epochs=1) for i in range(2)]
    test_model = copy.deepcopy(model)
    bits = federated_round_kashin_lloyd_update(test_model, deltas, 4, frame, iterations=2)
    finite = all(torch.isfinite(value).all().item() for value in test_model.state_dict().values())
    print(f"Kashin + Lloyd-Max 更新管线：模型参数有限值={finite}，通信量={bits:.6f} Bits/原始维度")
    return finite and math.isfinite(bits)

# ==========================================
# 6.5  Kashin 变换独立单元测试（不接入联邦训练）
# -------------------------------------------------
# 说明：这是"冗余紧框架 (redundant tight frame) + Kashin 系数求解"，
#       并不是单纯的随机旋转矩阵。
#   - 随机部分 Hadamard + 随机列符号 冗余紧框架 U (d × D, D > d)：
#       从 D×D Hadamard 矩阵中随机抽取 d 行，再乘上长度为 D 的随机 ±1 列符号，
#       构成框架上界 A = D/d 的冗余紧框架，满足 U U^T = A I_d。
#       全程不构造/不存储任何稠密矩阵，只保存 d 个行索引与 D 个列符号，
#       两个线性操作都通过快速沃尔什-哈达玛变换在 O(D log D) 内完成。
#   - 行索引与列符号由同一固定随机种子生成，客户端/服务器可复现同一框架。
#   - canonical 重构检查：x = U ((U^T x) / A) 应精确成立（误差 < 1e-6），证明框架正确。
#   - Kashin 表示的意义：任意 x ∈ R^d 都能写成 x = U a，且系数 a 的最大绝对值
#       小于普通 U^T x / A 系数的最大绝对值（能量被摊平到 D 个系数上）。
#   - 单元测试用 20 个独立随机向量，同时统计重构误差与系数动态范围改善。
# ==========================================

def _fwht_fast(t):
    """向量化的快速沃尔什-哈达玛变换（数学等价于现有 fast_walsh_hadamard_transform，
    仅在速度上优化，仅供本 Kashin 测试使用，不影响 SRK 等现有算法）。返回 H t / sqrt(n)。"""
    n = t.shape[0]
    y = t.clone()
    h = 1
    while h < n:
        y2 = y.view(-1, 2 * h)
        a = y2[:, :h].clone()
        b = y2[:, h:].clone()
        y2[:, :h] = a + b
        y2[:, h:] = a - b
        h *= 2
    return y.view(n) / math.sqrt(n)


class KashinFrame:
    """随机部分正交变换 + 随机列符号的冗余紧框架。

    D 为任意正整数：D 为 2 的幂时使用 FWHT；否则使用 FFT 相位构造的
    实值正交循环变换。两种变换都不构造稠密 d×D 矩阵，并满足 U U^T=A I。
    """
    def __init__(self, d, D, seed=2026):
        self.d = d
        self.D = D
        self.A = D / d  # 框架上界 (frame bound)
        gen = torch.Generator()
        gen.manual_seed(seed)
        # 随机抽取 d 个 Hadamard 行索引（不放回）
        self.indices = torch.randperm(D, generator=gen)[:d]
        # 长度为 D 的随机 ±1 列符号
        self.column_signs = (torch.rand(D, generator=gen) < 0.5).float() * 2.0 - 1.0
        self.use_fwht = (D & (D - 1)) == 0
        if not self.use_fwht:
            # 单位模复相位保证 FFT 变换及其逆变换均保持能量；保存相位即可复现。
            phase_angle = 2.0 * math.pi * torch.rand(D // 2 + 1, generator=gen)
            # rfft 的 DC 与 Nyquist 分量必须保持实数，才能得到严格的实值正交变换。
            phase_angle[0] = 0.0
            if D % 2 == 0:
                phase_angle[-1] = 0.0
            self.fft_phase = torch.polar(torch.ones_like(phase_angle), phase_angle)

    def _forward_transform(self, x):
        if self.use_fwht:
            return _fwht_fast(x)
        spectrum = torch.fft.rfft(x)
        return torch.fft.irfft(spectrum * self.fft_phase.to(spectrum.device), n=self.D)

    def _transpose_transform(self, x):
        if self.use_fwht:
            return _fwht_fast(x)
        spectrum = torch.fft.rfft(x)
        phase = self.fft_phase.to(spectrum.device)
        return torch.fft.irfft(spectrum * phase.conj(), n=self.D)

    def frame_analysis(self, x):
        """分析操作：x (R^d) -> 系数 U^T x (R^D)。U^T x = sqrt(A) * cs ⊙ FWHT(v)，v[idx]=x。"""
        v = torch.zeros(self.D)
        v[self.indices] = x
        transformed = self._transpose_transform(v)
        return math.sqrt(self.A) * self.column_signs * transformed

    def frame_synthesis(self, a):
        """合成操作：系数 a (R^D) -> 向量 U a (R^d)。U a = sqrt(A) * FWHT(cs ⊙ a)[idx]。"""
        transformed = self._forward_transform(self.column_signs * a)
        return math.sqrt(self.A) * transformed[self.indices]


def kashin_solve(frame, x, iterations=10, clipping_level=None):
    """
    迭代截断的 Kashin 系数求解：
        分析 -> 截断 -> 合成 -> 更新残差
    返回系数向量 a (长度 D)，最终满足 x ≈ frame.frame_synthesis(a)。
    clipping_level：截断阈值；默认取 2*||x||2/sqrt(D)（Kashin 常数量级）。
    """
    if clipping_level is None:
        clipping_level = 2.0 * torch.norm(x) / math.sqrt(frame.D)
    a = torch.zeros(frame.D)
    r = x.clone()
    for _ in range(iterations):
        c = frame.frame_analysis(r) / frame.A            # 残差的最小范数表示系数
        c_clip = torch.clamp(c, -clipping_level, clipping_level)  # 截断到 ±clipping_level
        a = a + c_clip
        r = r - frame.frame_synthesis(c_clip)            # 更新残差
    return a


def kashin_unit_test():
    """在联邦训练之前执行的独立单元测试（20 个独立随机向量）。
    验证：框架正确（canonical 重构 < 1e-6）+ Kashin 重构正确（< 1e-4）
          + 系数动态范围改善（kashin_peak/naive_peak 中位数 < 1）。"""
    model = MNIST_MLP()
    d = sum(p.numel() for p in model.parameters())  # 模型展平参数维度
    D = 65536                                       # 框架系数维度（固定）
    frame = KashinFrame(d, D)

    num_trials = 20
    # 用独立随机源（不扰动主程序 seed=42 的复现性）
    rng = torch.Generator()
    rng.manual_seed(1234)

    canon_errs, kashin_errs = [], []
    naive_norm_peaks, kashin_norm_peaks, peak_ratios = [], [], []

    for _ in range(num_trials):
        x = torch.randn(d, generator=rng)
        x_norm = torch.norm(x).item()

        # 1) canonical 重构：x = U ((U^T x) / A)，证明框架自身正确
        Ux = frame.frame_analysis(x)
        x_canonical = frame.frame_synthesis(Ux / frame.A)
        canon_errs.append((torch.norm(x - x_canonical) / torch.norm(x)).item())

        # 2) 普通（最小范数）表示系数峰值
        naive_peak = (Ux / frame.A).abs().max().item()

        # 3) Kashin 系数 a（迭代截断求解）及其峰值
        a = kashin_solve(frame, x)
        kashin_peak = a.abs().max().item()

        # 4) Kashin 重构误差
        x_hat = frame.frame_synthesis(a)
        kashin_errs.append((torch.norm(x - x_hat) / torch.norm(x)).item())

        # 5) 归一化峰值（去掉向量范数影响，便于跨向量比较）与比值
        scale = math.sqrt(D) / x_norm
        naive_norm_peaks.append(naive_peak * scale)
        kashin_norm_peaks.append(kashin_peak * scale)
        peak_ratios.append(kashin_peak / naive_peak)

    max_canon_err = max(canon_errs)
    max_kashin_err = max(kashin_errs)
    med_naive_norm = float(np.median(naive_norm_peaks))
    med_kashin_norm = float(np.median(kashin_norm_peaks))
    med_ratio = float(np.median(peak_ratios))

    print(f"  d (模型展平参数维度)             = {d}")
    print(f"  D (框架系数维度)                 = {D}")
    print(f"  D/d (冗余度)                     = {D / d:.4f}")
    print(f"  独立随机向量数                   = {num_trials}")
    print(f"  最大 canonical 重构误差          = {max_canon_err:.6e}")
    print(f"  最大 Kashin 重构误差             = {max_kashin_err:.6e}")
    print(f"  naive_normalized_peak 中位数     = {med_naive_norm:.6e}")
    print(f"  kashin_normalized_peak 中位数    = {med_kashin_norm:.6e}")
    print(f"  kashin_peak / naive_peak 中位数  = {med_ratio:.4f}")

    ok_canon = max_canon_err < 1e-6
    ok_kashin = max_kashin_err < 1e-4
    ok_range = med_ratio < 1.0

    if ok_canon and ok_kashin and ok_range:
        print("  [PASS] 框架正确且系数动态范围得到压缩：重构误差达标，Kashin 峰值相对下降。")
        return True
    elif ok_canon and ok_kashin:
        print("  [INFO] 重构正确但未观察到动态范围改善（kashin_peak/naive_peak 中位数 >= 1）。")
        return False
    else:
        print("  [FAIL] 框架校验未通过：canonical 或 Kashin 重构误差超限。")
        return False

# ==========================================
# 6.6  聚焦三算法（Original / SRK / Kashin）Δw 压缩对比实验
# ==========================================
def build_shared_kashin_codebook(base_model, frame, k_levels):
    """用所有客户端的一次校准更新拟合共享 Lloyd-Max 码本。"""
    calibration = []
    for client_id in range(num_clients):
        delta = local_train_delta(base_model, client_loaders[client_id], epochs=1)
        flat, _ = flatten_state_dict(delta)
        calibration.append(kashin_solve(frame, flat, iterations=5))
    centers = lloyd_max_codebook(torch.cat(calibration), k_levels, iterations=20)
    return centers / centers.abs().max().clamp_min(1e-12)


def build_shared_srk_codebook(base_model, k_levels, d):
    """用校准更新拟合 SRK 的共享归一化 Lloyd-Max 码本。"""
    d_pow2 = int(2 ** np.ceil(np.log2(d)))
    gen = torch.Generator(); gen.manual_seed(2026)
    signs = (torch.rand(d_pow2, generator=gen) < 0.5).float() * 2.0 - 1.0
    values = []
    for loader in client_loaders:
        delta = local_train_delta(base_model, loader, epochs=1)
        flat, _ = flatten_state_dict(delta)
        padded = torch.zeros(d_pow2); padded[:d] = flat
        coeff = _fwht_fast(padded * signs)
        values.append(coeff / coeff.abs().max().clamp_min(1e-12))
    centers = lloyd_max_codebook(torch.cat(values), k_levels, iterations=20)
    return centers / centers.abs().max().clamp_min(1e-12)


def run_focus_kashin_experiment(timestamp):
    """聚焦四算法（Original / SRK / Kashin / Kashin+Lloyd-Max）的 Δw 对比实验。
    - 仅使用已实现的 Δw 上传管线（local_train_delta + *_update 聚合）；
    - SK、SVK 不参与本聚焦实验；
    - 图1：固定通信预算 (FIXED_BITS) 下的收敛曲线；
    - 图2：相同通信预算下 (TRADEOFF_BITS) 的 trade-off 曲线；
    - 旧五/四算法流程保留在主程序的 else 分支（FOCUS_ON_KASHIN=False 时运行）。
    """
    # 调试累计器清空：保证本次运行从零累计（仅日志）
    _RANGE_ACC.clear()

    print("\n" + "="*80)
    print("【聚焦实验】Original / SRK+Uniform / SRK+Lloyd / Kashin+Uniform / Kashin+Lloyd")
    print("="*80)

    # ---------- 共享 Kashin 框架（全实验复用同一个，seed 固定，绝不按轮/按客户端重建） ----------
    initial_model = MNIST_MLP()
    model_dim_d = sum(p.numel() for p in initial_model.parameters())
    kashin_frame = KashinFrame(model_dim_d, D=65536, seed=2026)
    print(f"共享 Kashin 框架: d = {model_dim_d}, D = {kashin_frame.D}, D/d = {kashin_frame.D/model_dim_d:.4f}")

    # ========== 图1：固定通信预算的收敛曲线 ==========
    print("\n" + "="*80)
    print(f"【聚焦-图1】固定通信预算 FIXED_BITS={FIXED_BITS} (k={FIXED_K_LEVELS})，{NUM_ROUNDS_FOCUS} 轮收敛曲线")
    print("="*80)

    # 从同一个 initial_model 克隆三个模型
    model_original = copy.deepcopy(initial_model)
    model_srk = copy.deepcopy(initial_model)
    model_srk_lloyd = copy.deepcopy(initial_model)
    model_kashin = copy.deepcopy(initial_model)
    model_kashin_lloyd = copy.deepcopy(initial_model)

    num_rounds = NUM_ROUNDS_FOCUS
    k_levels = FIXED_K_LEVELS
    print("正在用所有客户端的校准更新拟合共享 Lloyd-Max 码本...")
    shared_lloyd_centers = build_shared_kashin_codebook(initial_model, kashin_frame, k_levels)
    shared_srk_lloyd_centers = build_shared_srk_codebook(initial_model, k_levels, model_dim_d)
    print(f"共享 Lloyd-Max 码本中心数: {len(shared_lloyd_centers)}")

    history_original = []
    history_srk = []
    history_srk_lloyd = []
    history_kashin = []
    history_kashin_lloyd = []
    bits_original_list = []
    bits_srk_list = []
    bits_srk_lloyd_list = []
    bits_kashin_list = []
    bits_kashin_lloyd_list = []

    for r in range(num_rounds):
        print(f"\n==================== 第 {r+1} 轮联邦实验 ====================")
        local_deltas_original = []
        local_deltas_srk = []
        local_deltas_srk_lloyd = []
        local_deltas_kashin = []
        local_deltas_kashin_lloyd = []

        for client_id in range(num_clients):
            loader = client_loaders[client_id]
            delta_original = local_train_delta(model_original, loader, epochs=2)
            delta_srk = local_train_delta(model_srk, loader, epochs=2)
            delta_srk_lloyd = local_train_delta(model_srk_lloyd, loader, epochs=2)
            delta_kashin = local_train_delta(model_kashin, loader, epochs=2)
            delta_kashin_lloyd = local_train_delta(model_kashin_lloyd, loader, epochs=2)
            local_deltas_original.append(delta_original)
            local_deltas_srk.append(delta_srk)
            local_deltas_srk_lloyd.append(delta_srk_lloyd)
            local_deltas_kashin.append(delta_kashin)
            local_deltas_kashin_lloyd.append(delta_kashin_lloyd)

        # Original：无压缩更新量聚合
        b_original = federated_round_original_update(model_original, local_deltas_original)
        acc_original = evaluate_model(model_original, test_loader)
        history_original.append(acc_original)
        bits_original_list.append(b_original)

        # SRK：旋转哈达玛 + 更新量量化压缩，每轮 D 可复现且不同
        b_srk = federated_round_srk_update(model_srk, local_deltas_srk, k_levels, rotation_seed=10000 + r)
        acc_srk = evaluate_model(model_srk, test_loader)
        history_srk.append(acc_srk)
        bits_srk_list.append(b_srk)

        b_srk_lloyd = federated_round_srk_lloyd_update(
            model_srk_lloyd, local_deltas_srk_lloyd, k_levels,
            shared_srk_lloyd_centers, rotation_seed=10000 + r)
        acc_srk_lloyd = evaluate_model(model_srk_lloyd, test_loader)
        history_srk_lloyd.append(acc_srk_lloyd)
        bits_srk_lloyd_list.append(b_srk_lloyd)

        # Kashin：共享 KashinFrame + 迭代截断求均匀系数，全轮复用同一 frame
        b_kashin = federated_round_kashin_update(model_kashin, local_deltas_kashin, k_levels, kashin_frame, iterations=10)
        acc_kashin = evaluate_model(model_kashin, test_loader)
        history_kashin.append(acc_kashin)
        bits_kashin_list.append(b_kashin)

        b_kashin_lloyd = federated_round_kashin_lloyd_update(
            model_kashin_lloyd, local_deltas_kashin_lloyd, k_levels,
            kashin_frame, iterations=10, shared_centers=shared_lloyd_centers)
        acc_kashin_lloyd = evaluate_model(model_kashin_lloyd, test_loader)
        history_kashin_lloyd.append(acc_kashin_lloyd)
        bits_kashin_lloyd_list.append(b_kashin_lloyd)

        print(f" -> Original (Δw, No Compression)      准确率: {acc_original*100:.2f}% | 单维通信开销: {b_original:.2f} Bits")
        print(f" -> SRK (Δw + Hadamard Rotation)       准确率: {acc_srk*100:.2f}% | 单维通信开销: {b_srk:.2f} Bits")
        print(f" -> SRK + Lloyd-Max                    准确率: {acc_srk_lloyd*100:.2f}% | 单维通信开销: {b_srk_lloyd:.2f} Bits")
        print(f" -> Kashin (Δw + Kashin Transform)     准确率: {acc_kashin*100:.2f}% | 单维通信开销: {b_kashin:.2f} Bits")
        print(f" -> Kashin + Lloyd-Max                 准确率: {acc_kashin_lloyd*100:.2f}% | 单维通信开销: {b_kashin_lloyd:.2f} Bits")

    # ---- 四算法汇总表 ----
    print("\n" + "="*120)
    print(f"          聚焦五算法汇总表 (Fixed b={FIXED_BITS}, k={FIXED_K_LEVELS}, {num_rounds} 轮)")
    print("="*120)
    print("  Round | Original | SRK-U | SRK-LM | Kashin-U | Kashin-LM | Bits")
    print("-"*120)
    for r in range(num_rounds):
        print(f" Round {r+1} | {history_original[r]*100:7.2f}% | {history_srk[r]*100:6.2f}% | {history_srk_lloyd[r]*100:7.2f}% | {history_kashin[r]*100:8.2f}% | {history_kashin_lloyd[r]*100:9.2f}%")
    print("="*120)

    srk_cost = bits_srk_list[-1]  # 与 Kashin 通信量相同（D = d_pow2）

    # ---- 图1：收敛曲线 ----
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    rounds = np.arange(1, num_rounds + 1)
    ax1.plot(rounds, [a*100 for a in history_original], marker='D', linestyle='-', linewidth=2, color='#8c564b', label='Original (Δw, No Compression)')
    ax1.plot(rounds, [a*100 for a in history_srk], marker='^', linestyle='--', linewidth=2, color='#ff7f0e', label='SRK (Δw + Hadamard Rotation)')
    ax1.plot(rounds, [a*100 for a in history_srk_lloyd], marker='v', linestyle=':', linewidth=2, color='#e377c2', label='SRK + Lloyd-Max')
    ax1.plot(rounds, [a*100 for a in history_kashin], marker='P', linestyle='-', linewidth=2, color='#9467bd', label='Kashin (Δw + Kashin Transform)')
    ax1.plot(rounds, [a*100 for a in history_kashin_lloyd], marker='s', linestyle=':', linewidth=2, color='#2ca02c', label='Kashin + Lloyd-Max')
    ax1.set_title("MNIST Accuracy with Model-Update Compression", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Federated Round", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_xticks(rounds)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # y 轴自动设置为 min(curves)-1 ~ max(curves)+1，确保能看清差异
    all_acc = [a*100 for a in history_original] + [a*100 for a in history_srk] + [a*100 for a in history_kashin] + [a*100 for a in history_kashin_lloyd]
    all_acc += [a*100 for a in history_srk_lloyd]
    ax1.set_ylim(min(all_acc) - 1.0, max(all_acc) + 1.0)

    # 图内文本框
    info_text_1 = f"Fixed b={FIXED_BITS} (k={FIXED_K_LEVELS})\nSRK/Kashin cost: {srk_cost:.3f} bits/dim/client\nLloyd-Max includes codebook overhead"
    ax1.text(0.03, 0.97, info_text_1, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    acc_filename = OUTPUT_DIR / f"kashin_vs_srk_accuracy_{timestamp}.png"
    fig1.savefig(acc_filename, dpi=300)
    print(f"\n图1已保存: '{acc_filename}'")

    # ========== 图2：相同通信预算下的 trade-off ==========
    print("\n" + "="*80)
    print(f"【聚焦-图2】相同通信预算 trade-off (TRADEOFF_BITS={TRADEOFF_BITS}，每点 {TRADEOFF_ROUNDS} 轮)")
    print("="*80)

    print("\n  bits |  k  |      method      | final_accuracy | classification_error | bits_per_dim_per_round | cumulative_bits_per_dim")
    print("-"*120)

    cum_srk_list = []
    cum_srk_lloyd_list = []
    cum_kashin_list = []
    cum_kashin_lloyd_list = []
    err_srk_list = []
    err_srk_lloyd_list = []
    err_kashin_list = []
    err_kashin_lloyd_list = []
    orig_errs = []

    for b in TRADEOFF_BITS:
        k_levels = 2 ** b
        # 每个 b 重新创建干净的 base_model，三种方法都从它克隆（起点绝对公平）
        base_model = MNIST_MLP()
        model_original = copy.deepcopy(base_model)
        model_srk = copy.deepcopy(base_model)
        model_srk_lloyd = copy.deepcopy(base_model)
        model_kashin = copy.deepcopy(base_model)
        model_kashin_lloyd = copy.deepcopy(base_model)
        print(f"  b={b}: 拟合共享 Lloyd-Max 码本...")
        shared_lloyd_centers = build_shared_kashin_codebook(base_model, kashin_frame, k_levels)
        shared_srk_lloyd_centers = build_shared_srk_codebook(base_model, k_levels, model_dim_d)

        res = {}
        for method, model in (('original', model_original), ('srk', model_srk),
                              ('srk_lloyd', model_srk_lloyd), ('kashin', model_kashin),
                              ('kashin_lloyd', model_kashin_lloyd)):
            bits_r = None
            for r in range(TRADEOFF_ROUNDS):
                deltas_r = [local_train_delta(model, client_loaders[c], epochs=2) for c in range(num_clients)]
                if method == 'original':
                    bits_r = federated_round_original_update(model, deltas_r)
                elif method == 'srk':
                    # SRK 每轮旋转种子稳定、可复现且每个 b / 每轮都不同
                    bits_r = federated_round_srk_update(model, deltas_r, k_levels, rotation_seed=20000 + 100*b + r)
                elif method == 'srk_lloyd':
                    bits_r = federated_round_srk_lloyd_update(
                        model, deltas_r, k_levels, shared_srk_lloyd_centers,
                        rotation_seed=20000 + 100*b + r)
                else:
                    if method == 'kashin':
                        bits_r = federated_round_kashin_update(model, deltas_r, k_levels, kashin_frame, iterations=10)
                    else:
                        bits_r = federated_round_kashin_lloyd_update(
                            model, deltas_r, k_levels, kashin_frame, iterations=10,
                            shared_centers=shared_lloyd_centers)
            acc = evaluate_model(model, test_loader)
            if method == 'kashin_lloyd':
                # 共享码本只发送一次，按本次 trade-off 的轮数摊销到每轮。
                bits_r += (32.0 * k_levels) / (model_dim_d * TRADEOFF_ROUNDS)
            res[method] = dict(acc=acc, err=1.0 - acc, bits=bits_r)

        for method in ('original', 'srk', 'srk_lloyd', 'kashin', 'kashin_lloyd'):
            cum = res[method]['bits'] * TRADEOFF_ROUNDS
            print(f"   {b:2d}  |  {k_levels:2d} | {method:15s} | {res[method]['acc']*100:12.2f}%  | {res[method]['err']*100:13.2f}%  | {res[method]['bits']:20.4f} | {cum:20.2f}")
        diff = abs(res['srk']['bits'] - res['kashin']['bits'])
        print(f"   SRK/Kashin same-budget difference = {diff:.3e}")
        if diff >= 1e-12:
            print("   [WARN] SRK/Kashin 通信量差值超过 1e-12！")

        orig_errs.append(res['original']['err'])
        cum_srk_list.append(res['srk']['bits'] * TRADEOFF_ROUNDS)
        cum_srk_lloyd_list.append(res['srk_lloyd']['bits'] * TRADEOFF_ROUNDS)
        cum_kashin_list.append(res['kashin']['bits'] * TRADEOFF_ROUNDS)
        cum_kashin_lloyd_list.append(res['kashin_lloyd']['bits'] * TRADEOFF_ROUNDS)
        err_srk_list.append(res['srk']['err'])
        err_srk_lloyd_list.append(res['srk_lloyd']['err'])
        err_kashin_list.append(res['kashin']['err'])
        err_kashin_lloyd_list.append(res['kashin_lloyd']['err'])

    # ---- 图2：按真实通信量的 trade-off 曲线 ----
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    # Original 是固定 32 bit/原始维度/轮，作为一个真实通信点，不画成水平线。
    orig_final_err = sum(orig_errs) / len(orig_errs)
    original_total_bits = 32.0 * TRADEOFF_ROUNDS
    ax2.scatter([original_total_bits], [orig_final_err], color='#8c564b', marker='D', s=70,
                label='Original (Δw, No Compression)')
    ax2.annotate('Original: 32 bit/param/round', xy=(original_total_bits, orig_final_err),
                 xytext=(-125, 10), textcoords='offset points', fontsize=9, color='#8c564b')

    ax2.plot(cum_srk_list, err_srk_list, marker='^', linestyle='--', linewidth=2, color='#ff7f0e', label='SRK (Δw + Hadamard Rotation)')
    ax2.plot(cum_srk_lloyd_list, err_srk_lloyd_list, marker='v', linestyle=':', linewidth=2, color='#e377c2', label='SRK + Lloyd-Max')
    ax2.plot(cum_kashin_list, err_kashin_list, marker='P', linestyle='-', linewidth=2, color='#9467bd', label='Kashin (Δw + Kashin Transform)')
    ax2.plot(cum_kashin_lloyd_list, err_kashin_lloyd_list, marker='s', linestyle=':', linewidth=2, color='#2ca02c', label='Kashin + Lloyd-Max')

    # 每个点标注 b；SRK / Kashin 用不同文字偏移避免重叠
    for i, b in enumerate(TRADEOFF_BITS):
        ax2.annotate(f'b={b}, D/d={kashin_frame.D/model_dim_d:.2f}', xy=(cum_srk_list[i], err_srk_list[i]), xytext=(10, 8),
                     textcoords='offset points', fontsize=9, color='#ff7f0e', ha='left')
        ax2.annotate(f'b={b}, D/d={kashin_frame.D/model_dim_d:.2f}', xy=(cum_kashin_list[i], err_kashin_list[i]), xytext=(10, -16),
                     textcoords='offset points', fontsize=9, color='#9467bd', ha='left')
        ax2.annotate(f'b={b}, D/d={kashin_frame.D/model_dim_d:.2f}', xy=(cum_kashin_lloyd_list[i], err_kashin_lloyd_list[i]), xytext=(10, 8),
                     textcoords='offset points', fontsize=9, color='#2ca02c', ha='left')

    ax2.set_xlabel("Cumulative Bits per Original Dimension / Client", fontsize=12)
    ax2.set_ylabel("Classification Error on MNIST", fontsize=12)
    ax2.set_title("Trade-off: Communication vs Accuracy (Original / SRK / Kashin / Lloyd-Max)", fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.6)

    info_text_2 = (f"x = cumulative actual bits / original dimension\n"
                   f"{TRADEOFF_ROUNDS} rounds per point; Lloyd-Max codebook sent once")
    ax2.text(0.03, 0.97, info_text_2, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    trade_filename = OUTPUT_DIR / f"kashin_vs_srk_tradeoff_{timestamp}.png"
    fig2.savefig(trade_filename, dpi=300)
    print(f"\n图2已保存: '{trade_filename}'")

    # 两张图都保存后再统一 plt.show() 一次（仅创建两张图、统一显示，不重复显示）
    print("\n【最终结果】两张图均已保存，统一显示中...")

    # 调试汇总：打印一张量化前动态范围 (xmax-xmin) 汇总表（仅日志）
    _print_range_summary()

    plt.show()

# ==========================================
# 6.7  变换系数分布诊断（Kashin + Lloyd-Max 研究第一步：只分析分布，不实现量化）
# ==========================================
def analyze_transform_coefficient_distribution():
    """分析 SRK 与 Kashin 变换后系数的分布（不量化、不训练、不评估、不出准确率/trade-off 图）。

    流程：
      1) 创建一个干净的 MNIST_MLP；
      2) 仅使用第 0 个客户端，local_train_delta(epochs=1) 得到模型更新量 Δw；
      3) 展平为 delta_flat，记录 d；
      4) SRK：固定随机种子生成 ±1 符号，固定 D_srk=65536，补零 -> 乘符号 -> _fwht_fast；
      5) Kashin：按 KASHIN_REDUNDANCY_SETTINGS 选择多个独立 D，
         每个 D 建 KashinFrame(d, D, seed=2026)，
         kashin_solve(frame, delta_flat, iterations=10)；
      6) 对 SRK 与每个 Kashin D 计算均值、标准差、极值、最大绝对值、
         标准化偏度 (mean(z^3))、标准化四阶矩 (mean(z^4))、归一化峰值 max|coeff|/std；
      7) 输出每个 Kashin D 的 D/d 与 b=1/2/3 每原始维度通信量；
      8) 保存一张多子图（标准化系数直方图 vs 标准正态 PDF）图片。

    SRK 仍固定使用 D=65536；Kashin 的 D 独立扫描，且允许不是 2 的幂。
    """
    print("\n" + "="*90)
    print("【分布诊断】SRK / Kashin 变换系数分布分析（第一步：不量化、不训练）")
    print("="*90)

    torch.manual_seed(2026)
    np.random.seed(2026)

    # ---------- 1. 干净的模型 + 第 0 个客户端的一次 Δw ----------
    model = MNIST_MLP()
    delta = local_train_delta(model, client_loaders[0], epochs=1)
    delta_flat, _ = flatten_state_dict(delta)
    d = delta_flat.shape[0]
    print(f"\n模型更新量 Δw：仅使用客户端 0，local_train_delta(epochs=1)")
    print(f"d (展平后维度) = {d}")

    # ---------- 2. SRK 系数（D_srk=65536 固定，随机 ±1 符号用固定种子，不量化） ----------
    D_srk = 65536
    srk_gen = torch.Generator()
    srk_gen.manual_seed(12345)  # 固定随机符号种子（与 SRK 聚合实现一致的 ±1 符号）
    padded = torch.zeros(D_srk)
    padded[:d] = delta_flat
    D_sign = (torch.rand(D_srk, generator=srk_gen) < 0.5).float() * 2.0 - 1.0
    srk_coeffs = _fwht_fast(padded * D_sign)

    # ---------- 3. Kashin 系数（每个 D 各建一个框架，seed 固定） ----------
    kashin_D_settings = [max(d, int(round(d * ratio)))
                         for ratio in KASHIN_REDUNDANCY_SETTINGS]
    kashin_coeffs = {}
    for D in kashin_D_settings:
        frame = KashinFrame(d, D=D, seed=2026)
        a = kashin_solve(frame, delta_flat, iterations=10)
        kashin_coeffs[D] = a
        print(f"Kashin D={D}: 系数长度 = {a.shape[0]}")

    # ---------- 4. 统计指标 ----------
    methods = [("SRK", D_srk, srk_coeffs)] + [("Kashin", D, kashin_coeffs[D]) for D in kashin_D_settings]

    print("\n" + "="*90)
    print("系数分布统计（z=(x-mean)/(std+1e-12)；skew=mean(z^3)；kurt4=mean(z^4)；norm_peak=max|coeff|/std）")
    print("参考：正态分布 skew≈0，四阶矩 kurt4≈3；norm_peak 越小越好（Kashin 的目标）")
    print("="*90)
    header = (f"  {'method':10s} {'D':>8s} {'mean':>12s} {'std':>12s} {'min':>12s} "
              f"{'max':>12s} {'max_abs':>12s} {'skew':>9s} {'kurt4':>9s} {'norm_peak':>11s}")
    print(header)
    print("-" * len(header))

    stats = {}
    for name, D, coeff in methods:
        c = coeff.float()
        mean = c.mean().item()
        std = c.std().item()
        cmin = c.min().item()
        cmax = c.max().item()
        max_abs = c.abs().max().item()
        z = (c - mean) / (std + 1e-12)
        skew = z.pow(3).mean().item()
        kurt4 = z.pow(4).mean().item()
        norm_peak = max_abs / (std + 1e-12)
        stats[(name, D)] = dict(mean=mean, std=std, min=cmin, max=cmax, max_abs=max_abs,
                                skew=skew, kurt4=kurt4, norm_peak=norm_peak, z=z)
        print(f"  {name:10s} {D:8d} {mean:12.6e} {std:12.6e} {cmin:12.6e} {cmax:12.6e} "
              f"{max_abs:12.6e} {skew:9.3f} {kurt4:9.3f} {norm_peak:11.3f}")

    # ---------- 5. 通信量（每个 Kashin D） ----------
    print("\n" + "="*90)
    print("通信量随 D 的变化（每原始维度 bit = (D*bits_per_coeff + 64) / d）")
    print("="*90)
    print(f"  {'D':>8s} {'D/d':>8s} {'b=1':>10s} {'b=2':>10s} {'b=3':>10s}")
    print("-" * 50)
    for D in kashin_D_settings:
        b1 = (D * 1 + 64) / d
        b2 = (D * 2 + 64) / d
        b3 = (D * 3 + 64) / d
        print(f"  {D:8d} {D/d:8.3f} {b1:10.3f} {b2:10.3f} {b3:10.3f}")

    # ---------- 6. 图片：三个子图的标准化系数直方图 vs 标准正态 PDF ----------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5))
    axes = np.atleast_1d(axes)
    for ax, (name, D, coeff) in zip(axes, methods):
        z = stats[(name, D)]["z"].numpy()
        lo = min(-5.0, float(z.min()))
        hi = max(5.0, float(z.max()))
        xgrid = np.linspace(lo, hi, 500)
        pdf = np.exp(-0.5 * xgrid ** 2) / math.sqrt(2 * math.pi)
        color = '#9467bd' if name == 'Kashin' else '#ff7f0e'
        ax.hist(z, bins=80, density=True, alpha=0.6, color=color,
                edgecolor='black', linewidth=0.4, label='standardized coeff')
        ax.plot(xgrid, pdf, 'r-', linewidth=1.8, label='N(0,1)')
        ax.set_xlabel("Standardized Coefficient")
        ax.set_ylabel("Density")
        ax.set_title(f"{name}, D={D}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)
    fig.suptitle("Transform Coefficient Distribution (standardized) vs Standard Normal", fontsize=14)
    fig.tight_layout()
    dist_filename = OUTPUT_DIR / f"transform_distribution_vs_D_{ts}.png"
    fig.savefig(dist_filename, dpi=300)
    plt.close(fig)
    print(f"\n分布图已保存: '{dist_filename}'")

    # ---------- 7. 结论提示 ----------
    print("\n" + "="*90)
    print("SRK 固定 D=65536；Kashin 独立扫描 D，不要求 D 为 2 的幂。")
    print("Kashin D 列表：" + ", ".join(str(D) for D in kashin_D_settings))
    print("="*90)


def test_lloyd_max_quantizer():
    """独立比较均匀量化与 Lloyd-Max 量化，不进入联邦训练。"""
    torch.manual_seed(1234)
    x = torch.cat([torch.randn(50000) * 0.8, torch.randn(5000) * 2.5])
    results = []
    print("\n" + "=" * 80)
    print("【Lloyd-Max 测试】均匀量化 vs 非均匀量化（仅独立测试）")
    print("=" * 80)
    print(f"{'k':>4s} {'uniform_mse':>16s} {'lloyd_mse':>16s} {'improvement':>14s}")
    print("-" * 80)

    for k in (2, 4, 8):
        uniform, _ = stochastic_k_level_quantize(x, k)
        lloyd, centers, _ = lloyd_max_quantize(x, k)
        uniform_mse = torch.mean((x - uniform) ** 2).item()
        lloyd_mse = torch.mean((x - lloyd) ** 2).item()
        improvement = 100.0 * (uniform_mse - lloyd_mse) / max(uniform_mse, 1e-12)
        results.append((k, uniform_mse, lloyd_mse))
        print(f"{k:4d} {uniform_mse:16.6e} {lloyd_mse:16.6e} {improvement:13.2f}%")

    fig, ax = plt.subplots(figsize=(7, 5))
    ks = [item[0] for item in results]
    ax.plot(ks, [item[1] for item in results], 'o--', linewidth=2,
            label='Uniform quantization')
    ax.plot(ks, [item[2] for item in results], 's-', linewidth=2,
            label='Lloyd-Max quantization')
    ax.set_xlabel('Quantization levels k')
    ax.set_ylabel('Mean squared quantization error')
    ax.set_title('Uniform vs Lloyd-Max Quantization')
    ax.set_xticks(ks)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    fig.tight_layout()
    filename = OUTPUT_DIR / f"lloyd_max_quantization_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"测试图已保存: '{filename}'")
    return results

# ==========================================
# 7. 主程序运行与对比实验
# ==========================================
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if RUN_LLOYD_MAX_TEST:
        test_lloyd_max_quantizer()
        print("\nLloyd-Max 独立测试完成，程序正常退出。")
        sys.exit(0)

    # ---------- 分布诊断开关：True 时只运行变换系数分布分析，不验证、不训练、不出旧图 ----------
    if RUN_DISTRIBUTION_TEST:
        analyze_transform_coefficient_distribution()
        RUN_DISTRIBUTION_TEST = False  # 一次性诊断，跑完恢复默认 False
        print("\n分布诊断完成，程序正常退出。")
        sys.exit(0)

    # 验证阶段先关闭量化前调试打印，保持验证输出清晰（正式实验前再恢复）
    DEBUG_RANGE = False

    # ---------- 验证 1：Δw 与完整参数聚合的通用等价性验证 ----------
    if not verify_fedavg_update_equivalence():
        print("\n更新量等价性验证失败，程序中止，不执行后续实验。")
        sys.exit(1)
    print("更新量等价性验证通过。\n")

    # ---------- 验证 2：第一阶段 Original Δw 管线（更贴近主循环）的等价性验证 ----------
    if not verify_first_round_original_update_path():
        print("\n第一阶段 Original Δw 管线验证失败，程序中止。")
        sys.exit(1)
    print("第一阶段 Original Δw 管线验证通过。\n")

    # ---------- 验证 3：SRK 的 Δw 压缩上传管线结构验证 ----------
    if not verify_srk_update_pipeline():
        print("\nSRK Δw 压缩上传管线验证失败，程序中止。")
        sys.exit(1)
    print("SRK Δw 压缩上传管线验证通过。\n")

    # ---------- 验证 4：Kashin 的 Δw 压缩上传管线结构验证 ----------
    if not verify_kashin_update_pipeline():
        print("\nKashin Δw 压缩上传管线验证失败，程序中止。")
        sys.exit(1)
    print("Kashin Δw 压缩上传管线验证通过。\n")

    # ---------- 轻量验证开关：False 时只运行以上四个验证后正常退出 ----------
    if not RUN_FULL_EXPERIMENT:
        print("RUN_FULL_EXPERIMENT = False：仅完成四个等价性验证，程序正常退出。")
        sys.exit(0)

    # 正式实验前恢复调试打印（验证阶段保持静默，输出清晰）
    DEBUG_RANGE = True

    # ---------- 聚焦实验开关：True 时只运行 Original / SRK / Kashin 三算法 Δw 对比 ----------
    if FOCUS_ON_KASHIN:
        run_focus_kashin_experiment(timestamp)
        print("聚焦三算法实验完成，程序正常退出。")
        sys.exit(0)

    # ---------- 旧五/四算法流程（FOCUS_ON_KASHIN=False 时运行，保留未删除） ----------
    # ---------- Kashin 独立单元测试（先于第一阶段，失败则不接入联邦训练） ----------
    print("\n" + "="*80)
    print("【Kashin 单元测试】冗余紧框架 + Kashin 系数求解（不接入联邦训练）")
    print("="*80)
    if not kashin_unit_test():
        print("\nKashin 单元测试失败，程序中止，不执行后续联邦训练。")
        sys.exit(1)
    print("Kashin 单元测试通过，继续进入第一阶段联邦训练...\n")

    initial_model = MNIST_MLP()

    # ---------- 共享 Kashin 框架（基于真实模型维度，seed 固定，全实验复用） ----------
    # 第一阶段全部 8 轮与第二阶段全部 k 值 / 客户端共用同一个框架，绝不按轮/按客户端重新生成。
    model_dim_d = sum(p.numel() for p in initial_model.parameters())
    kashin_frame = KashinFrame(model_dim_d, D=65536, seed=2026)
    print(f"共享 Kashin 框架: d = {model_dim_d}, D = {kashin_frame.D}, D/d = {kashin_frame.D/model_dim_d:.4f}")

    # ------------------------------------------
    # 阶段一：运行你原本的 8 轮仿真 (固定 k=4)
    # ------------------------------------------
    print("\n" + "="*80)
    print("【第一阶段】正在运行 8 轮标准联邦聚合实验 (固定 k=4)...")
    print("="*80)

    model_original = copy.deepcopy(initial_model)
    model_sk = copy.deepcopy(initial_model)
    model_srk = copy.deepcopy(initial_model)
    model_svk = copy.deepcopy(initial_model)
    model_kashin = copy.deepcopy(initial_model)

    init_acc = evaluate_model(initial_model, test_loader)
    print(f"\n初始全局模型识别准确率: {init_acc * 100:.2f}%")

    num_rounds = 8
    k_levels = 4

    history_original = []
    history_sk = []
    history_srk = []
    history_svk = []
    history_kashin = []

    bits_original_list = []
    bits_sk_list = []
    bits_srk_list = []
    bits_svk_list = []
    bits_kashin_list = []
    
    for r in range(num_rounds):
        print(f"\n==================== 第 {r+1} 轮联邦实验 ====================")
        
        local_deltas_original = []
        local_weights_sk = []
        local_deltas_srk = []
        local_weights_svk = []
        local_deltas_kashin = []

        for client_id in range(num_clients):
            loader = client_loaders[client_id]
            delta_original = local_train_delta(model_original, loader, epochs=2)
            w_sk = local_train(model_sk, loader, epochs=2)
            delta_srk = local_train_delta(model_srk, loader, epochs=2)
            w_svk = local_train(model_svk, loader, epochs=2)
            delta_kashin = local_train_delta(model_kashin, loader, epochs=2)

            local_deltas_original.append(delta_original)
            local_weights_sk.append(w_sk)
            local_deltas_srk.append(delta_srk)
            local_weights_svk.append(w_svk)
            local_deltas_kashin.append(delta_kashin)
            
        b_original = federated_round_original_update(model_original, local_deltas_original)
        acc_original = evaluate_model(model_original, test_loader)
        history_original.append(acc_original)
        bits_original_list.append(b_original)

        b_sk = federated_round_sk(model_sk, local_weights_sk, k_levels)
        acc_sk = evaluate_model(model_sk, test_loader)
        history_sk.append(acc_sk)
        bits_sk_list.append(b_sk)

        b_srk = federated_round_srk_update(model_srk, local_deltas_srk, k_levels, rotation_seed=10000 + r)
        acc_srk = evaluate_model(model_srk, test_loader)
        history_srk.append(acc_srk)
        bits_srk_list.append(b_srk)

        b_svk = federated_round_svk(model_svk, local_weights_svk, k_levels)
        acc_svk = evaluate_model(model_svk, test_loader)
        history_svk.append(acc_svk)
        bits_svk_list.append(b_svk)

        b_kashin = federated_round_kashin_update(
            model_kashin,
            local_deltas_kashin,
            k_levels,
            kashin_frame,
            iterations=10
        )
        acc_kashin = evaluate_model(model_kashin, test_loader)
        history_kashin.append(acc_kashin)
        bits_kashin_list.append(b_kashin)

        print(f" -> Original (Δw, No Compression)  准确率: {acc_original*100:.2f}% | 单维通信开销: {b_original:.2f} Bits")
        print(f" -> 算法1 (sk-标准量化)   准确率: {acc_sk*100:.2f}% | 单维通信开销: {b_sk:.2f} Bits")
        print(f" -> SRK (Δw + Hadamard Rotation)  准确率: {acc_srk*100:.2f}% | 单维通信开销: {b_srk:.2f} Bits")
        print(f" -> 算法3 (svk-熵编码)    准确率: {acc_svk*100:.2f}% | 单维通信开销: {b_svk:.4f} Bits")
        print(f" -> Kashin (Δw + Kashin Transform)  准确率: {acc_kashin*100:.2f}% | 单维通信开销: {b_kashin:.2f} Bits")

    # 打印总结表格
    print("\n" + "="*85)
    print("            DME (ICML 2017) 四种算法在 MNIST 联邦训练中的对比报告")
    print("="*85)
    print("  Round  |   Original (Acc / Bits)  |   sk 均匀量化 (Acc / Bits)  |   srk 旋转量化 (Acc / Bits)  |   svk 熵压缩量化 (Acc / Bits)")
    print("-"*85)
    for r in range(num_rounds):
        print(f" Round {r+1} |  {history_original[r]*100:.2f}% / {bits_original_list[r]:.1f} Bits  |      {history_sk[r]*100:.2f}% / {bits_sk_list[r]:.1f} Bits       |      {history_srk[r]*100:.2f}% / {bits_srk_list[r]:.1f} Bits       |      {history_svk[r]*100:.2f}% / {bits_svk_list[r]:.3f} Bits")
    print("="*85)

    # 自动绘制第一阶段的双子图并保存
    print("\n【绘图控制-1】正在生成标准对比图表...")
    rounds = np.arange(1, num_rounds + 1)
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#8c564b', '#1f77b4', '#ff7f0e', '#2ca02c']

    ax1.plot(rounds, [acc * 100 for acc in history_original], marker='D', linestyle='-', linewidth=2, color=colors[0], label='Original (No Compression)')
    ax1.plot(rounds, [acc * 100 for acc in history_sk], marker='o', linestyle='-', linewidth=2, color=colors[1], label='SK (Uniform Quantization)')
    ax1.plot(rounds, [acc * 100 for acc in history_srk], marker='^', linestyle='--', linewidth=2, color=colors[2], label='SRK (Hadamard Rotation)')
    ax1.plot(rounds, [acc * 100 for acc in history_svk], marker='s', linestyle=':', linewidth=2, color=colors[3], label='SVK (Entropy Coding)')

    ax1.set_xlabel('Federated Round', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('MNIST Classification Accuracy over Rounds', fontsize=13, fontweight='bold')
    ax1.set_xticks(rounds)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)

    # 柱状图
    algorithms = ['Original', 'SK', 'SRK', 'SVK']
    avg_bits = [bits_original_list[-1], bits_sk_list[-1], bits_srk_list[-1], bits_svk_list[-1]]
    bars = ax2.bar(algorithms, avg_bits, color=colors, alpha=0.8, edgecolor='black', width=0.5)
    ax2.set_ylabel('Average Bits per Dimension / Client', fontsize=12)
    ax2.set_title('Communication Cost comparison ', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f} Bits',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_filename = OUTPUT_DIR / f"mnist_icml2017_comparison_{timestamp}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"【第一阶段完成】图表已保存为: '{output_filename}'！")
    plt.close()

    # ==========================================
    # 🌟 第二阶段：量化级数 k 扫参实验（Figure 3 风格学术折衷图）
    # ==========================================
    print("\n" + "="*80)
    print("【第二阶段】正在启动量化级数 k 扫参实验以生成学术折衷曲线...")
    print("="*80)

    k_settings = [2, 4, 8, 16]   # 不同的量化层级对照组
    num_rounds_tradeoff = 4       # 每个 k 跑 4 轮以保证实验收敛
    
    # 记录点 (x: 累计比特, y: 分类错误率)
    points_sk = []
    points_srk = []
    points_svk = []
    
    # 提取哈达玛变换基础物理尺寸
    test_dict = MNIST_MLP().state_dict()
    flat_test, shapes_test = flatten_state_dict(test_dict)
    d_orig = flat_test.shape[0]
    d_pow2 = int(2 ** np.ceil(np.log2(d_orig)))
    D = (torch.rand(d_pow2) < 0.5).float() * 2.0 - 1.0

    for k in k_settings:
        print(f"\n---> 正在测试量化级数 k = {k}...")

        # 每次测试，重置三个模型的初始权重至完全一致
        # 使用全新的干净模型，确保起点绝对公平
        base_model = MNIST_MLP()
        base_model.load_state_dict(initial_model.state_dict())  # 从同一个初始点开始
        model_sk_trade = copy.deepcopy(base_model)
        model_srk_trade = copy.deepcopy(base_model)
        model_svk_trade = copy.deepcopy(base_model)
        
        cum_bits_sk, cum_bits_srk, cum_bits_svk = 0, 0, 0
        
        # 进行联邦训练
        for r in range(num_rounds_tradeoff):
            local_weights_sk = [local_train(model_sk_trade, client_loaders[c_id], epochs=1) for c_id in range(num_clients)]
            local_weights_srk = [local_train(model_srk_trade, client_loaders[c_id], epochs=1) for c_id in range(num_clients)]
            local_weights_svk = [local_train(model_svk_trade, client_loaders[c_id], epochs=1) for c_id in range(num_clients)]
            
            # 聚合
            b_sk = federated_round_sk(model_sk_trade, local_weights_sk, k)
            b_srk = federated_round_srk(model_srk_trade, local_weights_srk, k)
            b_svk = federated_round_svk(model_svk_trade, local_weights_svk, k)
            
            # 累计通信比特
            cum_bits_sk += b_sk
            cum_bits_srk += b_srk
            cum_bits_svk += b_svk
            
        # 计算 4 轮训练结束后的“分类错误率 (Error = 1 - Accuracy)”
        err_sk = 1.0 - evaluate_model(model_sk_trade, test_loader)
        err_srk = 1.0 - evaluate_model(model_srk_trade, test_loader)
        err_svk = 1.0 - evaluate_model(model_svk_trade, test_loader)
        
        points_sk.append((cum_bits_sk, err_sk))
        points_srk.append((cum_bits_srk, err_srk))
        points_svk.append((cum_bits_svk, err_svk))

    # 自动绘制学术折衷曲线并保存
    print("\n【绘图控制-2】正在生成学术折衷曲线图...")
    fig_trade, ax_trade = plt.subplots(figsize=(8, 6))
    
    xs_sk, ys_sk = zip(*points_sk)
    xs_srk, ys_srk = zip(*points_srk)
    xs_svk, ys_svk = zip(*points_svk)
    
    # 绘制曲线 (Y轴代表错误率，越往下越好；X轴代表累计比特数)
    ax_trade.plot(xs_sk, ys_sk, marker='o', color='#1f77b4', linestyle='-', linewidth=2, label='SK (Uniform Quantization)')
    ax_trade.plot(xs_srk, ys_srk, marker='^', color='#ff7f0e', linestyle='--', linewidth=2, label='SRK (Hadamard Rotation)')
    ax_trade.plot(xs_svk, ys_svk, marker='s', color='#2ca02c', linestyle=':', linewidth=2, label='SVK (Entropy Coding)')

    # 为每一个数据点标注对应的 k 值 (k=2, 4, 8, 16)
    # 不同算法错开标注位置，避免标签互相遮挡
    annotate_offsets = [(5, 6), (5, 14), (5, -12)]
    for (algo_points, off) in zip((points_sk, points_srk, points_svk), annotate_offsets):
        for (x_val, y_val), k_val in zip(algo_points, k_settings):
            ax_trade.annotate(f'k={k_val}', xy=(x_val, y_val), xytext=off,
                              textcoords='offset points', fontsize=9,
                              ha='left', va='center', color='dimgray')

    # 添加k值信息作为文本注释
    info_text = f"Quantization levels: k = {', '.join(map(str, k_settings))}\nRounds per k: {num_rounds_tradeoff}"
    ax_trade.text(0.02, 0.98, info_text, transform=ax_trade.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=10)

    ax_trade.set_xlabel('Total Bits per Original Dimension / Client (cumulative)', fontsize=12)
    ax_trade.set_ylabel('Classification Error on MNIST', fontsize=12)
    ax_trade.set_title('DME Quantization Trade-off on MNIST (ICML 2017)', fontsize=13, fontweight='bold')
    ax_trade.grid(True, linestyle='--', alpha=0.6)
    ax_trade.legend(fontsize=11)
    
    output_filename_trade = OUTPUT_DIR / f"mnist_icml2017_tradeoff_{timestamp}.png"
    plt.savefig(output_filename_trade, dpi=300)
    print(f"【第二阶段完成】图表已保存为: '{output_filename_trade}'！")

    # ==========================================
    # 🌟 同时显示两张图表
    # ==========================================
    print("\n【最终结果】正在同时显示两张图表...")

    # 创建一个包含两个子图的窗口
    fig_all, (ax1_all, ax2_all) = plt.subplots(1, 2, figsize=(16, 6))

    # 复制第一阶段的图表内容
    rounds = np.arange(1, num_rounds + 1)
    colors = ['#8c564b', '#1f77b4', '#ff7f0e', '#2ca02c']

    # 左子图：第一阶段的结果
    ax1_all.plot(rounds, [acc * 100 for acc in history_original], marker='D', linestyle='-', linewidth=2, color=colors[0], label='Original (No Compression)')
    ax1_all.plot(rounds, [acc * 100 for acc in history_sk], marker='o', linestyle='-', linewidth=2, color=colors[1], label='SK (Uniform Quantization)')
    ax1_all.plot(rounds, [acc * 100 for acc in history_srk], marker='^', linestyle='--', linewidth=2, color=colors[2], label='SRK (Hadamard Rotation)')
    ax1_all.plot(rounds, [acc * 100 for acc in history_svk], marker='s', linestyle=':', linewidth=2, color=colors[3], label='SVK (Entropy Coding)')
    ax1_all.set_xlabel('Federated Round', fontsize=12)
    ax1_all.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1_all.set_title('MNIST Classification Accuracy over Rounds', fontsize=13, fontweight='bold')
    ax1_all.set_xticks(rounds)
    ax1_all.grid(True, linestyle='--', alpha=0.6)
    ax1_all.legend(fontsize=10)

    # 右子图：第二阶段的折衷曲线
    ax2_all.plot(xs_sk, ys_sk, marker='o', color='#1f77b4', linestyle='-', linewidth=2, label='SK (Uniform Quantization)')
    ax2_all.plot(xs_srk, ys_srk, marker='^', color='#ff7f0e', linestyle='--', linewidth=2, label='SRK (Hadamard Rotation)')
    ax2_all.plot(xs_svk, ys_svk, marker='s', color='#2ca02c', linestyle=':', linewidth=2, label='SVK (Entropy Coding)')

    # 为每一个数据点标注对应的 k 值
    for (algo_points, off) in zip((points_sk, points_srk, points_svk), annotate_offsets):
        for (x_val, y_val), k_val in zip(algo_points, k_settings):
            ax2_all.annotate(f'k={k_val}', xy=(x_val, y_val), xytext=off,
                             textcoords='offset points', fontsize=8,
                             ha='left', va='center', color='dimgray')

    ax2_all.set_xlabel('Total Bits per Original Dimension / Client (cumulative)', fontsize=12)
    ax2_all.set_ylabel('Classification Error on MNIST ', fontsize=12)
    ax2_all.set_title('DME Quantization Trade-off on MNIST', fontsize=13, fontweight='bold')
    ax2_all.grid(True, linestyle='--', alpha=0.6)
    ax2_all.legend(fontsize=11)

    plt.tight_layout()
    output_filename_combined = OUTPUT_DIR / f"mnist_icml2017_combined_{timestamp}.png"
    plt.savefig(output_filename_combined, dpi=300)
    print(f"【联合图表已保存为】 '{output_filename_combined}'！")
    plt.show()
