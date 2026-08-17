import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import copy

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
# 2. 准备数据：增加客户端至 120 个
# ==========================================
print("【数据准备】正在下载并分发数据至 120 个客户端...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

num_clients = 120  # 增加客户端数量以满足两阶段统计需求
client_loaders = []
images_per_client = 100 # 每个客户端分 100 张图，总共消耗 12000 张训练图

for i in range(num_clients):
    start_idx = i * images_per_client
    end_idx = start_idx + images_per_client
    subset = Subset(train_dataset, list(range(start_idx, end_idx)))
    loader = DataLoader(subset, batch_size=32, shuffle=True)
    client_loaders.append(loader)

test_subset = Subset(test_dataset, list(range(1000)))
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
print("【数据准备】120 个客户端数据准备完毕！")

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
# 🌟 4. 第一回合：基于格雷码（Gray Code）的非自适应定位
# ==========================================

def client_gray_query(state_dict, bit_level, lam=1.0):
    """
    客户端本地操作：计算第 bit_level 位的格雷函数值 g_l(X')
    """
    queried_dict = {}
    for key, var in state_dict.items():
        # 1. 归一化到 [0, 1] 空间 (公式 52 附近的 Rescaling)
        X_prime = torch.clamp((var + lam) / (2 * lam), 0.0, 1.0)
        
        # 2. 严格执行论文第 34 页 Definition 19 公式：
        # g_l(x) = 1  若  floor(2^l * x) mod 4 in {1, 2} 否则为 0
        ell = bit_level + 1 # 对应公式中的 l-th
        term = torch.floor((2 ** ell) * X_prime)
        val_mod = torch.remainder(term, 4)
        
        # 产生 1-bit 信号 (0.0 或 1.0)
        queried_dict[key] = ((val_mod == 1.0) | (val_mod == 2.0)).float()
    return queried_dict

def server_decode_localization(client_gray_responses, lam=1.0, sigma=0.1):
    """
    服务器端操作：聚合格雷码比特并解码出 O(sigma) 的区间 [L', U']
    """
    M = 4  # 4 位格雷码，可将空间切分为 16 个格子（格子宽度 = 2/16 = 0.125，符合 O(sigma) = 0.1）
    template_keys = client_gray_responses[0][0].keys()
    
    # 1. 对每一组客户端返回的比特进行多数投票（Majority Vote）
    voted_bits = {key: [None]*M for key in template_keys}
    group_size = len(client_gray_responses) // M # 每组约 12 个客户端
    
    for key in template_keys:
        for bit in range(M):
            # 收集该组客户端对该比特位的回答
            bit_sum = torch.zeros_like(client_gray_responses[0][0][key])
            for i in range(bit * group_size, (bit + 1) * group_size):
                bit_sum += client_gray_responses[i][0][key]
            # 多数投票判定
            voted_bits[key][bit] = (bit_sum >= (group_size / 2)).float()
            
    # 2. 严格执行格雷码向二进制解码的数学逻辑
    decoded_centers = {}
    for key in template_keys:
        # 格雷码转二进制：b_i = b_{i-1} ^ z_i
        binary_bits = [None] * M
        binary_bits[0] = voted_bits[key][0]
        for r in range(1, M):
            binary_bits[r] = torch.bitwise_xor(binary_bits[r-1].int(), voted_bits[key][r].int()).float()
            
        # 二进制转十进制网格索引
        grid_idx = torch.zeros_like(voted_bits[key][0])
        for r in range(M):
            grid_idx += binary_bits[r] * (2 ** (M - 1 - r))
            
        # 计算粗略区间边界 [L, U]
        grid_width = (2 * lam) / (2 ** M)
        L = -lam + grid_idx * grid_width
        U = L + grid_width
        
        # 3. 严格执行公式 (54) 的区间拓宽（Widening）来消除潜在的边界比特错误
        widening_step = (2 * lam) / (2 ** (M + 2))
        L_prime = torch.clamp(L - widening_step, -lam, lam)
        U_prime = torch.clamp(U + widening_step, -lam, lam)
        
        # 取拓宽后区间的中心点作为第二回合的平移原点
        decoded_centers[key] = (L_prime + U_prime) / 2.0
        
    return decoded_centers

# ==========================================
# 🌟 5. 第二回合：基于几何网格与分配分流的精细估计
# ==========================================

def client_refinement_query(state_dict, center, sigma=0.1, t=0.4):
    """
    客户端本地操作：根据服务器分派的区域边界 [a_i, b_i]，在本地进行四种阈值对比（Step 4）
    """
    refinement_responses = {}
    # 定义 6 个几何网格区间的边界（i = -3, -2, -1, 1, 2, 3）
    # m_0 = 0, m_1 = 1, m_2 = 2, m_3 = 4 (对应倍数的 sigma)
    # 本函数需要被不同的客户端群调用（不同客户端群测试不同的区域）
    
    for key, var in state_dict.items():
        # 1. 坐标系平移到第一阶段估算出的中心点（Step 1 的 Shift）
        var_shifted = var - center[key]
        
        # 为了演示，我们将每个客户端本地的四种查询打包在返回字典中
        # 客户端会根据服务器分配的网格编号 i 来做对比
        # 这里我们在代码里让客户端同时备好 6 个区域的应答，方便服务器按需抽取
        refinement_responses[key] = {}
        regions = {
            1: (0.0, 1.0*sigma), 2: (1.0*sigma, 2.0*sigma), 3: (2.0*sigma, 4.0*sigma),
            -1: (-1.0*sigma, 0.0), -2: (-2.0*sigma, -1.0*sigma), -3: (-4.0*sigma, -2.0*sigma)
        }
        for r_id, (a_i, b_i) in regions.items():
            # 裁剪并生成随机阈值 T_i ~ Unif(a_i, b_i)
            clipped = torch.clamp(var_shifted, a_i, b_i)
            T_i = torch.FloatTensor(var.shape).uniform_(a_i, b_i)
            
            # 计算四种 1-bit 查询：1{X>=a}, 1{X>=T}, 1{X<=b}, 1{X<=T}
            A = (clipped >= a_i).float()
            B = (clipped >= T_i).float()
            C = (clipped <= b_i).float()
            D = (clipped <= T_i).float()
            refinement_responses[key][r_id] = (A, B, C, D)
            
    return refinement_responses

def server_refinement_aggregation(client_refine_responses, center, sigma=0.1):
    """
    服务器端操作：收集后 70 个客户端分流后的应答，利用公式 (18) 和 (19) 重构高精度均值
    """
    template_keys = center.keys()
    aggregated_weights = {}
    
    # 严格按照 n_i 比例分配 70 个客户端到 6 个网格中
    # 权重比例：1/2 (R_1), 1/4 (R_2), 1/8 (R_3)，两侧对称。
    # 70个客户端分配方案：R_1和R_-1各20人，R_2和R_-2各10人，R_3和R_-3各5人。
    allocation = {
        1: list(range(0, 20)),
        -1: list(range(20, 40)),
        2: list(range(40, 50)),
        -2: list(range(50, 60)),
        3: list(range(60, 65)),
        -3: list(range(65, 70))
    }
    regions_bound = {
        1: (0.0, 1.0*sigma), 2: (1.0*sigma, 2.0*sigma), 3: (2.0*sigma, 4.0*sigma),
        -1: (-1.0*sigma, 0.0), -2: (-2.0*sigma, -1.0*sigma), -3: (-4.0*sigma, -2.0*sigma)
    }
    
    for key in template_keys:
        mu_base = torch.zeros_like(center[key])
        
        # 分别对 6 个区域进行概率重构
        for r_id, indices in allocation.items():
            a_i, b_i = regions_bound[r_id]
            n_i = len(indices)
            
            # 累加该区域分配的所有客户端的回答
            sum_A = torch.zeros_like(center[key])
            sum_B = torch.zeros_like(center[key])
            sum_C = torch.zeros_like(center[key])
            sum_D = torch.zeros_like(center[key])
            
            for idx in indices:
                A, B, C, D = client_refine_responses[idx][key][r_id]
                sum_A += A
                sum_B += B
                sum_C += C
                sum_D += D
                
            # 计算经验概率估计值 p_a_hat 和 p_b_hat （严格执行公式 18 & 19）
            p_a_hat = (sum_A - sum_B) / n_i
            p_b_hat = (sum_C - sum_D) / n_i
            
            # 计算局部均值贡献并累加：mu_i = a_i * p_a_hat + b_i * p_b_hat
            mu_i = a_i * p_a_hat + b_i * p_b_hat
            mu_base += mu_i
            
        # 加上第一轮的平移原点，得到最终的精确重构权重
        aggregated_weights[key] = mu_base + center[key]
        
    return aggregated_weights

# ==========================================
# 6. 服务器端评估函数
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
# 7. 主程序运行：120 个客户端协同进行 2-Stage 联邦训练
# ==========================================
if __name__ == "__main__":
    global_model = MNIST_MLP()
    
    init_acc = evaluate_model(global_model, test_loader)
    print(f"\n -> 初始未训练的全局模型识别准确率: {init_acc * 100:.2f}%")
    
    num_rounds = 10
    lam_bound = 1.0    # 初始搜索边界 lambda = 1.0
    sigma_noise = 0.1  # 噪声规模参数 sigma = 0.1
    
    for r in range(num_rounds):
        print(f"\n================ 第 {r+1} 轮【论文双阶段 1-Bit 联邦】开始 ================")
        
        # 1. 120个客户端本地训练
        local_trained_dicts = []
        for client_id in range(num_clients):
            trained_dict = local_train(global_model, client_loaders[client_id], epochs=2)
            local_trained_dicts.append(trained_dict)
        print(f" -> [客户端] 所有 120 个客户端本地训练结束。")
        
        # 2. 【第一回合：粗略定位】
        # 前 50 个客户端分别针对第 0, 1, 2, 3 位格雷码进行非自适应问答
        client_gray_responses = []
        for i in range(50):
            bit_level = i // 12  # 将50个客户端分为4个小组
            if bit_level > 3: bit_level = 3
            response = client_gray_query(local_trained_dicts[i], bit_level=bit_level, lam=lam_bound)
            client_gray_responses.append((response, bit_level))
            
        # 服务器解码得到每个参数坐标的 O(sigma) 级别中心点
        print(" -> [服务器-第1轮] 收集前 50 个客户端的格雷码应答，成功解码并定位出 O(sigma) 均值区间...")
        decoded_centers = server_decode_localization(client_gray_responses, lam=lam_bound, sigma=sigma_noise)
        
        # 3. 【第二回合：精细重构】
        # 后 70 个客户端根据自适应分配公式，对平移后的几何网格执行四种 1-bit 对比
        client_refine_responses = []
        for i in range(50, 120):
            response = client_refinement_query(local_trained_dicts[i], center=decoded_centers, sigma=sigma_noise)
            client_refine_responses.append(response)
            
        # 服务器收集应答，使用公式 (18) 和 (19) 拼接高精度均值参数
        print(" -> [服务器-第2轮] 收集后 70 个客户端的自适应网格应答，进行局部概率无偏重构...")
        new_global_weights = server_refinement_aggregation(client_refine_responses, center=decoded_centers, sigma=sigma_noise)
        
        # 4. 更新全局模型并评估
        global_model.load_state_dict(new_global_weights)
        round_acc = evaluate_model(global_model, test_loader)
        print(f"====== 第 {r+1} 轮结束 | 严格双阶段 1-Bit 重构后的全局准确率: {round_acc * 100:.2f}% ======")