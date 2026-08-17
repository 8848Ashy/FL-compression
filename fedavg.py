import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import copy

# ==========================================
# 1. 定义模型：简单的多层感知机 (MLP)
# ==========================================
class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        self.fc = nn.Sequential(
            # 输入：28x28 像素的图片，展平成 784 维向量
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            # 输出：10 维（对应数字 0-9 的概率）
            nn.Linear(64, 10)
        )
    def forward(self, x):
        # 展平图片：将 [batch_size, 1, 28, 28] 展平为 [batch_size, 784]
        x = x.view(-1, 28 * 28)
        return self.fc(x)

# ==========================================
# 2. 下载并分配数据：模拟 3 个客户端的本地 MNIST 数据
# ==========================================
print("【数据准备】正在下载并加载 MNIST 真实数据集...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # 标准化
])

# 自动在当前目录下下载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

num_clients = 10
client_loaders = []
images_per_client = 600 # 每个客户端分配 600 张图，既是真数据，又保证 CPU 运行极快

# 给 3 个客户端分配互不重合的数据切片
for i in range(num_clients):
    start_idx = i * images_per_client
    end_idx = start_idx + images_per_client
    indices = list(range(start_idx, end_idx))
    
    subset = Subset(train_dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=True)
    client_loaders.append(loader)

# 服务器测试集：取 1000 张图片用来测试全局模型
test_subset = Subset(test_dataset, list(range(1000)))
test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
print("【数据准备】数据切片分配完毕！")

# ==========================================
# 3. 客户端本地训练函数（多分类任务）
# ==========================================
def local_train(global_model, dataloader, epochs=2, lr=0.05):
    local_model = copy.deepcopy(global_model)
    optimizer = optim.SGD(local_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() # 多分类交叉熵损失函数
    
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
# 4. 服务器评估函数（计算全局模型对手写数字的识别准确率）
# ==========================================
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # 取概率最大的那个数字作为预测值
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ==========================================
# 5. 服务器参数聚合函数（FedAvg 平均）
# ==========================================
def server_aggregate(local_weights_list):
    aggregated_weights = copy.deepcopy(local_weights_list[0])
    for key in aggregated_weights.keys():
        for i in range(1, len(local_weights_list)):
            aggregated_weights[key] += local_weights_list[i][key]
        aggregated_weights[key] /= len(local_weights_list)
    return aggregated_weights

# ==========================================
# 6. 主程序运行
# ==========================================
if __name__ == "__main__":
    global_model = MNIST_MLP()
    
    # 初始评估
    init_acc = evaluate_model(global_model, test_loader)
    print(f"\n -> 初始未训练的全局模型，数字识别准确率: {init_acc * 100:.2f}% ")
    
    num_rounds = 5  # 进行 5 轮联邦学习
    
    for r in range(num_rounds):
        print(f"\n================ 第 {r+1} 轮联邦学习开始 ================")
        local_weights_list = []
        
        # 1. 客户端下载模型并在本地 MNIST 切片上训练
        for client_id in range(num_clients):
            loader = client_loaders[client_id]
            updated_weights = local_train(global_model, loader, epochs=2)
            local_weights_list.append(updated_weights)
            print(f" -> 客户端 {client_id} 已在 600 张手写数字图片上训练完成，并上传参数...")
            
        # 2. 服务器聚合权重
        print(" -> [服务器] 正在对 3 个客户端上传的参数进行聚合求平均...")
        new_global_weights = server_aggregate(local_weights_list)
        global_model.load_state_dict(new_global_weights)
        
        # 3. 评估最新全局模型
        round_acc = evaluate_model(global_model, test_loader)
        print(f"====== 第 {r+1} 轮结束 | 全局模型的手写数字识别准确率提升至: {round_acc * 100:.2f}% ======")