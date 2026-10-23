"""Control Experiment: Static vs. Random Sign vs. Pure Raw 1-bit vs. SRK vs. No-Scale Kashin."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

# 1. 确保项目根目录在系统路径中，以便顺利导入自定义模块
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from archive.root import config
from archive.compression.srk_legacy import federated_round_srk_update
from compression.kashin_solver import kashin_solve
from compression.kashin_frame import FourierKashinFrame
from data.mnist_federated import build_mnist_federated_data
from federated.aggregation import federated_round_original_update
from federated.evaluation import evaluate_model
from federated.local_training import local_train_delta
from models.mnist_mlp import MNIST_MLP
from utils.communication import normalized_bits_per_dim
from utils.state_dict import (flatten_state_dict, state_dict_add,
                              unflatten_state_dict)
from compression.quantization import stochastic_k_level_quantize


# 从配置文件中读取联邦训练的焦点轮数
ROUNDS = config.NUM_ROUNDS_FOCUS

# 定义参与对比的五种策略及其在图表中的显示名称、颜色与线型
METHODS = (
    ("不更新", "gray", "--"),                 # 对照组1：完全不更新模型的静态基线
    ("随机符号", "orange", "--"),      # 对照组2：随机抛硬币的符号控制组
    ("signsgd", "red", "-"),                 # 对照组3：纯正负号、无任何优化的 Raw 1-bit
    ("hardmard旋转 ", "blue", "-"),      # 对照组4：无缩放、仅保留旋转的 SRK 1-bit
    ("Kashin ", "green", "-"), # 对照组5：无缩放、仅保留傅里叶框架的 Kashin 1-bit
)


def markdown_table(rows: list[dict]) -> str:
    """生成用于在终端打印的 Markdown 表格，直观展示每一轮各方法的准确率与误差。"""
    headers = ["Round"]
    for label, _, _ in METHODS:
        headers.extend((f"{label} Test Accuracy (%)", f"{label} relerr (NMSE)"))
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        values = [str(row["round"])]
        for label, _, _ in METHODS:
            acc = row[label + ".accuracy"]
            err = row[label + ".relerr"]
            values.extend((f"{acc:.2f}", f"{err:.6g}"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def reset_client_shuffle(client_loaders, round_index: int) -> None:
    """重置客户端数据加载器的随机种子，确保每轮数据的随机打乱过程具有可复现性。"""
    for client_index, loader in enumerate(client_loaders):
        seed = config.EXPERIMENT_SEED + round_index * 1009 + client_index
        if loader.generator is not None:
            loader.generator.manual_seed(seed)
        else:
            torch.manual_seed(seed)


def client_delta(model, loader, round_index: int):
    """计算单个客户端在本地训练后的模型参数变化量（Delta）。"""
    reset_client_shuffle([loader], round_index)
    return local_train_delta(model, loader, epochs=2, lr=0.05)


def static_model_round(global_model, delta) -> float:
    """对照组 1：静态模型。故意不更新任何参数（更新量全为 0），验证准确率是否恒定。"""
    flat, shapes = flatten_state_dict(delta)
    global_model.load_state_dict(state_dict_add(
        global_model.state_dict(), unflatten_state_dict(torch.zeros_like(flat), shapes)
    ))
    return 0.0


def random_sign_round(global_model, delta, round_index: int) -> float:
    """对照组 2：随机符号。强行赋予随机正负号（抛硬币），用来证明真实梯度方向的重要性。"""
    flat, shapes = flatten_state_dict(delta)
    g = torch.Generator().manual_seed(config.EXPERIMENT_SEED + round_index * 777)
    random_bits = torch.randint(0, 2, flat.shape, generator=g, device=flat.device)
    quantized = torch.where(random_bits == 1, torch.ones_like(flat), -torch.ones_like(flat))
    
    # 计算相对误差 (NMSE)
    relerr = float(
        (quantized - flat).square().sum()
        / flat.square().sum().clamp_min(1e-12)
    )
    global_model.load_state_dict(state_dict_add(
        global_model.state_dict(), unflatten_state_dict(quantized, shapes)
    ))
    return relerr


def pure_raw_1bit_round(global_model, delta) -> float:
    """对照组 3：纯 Raw 1-bit。只取真实梯度的正负号，没有任何额外的旋转、缩放或元数据。"""
    flat, shapes = flatten_state_dict(delta)
    quantized = torch.where(flat >= 0, torch.ones_like(flat), -torch.ones_like(flat))
    
    relerr = float(
        (quantized - flat).square().sum()
        / flat.square().sum().clamp_min(1e-12)
    )
    global_model.load_state_dict(state_dict_add(
        global_model.state_dict(), unflatten_state_dict(quantized, shapes)
    ))
    return relerr


def federated_round_kashin_noscale_update(global_model, client_deltas, k_levels, frame, iterations=10, quant_seeds=None, relerr_out=None):
    """自定义的无缩放、纯符号传输的 Kashin 1-bit 更新函数。
    核心逻辑：通过 Kashin 框架求解系数后，强制剥离所有幅值缩放，只传输纯 ±1 符号。
    """
    flat0, shapes = flatten_state_dict(client_deltas[0])
    aggregate = None
    
    for index, delta in enumerate(client_deltas):
        flat, _ = flatten_state_dict(delta)
        # 1. 求解 Kashin 傅里叶框架下的变换系数
        coefficients = kashin_solve(frame, flat, iterations=iterations)
        
        quant_generator = None
        if quant_seeds is not None:
            quant_generator = torch.Generator(device=flat.device)
            quant_generator.manual_seed(quant_seeds[index])
            
        # 2. 执行常规的 k 级随机量化
        quantized, _ = stochastic_k_level_quantize(coefficients, k_levels, generator=quant_generator)
        
        # 3. 核心修改：强制剥离所有幅值缩放，只保留纯正负号（±1）作为传输原始数据
        quantized = torch.where(quantized >= 0, torch.ones_like(quantized), -torch.ones_like(quantized))
        
        aggregate = quantized if aggregate is None else aggregate + quantized
        
        if relerr_out is not None:
            reconstructed = frame.frame_synthesis(quantized)
            relerr_out.append(float(((reconstructed - flat) ** 2).sum() / (flat ** 2).sum().clamp_min(1e-12)))
            
    aggregate /= len(client_deltas)
    
    # 在聚合后同样对整体取符号，确保通信双端传输的完全是纯 1-bit 符号流
    aggregate = torch.where(aggregate >= 0, torch.ones_like(aggregate), -torch.ones_like(aggregate))
    
    delta = unflatten_state_dict(frame.frame_synthesis(aggregate), shapes)
    global_model.load_state_dict(state_dict_add(global_model.state_dict(), delta))
    return normalized_bits_per_dim(frame.D, flat0.shape[0], int(math.ceil(math.log2(k_levels))))


def plot_results(rows: list[dict], output_path: Path) -> None:
    """绘制实验对比图：左边展示测试准确率曲线，右边展示对数坐标下的相对误差 (NMSE)。"""
    rounds = [row["round"] for row in rows]
    fig, (accuracy_ax, error_ax) = plt.subplots(1, 2, figsize=(14, 5.2))
    
    # === 在这里添加全局图名（Title） ===
    fig.suptitle("1-bit no scale quantization", fontsize=14, fontweight="bold", y=0.98)
    
    for label, color, style in METHODS:
        accuracies = [row[label + ".accuracy"] for row in rows]
        errors = [max(row[label + ".relerr"], 1e-16) for row in rows]
        accuracy_ax.plot(rounds, accuracies, color=color, linestyle=style,
                         linewidth=1.5, label=label)
        error_ax.plot(rounds, [math.log10(value) for value in errors],
                      color=color, linestyle=style,
                      linewidth=1.5, label=label)
    accuracy_ax.set(xlabel="Communication Rounds", ylabel="Test Accuracy (%)",
                    ylim=(0, 100), xticks=rounds[::max(1, len(rounds)//10)])
    accuracy_ax.grid(True, linestyle="--", alpha=0.4)
    accuracy_ax.legend(fontsize=8)
    error_ax.set(xlabel="Communication Rounds",
                 ylabel="log10(Relative Error / NMSE)", xticks=rounds[::max(1, len(rounds)//10)])
    error_ax.grid(True, linestyle="--", alpha=0.4)
    error_ax.legend(fontsize=8)
    
    # 调整布局，为顶部的 suptitle 留出空间
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run() -> list[dict]:
    """主函数：初始化联邦环境、构建各方法模型实例、执行多轮通信训练并保存图像。"""
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    torch.manual_seed(config.MODEL_SEED)
    
    # 建立单客户端的 MNIST 联邦数据集与测试集
    client_loaders, test_loader = build_mnist_federated_data(
        root=str(ROOT / "data"), num_clients=1,
        images_per_client=config.IMAGES_PER_CLIENT, paired_shuffle=True,
    )
    train_loader = client_loaders[0]

    initial_model = MNIST_MLP()
    dimension = sum(parameter.numel() for parameter in initial_model.parameters())
    if dimension != 50890:
        raise RuntimeError(f"Expected MNIST_MLP dimension 50890, found {dimension}")

    # 初始化 Kashin 框架（基于配置中的冗余度扩展维度 D）
    d = dimension
    D = int(math.ceil(d * config.KASHIN_LAMBDA))
    kashin_frame = FourierKashinFrame(d=d, D=D, seed=config.EXPERIMENT_SEED)

    # 为每种对比方法深度复制一份独立的模型副本
    models = [copy.deepcopy(initial_model) for _ in METHODS]
    initial_row = {"round": 0}
    for label, model in zip((method[0] for method in METHODS), models):
        accuracy = evaluate_model(model, test_loader) * 100
        initial_row[label + ".accuracy"] = accuracy
        initial_row[label + ".relerr"] = 0.0
    rows = [initial_row]

    # 开始多轮联邦通信训练迭代
    for round_index in range(1, ROUNDS + 1):
        round_row = {"round": round_index}
        for branch, ((label, _, _), model) in enumerate(zip(METHODS, models)):
            delta = client_delta(model, train_loader, round_index)
            if branch == 0:
                relerr = static_model_round(model, delta)
            elif branch == 1:
                relerr = random_sign_round(model, delta, round_index)
            elif branch == 2:
                relerr = pure_raw_1bit_round(model, delta)
            elif branch == 3:
                # 方案 4：无缩放的 SRK 1-bit 更新
                relative_errors = []
                federated_round_srk_update(
                    model, [delta], k_levels=2,
                    rotation_seeds=[config.EXPERIMENT_SEED + round_index],
                    quant_seeds=[config.EXPERIMENT_SEED + round_index * 1009],
                    relerr_out=relative_errors,
                )
                relerr = relative_errors[0] if relative_errors else 0.0
            else:
                # 方案 5：无缩放的 Kashin 1-bit 更新
                relative_errors = []
                federated_round_kashin_noscale_update(
                    model, [delta], k_levels=2, frame=kashin_frame, iterations=10,
                    quant_seeds=[config.EXPERIMENT_SEED + round_index * 1009],
                    relerr_out=relative_errors,
                )
                relerr = relative_errors[0] if relative_errors else 0.0

            round_row[label + ".accuracy"] = evaluate_model(model, test_loader) * 100
            round_row[label + ".relerr"] = relerr
        rows.append(round_row)
        if round_index % 10 == 0 or round_index == ROUNDS:
            print(f"Completed communication round {round_index}/{ROUNDS}", flush=True)

    # 打印最终的 Markdown 表格并保存对比图
    table = markdown_table(rows)
    print(table, flush=True)
    output_path = ROOT / "outputs" / "kashin_noscale_vs_srk3.png"
    plot_results(rows, output_path)
    print(f"Figure: {output_path}")
    return rows


if __name__ == "__main__":
    run()
