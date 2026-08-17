"""Kashin 中均匀量化与共享 Lloyd-Max 量化的联邦学习对比实验。

本脚本复用 2017.py 中的 MNIST、MLP、本地训练和 Kashin 实现，但只运行两种方法：
    1. Kashin + 随机均匀量化
    2. Kashin + 共享 Lloyd-Max 量化

两种方法使用同一个初始模型、同一批客户端、同一个 KashinFrame 和相同训练轮数。
Lloyd-Max 码本只在实验开始时用校准更新拟合一次，之后所有客户端共用。
"""

import copy
import importlib.util
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch


CURRENT_FILE = r"D:\FL\2017.py"
spec = importlib.util.spec_from_file_location("fl2017", CURRENT_FILE)
fl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fl)


# 实验配置：只改这些参数即可重新实验
NUM_ROUNDS = 8
LOCAL_EPOCHS = 2
K_LEVELS = 4
KASHIN_D = 65536
KASHIN_SEED = 2026
CALIBRATION_EPOCHS = 1
KASHIN_ITERATIONS = 10
LLOYD_ITERATIONS = 20


def fit_shared_codebook(model, frame):
    """用所有客户端的一次校准更新拟合共享 Lloyd-Max 码本。"""
    coefficients = []
    print("正在拟合共享 Lloyd-Max 码本...")
    for client_id, loader in enumerate(fl.client_loaders):
        delta = fl.local_train_delta(model, loader, epochs=CALIBRATION_EPOCHS)
        flat, _ = fl.flatten_state_dict(delta)
        coefficients.append(fl.kashin_solve(frame, flat, iterations=5))
        print(f"  已完成校准客户端 {client_id + 1}/{fl.num_clients}")
    return fl.lloyd_max_codebook(torch.cat(coefficients), K_LEVELS, LLOYD_ITERATIONS)


def run_experiment():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 90)
    print("Kashin: Uniform Quantization vs Shared Lloyd-Max Quantization")
    print("=" * 90)
    print(f"rounds={NUM_ROUNDS}, local_epochs={LOCAL_EPOCHS}, k={K_LEVELS}, D={KASHIN_D}")

    # 两种方法严格从同一个初始模型开始
    initial_model = fl.MNIST_MLP()
    model_uniform = copy.deepcopy(initial_model)
    model_lloyd = copy.deepcopy(initial_model)
    d = sum(parameter.numel() for parameter in initial_model.parameters())
    frame = fl.KashinFrame(d, D=KASHIN_D, seed=KASHIN_SEED)
    shared_centers = fit_shared_codebook(copy.deepcopy(initial_model), frame)

    history_uniform = []
    history_lloyd = []
    uniform_bits = None
    lloyd_bits = None

    for round_id in range(NUM_ROUNDS):
        print(f"\n========== Federated Round {round_id + 1}/{NUM_ROUNDS} ==========")
        deltas_uniform = []
        deltas_lloyd = []

        for client_id, loader in enumerate(fl.client_loaders):
            # 两种方法使用相同客户端数据；分别从各自当前全局模型本地训练
            deltas_uniform.append(fl.local_train_delta(model_uniform, loader, epochs=LOCAL_EPOCHS))
            deltas_lloyd.append(fl.local_train_delta(model_lloyd, loader, epochs=LOCAL_EPOCHS))

        uniform_bits = fl.federated_round_kashin_update(
            model_uniform, deltas_uniform, K_LEVELS, frame, iterations=KASHIN_ITERATIONS)
        lloyd_bits = fl.federated_round_kashin_lloyd_update(
            model_lloyd, deltas_lloyd, K_LEVELS, frame,
            iterations=KASHIN_ITERATIONS,
            shared_centers=shared_centers,
            include_codebook_bits=False)

        acc_uniform = fl.evaluate_model(model_uniform, fl.test_loader)
        acc_lloyd = fl.evaluate_model(model_lloyd, fl.test_loader)
        history_uniform.append(acc_uniform)
        history_lloyd.append(acc_lloyd)
        print(f"Uniform Kashin : {acc_uniform * 100:.2f}% | {uniform_bits:.6f} bits/dim/round")
        print(f"Shared Lloyd  : {acc_lloyd * 100:.2f}% | {lloyd_bits:.6f} bits/dim/round")

    codebook_bits = 32 * K_LEVELS
    amortized_lloyd_bits = lloyd_bits + codebook_bits / (d * NUM_ROUNDS)
    print("\n" + "=" * 90)
    print("最终结果")
    print(f"Uniform Kashin final accuracy : {history_uniform[-1] * 100:.2f}%")
    print(f"Shared Lloyd final accuracy   : {history_lloyd[-1] * 100:.2f}%")
    print(f"Uniform cumulative bits       : {uniform_bits * NUM_ROUNDS:.6f}")
    print(f"Lloyd cumulative bits         : {lloyd_bits * NUM_ROUNDS:.6f}")
    print(f"Lloyd codebook overhead       : {codebook_bits} bits/client (one-time)")
    print(f"Lloyd amortized cumulative    : {amortized_lloyd_bits * NUM_ROUNDS:.6f}")
    print("=" * 90)

    rounds = np.arange(1, NUM_ROUNDS + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, np.array(history_uniform) * 100, "o--", linewidth=2,
            color="#9467bd", label="Kashin + Uniform Quantization")
    ax.plot(rounds, np.array(history_lloyd) * 100, "s-", linewidth=2,
            color="#2ca02c", label="Kashin + Shared Lloyd-Max")
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Kashin Quantizer Comparison on MNIST")
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    ax.set_ylim(min(np.r_[history_uniform, history_lloyd]) * 100 - 1,
                max(np.r_[history_uniform, history_lloyd]) * 100 + 1)
    fig.tight_layout()
    filename = f"kashin_uniform_vs_lloyd_{timestamp}.png"
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"准确率图已保存: {filename}")


if __name__ == "__main__":
    run_experiment()
