"""Kashin 中均匀量化与共享 Lloyd-Max 量化的联邦学习对比实验。

本脚本复用 2017.py 中的 MNIST、MLP、本地训练和 Kashin 实现，但只运行两种方法：
    1. Kashin + 随机均匀量化
    2. Kashin + 共享 Lloyd-Max 量化

两种方法使用同一个初始模型、同一批客户端、同一个 KashinFrame 和相同训练轮数。
Lloyd-Max 码本只在实验开始时用校准更新拟合一次，之后所有客户端共用。
"""

import copy
import importlib.util
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch


CURRENT_FILE = r"D:\FL\2017.py"
spec = importlib.util.spec_from_file_location("fl2017", CURRENT_FILE)
fl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fl)

OUTPUT_DIR = Path(__file__).resolve().parent / "figures_kashin_lloyd_vs_uniform"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
        coefficients_i = fl.kashin_solve(frame, flat, iterations=5)
        coefficients.append(coefficients_i / (coefficients_i.abs().max() + 1e-12))
        print(f"  已完成校准客户端 {client_id + 1}/{fl.num_clients}")
    centers = fl.lloyd_max_codebook(torch.cat(coefficients), K_LEVELS, LLOYD_ITERATIONS)
    # 系数按每客户端最大绝对值归一化到 [-1, 1]，码本也必须覆盖同一范围。
    return centers / centers.abs().max().clamp_min(1e-12)


def federated_round_scaled_lloyd(global_model, client_deltas, frame, centers):
    """共享归一化 Lloyd-Max：每客户端发送一个 float32 scale，避免码本失配。"""
    aggregated = None
    flat0, shapes = fl.flatten_state_dict(client_deltas[0])
    for delta in client_deltas:
        flat, _ = fl.flatten_state_dict(delta)
        coeff = fl.kashin_solve(frame, flat, iterations=KASHIN_ITERATIONS)
        scale = coeff.abs().max().clamp_min(1e-12)
        normalized = coeff / scale
        quantized, _ = fl.quantize_with_codebook(normalized, centers)
        restored = quantized * scale
        aggregated = restored if aggregated is None else aggregated + restored
    aggregated /= len(client_deltas)
    average_delta = fl.unflatten_state_dict(frame.frame_synthesis(aggregated), shapes)
    global_model.load_state_dict(fl.state_dict_add(global_model.state_dict(), average_delta))
    # D 个索引 + 每客户端一个 float32 scale。
    bits = (frame.D * int(np.ceil(np.log2(K_LEVELS))) + 32.0) / flat0.shape[0]
    return bits


def fixed_middle_dense_codebook(k_levels):
    """固定的中间密、两端稀码本，基于标准正态分布分位点构造。"""
    normal = torch.distributions.Normal(0.0, 1.0)
    probabilities = (torch.arange(k_levels, dtype=torch.float32) + 0.5) / k_levels
    centers = normal.icdf(probabilities)
    centers = centers / centers.abs().max()
    return centers


def federated_round_fixed_nonuniform(global_model, client_deltas, frame, centers):
    """固定非均匀码本 + 每客户端尺度因子的 Kashin 聚合。"""
    aggregated = None
    flat0, shapes = fl.flatten_state_dict(client_deltas[0])
    for delta in client_deltas:
        flat, _ = fl.flatten_state_dict(delta)
        coeff = fl.kashin_solve(frame, flat, iterations=KASHIN_ITERATIONS)
        scale = coeff.abs().max().clamp_min(1e-12)
        quantized, _ = fl.quantize_with_codebook(coeff / scale, centers)
        restored = quantized * scale
        aggregated = restored if aggregated is None else aggregated + restored
    aggregated /= len(client_deltas)
    average_delta = fl.unflatten_state_dict(frame.frame_synthesis(aggregated), shapes)
    global_model.load_state_dict(fl.state_dict_add(global_model.state_dict(), average_delta))
    return (frame.D * int(np.ceil(np.log2(K_LEVELS))) + 32.0) / flat0.shape[0]


def get_layer_slices(state_dict):
    slices = []
    start = 0
    for key, value in state_dict.items():
        end = start + value.numel()
        slices.append((key, start, end))
        start = end
    return slices


def fit_layer_codebooks(model, frame):
    """按原始参数层拟合共享码本；实际只需聚合每层 count/sum。"""
    slices = get_layer_slices(model.state_dict())
    total_dim = sum(end - start for _, start, end in slices)
    layer_frames = [fl.KashinFrame(end - start,
                                   max(end - start, int(round(frame.D * (end - start) / total_dim))),
                                   seed=KASHIN_SEED + index)
                    for index, (_, start, end) in enumerate(slices)]
    collected = [[] for _ in slices]
    for loader in fl.client_loaders:
        delta = fl.local_train_delta(model, loader, epochs=CALIBRATION_EPOCHS)
        flat, _ = fl.flatten_state_dict(delta)
        for index, (_, start, end) in enumerate(slices):
            layer_frame = layer_frames[index]
            part = flat[start:end]
            coeff = fl.kashin_solve(layer_frame, part, iterations=5)
            collected[index].append(coeff / coeff.abs().max().clamp_min(1e-12))
    codebooks = []
    for values in collected:
        centers = fl.lloyd_max_codebook(torch.cat(values), K_LEVELS, LLOYD_ITERATIONS)
        codebooks.append(centers / centers.abs().max().clamp_min(1e-12))
    return slices, layer_frames, codebooks


def federated_round_layer_lloyd(global_model, client_deltas, slices, layer_frames, codebooks):
    """每层共享 Lloyd-Max 码本 + 每层 scale 的 Kashin 聚合。"""
    aggregated = None
    flat0, shapes = fl.flatten_state_dict(client_deltas[0])
    for delta in client_deltas:
        flat, _ = fl.flatten_state_dict(delta)
        restored_parts = []
        for (_, start, end), layer_frame, centers in zip(slices, layer_frames, codebooks):
            part = flat[start:end]
            coeff = fl.kashin_solve(layer_frame, part, iterations=KASHIN_ITERATIONS)
            scale = coeff.abs().max().clamp_min(1e-12)
            quantized, _ = fl.quantize_with_codebook(coeff / scale, centers)
            restored_parts.append(layer_frame.frame_synthesis(quantized * scale))
        restored = torch.cat(restored_parts)
        aggregated = restored if aggregated is None else aggregated + restored
    aggregated /= len(client_deltas)
    average_delta = fl.unflatten_state_dict(aggregated, shapes)
    global_model.load_state_dict(fl.state_dict_add(global_model.state_dict(), average_delta))
    # 每层一个 scale；码本只更新/广播一次。
    total_D = sum(layer_frame.D for layer_frame in layer_frames)
    bits = (total_D * int(np.ceil(np.log2(K_LEVELS))) + 32.0 * len(slices)) / flat0.shape[0]
    return bits


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
    model_fixed = copy.deepcopy(initial_model)
    model_layer = copy.deepcopy(initial_model)
    d = sum(parameter.numel() for parameter in initial_model.parameters())
    frame = fl.KashinFrame(d, D=KASHIN_D, seed=KASHIN_SEED)
    shared_centers = fit_shared_codebook(copy.deepcopy(initial_model), frame)
    layer_slices, layer_frames, layer_centers = fit_layer_codebooks(copy.deepcopy(initial_model), frame)
    fixed_centers = fixed_middle_dense_codebook(K_LEVELS)
    print(f"固定中间密集码本: {fixed_centers.tolist()}")

    history_uniform = []
    history_lloyd = []
    history_fixed = []
    history_layer = []
    uniform_bits = None
    lloyd_bits = None
    diagnostics = []
    round_codebooks = []

    for round_id in range(NUM_ROUNDS):
        print(f"\n========== Federated Round {round_id + 1}/{NUM_ROUNDS} ==========")
        deltas_uniform = []
        deltas_lloyd = []
        deltas_fixed = []
        deltas_layer = []

        # 两种方法使用完全相同的 DataLoader shuffle 随机状态。
        rng_state = torch.get_rng_state()
        for client_id, loader in enumerate(fl.client_loaders):
            # 两种方法使用相同客户端数据；分别从各自当前全局模型本地训练
            deltas_uniform.append(fl.local_train_delta(model_uniform, loader, epochs=LOCAL_EPOCHS))
        torch.set_rng_state(rng_state)
        for client_id, loader in enumerate(fl.client_loaders):
            deltas_lloyd.append(fl.local_train_delta(model_lloyd, loader, epochs=LOCAL_EPOCHS))
        torch.set_rng_state(rng_state)
        for client_id, loader in enumerate(fl.client_loaders):
            deltas_fixed.append(fl.local_train_delta(model_fixed, loader, epochs=LOCAL_EPOCHS))
        torch.set_rng_state(rng_state)
        for client_id, loader in enumerate(fl.client_loaders):
            deltas_layer.append(fl.local_train_delta(model_layer, loader, epochs=LOCAL_EPOCHS))

        # 在聚合前诊断两种量化器，不改变聚合流程。
        mse_uniform = []
        mse_lloyd = []
        bias_uniform = []
        bias_lloyd = []
        cosine_uniform = []
        cosine_lloyd = []
        out_of_range = []
        round_coefficients = []
        boundaries = (shared_centers[:-1] + shared_centers[1:]) / 2.0
        for delta_u, delta_l in zip(deltas_uniform, deltas_lloyd):
            flat_u, _ = fl.flatten_state_dict(delta_u)
            flat_l, _ = fl.flatten_state_dict(delta_l)
            coeff_u = fl.kashin_solve(frame, flat_u, iterations=KASHIN_ITERATIONS)
            coeff_l = fl.kashin_solve(frame, flat_l, iterations=KASHIN_ITERATIONS)
            round_coefficients.append(coeff_l / (coeff_l.abs().max() + 1e-12))

            torch.set_rng_state(rng_state)
            quant_u, _ = fl.stochastic_k_level_quantize(coeff_u, K_LEVELS)
            scale_l = coeff_l.abs().max().clamp_min(1e-12)
            quant_l, _ = fl.quantize_with_codebook(coeff_l / scale_l, shared_centers)
            quant_l = quant_l * scale_l
            mse_uniform.append(torch.mean((coeff_u - quant_u) ** 2).item())
            mse_lloyd.append(torch.mean((coeff_l - quant_l) ** 2).item())
            bias_uniform.append(torch.norm(quant_u - coeff_u).item() / (torch.norm(coeff_u).item() + 1e-12))
            bias_lloyd.append(torch.norm(quant_l - coeff_l).item() / (torch.norm(coeff_l).item() + 1e-12))
            cosine_uniform.append(torch.nn.functional.cosine_similarity(coeff_u, quant_u, dim=0).item())
            cosine_lloyd.append(torch.nn.functional.cosine_similarity(coeff_l, quant_l, dim=0).item())
            out_of_range.append((((coeff_l / scale_l) < boundaries[0]) |
                                 ((coeff_l / scale_l) > boundaries[-1])).float().mean().item())

        diagnostics.append({
            "round": round_id + 1,
            "uniform_mse": float(np.mean(mse_uniform)),
            "lloyd_mse": float(np.mean(mse_lloyd)),
            "uniform_rel_error": float(np.mean(bias_uniform)),
            "lloyd_rel_error": float(np.mean(bias_lloyd)),
            "uniform_cosine": float(np.mean(cosine_uniform)),
            "lloyd_cosine": float(np.mean(cosine_lloyd)),
            "lloyd_out_of_range": float(np.mean(out_of_range)),
        })
        oracle_centers = fl.lloyd_max_codebook(torch.cat(round_coefficients), K_LEVELS, LLOYD_ITERATIONS)
        oracle_centers = oracle_centers / oracle_centers.abs().max().clamp_min(1e-12)
        round_codebooks.append(oracle_centers.detach().cpu().numpy())
        item = diagnostics[-1]
        print(f"诊断：MSE U/L={item['uniform_mse']:.3e}/{item['lloyd_mse']:.3e}, "
              f"cos U/L={item['uniform_cosine']:.6f}/{item['lloyd_cosine']:.6f}, "
              f"Lloyd超范围={item['lloyd_out_of_range'] * 100:.2f}%")
        print("  固定码本:      " + ", ".join(f"{v:.4f}" for v in fixed_centers.tolist()))
        print("  共享Lloyd码本: " + ", ".join(f"{v:.4f}" for v in shared_centers.tolist()))
        print("  当轮理想码本:  " + ", ".join(f"{v:.4f}" for v in oracle_centers.tolist()))

        uniform_bits = fl.federated_round_kashin_update(
            model_uniform, deltas_uniform, K_LEVELS, frame, iterations=KASHIN_ITERATIONS)
        lloyd_bits = federated_round_scaled_lloyd(
            model_lloyd, deltas_lloyd, frame, shared_centers)
        fixed_bits = federated_round_fixed_nonuniform(
            model_fixed, deltas_fixed, frame, fixed_centers)
        layer_bits = federated_round_layer_lloyd(
            model_layer, deltas_layer, layer_slices, layer_frames, layer_centers)

        acc_uniform = fl.evaluate_model(model_uniform, fl.test_loader)
        acc_lloyd = fl.evaluate_model(model_lloyd, fl.test_loader)
        acc_fixed = fl.evaluate_model(model_fixed, fl.test_loader)
        acc_layer = fl.evaluate_model(model_layer, fl.test_loader)
        history_uniform.append(acc_uniform)
        history_lloyd.append(acc_lloyd)
        history_fixed.append(acc_fixed)
        history_layer.append(acc_layer)
        print(f"Uniform Kashin : {acc_uniform * 100:.2f}% | {uniform_bits:.6f} bits/dim/round")
        print(f"Shared Lloyd  : {acc_lloyd * 100:.2f}% | {lloyd_bits:.6f} bits/dim/round")
        print(f"Fixed nonuniform: {acc_fixed * 100:.2f}% | {fixed_bits:.6f} bits/dim/round")
        print(f"Layer-wise Lloyd: {acc_layer * 100:.2f}% | {layer_bits:.6f} bits/dim/round")

    codebook_bits = 32 * K_LEVELS
    amortized_lloyd_bits = lloyd_bits + codebook_bits / (d * NUM_ROUNDS)
    print("\n" + "=" * 90)
    print("最终结果")
    print(f"Uniform Kashin final accuracy : {history_uniform[-1] * 100:.2f}%")
    print(f"Shared Lloyd final accuracy   : {history_lloyd[-1] * 100:.2f}%")
    print(f"Fixed nonuniform final accuracy: {history_fixed[-1] * 100:.2f}%")
    print(f"Layer-wise Lloyd final accuracy  : {history_layer[-1] * 100:.2f}%")
    print(f"Uniform cumulative bits       : {uniform_bits * NUM_ROUNDS:.6f}")
    print(f"Lloyd cumulative bits         : {lloyd_bits * NUM_ROUNDS:.6f}")
    print(f"Lloyd codebook overhead       : {codebook_bits} bits/client (one-time)")
    print(f"Lloyd amortized cumulative    : {amortized_lloyd_bits * NUM_ROUNDS:.6f}")
    print("=" * 90)

    print("\n每轮量化诊断：")
    print("round | uniform MSE | Lloyd MSE | uniform cosine | Lloyd cosine | Lloyd out-of-range")
    for item in diagnostics:
        print(f"{item['round']:5d} | {item['uniform_mse']:.3e} | {item['lloyd_mse']:.3e} | "
              f"{item['uniform_cosine']:.6f} | {item['lloyd_cosine']:.6f} | "
              f"{item['lloyd_out_of_range'] * 100:.2f}%")

    diagnostic_array = np.array([[item[key] for key in (
        "uniform_mse", "lloyd_mse", "uniform_cosine", "lloyd_cosine",
        "lloyd_out_of_range")] for item in diagnostics])
    np.savetxt(OUTPUT_DIR / f"kashin_quantization_diagnostics_{timestamp}.csv", diagnostic_array,
               delimiter=",", header="uniform_mse,lloyd_mse,uniform_cosine,lloyd_cosine,lloyd_out_of_range",
               comments="")

    codebook_array = np.vstack(round_codebooks)
    np.savetxt(OUTPUT_DIR / f"kashin_codebooks_by_round_{timestamp}.csv", codebook_array,
               delimiter=",", header=",".join(f"center_{i + 1}" for i in range(K_LEVELS)),
               comments="")
    fig_codebook, ax_codebook = plt.subplots(figsize=(9, 5))
    for level in range(K_LEVELS):
        ax_codebook.plot(range(1, NUM_ROUNDS + 1), codebook_array[:, level],
                         marker="o", linewidth=2, label=f"Lloyd center {level + 1}")
    ax_codebook.axhline(0, color="black", linewidth=0.8)
    ax_codebook.set_xlabel("Federated Round")
    ax_codebook.set_ylabel("Normalized quantization center")
    ax_codebook.set_title("Ideal Lloyd-Max Codebook by Round")
    ax_codebook.set_xticks(range(1, NUM_ROUNDS + 1))
    ax_codebook.grid(True, linestyle="--", alpha=0.5)
    ax_codebook.legend()
    fig_codebook.tight_layout()
    codebook_filename = OUTPUT_DIR / f"kashin_codebooks_by_round_{timestamp}.png"
    fig_codebook.savefig(codebook_filename, dpi=300)
    plt.close(fig_codebook)
    print(f"每轮理想 Lloyd-Max 码本图已保存: {codebook_filename}")

    rounds = np.arange(1, NUM_ROUNDS + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, np.array(history_uniform) * 100, "o--", linewidth=2,
            color="#9467bd", label="Kashin + Uniform Quantization")
    ax.plot(rounds, np.array(history_lloyd) * 100, "s-", linewidth=2,
            color="#2ca02c", label="Kashin + Shared Lloyd-Max")
    ax.plot(rounds, np.array(history_fixed) * 100, "^-.", linewidth=2,
            color="#ff7f0e", label="Kashin + Fixed Middle-Dense")
    ax.plot(rounds, np.array(history_layer) * 100, "d-.", linewidth=2,
            color="#d62728", label="Kashin + Layer-wise Lloyd-Max")
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Kashin Quantizer Comparison on MNIST")
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    ax.set_ylim(min(np.r_[history_uniform, history_lloyd, history_fixed, history_layer]) * 100 - 1,
                max(np.r_[history_uniform, history_lloyd, history_fixed, history_layer]) * 100 + 1)
    fig.tight_layout()
    filename = OUTPUT_DIR / f"kashin_uniform_vs_lloyd_{timestamp}.png"
    fig.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"准确率图已保存: {filename}")


if __name__ == "__main__":
    run_experiment()
