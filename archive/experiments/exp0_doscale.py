"""Control Experiment: 1-bit Methods with Optimal Dynamic Scale (L1-Mean) Metadata per Round."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from archive.root import config
from compression.kashin_solver import kashin_solve
from compression.kashin_frame import FourierKashinFrame
from data.mnist_federated import build_mnist_federated_data
from federated.evaluation import evaluate_model
from federated.local_training import local_train_delta
from models.mnist_mlp import MNIST_MLP
from utils.communication import normalized_bits_per_dim
from utils.state_dict import (flatten_state_dict, state_dict_add,
                              unflatten_state_dict)

ROUNDS = config.NUM_ROUNDS_FOCUS
KASHIN_ITERS = 25  # 对齐 25 次迭代以保证紧框架收敛精度

# 使用英文标签避免 Matplotlib 缺少中文字体导致显示方框乱码
METHODS = (
    ("Static (0-bit)", "gray", "--"),                 
    ("Random Sign (with scale)", "orange", "--"),      
    ("SignSGD (with scale)", "red", "-"),                 
    ("Hadamard (with scale)", "blue", "-"),      
    ("Kashin (with scale)", "green", "-"), 
)


def markdown_table(rows: list[dict]) -> str:
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
    for client_index, loader in enumerate(client_loaders):
        seed = config.EXPERIMENT_SEED + round_index * 1009 + client_index
        if loader.generator is not None:
            loader.generator.manual_seed(seed)
        else:
            torch.manual_seed(seed)


def client_delta(model, loader, round_index: int):
    reset_client_shuffle([loader], round_index)
    return local_train_delta(model, loader, epochs=2, lr=0.05)


def quantize_1bit_with_dynamic_scale(tensor: torch.Tensor, mode: str = "mean") -> tuple[torch.Tensor, float]:
    """
    计算每轮动态 Scale 并执行 1-bit 符号量化：
    - mode="mean": 最小化均方误差 (MSE) 的理论最优缩放因子 s* = E[|x|]，免疫极端离群值爆炸
    - mode="quantile": 99.5% 分位数动态缩放
    - mode="max": 最大绝对值（在原始梯度空间极易被单个 Outlier 劫持导致模型崩溃）
    """
    if mode == "mean":
        scale = float(tensor.abs().mean().item())
    elif mode == "quantile":
        scale = float(tensor.abs().quantile(0.995).item())
    elif mode == "max":
        scale = float(tensor.abs().max().item())
    else:
        raise ValueError(f"Unknown scale mode: {mode}")

    if scale < 1e-12:
        scale = 1e-12
    quantized = torch.where(tensor >= 0,
                            torch.full_like(tensor, scale),
                            torch.full_like(tensor, -scale))
    return quantized, scale


def fwht_1d(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"FWHT length must be a power of 2, got {n}")
    y = x.clone()
    h = 1
    while h < n:
        y = y.view(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :].clone()
        b = y[:, :, 1, :].clone()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        h *= 2
    return y.view_as(x) / math.sqrt(n)


def hadamard_forward(flat: torch.Tensor, seed: int) -> tuple[torch.Tensor, int]:
    d = flat.numel()
    D_had = 1 << (d - 1).bit_length()
    padded = torch.zeros(D_had, device=flat.device, dtype=flat.dtype)
    padded[:d] = flat
    g = torch.Generator(device=flat.device).manual_seed(seed)
    signs = torch.randint(0, 2, (D_had,), generator=g, device=flat.device, dtype=flat.dtype) * 2.0 - 1.0
    return fwht_1d(padded * signs), d


def hadamard_inverse(rotated_q: torch.Tensor, d: int, seed: int) -> torch.Tensor:
    D_had = rotated_q.numel()
    g = torch.Generator(device=rotated_q.device).manual_seed(seed)
    signs = torch.randint(0, 2, (D_had,), generator=g, device=rotated_q.device, dtype=rotated_q.dtype) * 2.0 - 1.0
    unrotated = fwht_1d(rotated_q) * signs
    return unrotated[:d]


def static_model_round(global_model, delta) -> float:
    flat, shapes = flatten_state_dict(delta)
    global_model.load_state_dict(state_dict_add(
        global_model.state_dict(), unflatten_state_dict(torch.zeros_like(flat), shapes)
    ))
    return 0.0


def random_sign_round(global_model, delta, round_index: int) -> float:
    flat, shapes = flatten_state_dict(delta)
    g = torch.Generator(device=flat.device).manual_seed(config.EXPERIMENT_SEED + round_index * 777)
    random_bits = torch.randint(0, 2, flat.shape, generator=g, device=flat.device)
    # 与 SignSGD 保持完全相同的动态量级（Mean Scale），仅方向纯随机
    scale = max(float(flat.abs().mean().item()), 1e-12)
    quantized = torch.where(random_bits == 1, torch.full_like(flat, scale), torch.full_like(flat, -scale))
    
    relerr = float((quantized - flat).square().sum() / flat.square().sum().clamp_min(1e-12))
    global_model.load_state_dict(state_dict_add(
        global_model.state_dict(), unflatten_state_dict(quantized, shapes)
    ))
    return relerr


def pure_raw_1bit_with_scale_round(global_model, delta) -> float:
    flat, shapes = flatten_state_dict(delta)
    quantized, _ = quantize_1bit_with_dynamic_scale(flat, mode="mean")
    
    relerr = float((quantized - flat).square().sum() / flat.square().sum().clamp_min(1e-12))
    global_model.load_state_dict(state_dict_add(
        global_model.state_dict(), unflatten_state_dict(quantized, shapes)
    ))
    return relerr


def hadamard_1bit_with_scale_round(global_model, delta, seed: int) -> float:
    flat, shapes = flatten_state_dict(delta)
    rotated, orig_d = hadamard_forward(flat, seed=seed)
    q_rot, _ = quantize_1bit_with_dynamic_scale(rotated, mode="mean")
    reconstructed = hadamard_inverse(q_rot, d=orig_d, seed=seed)

    relerr = float((reconstructed - flat).square().sum() / flat.square().sum().clamp_min(1e-12))
    global_model.load_state_dict(state_dict_add(
        global_model.state_dict(), unflatten_state_dict(reconstructed, shapes)
    ))
    return relerr


def federated_round_kashin_scale_update(global_model, client_deltas, k_levels, frame, iterations=KASHIN_ITERS, relerr_out=None):
    """带有每轮动态 Scale 的 Kashin 1-bit 更新函数"""
    flat0, shapes = flatten_state_dict(client_deltas[0])
    aggregate = None
    
    for delta in client_deltas:
        flat, _ = flatten_state_dict(delta)
        coefficients = kashin_solve(frame, flat, iterations=iterations)
        
        # 计算当前轮次高维系数的最优动态 Scale (Mean Scale)
        quantized, _ = quantize_1bit_with_dynamic_scale(coefficients, mode="mean")
        
        aggregate = quantized if aggregate is None else aggregate + quantized
        
        if relerr_out is not None:
            reconstructed = frame.frame_synthesis(quantized)
            relerr_out.append(float(((reconstructed - flat) ** 2).sum() / (flat ** 2).sum().clamp_min(1e-12)))
            
    aggregate /= len(client_deltas)
    
    delta = unflatten_state_dict(frame.frame_synthesis(aggregate), shapes)
    global_model.load_state_dict(state_dict_add(global_model.state_dict(), delta))
    return normalized_bits_per_dim(frame.D, flat0.shape[0], int(math.ceil(math.log2(k_levels))))


def plot_results(rows: list[dict], output_path: Path) -> None:
    rounds = [row["round"] for row in rows]
    fig, (accuracy_ax, error_ax) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle("1-bit Dynamic Scale Quantization (Optimal L1-Mean Scale)", fontsize=14, fontweight="bold", y=0.98)
    
    for label, color, style in METHODS:
        accuracies = [row[label + ".accuracy"] for row in rows]
        errors = [max(row[label + ".relerr"], 1e-16) for row in rows]
        accuracy_ax.plot(rounds, accuracies, color=color, linestyle=style, linewidth=1.6, label=label)
        error_ax.plot(rounds, [math.log10(value) for value in errors], color=color, linestyle=style, linewidth=1.6, label=label)
        
    accuracy_ax.set(xlabel="Communication Rounds", ylabel="Test Accuracy (%)", ylim=(0, 100))
    accuracy_ax.grid(True, linestyle="--", alpha=0.4)
    accuracy_ax.legend(fontsize=8.5, loc="lower right")
    
    error_ax.set(xlabel="Communication Rounds", ylabel="log10(Relative Error / NMSE)")
    error_ax.grid(True, linestyle="--", alpha=0.4)
    error_ax.legend(fontsize=8.5, loc="upper right")
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run() -> list[dict]:
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
    torch.manual_seed(config.MODEL_SEED)
    
    client_loaders, test_loader = build_mnist_federated_data(
        root=str(ROOT / "data"), num_clients=1,
        images_per_client=config.IMAGES_PER_CLIENT, paired_shuffle=True,
    )
    train_loader = client_loaders[0]

    initial_model = MNIST_MLP()
    d = sum(parameter.numel() for parameter in initial_model.parameters())
    D = int(math.ceil(d * config.KASHIN_LAMBDA))
    kashin_frame = FourierKashinFrame(d=d, D=D, seed=config.EXPERIMENT_SEED)
    fixed_had_seed = config.EXPERIMENT_SEED + 42

    models = [copy.deepcopy(initial_model) for _ in METHODS]
    initial_row = {"round": 0}
    init_acc = evaluate_model(initial_model, test_loader) * 100
    for label, _, _ in METHODS:
        initial_row[label + ".accuracy"] = init_acc
        initial_row[label + ".relerr"] = 0.0
    rows = [initial_row]

    for round_index in range(1, ROUNDS + 1):
        round_row = {"round": round_index}
        for branch, ((label, _, _), model) in enumerate(zip(METHODS, models)):
            if branch == 0:
                round_row[label + ".accuracy"] = init_acc
                round_row[label + ".relerr"] = 0.0
                continue

            delta = client_delta(model, train_loader, round_index)
            if branch == 1:
                relerr = random_sign_round(model, delta, round_index)
            elif branch == 2:
                relerr = pure_raw_1bit_with_scale_round(model, delta)
            elif branch == 3:
                relerr = hadamard_1bit_with_scale_round(model, delta, seed=fixed_had_seed)
            else:
                relative_errors = []
                federated_round_kashin_scale_update(
                    model, [delta], k_levels=2, frame=kashin_frame, iterations=KASHIN_ITERS,
                    relerr_out=relative_errors,
                )
                relerr = relative_errors[0] if relative_errors else 0.0

            round_row[label + ".accuracy"] = evaluate_model(model, test_loader) * 100
            round_row[label + ".relerr"] = relerr
            
        rows.append(round_row)
        if round_index % 10 == 0 or round_index == ROUNDS:
            print(f"Completed communication round {round_index}/{ROUNDS}", flush=True)

    table = markdown_table(rows)
    print(table, flush=True)
    output_path = ROOT / "outputs" / "1bit_doscale.png"
    plot_results(rows, output_path)
    print(f"Figure: {output_path}")
    return rows


if __name__ == "__main__":
    run()