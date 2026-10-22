"""Control Experiment: Static vs. Random Sign vs. Pure Raw 1-bit vs. Hadamard vs. Kashin (Strictly Unified Static Threshold)."""
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

import config
from compression.kashin_solver import kashin_solve
from compression.kashin_frame import FourierKashinFrame
from data.mnist_federated import build_mnist_federated_data
from federated.evaluation import evaluate_model
from federated.local_training import local_train_delta
from models.mnist_mlp import MNIST_MLP
from utils.state_dict import (flatten_state_dict, state_dict_add,
                              unflatten_state_dict)

ROUNDS = config.NUM_ROUNDS_FOCUS
KASHIN_ITERS = 25  # 对齐 exp1 的 25 次迭代以保证紧框架收敛精度

METHODS = (
    ("Static (0-bit)", "gray", "--"),                 
    ("Random Sign", "orange", "--"),      
    ("SignSGD (Raw 1-bit)", "red", "-"),                 
    ("Hadamard 1-bit", "blue", "-"),      
    ("Kashin 1-bit", "green", "-"), 
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


def quantize_1bit_with_threshold(tensor: torch.Tensor, clip_val: float) -> torch.Tensor:
    """带静态截断边界（Threshold / clip_val）的 1-bit 符号量化"""
    return torch.where(tensor >= 0,
                       torch.full_like(tensor, clip_val),
                       torch.full_like(tensor, -clip_val))


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


def plot_results(rows: list[dict], output_path: Path) -> None:
    rounds = [row["round"] for row in rows]
    fig, (accuracy_ax, error_ax) = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle("1-bit No-Scale Quantization (Unified Static Threshold)", fontsize=14, fontweight="bold", y=0.98)
    
    for label, color, style in METHODS:
        accuracies = [row[label + ".accuracy"] for row in rows]
        errors = [max(row[label + ".relerr"], 1e-16) for row in rows]
        accuracy_ax.plot(rounds, accuracies, color=color, linestyle=style, linewidth=1.5, label=label)
        error_ax.plot(rounds, [math.log10(value) for value in errors], color=color, linestyle=style, linewidth=1.5, label=label)
        
    accuracy_ax.set(xlabel="Communication Rounds", ylabel="Test Accuracy (%)",
                    ylim=(0, 100), xticks=rounds[::max(1, len(rounds)//10)])
    accuracy_ax.grid(True, linestyle="--", alpha=0.4)
    accuracy_ax.legend(fontsize=8)
    error_ax.set(xlabel="Communication Rounds",
                 ylabel="log10(Relative Error / NMSE)", xticks=rounds[::max(1, len(rounds)//10)])
    error_ax.grid(True, linestyle="--", alpha=0.4)
    error_ax.legend(fontsize=8)
    
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
    d = sum(p.numel() for p in initial_model.parameters())
    if d != 50890:
        raise RuntimeError(f"Expected MNIST_MLP dimension 50890, found {d}")

    D = int(math.ceil(d * config.KASHIN_LAMBDA))
    kashin_frame = FourierKashinFrame(d=d, D=D, seed=config.EXPERIMENT_SEED)
    fixed_had_seed = config.EXPERIMENT_SEED + 42

    models = [copy.deepcopy(initial_model) for _ in METHODS]
    static_clips: dict[str, float] = {}

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
            flat, shapes = flatten_state_dict(delta)

            if branch == 1:  # Random Sign (使用与 Raw 相同的静态幅度标尺，仅方向随机)
                if label not in static_clips:
                    static_clips[label] = float(flat.abs().quantile(0.995))
                curr_clip = static_clips[label]
                g = torch.Generator(device=flat.device).manual_seed(config.EXPERIMENT_SEED + round_index * 777)
                random_bits = torch.randint(0, 2, flat.shape, generator=g, device=flat.device)
                reconstructed = torch.where(random_bits == 1, torch.full_like(flat, curr_clip), torch.full_like(flat, -curr_clip))

            elif branch == 2:  # SignSGD (Raw 1-bit)
                if label not in static_clips:
                    static_clips[label] = float(flat.abs().quantile(0.995))
                reconstructed = quantize_1bit_with_threshold(flat, static_clips[label])

            elif branch == 3:  # Hadamard 1-bit (真正应用静态 Threshold)
                rotated, orig_d = hadamard_forward(flat, seed=fixed_had_seed)
                if label not in static_clips:
                    static_clips[label] = float(rotated.abs().quantile(0.995))
                q_rot = quantize_1bit_with_threshold(rotated, static_clips[label])
                reconstructed = hadamard_inverse(q_rot, d=orig_d, seed=fixed_had_seed)

            elif branch == 4:  # Kashin 1-bit
                coeffs = kashin_solve(kashin_frame, flat, iterations=KASHIN_ITERS)
                if label not in static_clips:
                    static_clips[label] = float(coeffs.abs().quantile(0.995))
                q_coeffs = quantize_1bit_with_threshold(coeffs, static_clips[label])
                reconstructed = kashin_frame.frame_synthesis(q_coeffs)

            relerr = float((reconstructed - flat).square().sum() / flat.square().sum().clamp_min(1e-12))
            model.load_state_dict(state_dict_add(model.state_dict(), unflatten_state_dict(reconstructed, shapes)))

            round_row[label + ".accuracy"] = evaluate_model(model, test_loader) * 100
            round_row[label + ".relerr"] = relerr

        rows.append(round_row)
        if round_index % 10 == 0 or round_index == ROUNDS:
            print(f"Completed communication round {round_index}/{ROUNDS}", flush=True)

    table = markdown_table(rows)
    print(table, flush=True)
    output_path = ROOT / "outputs" / "exp0threshold.png"
    plot_results(rows, output_path)
    print(f"Figure: {output_path}")
    return rows


if __name__ == "__main__":
    run()