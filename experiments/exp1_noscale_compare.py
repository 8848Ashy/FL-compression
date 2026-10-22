"""Comprehensive Experiment: No-Scale Multi-bit (1, 2, 3 bit) & Variable Kashin Lambda.
Fixed: Fixed Hadamard rotation seed (prevents NMSE explosion) + Stochastic Quantization (monotonic rate-accuracy).
"""
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
from compression.kashin_frame import FourierKashinFrame
from compression.kashin_solver import kashin_solve
from data.mnist_federated import build_mnist_federated_data
from federated.evaluation import evaluate_model
from federated.local_training import local_train_delta
from models.mnist_mlp import MNIST_MLP
from utils.state_dict import (flatten_state_dict, state_dict_add,
                              unflatten_state_dict)

# ==================== 实验全局超参数配置 ====================
ROUNDS = config.NUM_ROUNDS_FOCUS
CLIP_MODE = "auto_warmup"
HADAMARD_COUNT_PADDING_IN_BPD = True


# ==================== 核心数学工具：随机量化与固定旋转 FWHT ====================

def quantize_static_noscale(tensor: torch.Tensor, bits: int, clip_val: float) -> torch.Tensor:
    """引入无偏随机量化（Stochastic Quantization），彻底消除后期死区效应，保证多比特单调增益"""
    if bits < 1:
        return torch.zeros_like(tensor)

    if bits == 1:
        return torch.where(tensor >= 0,
                           torch.full_like(tensor, clip_val),
                           torch.full_like(tensor, -clip_val))

    num_levels = 1 << bits  # 2^bits
    clipped = torch.clamp(tensor, -clip_val, clip_val)
    step = (2.0 * clip_val) / (num_levels - 1)
    
    # 随机舍入（Stochastic Rounding）
    scaled_pos = (clipped + clip_val) / step
    lower_idx = scaled_pos.floor().clamp(0, num_levels - 2)
    upper_idx = lower_idx + 1.0
    prob = scaled_pos - lower_idx
    rand_mask = torch.rand_like(tensor) < prob
    chosen_idx = torch.where(rand_mask, upper_idx, lower_idx)
    
    quantized = -clip_val + chosen_idx * step
    return quantized


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
    """客户端：使用固定种子生成正交随机哈达玛旋转矩阵，保证全过程坐标系稳定"""
    d = flat.numel()
    D_had = 1 << (d - 1).bit_length()  # 50890 -> 65536
    padded = torch.zeros(D_had, device=flat.device, dtype=flat.dtype)
    padded[:d] = flat

    g = torch.Generator(device=flat.device).manual_seed(seed)
    signs = torch.randint(0, 2, (D_had,), generator=g, device=flat.device, dtype=flat.dtype) * 2.0 - 1.0
    rotated = fwht_1d(padded * signs)
    return rotated, d


def hadamard_inverse(rotated_q: torch.Tensor, d: int, seed: int) -> torch.Tensor:
    D_had = rotated_q.numel()
    g = torch.Generator(device=rotated_q.device).manual_seed(seed)
    signs = torch.randint(0, 2, (D_had,), generator=g, device=rotated_q.device, dtype=rotated_q.dtype) * 2.0 - 1.0
    unrotated = fwht_1d(rotated_q) * signs
    return unrotated[:d]


# ==================== 实验分支定义 ====================

def build_experiment_specs(d: int) -> list[dict]:
    D_had = 1 << (d - 1).bit_length()
    had_ratio = (D_had / d) if HADAMARD_COUNT_PADDING_IN_BPD else 1.0

    specs = [
        {"name": "Static (0-bit)", "family": "Static", "bits": 0, "lam": 0.0, "bpd": 0.0,
         "color": "gray", "ls": "--", "marker": "x"},

        {"name": "Raw/SignSGD (1-bit)", "family": "SignSGD (Raw)", "bits": 1, "lam": 1.0, "bpd": 1.0,
         "color": "#d62728", "ls": ":", "marker": "o"},
        {"name": "Raw/SignSGD (2-bit)", "family": "SignSGD (Raw)", "bits": 2, "lam": 1.0, "bpd": 2.0,
         "color": "#d62728", "ls": "--", "marker": "o"},
        {"name": "Raw/SignSGD (3-bit)", "family": "SignSGD (Raw)", "bits": 3, "lam": 1.0, "bpd": 3.0,
         "color": "#d62728", "ls": "-", "marker": "o"},

        {"name": "Hadamard (1-bit)", "family": "Hadamard", "bits": 1, "lam": had_ratio, "bpd": 1.0 * had_ratio,
         "color": "#1f77b4", "ls": ":", "marker": "s"},
        {"name": "Hadamard (2-bit)", "family": "Hadamard", "bits": 2, "lam": had_ratio, "bpd": 2.0 * had_ratio,
         "color": "#1f77b4", "ls": "--", "marker": "s"},
        {"name": "Hadamard (3-bit)", "family": "Hadamard", "bits": 3, "lam": had_ratio, "bpd": 3.0 * had_ratio,
         "color": "#1f77b4", "ls": "-", "marker": "s"},

        {"name": "Kashin (lam=1.3, 1-bit)", "family": "Kashin (Fixed lam=1.3, Vary Bits)", "bits": 1, "lam": 1.3, "bpd": 1.3 * 1,
         "color": "#2ca02c", "ls": ":", "marker": "^"},
        {"name": "Kashin (lam=1.3, 2-bit)", "family": "Kashin (Fixed lam=1.3, Vary Bits)", "bits": 2, "lam": 1.3, "bpd": 1.3 * 2,
         "color": "#2ca02c", "ls": "--", "marker": "^"},
        {"name": "Kashin (lam=1.3, 3-bit)", "family": "Kashin (Fixed lam=1.3, Vary Bits)", "bits": 3, "lam": 1.3, "bpd": 1.3 * 3,
         "color": "#2ca02c", "ls": "-", "marker": "^"},

        {"name": "Kashin (lam=1.15, 2-bit)", "family": "Kashin (Fixed 2-bit, Vary lam)", "bits": 2, "lam": 1.15, "bpd": 1.15 * 2,
         "color": "#9467bd", "ls": ":", "marker": "D"},
        {"name": "Kashin (lam=1.60, 2-bit)", "family": "Kashin (Fixed 2-bit, Vary lam)", "bits": 2, "lam": 1.60, "bpd": 1.60 * 2,
         "color": "#9467bd", "ls": "--", "marker": "D"},
        {"name": "Kashin (lam=2.00, 2-bit)", "family": "Kashin (Fixed 2-bit, Vary lam)", "bits": 2, "lam": 2.00, "bpd": 2.00 * 2,
         "color": "#9467bd", "ls": "-", "marker": "D"},
    ]
    return specs


# ==================== 绘图函数 ====================

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


def plot_training_curves(rows: list[dict], specs: list[dict], output_path: Path) -> None:
    rounds = [row["round"] for row in rows]
    fig, (acc_ax, err_ax) = plt.subplots(1, 2, figsize=(16, 6.2))
    fig.suptitle("No-Scale Multi-bit Compression: Training Dynamics Over Rounds",
                 fontsize=14, fontweight="bold", y=0.98)

    for spec in specs:
        label = spec["name"]
        accs = [row[label + ".accuracy"] for row in rows]
        errs = [max(row[label + ".relerr"], 1e-16) for row in rows]
        acc_ax.plot(rounds, accs, color=spec["color"], linestyle=spec["ls"], linewidth=1.6, label=label)
        err_ax.plot(rounds, [math.log10(v) for v in errs], color=spec["color"], linestyle=spec["ls"], linewidth=1.6, label=label)

    acc_ax.set(xlabel="Communication Rounds", ylabel="Test Accuracy (%)", ylim=(0, 100))
    acc_ax.grid(True, linestyle="--", alpha=0.4)
    acc_ax.legend(fontsize=7.5, loc="lower right", ncol=2)

    err_ax.set(xlabel="Communication Rounds", ylabel="log10(Relative Error / NMSE)")
    err_ax.grid(True, linestyle="--", alpha=0.4)
    err_ax.legend(fontsize=7.5, loc="upper right", ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_bpd(rows: list[dict], specs: list[dict], output_path: Path) -> None:
    final_row = rows[-1]
    fig, ax = plt.subplots(figsize=(11.5, 6.8))

    families = {}
    for spec in specs:
        fam = spec["family"]
        if fam == "Static":
            continue
        families.setdefault(fam, []).append(spec)

    kashin_lam13_2bit = next(s for s in specs if s["name"] == "Kashin (lam=1.3, 2-bit)")
    if "Kashin (Fixed 2-bit, Vary lam)" in families:
        families["Kashin (Fixed 2-bit, Vary lam)"].append(kashin_lam13_2bit)

    family_styles = {
        "SignSGD (Raw)": ("#d62728", "-o"),
        "Hadamard": ("#1f77b4", "-s"),
        "Kashin (Fixed lam=1.3, Vary Bits)": ("#2ca02c", "-^"),
        "Kashin (Fixed 2-bit, Vary lam)": ("#9467bd", "--D"),
    }

    custom_offsets = {
        "Raw/SignSGD (1-bit)": (-10, 22),
        "Raw/SignSGD (2-bit)": (-18, 24),
        "Raw/SignSGD (3-bit)": (-15, 24),
        "Hadamard (1-bit)": (-15, -34),
        "Hadamard (2-bit)": (-28, -36),
        "Hadamard (3-bit)": (-35, -36),
        "Kashin (lam=1.3, 1-bit)": (28, -12),
        "Kashin (lam=1.3, 2-bit)": (-5, 26),
        "Kashin (lam=1.3, 3-bit)": (-32, 24),
        "Kashin (lam=1.15, 2-bit)": (22, -28),
        "Kashin (lam=1.60, 2-bit)": (10, 24),
        "Kashin (lam=2.00, 2-bit)": (26, -20),
    }

    annotated_names = set()
    all_active_ys = []

    for fam_name, fam_specs in families.items():
        fam_specs_sorted = sorted(fam_specs, key=lambda s: s["bpd"])
        xs = [s["bpd"] for s in fam_specs_sorted]
        ys = [final_row[s["name"] + ".accuracy"] for s in fam_specs_sorted]
        all_active_ys.extend(ys)
        color, fmt = family_styles.get(fam_name, ("black", "-o"))

        ax.plot(xs, ys, fmt, color=color, linewidth=2.2, markersize=8.5, label=fam_name, zorder=3)

        for s, x, y in zip(fam_specs_sorted, xs, ys):
            name = s["name"]
            if name in annotated_names:
                continue
            annotated_names.add(name)

            if "SignSGD" in name:
                tag = f"Raw {s['bits']}b\n({y:.1f}%)"
            elif "Hadamard" in name:
                tag = f"Had {s['bits']}b [{x:.2f}]\n({y:.1f}%)"
            else:
                tag = f"λ={s['lam']}, {s['bits']}b\n({y:.1f}%)"

            dx, dy = custom_offsets.get(name, (0, 18))
            ax.annotate(
                tag,
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8.5,
                color=color,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color, alpha=0.88, lw=0.8),
                arrowprops=dict(arrowstyle="-", color=color, lw=0.9, alpha=0.7),
                zorder=5,
            )

    min_y = max(0.0, math.floor((min(all_active_ys) - 7.0) / 5.0) * 5.0)
    max_y = min(100.0, math.ceil((max(all_active_ys) + 7.0) / 5.0) * 5.0)

    static_acc = final_row["Static (0-bit).accuracy"]
    ax.plot([], [], color="gray", linestyle="--", label=f"Static 0-bit Baseline ({static_acc:.1f}% - off-scale)")

    ax.set_title("Rate-Accuracy Trade-off (No-Scale): Test Accuracy vs. Bits per Dimension",
                 fontsize=13.5, fontweight="bold", pad=14)
    ax.set_xlabel("Communication Budget: Bits per Dimension (bpd = D/d × bits)", fontsize=11.5)
    ax.set_ylabel(f"Final Test Accuracy at Round {ROUNDS} (%)", fontsize=11.5)
    ax.set_xlim(0.75, 4.35)
    ax.set_ylim(min_y, max_y)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax.legend(fontsize=9.5, loc="lower right", framealpha=0.95)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


# ==================== 主流程执行 ====================

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

    specs = build_experiment_specs(d)

    kashin_frames: dict[float, FourierKashinFrame] = {}
    for spec in specs:
        if "Kashin" in spec["name"]:
            lam = spec["lam"]
            if lam not in kashin_frames:
                D_kashin = int(math.ceil(d * lam))
                kashin_frames[lam] = FourierKashinFrame(d=d, D=D_kashin, seed=config.EXPERIMENT_SEED)

    models = [copy.deepcopy(initial_model) for _ in specs]
    static_clips: dict[str, float] = {}

    initial_row = {"round": 0}
    init_acc = evaluate_model(initial_model, test_loader) * 100
    for spec in specs:
        initial_row[spec["name"] + ".accuracy"] = init_acc
        initial_row[spec["name"] + ".relerr"] = 0.0
    rows = [initial_row]

    print(f"Starting fixed experiment with {len(specs)} configurations over {ROUNDS} rounds...", flush=True)

    for round_index in range(1, ROUNDS + 1):
        round_row = {"round": round_index}
        # 【核心修复】：固定 Hadamard 的旋转种子，防止全过程坐标系漂移导致 3-bit  NMSE 爆炸
        fixed_had_seed = config.EXPERIMENT_SEED + 42

        for idx, spec in enumerate(specs):
            name = spec["name"]
            bits = spec["bits"]
            model = models[idx]

            if bits == 0:
                round_row[name + ".accuracy"] = init_acc
                round_row[name + ".relerr"] = 0.0
                continue

            delta = client_delta(model, train_loader, round_index)
            flat, shapes = flatten_state_dict(delta)

            if "SignSGD" in spec["family"]:
                if name not in static_clips:
                    static_clips[name] = float(flat.abs().quantile(0.995))
                q_flat = quantize_static_noscale(flat, bits=bits, clip_val=static_clips[name])
                reconstructed = q_flat

            elif "Hadamard" in spec["family"]:
                rotated, orig_d = hadamard_forward(flat, seed=fixed_had_seed)
                if name not in static_clips:
                    static_clips[name] = float(rotated.abs().quantile(0.995))
                q_rot = quantize_static_noscale(rotated, bits=bits, clip_val=static_clips[name])
                reconstructed = hadamard_inverse(q_rot, d=orig_d, seed=fixed_had_seed)

            elif "Kashin" in spec["family"]:
                frame = kashin_frames[spec["lam"]]
                # 【核心修复】：将迭代次数提升至 25，确保高维紧框架重构精度
                coeffs = kashin_solve(frame, flat, iterations=25)
                if name not in static_clips:
                    static_clips[name] = float(coeffs.abs().quantile(0.995))
                q_coeffs = quantize_static_noscale(coeffs, bits=bits, clip_val=static_clips[name])
                reconstructed = frame.frame_synthesis(q_coeffs)

            else:
                raise ValueError(f"Unknown spec family: {spec['family']}")

            relerr = float((reconstructed - flat).square().sum() / flat.square().sum().clamp_min(1e-12))
            model.load_state_dict(state_dict_add(model.state_dict(), unflatten_state_dict(reconstructed, shapes)))

            round_row[name + ".accuracy"] = evaluate_model(model, test_loader) * 100
            round_row[name + ".relerr"] = relerr

        rows.append(round_row)
        if round_index % 5 == 0 or round_index == ROUNDS:
            print(f"Completed communication round {round_index}/{ROUNDS}", flush=True)

    print("\n================ Final Round Summary ================")
    print(f"{'Method Name':<28} | {'bpd':<6} | {'Final Acc (%)':<14} | {'Final NMSE':<12}")
    print("-" * 68)
    for spec in specs:
        name = spec["name"]
        bpd = spec["bpd"]
        acc = rows[-1][name + ".accuracy"]
        err = rows[-1][name + ".relerr"]
        print(f"{name:<28} | {bpd:<6.2f} | {acc:<14.2f} | {err:<12.6g}")

    curves_path = ROOT / "outputs" / "noscale_multibit_training_curves.png"
    bpd_path = ROOT / "outputs" / "noscale_accuracy_vs_bpd.png"
    plot_training_curves(rows, specs, curves_path)
    plot_accuracy_vs_bpd(rows, specs, bpd_path)
    print(f"\nSaved Training Curves Figure to: {curves_path}")
    print(f"Saved Accuracy vs. bpd Figure to: {bpd_path}")
    return rows


if __name__ == "__main__":
    run()