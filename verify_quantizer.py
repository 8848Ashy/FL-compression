"""Directly verify quantizer theory on a REAL client delta.

Measures relative reconstruction error E||Q(g)-g||^2 / ||g||^2 for SRK and
Kashin pipelines at several bit-widths. Theory predicts decay ~ 1/(2^b - 1)^2.
"""
import csv
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.plot_style import configure_chinese_plotting

configure_chinese_plotting()

import config
from data.mnist_federated import build_mnist_federated_data
from models.mnist_mlp import MNIST_MLP
from federated.local_training import local_train_delta
from compression.srk import _fwht_fast
from compression.quantization import stochastic_k_level_quantize
from compression.kashin_frame import FourierKashinFrame
from compression.kashin_solver import kashin_solve
from utils.communication import srk_encoded_dimension
from utils.state_dict import flatten_state_dict

TRIALS_SRK = 20
TRIALS_KASHIN = 5
BITS = [1, 2, 4, 8, 16]


def srk_reconstruct(flat, k_levels, seed):
    d = flat.shape[0]; D = srk_encoded_dimension(d)
    generator = torch.Generator(); generator.manual_seed(seed)
    signs = ((torch.rand(D, generator=generator) < .5).float() * 2 - 1).to(flat.device)
    padded = torch.zeros(D, device=flat.device); padded[:d] = flat
    rotated = _fwht_fast(padded * signs)
    quantized, _ = stochastic_k_level_quantize(rotated, k_levels)
    return (_fwht_fast(quantized) * signs)[:d]


def kashin_reconstruct(flat, k_levels, frame):
    coefficients = kashin_solve(frame, flat, iterations=10)
    quantized, _ = stochastic_k_level_quantize(coefficients, k_levels)
    return frame.frame_synthesis(quantized)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.MODEL_SEED); np.random.seed(config.MODEL_SEED)
    client_loaders, _ = build_mnist_federated_data(num_clients=config.NUM_CLIENTS,
                                                   images_per_client=config.IMAGES_PER_CLIENT)
    model = MNIST_MLP().to(device)
    delta = local_train_delta(model, client_loaders[0], epochs=2)
    flat, _ = flatten_state_dict(delta)
    d = flat.shape[0]; g2 = float((flat ** 2).sum())
    print(f"real client delta: d={d}, ||g||={math.sqrt(g2):.4f}, device={device}", flush=True)

    frame = FourierKashinFrame(d, int(round(2.0 * d)), seed=config.EXPERIMENT_SEED)
    results = {"SRK": [], "Kashin λ=2": []}
    for bits in BITS:
        k_levels = 2 ** bits
        errs = []
        for t in range(TRIALS_SRK):
            rec = srk_reconstruct(flat, k_levels, seed=1000 + t)
            errs.append(float(((rec - flat) ** 2).sum()) / g2)
        results["SRK"].append(float(np.mean(errs)))
        errs = []
        for _ in range(TRIALS_KASHIN):
            rec = kashin_reconstruct(flat, k_levels, frame)
            errs.append(float(((rec - flat) ** 2).sum()) / g2)
        results["Kashin λ=2"].append(float(np.mean(errs)))
        print(f"bits={bits:2d}  SRK_relerr={results['SRK'][-1]:.6f}  Kashin_relerr={results['Kashin λ=2'][-1]:.6f}", flush=True)

    theory = [1.0 / (2 ** b - 1) ** 2 for b in BITS]
    theory = [t * results["SRK"][2] / theory[2] for t in theory]  # align at b=4

    out_dir = Path(__file__).resolve().parent
    with (out_dir / "results" / "quantizer_distortion.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bits", "k_levels", "SRK_relerr", "Kashin_relerr", "theory_relerr"])
        for i, bits in enumerate(BITS):
            w.writerow([bits, 2 ** bits, results["SRK"][i], results["Kashin λ=2"][i], theory[i]])
    print("[PASS] saved", out_dir / "results" / "quantizer_distortion.csv")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, errs in results.items():
        ax.plot(BITS, errs, marker="o", label=name)
    ax.plot(BITS, theory, "k--", alpha=.6, label="理论值 ~ 1/(2^b-1)^2")
    ax.set_yscale("log"); ax.set_xlabel("每个坐标的 bit 数")
    ax.set_ylabel("相对误差  E||Q(g)-g||^2 / ||g||^2")
    ax.set_title("量化器失真与 bit 数（真实客户端更新）")
    ax.grid(True, linestyle="--", alpha=.5); ax.legend(); fig.tight_layout()
    out = Path(__file__).resolve().parent / "plots" / "quantizer_distortion.png"
    fig.savefig(out, dpi=300); print("[PASS] saved", out)


if __name__ == "__main__":
    main()
