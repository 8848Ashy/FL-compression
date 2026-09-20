"""Standalone diagnostics for the existing SRK quantizer.

This file does not modify or invoke the main experiment.  Run from D:\\FL with
the project's fl-mnist environment activated:

    python diagnose_srk_quantization.py

Use --real-update to perform one small client update, and --run-fl for a tiny
two-round diagnostic (both are opt-in because they touch the MNIST loader).
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.plot_style import configure_chinese_plotting

configure_chinese_plotting()

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from compression.quantization import stochastic_k_level_quantize
from compression.srk import _fwht_fast, federated_round_srk_update
from data.mnist_federated import build_mnist_federated_data
from federated.local_training import local_train_delta
from federated.evaluation import evaluate_model
from models.mnist_mlp import MNIST_MLP
from utils.state_dict import flatten_state_dict
from utils.communication import srk_encoded_dimension


BITS = (2, 4, 8)
REPEATS = 100


def relative_qe(x, q):
    return float(torch.sum((x - q) ** 2) / torch.sum(x ** 2).clamp_min(1e-30))


def quantize_srk_vector(x, bits, seed=None):
    """Use the same transform path as SRK, without duplicating the algorithm."""
    d = x.numel(); D = srk_encoded_dimension(d)
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    signs = (torch.rand(D, generator=g) < 0.5).to(x.dtype).mul(2).sub(1)
    padded = torch.zeros(D, dtype=x.dtype); padded[:d] = x
    rotated = _fwht_fast(padded * signs)
    q, _ = stochastic_k_level_quantize(rotated, 2 ** bits)
    recovered = _fwht_fast(q) * signs
    return recovered[:d]


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), std=float(a.std(ddof=1)), median=float(np.median(a)),
                min=float(a.min()), max=float(a.max()))


def make_vectors(n=4096):
    g = torch.Generator().manual_seed(123)
    return {
        "gaussian": torch.randn(n, generator=g),
        "sparse": torch.where(torch.rand(n, generator=g) < .05, torch.randn(n, generator=g), torch.zeros(n)),
        "small_scale": torch.randn(n, generator=g) * 1e-3,
        "large_scale": torch.randn(n, generator=g) * 1e3,
    }


def run_vector_tests(vectors, repeats=REPEATS):
    rows = []
    for kind, x in vectors.items():
        for bits in BITS:
            values = [relative_qe(x, quantize_srk_vector(x, bits, 1000 + i)) for i in range(repeats)]
            s = stats(values); s.update(vector=kind, bits=bits)
            rows.append(s)
    return rows


def run_unbiasedness(x, repeats=1000):
    rows = []
    for bits in BITS:
        samples = torch.stack([quantize_srk_vector(x, bits, 2000 + i) for i in range(repeats)])
        bias = torch.linalg.vector_norm(samples.mean(0) - x) / torch.linalg.vector_norm(x).clamp_min(1e-30)
        rows.append({"bits": bits, "relative_bias": float(bias)})
    return rows


def run_real_update():
    loaders, _ = build_mnist_federated_data(root=str(ROOT / "data"), num_clients=1,
                                             images_per_client=64, test_size=32)
    model = MNIST_MLP(); delta = local_train_delta(model, loaders[0], epochs=1, lr=0.05)
    x, _ = flatten_state_dict(delta)
    rows = []
    for bits in BITS:
        vals = []
        for i in range(REPEATS):
            q = quantize_srk_vector(x, bits, 3000 + i)
            vals.append((float(torch.linalg.vector_norm(x - q)), relative_qe(x, q),
                         float(torch.nn.functional.cosine_similarity(x, q, dim=0))))
        arr = np.asarray(vals)
        rows.append({"bits": bits, "delta_norm": float(torch.linalg.vector_norm(x)),
                     "error_norm_mean": float(arr[:, 0].mean()), "error_norm_std": float(arr[:, 0].std(ddof=1)),
                     "normalized_qe_mean": float(arr[:, 1].mean()), "normalized_qe_std": float(arr[:, 1].std(ddof=1)),
                     "cosine_mean": float(arr[:, 2].mean()), "cosine_std": float(arr[:, 2].std(ddof=1))})
    return x, rows


def run_tiny_fl(x_delta=None, rounds=2):
    """Optional tiny diagnostic; same model/data seed, only bits vary."""
    loaders, test_loader = build_mnist_federated_data(root=str(ROOT / "data"), num_clients=1,
                                                      images_per_client=64, test_size=200)
    models = {bits: MNIST_MLP() for bits in BITS}
    torch.manual_seed(777)
    base = MNIST_MLP(); models = {bits: MNIST_MLP() for bits in BITS}
    for bits in BITS: models[bits].load_state_dict(base.state_dict())
    rows = []
    for r in range(rounds):
        for bits in BITS:
            delta = local_train_delta(models[bits], loaders[0], epochs=1, lr=0.05)
            flat, _ = flatten_state_dict(delta); q = quantize_srk_vector(flat, bits, 9000 + r)
            federated_round_srk_update(models[bits], [delta], 2 ** bits, rotation_seed=9000 + r)
            rows.append({"round": r + 1, "bits": bits,
                         "test_accuracy": evaluate_model(models[bits], test_loader),
                         "train_loss": float("nan"), "quantization_error": relative_qe(flat, q)})
    return rows


def write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-real-update", action="store_true")
    ap.add_argument("--run-fl", action="store_true"); ap.add_argument("--repeats", type=int, default=REPEATS)
    args = ap.parse_args(); out = ROOT / "diagnostic_outputs"; out.mkdir(exist_ok=True)
    print("========== SRK DIAGNOSTIC SUMMARY ==========")
    x = torch.randn(4096, generator=torch.Generator().manual_seed(42))
    y = _fwht_fast(_fwht_fast(x)); fwht_err = float(torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(x))
    print(f"FWHT reconstruction: {'PASS' if fwht_err < 1e-6 else 'FAIL'} (relative error={fwht_err:.3e})")
    print("Quantizer mapping: stochastic_k_level_quantize receives k_levels=2**bits")
    for b in BITS: print(f"bits = {b} -> levels = {2 ** b}")
    print("Thus SRK's federated_round_srk_update argument is levels, not bits.")
    print("Callers must pass 2**bits; passing 2, 4, 8 directly means 1, 2, 3 bits.")
    rows = run_vector_tests(make_vectors(), args.repeats); write_csv(out / "srk_quantization_vector_stats.csv", rows)
    print("\nQuantization error (normalized QE mean across vector classes):")
    means = {}
    for b in BITS:
        vals = [r["mean"] for r in rows if r["bits"] == b]; means[b] = float(np.mean(vals)); print(f"{b}-bit: {means[b]:.6g}")
    mono = means[8] < means[4] < means[2]; print(f"Monotonic quantization improvement: {'PASS' if mono else 'FAIL'}")
    unbiased = run_unbiasedness(x); write_csv(out / "srk_unbiasedness.csv", unbiased)
    for r in unbiased: print(f"Unbiasedness {r['bits']}-bit relative bias: {r['relative_bias']:.6g}")
    if not args.skip_real_update:
        _, real = run_real_update(); write_csv(out / "srk_real_update_stats.csv", real)
        for r in real: print(f"Real update {r['bits']}-bit QE={r['normalized_qe_mean']:.6g}±{r['normalized_qe_std']:.3g}, cosine={r['cosine_mean']:.6f}")
    if args.run_fl:
        fl = run_tiny_fl(rounds=2); write_csv(out / "srk_diagnostic_results.csv", fl)
        fig, ax = plt.subplots(figsize=(7, 4))
        for b in BITS:
            s = [r for r in fl if r["bits"] == b]; ax.plot([r["round"] for r in s], [r["test_accuracy"] * 100 for r in s], marker="o", label=f"{b}-bit")
        ax.set(xlabel="轮次", ylabel="测试准确率（%）"); ax.grid(True, linestyle="--", alpha=.5); ax.legend(); fig.tight_layout(); fig.savefig(out / "srk_accuracy_vs_round.png", dpi=200); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 4))
        for b in BITS:
            s = [r for r in fl if r["bits"] == b]; ax.plot([r["round"] for r in s], [r["quantization_error"] for r in s], marker="o", label=f"{b}-bit")
        ax.set(xlabel="轮次", ylabel="归一化量化误差"); ax.grid(True, linestyle="--", alpha=.5); ax.legend(); fig.tight_layout(); fig.savefig(out / "srk_quantization_error_vs_round.png", dpi=200); plt.close(fig)
    print("\nFinal diagnosis:")
    if not mono: print("SRK quantization implementation may be incorrect; inspect levels, stochastic rounding, scale, clipping, and FWHT normalization.")
    elif max(r["relative_bias"] for r in unbiased) > 0.05: print("QE improves, but unbiasedness bias is non-negligible; inspect stochastic rounding/scale.")
    else: print("Quantization behaves as expected; any non-monotonic FL accuracy should be evaluated as training variance with multiple seeds.")
    print("==============================================")


if __name__ == "__main__": main()
