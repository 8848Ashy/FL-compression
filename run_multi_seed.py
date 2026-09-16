"""Multi-seed version of the low-bit SRK/Kashin trade-off experiment.

Runs the same experiment with several random seeds, then reports
mean +/- std of the final-round test accuracy for every configuration
and plots the trade-off figure with error bars.
"""
import csv
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import config
from data.mnist_federated import build_mnist_federated_data
from experiments.lowbit import run_kashin_lowbit_experiment

SEEDS = [0, 1, 2, 3, 4]


def main():
    root = Path(__file__).resolve().parent
    results = root / "results"; plots = root / "plots"
    results.mkdir(exist_ok=True); plots.mkdir(exist_ok=True)
    client_loaders, test_loader = build_mnist_federated_data(num_clients=config.NUM_CLIENTS, images_per_client=config.IMAGES_PER_CLIENT,
                                                             paired_shuffle=config.CRN_PAIRED)

    final_rows = []
    for seed in SEEDS:
        print(f"===== seed {seed} =====", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        rows = run_kashin_lowbit_experiment(client_loaders, test_loader, num_clients=config.NUM_CLIENTS,
                                            rounds=config.NUM_ROUNDS_FOCUS, seed=seed, save_outputs=False,
                                            crn_paired=config.CRN_PAIRED)
        last = config.NUM_ROUNDS_FOCUS
        final_rows += [{"seed": seed, "label": r["label"], "method": r["method"], "bits": r["bits"],
                        "lambda": r["lambda"], "normalized_bits": r["normalized_bits"],
                        "accuracy": r["accuracy"], "relerr2": r["relerr2"]} for r in rows if r["round"] == last]

    labels = []
    for r in final_rows:
        if r["label"] not in labels: labels.append(r["label"])
    summary = []
    for label in labels:
        group = [r for r in final_rows if r["label"] == label]
        accs = np.array([g["accuracy"] for g in group]) * 100
        errs = np.array([g["relerr2"] for g in group])
        g0 = group[0]
        summary.append({"label": label, "method": g0["method"], "bits": g0["bits"], "lambda": g0["lambda"],
                        "normalized_bits": g0["normalized_bits"], "mean_acc": float(accs.mean()),
                        "std_acc": float(accs.std()), "mean_relerr2": float(errs.mean()),
                        "std_relerr2": float(errs.std()), "n_seeds": len(accs)})

    with (results / "multi_seed_raw.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(final_rows[0])); w.writeheader(); w.writerows(final_rows)
    with (results / "multi_seed_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {2: "tab:blue", 4: "tab:orange", 8: "tab:green", 16: "tab:red"}
    markers = {2: "x", 4: "s", 8: "^", 16: "D"}
    for bits in (2, 4, 8, 16):
        srk = [s for s in summary if s["method"] == "SRK" and s["bits"] == bits]
        if srk:
            ax.errorbar(srk[0]["normalized_bits"], srk[0]["mean_acc"], yerr=srk[0]["std_acc"],
                        color="black", marker=markers[bits], markersize=9, linestyle="none",
                        capsize=4, label=f"SRK b={bits}")
        k = sorted([s for s in summary if s["method"] == "Kashin" and s["bits"] == bits], key=lambda x: x["lambda"])
        if k:
            ax.errorbar([x["normalized_bits"] for x in k], [x["mean_acc"] for x in k],
                        yerr=[x["std_acc"] for x in k], marker="o", color=colors[bits],
                        capsize=4, label=f"Kashin b={bits}")
            for x in k:
                ax.annotate(f"λ={x['lambda']:g}", (x["normalized_bits"], x["mean_acc"]),
                            xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Per-round Communication (bit/dim/client)"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"Multi-seed (n={len(SEEDS)}) Accuracy-Communication Trade-off: mean +/- std")
    ax.grid(True, linestyle="--", alpha=.5); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(plots / "multi_seed_tradeoff.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for bits in (2, 4, 8, 16):
        srk = [s for s in summary if s["method"] == "SRK" and s["bits"] == bits]
        if srk:
            ax.errorbar(srk[0]["normalized_bits"], srk[0]["mean_relerr2"], yerr=srk[0]["std_relerr2"],
                        color="black", marker=markers[bits], markersize=9, linestyle="none",
                        capsize=4, label=f"SRK b={bits}")
        k = sorted([s for s in summary if s["method"] == "Kashin" and s["bits"] == bits], key=lambda x: x["lambda"])
        if k:
            ax.errorbar([x["normalized_bits"] for x in k], [x["mean_relerr2"] for x in k],
                        yerr=[x["std_relerr2"] for x in k], marker="o", color=colors[bits],
                        capsize=4, label=f"Kashin b={bits}")
            for x in k:
                ax.annotate(f"λ={x['lambda']:g}", (x["normalized_bits"], x["mean_relerr2"]),
                            xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_yscale("log"); ax.set_xlabel("Per-round Communication (bit/dim/client)")
    ax.set_ylabel("Mean relative squared error")
    ax.set_title(f"Multi-seed (n={len(SEEDS)}) Compression Distortion: mean +/- std")
    ax.grid(True, linestyle="--", alpha=.5); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(plots / "multi_seed_relerr2.png", dpi=300); plt.close(fig)

    print("[PASS] multi-seed summary + figures saved")
    for s in summary:
        print(f"{s['label']:28s} {s['mean_acc']:.2f} +/- {s['std_acc']:.2f}")


if __name__ == "__main__":
    main()
