"""Multi-seed version of the low-bit SRK/Kashin trade-off experiment.

Runs the same experiment with several random seeds, then reports
mean +/- std of the final-round test accuracy for every configuration
and plots the trade-off figure with error bars.
"""
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.plot_style import configure_chinese_plotting

configure_chinese_plotting()

import config
from data.mnist_federated import build_mnist_federated_data
from experiments.lowbit import run_kashin_lowbit_experiment

SEEDS = [0, 1, 2, 3, 4]
FIXED_BUDGETS = [8, 16, 32, 64, 128, 256, 512]


def main():
    root = Path(__file__).resolve().parent
    results = root / "results"; plots = root / "plots"
    results.mkdir(exist_ok=True); plots.mkdir(exist_ok=True)
    client_loaders, test_loader = build_mnist_federated_data(num_clients=config.LOWBIT_NUM_CLIENTS, images_per_client=config.LOWBIT_IMAGES_PER_CLIENT,
                                                             paired_shuffle=config.CRN_PAIRED)

    final_rows = []
    fixed_budget_rows = []
    for seed in SEEDS:
        print(f"===== seed {seed} =====", flush=True)
        torch.manual_seed(seed); np.random.seed(seed)
        rows = run_kashin_lowbit_experiment(client_loaders, test_loader, num_clients=config.LOWBIT_NUM_CLIENTS,
                                            rounds=config.NUM_ROUNDS_FOCUS, seed=seed, save_outputs=False,
                                            crn_paired=config.CRN_PAIRED)
        last = config.NUM_ROUNDS_FOCUS
        final_rows += [{"seed": seed, "label": r["label"], "method": r["method"], "bits": r["bits"],
                        "lambda": r["lambda"], "normalized_bits": r["normalized_bits"],
                        "accuracy": r["accuracy"], "relerr2": r["relerr2"]} for r in rows if r["round"] == last]
        for budget in FIXED_BUDGETS:
            for label in sorted({r["label"] for r in rows}):
                eligible = [r for r in rows if r["label"] == label and r["cumulative_normalized_bits"] <= budget]
                if not eligible:
                    continue
                point = max(eligible, key=lambda r: r["round"])
                fixed_budget_rows.append({"seed": seed, "budget_normalized_bits": budget,
                                          "label": point["label"], "method": point["method"],
                                          "bits": point["bits"], "lambda": point["lambda"],
                                          "round": point["round"], "accuracy": point["accuracy"]})

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
    with (results / "multi_seed_fixed_budget_raw.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fixed_budget_rows[0])); w.writeheader(); w.writerows(fixed_budget_rows)

    frontier_by_seed = []
    for seed in SEEDS:
        for budget in FIXED_BUDGETS:
            for method in ("SRK", "Kashin", "Original"):
                candidates = [x for x in fixed_budget_rows if x["seed"] == seed
                              and x["budget_normalized_bits"] == budget and x["method"] == method]
                if candidates:
                    frontier_by_seed.append({**max(candidates, key=lambda x: x["accuracy"]),
                                             "frontier_method": method})
    frontier_summary = []
    for budget in FIXED_BUDGETS:
        for method in ("SRK", "Kashin", "Original"):
            values = [x["accuracy"] * 100 for x in frontier_by_seed
                      if x["budget_normalized_bits"] == budget and x["frontier_method"] == method]
            if values:
                frontier_summary.append({"budget_normalized_bits": budget, "method": method,
                                         "mean_acc": float(np.mean(values)),
                                         "std_acc": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                                         "n_seeds": len(values)})
    with (results / "multi_seed_fixed_budget_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(frontier_summary[0])); w.writeheader(); w.writerows(frontier_summary)

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
    ax.set_xlabel("每轮通信量（bit/维度/客户端）"); ax.set_ylabel("测试准确率（%）")
    ax.set_title(f"多种子（n={len(SEEDS)}）准确率-通信量权衡：均值 ± 标准差")
    ax.grid(True, linestyle="--", alpha=.5); ax.legend(fontsize=8); fig.tight_layout()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(plots / f"{run_stamp}_multi_seed_tradeoff.png", dpi=300); plt.close(fig)

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
    ax.set_yscale("log"); ax.set_xlabel("每轮通信量（bit/维度/客户端）")
    ax.set_ylabel("平均相对平方误差")
    ax.set_title(f"多种子（n={len(SEEDS)}）压缩失真：均值 ± 标准差")
    ax.grid(True, linestyle="--", alpha=.5); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(plots / f"{run_stamp}_multi_seed_relerr2.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    frontier_colors = {"SRK": "tab:blue", "Kashin": "tab:orange", "Original": "tab:green"}
    for method, color in frontier_colors.items():
        series = [x for x in frontier_summary if x["method"] == method]
        if series:
            ax.errorbar([x["budget_normalized_bits"] for x in series], [x["mean_acc"] for x in series],
                        yerr=[x["std_acc"] for x in series], marker="o", capsize=4,
                        color=color, label=f"{method} 前沿")
    ax.set_xlabel("固定累计通信预算（bit/维度/客户端）")
    ax.set_ylabel("预算内最后一轮的准确率（%）")
    ax.set_title(f"多种子固定预算下的准确率权衡（n={len(SEEDS)}）")
    ax.grid(True, linestyle="--", alpha=.5); ax.legend(); fig.tight_layout()
    fig.savefig(plots / f"{run_stamp}_multi_seed_fixed_budget.png", dpi=300); plt.close(fig)

    print("[PASS] multi-seed summary + figures saved")
    for s in summary:
        print(f"{s['label']:28s} {s['mean_acc']:.2f} +/- {s['std_acc']:.2f}")


if __name__ == "__main__":
    main()
