"""Plot accuracy and compression metrics from ``run_experiment.py`` outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runs(root: Path):
    runs = []
    for hparams_path in sorted(root.glob("**/hparams.json")):
        metrics_path = hparams_path.with_name("metrics.csv")
        if not metrics_path.exists():
            continue
        params = json.loads(hparams_path.read_text(encoding="utf-8"))
        with metrics_path.open(encoding="utf-8", newline="") as stream:
            rows = [{k: float(v) if v else float("nan") for k, v in row.items()}
                    for row in csv.DictReader(stream)]
        if rows:
            runs.append((params, rows))
    return runs


def plot(root: Path, output_dir: Path):
    runs = load_runs(root)
    if not runs:
        raise FileNotFoundError(f"No completed runs with hparams.json and metrics.csv under {root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for index, (params, rows) in enumerate(runs):
        label = params.get("exp_name", params.get("transform", f"run-{index}"))
        color = colors[index % len(colors)]
        rounds = [r["round"] for r in rows]
        cumulative = [r["cumulative_bpd"] for r in rows]
        acc = [r["test_acc"] for r in rows]
        nmse = [max(r["mean_nmse"], 1e-18) for r in rows]
        axes[0, 0].plot(rounds, acc, label=label, color=color)
        axes[0, 1].plot(rounds, nmse, label=label, color=color)
        axes[1, 0].plot(cumulative, acc, label=label, color=color)
        axes[1, 1].plot(cumulative, nmse, label=label, color=color)
    axes[0, 0].set(xlabel="Communication rounds", ylabel="Test accuracy", title="Accuracy vs rounds")
    axes[0, 1].set(xlabel="Communication rounds", ylabel="Mean client NMSE", title="NMSE vs rounds", yscale="log")
    axes[1, 0].set(xlabel="Cumulative bits per dimension", ylabel="Test accuracy", title="Accuracy vs cumulative BPD")
    axes[1, 1].set(xlabel="Cumulative bits per dimension", ylabel="Mean client NMSE", title="NMSE vs cumulative BPD", yscale="log")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_rounds.png", dpi=220)
    plt.close(fig)

    grouped = {}
    for params, rows in runs:
        transform = params.get("transform", "unknown")
        tail = rows[-10:]
        x = sum(r["bpd"] for r in tail) / len(tail)
        y = sum(r["test_acc"] for r in tail) / len(tail)
        grouped.setdefault(transform, []).append((x, y, params.get("bits", 1), params.get("lam", 1.0)))
    fig, ax = plt.subplots(figsize=(9, 6))
    for transform, points in sorted(grouped.items()):
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=transform)
        for x, y, bits, lam in points:
            ax.annotate(f"{bits}b, λ={lam:g}", (x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7)
    ax.set(xlabel="Bits per dimension", ylabel="Final test accuracy (last 10 round mean)",
           title="Accuracy vs communication budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_bpd.png", dpi=220)
    plt.close(fig)
    return output_dir / "accuracy_vs_rounds.png", output_dir / "accuracy_vs_bpd.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("./results/default"))
    parser.add_argument("--figure_dir", type=Path, default=None)
    args = parser.parse_args()
    paths = plot(args.output_dir, args.figure_dir or args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
