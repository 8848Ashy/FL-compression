"""Plot accuracy and compression metrics from ``run_experiment.py`` outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
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

    points = []
    for params, rows in runs:
        tail = rows[-10:]
        points.append({
            "transform": params.get("transform", "unknown"),
            "scale_mode": params.get("scale_mode", "unknown"),
            "bits": int(params.get("bits", 1)),
            "lam": float(params.get("lam", 1.0)),
            "bpd": sum(r["bpd"] for r in tail) / len(tail),
            "accuracy": sum(r["test_acc"] for r in tail) / len(tail),
            "nmse": sum(r["mean_nmse"] for r in tail) / len(tail),
        })

    colors = {"raw": "#d62728", "hadamard": "#1f77b4",
              "fourier_kashin": "#2ca02c", "fourier_kashin_balanced": "#9467bd"}
    scale_modes = sorted({point["scale_mode"] for point in points})

    def draw_metric(ax, metric):
        for scale_index, scale_mode in enumerate(scale_modes):
            subset = [p for p in points if p["scale_mode"] == scale_mode]
            suffix = f" ({scale_mode})" if len(scale_modes) > 1 else ""
            for transform in ("raw", "hadamard"):
                series = sorted((p for p in subset if p["transform"] == transform),
                                key=lambda p: p["bits"])
                if series:
                    ax.plot([p["bpd"] for p in series], [p[metric] for p in series],
                            marker="o", color=colors[transform],
                            linestyle="-" if transform == "raw" else "--",
                            label=f"{transform}{suffix}")

            for transform in ("fourier_kashin", "fourier_kashin_balanced"):
                kashin = [p for p in subset if p["transform"] == transform]
                if not kashin:
                    continue
                fixed_bit = sorted((p for p in kashin if p["bits"] == 1), key=lambda p: p["lam"])
                fixed_lam = sorted((p for p in kashin if math.isclose(p["lam"], 1.3, abs_tol=1e-6)),
                                   key=lambda p: p["bits"])
                color = colors[transform]
                if fixed_bit:
                    ax.plot([p["bpd"] for p in fixed_bit], [p[metric] for p in fixed_bit],
                            marker="^", color=color, linestyle="-",
                            label=f"{transform}, 1 bit, varying λ{suffix}")
                if fixed_lam:
                    ax.plot([p["bpd"] for p in fixed_lam], [p[metric] for p in fixed_lam],
                            marker="s", color=color, linestyle=":",
                            label=f"{transform}, λ=1.3, varying bits{suffix}")
                # Pareto frontier: minimize BPD and maximize the selected metric.
                unique = {}
                for p in kashin:
                    key = round(p["bpd"], 9)
                    if key not in unique:
                        unique[key] = p
                    elif ((metric == "accuracy" and p[metric] > unique[key][metric])
                          or (metric == "nmse" and p[metric] < unique[key][metric])):
                        unique[key] = p
                frontier = []
                best_score = -math.inf
                for p in sorted(unique.values(), key=lambda row: row["bpd"]):
                    score = p[metric] if metric == "accuracy" else -p[metric]
                    if score > best_score:
                        frontier.append(p)
                        best_score = score
                if len(frontier) > 1:
                    ax.plot([p["bpd"] for p in frontier], [p[metric] for p in frontier],
                            color=color, linewidth=2.4, alpha=0.75,
                            label=f"{transform} Pareto frontier{suffix}")

        baseline = [p[metric] for p in points if p["transform"] == "uncompressed"]
        if baseline and metric == "accuracy":
            ax.axhline(sum(baseline) / len(baseline), color="black", linestyle="--",
                       label="Uncompressed FedAvg (32 BPD)")
        ax.set_xlabel("Bits per dimension (BPD)")
        ax.set_ylabel("Last-10-round mean accuracy" if metric == "accuracy"
                      else "Last-10-round mean client NMSE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    draw_metric(ax, "accuracy")
    ax.set_title("Accuracy vs communication budget")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_bpd.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    draw_metric(ax, "nmse")
    ax.set_yscale("log")
    ax.set_title("Round mean NMSE vs communication budget")
    fig.tight_layout()
    fig.savefig(output_dir / "nmse_vs_bpd.png", dpi=220)
    plt.close(fig)
    return (output_dir / "accuracy_vs_rounds.png", output_dir / "accuracy_vs_bpd.png",
            output_dir / "nmse_vs_bpd.png")


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
