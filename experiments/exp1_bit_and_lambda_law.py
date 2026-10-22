"""Isolated single-vector study of quantization error versus bit width/lambda."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compression.vector_coders import (
    METHODS,
    evaluate_prepared_vector_compression,
    prepare_single_vector_compression,
)
from compression.kashin_frame import FourierKashinFrame

D = 50890
BITS = (1, 2, 3, 4, 6, 8)
LAMBDAS = (1.0, 1.5, 2.0)
QUANT_SEEDS = (11, 23, 37, 51, 79)
METHOD_NAMES = {
    "hadamard_srk": "SRK (Hadamard)",
    "fourier_srk": "Fourier-SRK",
    "fourier_kashin_historical": "Fourier-Kashin historical",
    "fourier_kashin_balanced": "Fourier-Kashin balanced",
}


def build_vectors(root: Path, synthetic_only: bool = False) -> dict[str, tuple[torch.Tensor, str]]:
    generator = torch.Generator().manual_seed(20261018)
    heavy_tail = torch.randn(D, generator=generator).pow(3)
    sparse_mask = torch.rand(D, generator=generator) < 0.005
    background = torch.randn(D, generator=generator) * 0.01
    spikes = torch.randn(D, generator=generator) * 10.0
    spike_vector = torch.where(sparse_mask, spikes, background)
    vectors = {
        "HeavyTail_Vector": (heavy_tail, "torch.randn(d, seed=20261018)**3"),
        "Real_Or_Spike_Vector": (spike_vector, "synthetic sparse-spike fallback"),
    }

    mnist_raw = root / "data" / "MNIST" / "raw" / "train-images-idx3-ubyte"
    if not synthetic_only and mnist_raw.exists():
        from data.mnist_federated import build_mnist_federated_data
        from federated.local_training import local_train_delta
        from models.mnist_mlp import MNIST_MLP
        from utils.state_dict import flatten_state_dict

        torch.manual_seed(20261018)
        clients, _ = build_mnist_federated_data(
            root=str(root / "data"), num_clients=1, images_per_client=64,
            test_size=1, batch_size_train=32, batch_size_test=1,
        )
        model = MNIST_MLP()
        delta = local_train_delta(model, clients[0], epochs=1, lr=0.05)
        real_update, _ = flatten_state_dict(delta)
        real_update = real_update.detach().cpu().float().flatten()
        if real_update.numel() != D:
            raise RuntimeError(f"MNIST update has {real_update.numel()} values; expected {D}")
        vectors["Real_Or_Spike_Vector"] = (real_update, "MNIST client 0, 64 examples, 1 local epoch")
    return vectors


def summarize(raw_rows: list[dict]) -> list[dict]:
    keys = ("vector", "source", "method", "method_name", "requested_lambda", "bits")
    groups: dict[tuple, list[dict]] = {}
    for row in raw_rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    metrics = ("actual_bits", "D", "actual_lambda", "kappa_inf", "alg_nmse",
               "quant_nmse", "cross_term", "total_nmse")
    for key, group in groups.items():
        item = dict(zip(keys, key))
        for metric in metrics:
            values = np.asarray([r[metric] for r in group], dtype=np.float64)
            item[metric] = float(values.mean())
            if metric in {"kappa_inf", "alg_nmse", "quant_nmse", "cross_term", "total_nmse"}:
                item[metric + "_std"] = float(values.std(ddof=1))
        item["n_seeds"] = len(group)
        out.append(item)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict]) -> str:
    columns = ("vector", "method", "requested_lambda", "actual_lambda", "bits",
               "actual_bits", "kappa_inf", "alg_nmse", "quant_nmse", "total_nmse")
    headers = ("vector", "method", "lam", "actual_lambda", "bits", "actual_bits",
               "kappa_inf", "alg_nmse", "quant_nmse", "total_nmse")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_results(rows: list[dict], path: Path) -> None:
    vectors = list(dict.fromkeys(row["vector"] for row in rows))
    fig, axes = plt.subplots(2, 2, figsize=(17, 11), squeeze=False)
    colors = plt.cm.tab20(np.linspace(0, 1, len(METHODS) * len(LAMBDAS)))
    for row_index, vector in enumerate(vectors):
        vector_rows = [r for r in rows if r["vector"] == vector]
        ax_total, ax_components = axes[row_index]
        color_index = 0
        for method in METHODS:
            for lam in LAMBDAS:
                series = sorted(
                    (r for r in vector_rows if r["method"] == method and r["requested_lambda"] == lam),
                    key=lambda r: r["bits"],
                )
                color = colors[color_index]
                color_index += 1
                if not series:
                    continue
                label = f"{METHOD_NAMES[method]}, λ={lam:g}"
                ax_total.plot([r["bits"] for r in series],
                              [math.log10(max(r["total_nmse"], 1e-18)) for r in series],
                              marker="o", color=color, label=label)
                ax_components.plot([r["bits"] for r in series],
                                   [r["alg_nmse"] for r in series],
                                   marker="o", linestyle="-", color=color,
                                   label=f"alg: {label}")
                ax_components.plot([r["bits"] for r in series],
                                   [r["quant_nmse"] for r in series],
                                   marker="x", linestyle="--", color=color,
                                   label=f"quant: {label}")
        ax_total.set_title(f"{vector}: total error")
        ax_total.set_xlabel("bits per coefficient")
        ax_total.set_ylabel("log10(total_nmse)")
        ax_total.grid(True, alpha=0.3)
        ax_total.legend(fontsize=7, ncol=2)
        ax_components.set_title(f"{vector}: algorithm and quantization components")
        ax_components.set_xlabel("bits per coefficient")
        ax_components.set_ylabel("relative squared error (NMSE)")
        ax_components.set_yscale("log")
        ax_components.grid(True, alpha=0.3)
        ax_components.legend(fontsize=6, ncol=2)
    fig.suptitle("Pure vector compression: bit-width and redundancy study", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run(synthetic_only: bool = False) -> list[dict]:
    output = ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    vectors = build_vectors(ROOT, synthetic_only=synthetic_only)
    raw_rows = []
    for vector_name, (x, source) in vectors.items():
        x = x.detach().cpu().float().flatten()
        if x.numel() != D:
            raise ValueError(f"{vector_name} dimension is {x.numel()}, expected {D}")
        print(f"Preparing {vector_name}: {source}; d={x.numel()}", flush=True)
        for lam in LAMBDAS:
            frame_cache = {}
            for method in METHODS:
                if method == "hadamard_srk":
                    prepared = prepare_single_vector_compression(x, method, lam, frame_seed=4107)
                else:
                    if lam not in frame_cache:
                        frame_cache[lam] = FourierKashinFrame(
                            D, int(round(lam * D)), seed=4107
                        )
                    frame = frame_cache[lam]
                    prepared = prepare_single_vector_compression(x, method, lam, frame_seed=4107, frame=frame)
                for bits in BITS:
                    for seed in QUANT_SEEDS:
                        metrics = evaluate_prepared_vector_compression(x, prepared, bits, seed)
                        raw_rows.append({
                            "vector": vector_name, "source": source, "method": method,
                            "method_name": METHOD_NAMES[method], "requested_lambda": lam,
                            "actual_lambda": metrics["actual_lambda"], "D": metrics["D"],
                            "bits": bits, "quant_seed": seed,
                            **{key: metrics[key] for key in (
                                "actual_bits", "kappa_inf", "alg_nmse", "quant_nmse",
                                "cross_term", "total_nmse")},
                        })
    summary = summarize(raw_rows)
    write_csv(output / "exp1_bit_law_raw.csv", raw_rows)
    write_csv(output / "exp1_bit_law_summary.csv", summary)
    plot_results(summary, output / "exp1_bit_law.png")
    table = markdown_table(summary)
    (output / "exp1_bit_law.md").write_text(table + "\n", encoding="utf-8")
    print(table, flush=True)
    print(f"\nRaw samples: {output / 'exp1_bit_law_raw.csv'}")
    print(f"Summary: {output / 'exp1_bit_law_summary.csv'}")
    print(f"Figure: {output / 'exp1_bit_law.png'}")
    print(f"Markdown table: {output / 'exp1_bit_law.md'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-only", action="store_true",
                        help="use the sparse-spike vector instead of training a local MNIST client")
    args = parser.parse_args()
    run(synthetic_only=args.synthetic_only)


if __name__ == "__main__":
    main()
