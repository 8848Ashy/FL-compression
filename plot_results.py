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
        params = json.loads(hparams_path.read_text(encoding="utf-8-sig"))
        with metrics_path.open(encoding="utf-8", newline="") as stream:
            rows = [{k: float(v) if v else float("nan") for k, v in row.items()}
                    for row in csv.DictReader(stream)]
        if rows:
            runs.append((params, rows))
    return runs


def _layout_annotations_above(fig, ax, specs, gap=12.0):
    """Place all annotation boxes in rows ABOVE the axes (never over the
    curves), ordered by data-x, with arrows pointing down to the points.

    ``specs`` is a list of dicts with keys: text, xy (data coords), color.
    Returns (n_rows, row_height_px) so the caller can make room (title pad)."""
    if not specs:
        return 0, 0.0
    dpi = fig.dpi

    def make_ann(spec, pos, coords):
        return ax.annotate(spec["text"], xy=spec["xy"], xycoords="data",
                           xytext=pos, textcoords=coords,
                           ha="center", va="center", fontsize=8,
                           color=spec["color"], fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                     ec=spec["color"], alpha=0.95),
                           arrowprops=dict(arrowstyle="->", color=spec["color"],
                                           lw=1.0, shrinkA=0, shrinkB=4))

    # Pass 1: create temporary annotations (hidden position) to measure text sizes
    tmps = [make_ann(s, (0, 0), "offset points") for s in specs]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    sizes = [t.get_window_extent(renderer) for t in tmps]
    sizes = [(bb.width, bb.height) for bb in sizes]
    for t in tmps:
        t.remove()

    def pack(renderer):
        fig_w, _ = fig.canvas.get_width_height()
        order = sorted(range(len(specs)),
                       key=lambda i: ax.transData.transform(specs[i]["xy"])[0])
        row_h = max(h for _, h in sizes) + 12
        rows = []          # list of list of (x0, x1)
        place = {}
        for i in order:
            w, _ = sizes[i]
            px = ax.transData.transform(specs[i]["xy"])[0]
            want = min(max(px - w / 2, 4.0), fig_w - 4.0 - w)
            for r, row in enumerate(rows):
                x = want
                for (x0, x1) in sorted(row):
                    if x < x1 + gap and x + w > x0 - gap:
                        x = x1 + gap
                if x + w <= fig_w - 4.0:
                    row.append((x, x + w))
                    place[i] = (x + w / 2, r)
                    break
            else:
                rows.append([(want, want + w)])
                place[i] = (want + w / 2, len(rows) - 1)
        return place, len(rows), row_h

    _, n_rows, row_h = pack(renderer)

    # Deterministically reserve a strip above the axes for label rows + title:
    # figure height = plotting area + reserve, axes top fixed below the strip.
    title_room = 54
    reserve = 16 + n_rows * row_h + title_room
    fig_h = 6.5 * dpi + reserve + 10
    fig.set_size_inches(fig.get_size_inches()[0], fig_h / dpi)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.09,
                        top=1.0 - reserve / fig_h)
    fig.text(0.5, 1.0 - 24.0 / fig_h, "Accuracy vs communication budget",
             ha="center", va="center", fontsize=13)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(renderer)
    place, n_rows, row_h = pack(renderer)
    fig_w, fig_h = fig.canvas.get_width_height()  # dpi-invariant via fractions
    top0 = ax_bb.y1 + 16
    for i, spec in enumerate(specs):
        cx, r = place[i]
        y_px = top0 + r * row_h + sizes[i][1] * 0.5
        make_ann(spec, (cx / fig_w, y_px / fig_h), "figure fraction")
    return n_rows, row_h


def plot(root: Path, output_dir: Path):
    runs = load_runs(root)
    if not runs:
        raise FileNotFoundError(f"No completed runs with hparams.json and metrics.csv under {root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Style rules (matching the reference figure):
    #   raw / hadamard: one color per method (raw red, hadamard blue),
    #   linestyle = bits (1b ':', 2b '--', 3b '-').
    #   kashin: one color per bits (1b green, 2b purple), linestyle = lambda
    #   (1.15 ':', 1.6 '--', 2.0 '-'); only bits 1/2 and lam 1.15/1.6/2.0 shown.
    method_colors = {"raw": "#d62728", "hadamard": "#1f77b4"}
    bits_linestyles = {1: ":", 2: "--", 3: "-"}
    kashin_colors = {1: "#2ca02c", 2: "#9467bd"}
    kashin_lam_styles = {1.15: ":", 1.6: "--", 2.0: "-"}

    def series_label_style(transform, bits, lam):
        """Returns (color, linestyle, label)."""
        if transform == "raw":
            return (method_colors["raw"], bits_linestyles[bits],
                    f"Raw/SignSGD ({bits}-bit)")
        if transform == "hadamard":
            return (method_colors["hadamard"], bits_linestyles[bits],
                    f"Hadamard ({bits}-bit)")
        if transform == "uncompressed":
            return "#7f7f7f", "-", "Uncompressed FedAvg (32-bit)"
        if transform.startswith("fourier_kashin"):
            lam = round(float(lam), 2)
            return (kashin_colors[bits], kashin_lam_styles[lam],
                    f"Kashin (lam={lam:g}, {bits}-bit)")
        return "#333333", "-", f"{transform} ({bits}-bit)"

    def _rounds_sort_key(item):
        params, _ = item
        transform = params.get("transform", "")
        bits = int(params.get("bits", 1))
        lam = float(params.get("lam", 0.0))
        if transform == "raw":
            return (0, bits, lam)
        if transform == "hadamard":
            return (1, bits, lam)
        if transform.startswith("fourier_kashin"):
            return (2, bits, lam)
        return (3, bits, lam)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for index, (params, rows) in enumerate(sorted(runs, key=_rounds_sort_key)):
        transform = params.get("transform", f"run-{index}")
        bits = int(params.get("bits", 1))
        lam = float(params.get("lam", 1.0))
        if transform.startswith("fourier_kashin"):
            if bits not in kashin_colors or round(lam, 2) not in kashin_lam_styles:
                continue  # only bits 1/2 and lam 1.15/1.6/2.0 are plotted here
        if transform == "uncompressed":
            continue  # reference curve, drawn on the accuracy-vs-rounds axes only
        color, linestyle, label = series_label_style(transform, bits, lam)
        rounds = [r["round"] for r in rows]
        cumulative = [r["cumulative_bpd"] for r in rows]
        acc = [r["test_acc"] for r in rows]
        nmse = [max(r["mean_nmse"], 1e-18) for r in rows]
        axes[0, 0].plot(rounds, acc, label=label, color=color, linestyle=linestyle)
        axes[0, 1].plot(rounds, nmse, label=label, color=color, linestyle=linestyle)
        axes[1, 0].plot(cumulative, acc, label=label, color=color, linestyle=linestyle)
        axes[1, 1].plot(cumulative, nmse, label=label, color=color, linestyle=linestyle)
    uncompressed_label = "Uncompressed FedAvg (32-bit)"
    for params, rows in runs:
        if params.get("transform") != "uncompressed":
            continue
        axes[0, 0].plot([r["round"] for r in rows], [r["test_acc"] for r in rows],
                        color="#7f7f7f", linestyle="--", label=uncompressed_label)
        uncompressed_label = None
    axes[0, 0].set(xlabel="Communication rounds", ylabel="Test accuracy", title="Accuracy vs rounds")
    axes[0, 1].set(xlabel="Communication rounds", ylabel="Mean client NMSE", title="NMSE vs rounds", yscale="log")
    axes[1, 0].set(xlabel="Cumulative bits per dimension", ylabel="Test accuracy", title="Accuracy vs cumulative BPD")
    axes[1, 1].set(xlabel="Cumulative bits per dimension", ylabel="Mean client NMSE", title="NMSE vs cumulative BPD", yscale="log")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="best", framealpha=0.9)
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

    # Same colors as accuracy_vs_rounds:
    #   raw = red, hadamard = blue, kashin b1 = green, b2 = purple, b3 = orange.
    #   raw / hadamard points are each connected into one curve per method;
    #   kashin draws one curve per bits across ALL lambda values.
    #   No data-value annotations.
    method_colors = {"raw": "#d62728", "hadamard": "#1f77b4"}
    kashin_colors = {1: "#2ca02c", 2: "#9467bd", 3: "#ff7f0e"}
    method_markers = {"raw": "o", "hadamard": "s", "fourier_kashin": "^",
                      "fourier_kashin_balanced": "D"}
    method_names = {"raw": "Raw/SignSGD", "hadamard": "Hadamard",
                    "fourier_kashin": "Kashin", "fourier_kashin_balanced": "Kashin(bal)"}
    scale_modes = sorted({point["scale_mode"] for point in points})

    def draw_metric(ax, metric):
        for scale_mode in scale_modes:
            subset = [p for p in points if p["scale_mode"] == scale_mode]
            suffix = f" ({scale_mode})" if len(scale_modes) > 1 else ""
            # raw / hadamard: one connected curve per method across bits
            for transform in ("raw", "hadamard"):
                series = sorted((p for p in subset
                                 if p["transform"] == transform
                                 and p["bits"] in (1, 2, 3)),
                                key=lambda p: p["bpd"])
                if not series:
                    continue
                ax.plot([p["bpd"] for p in series], [p[metric] for p in series],
                        marker=method_markers[transform], color=method_colors[transform],
                        linestyle="-", label=f"{method_names[transform]}{suffix}")

            # kashin: one curve per bits across all lambda values
            for transform in ("fourier_kashin", "fourier_kashin_balanced"):
                kashin = [p for p in subset
                          if p["transform"] == transform
                          and p["bits"] in kashin_colors]
                if not kashin:
                    continue
                for bits in (1, 2, 3):
                    series = sorted((p for p in kashin if p["bits"] == bits),
                                    key=lambda p: p["lam"])
                    if not series:
                        continue
                    ax.plot([p["bpd"] for p in series], [p[metric] for p in series],
                            marker=method_markers[transform], color=kashin_colors[bits],
                            linestyle="-",
                            label=f"{method_names[transform]} ({bits}-bit){suffix}")

        baseline = [p[metric] for p in points if p["transform"] == "uncompressed"]
        if baseline and metric == "accuracy":
            ax.axhline(sum(baseline) / len(baseline), color="black", linestyle="--",
                       label="Uncompressed FedAvg (32-bit)")
        ax.set_xlabel("Bits per dimension (BPD)")
        ax.set_ylabel("Last-10-round mean accuracy" if metric == "accuracy"
                      else "Last-10-round mean client NMSE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2, loc="best", framealpha=0.9)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    draw_metric(ax, "accuracy")
    ax.set_title("Last-10-round mean accuracy vs communication budget")
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
