"""Plot accuracy-vs-round curves from kashin_lowbit_round_metrics.csv.

2x2 subplots grouped by bits; each shows SRK (black) + Kashin lambda sweep.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
csv_path = root / "results" / "kashin_lowbit_round_metrics.csv"

rows = []
with csv_path.open(newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append({"label": r["label"], "method": r["method"], "bits": int(r["bits"]),
                     "lambda": float(r["lambda"]) if r["lambda"] else None,
                     "round": int(r["round"]), "acc": float(r["accuracy"]) * 100})

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
lam_colors = {1.0: "tab:blue", 1.5: "tab:orange", 2.0: "tab:green", 2.5: "tab:red", 3.0: "tab:purple"}
for ax, bits in zip(axes.flat, (2, 4, 8, 16)):
    srk = sorted([r for r in rows if r["method"] == "SRK" and r["bits"] == bits], key=lambda x: x["round"])
    if srk:
        ax.plot([r["round"] for r in srk], [r["acc"] for r in srk],
                color="black", linewidth=2.2, label="SRK")
    for lam in (1.0, 1.5, 2.0, 2.5, 3.0):
        k = sorted([r for r in rows if r["method"] == "Kashin" and r["bits"] == bits and r["lambda"] == lam],
                   key=lambda x: x["round"])
        if k:
            ax.plot([r["round"] for r in k], [r["acc"] for r in k],
                    color=lam_colors[lam], alpha=.8, label=f"Kashin λ={lam:g}")
    ax.set_title(f"b = {bits}"); ax.grid(True, linestyle="--", alpha=.5)
    ax.legend(fontsize=8); ax.set_xlabel("Round"); ax.set_ylabel("Test Accuracy (%)")
fig.suptitle(f"Accuracy vs Round (single seed, {max(r['round'] for r in rows)} rounds)")
fig.tight_layout()
out = root / "plots" / "rounds_curve.png"
fig.savefig(out, dpi=300)
print("[PASS] saved", out)
