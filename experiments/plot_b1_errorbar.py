# -*- coding: utf-8 -*-
"""b=1 误差棒图（跨 seed 聚合）。Q1 完成后作为阶段产出运行。
用法: python experiments/plot_b1_errorbar.py --root results/stage4_mid_full_lr10"""
import argparse, csv, io, os, json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
args = ap.parse_args()

groups = defaultdict(list)
for d in os.listdir(args.root):
    mc = os.path.join(args.root, d, "metrics.csv")
    hp = os.path.join(args.root, d, "hparams.json")
    if not (os.path.exists(mc) and os.path.exists(hp)):
        continue
    with io.open(hp, encoding="utf-8-sig") as f:
        h = json.load(f)
    if int(h.get("bits", 0)) != 1:
        continue
    with io.open(mc, encoding="utf-8") as f:
        m = list(csv.DictReader(f))
    if len(m) < 80:
        continue
    tail = m[-10:]
    key = (h["transform"], float(h.get("lam", 1.0)))
    groups[key].append((sum(float(r["bpd"]) for r in tail) / 10,
                        sum(float(r["test_acc"]) for r in tail) / 10 * 100))

fig, ax = plt.subplots(figsize=(10, 6.4))
style = {"raw": ("#d62728", "o", "Raw (det + LS-alpha*)"),
         "hadamard": ("#1f77b4", "s", "Hadamard"),
         "fourier_kashin_balanced": ("#2ca02c", "^", "Kashin balanced (b=1, lam sweep)"),
         "uncompressed": ("black", "D", "Uncompressed")}
for key, pts in sorted(groups.items()):
    if len(pts) < 2:
        continue
    import statistics
    xs = statistics.mean(p[0] for p in pts)
    ys = [p[1] for p in pts]
    color, marker, lbl = style.get(key[0], ("gray", "x", key[0]))
    label = lbl if key[0] != "fourier_kashin_balanced" else ("%s (lam=%g)" % (lbl, key[1]))
    ax.errorbar([xs], [statistics.mean(ys)], yerr=[statistics.stdev(ys)],
                fmt=marker, color=color, markersize=7, capsize=4, label=label, linewidth=1.4)
ax.set(xlabel="Bits per dimension (BPD)", ylabel="Last-10-round mean test accuracy (%)",
       title="b=1 configs, mean +/- std over seeds (stage: %s)" % os.path.basename(args.root.rstrip('/\\')))
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
out = os.path.join(args.root, "b1_errorbar.png")
fig.tight_layout()
fig.savefig(out, dpi=220)
print("fig ->", out, "| configs with >=2 seeds:", sum(1 for v in groups.values() if len(v) >= 2))
