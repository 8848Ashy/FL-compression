# -*- coding: utf-8 -*-
"""生成 3-seed 聚合汇总表（mean/std）。
- uncompressed 元数据修正为 bits=32, lam=1.0, actual_bpd=32.0
用法: python experiments/make_summary.py --root results/stage4_mid_full_lr10 --out results/stage4_mid_full_lr10/master_table_lr10_summary.csv"""
import argparse, csv, io, os, json, statistics
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

METRICS = ["actual_bpd", "r20_acc", "r40_acc", "r60_acc", "r80_acc", "tail10_acc", "tail10_loss", "tail10_nmse"]
groups = defaultdict(list)
n_runs = 0
for d in os.listdir(args.root):
    mc = os.path.join(args.root, d, "metrics.csv")
    hp = os.path.join(args.root, d, "hparams.json")
    if not (os.path.exists(mc) and os.path.exists(hp)):
        continue
    with io.open(mc, encoding="utf-8") as f:
        m = list(csv.DictReader(f))
    if len(m) < 80:
        continue
    n_runs += 1
    with io.open(hp, encoding="utf-8-sig") as f:
        h = json.load(f)
    method = h["transform"]
    bits, lam = int(h["bits"]), float(h.get("lam", 1.0))
    if method == "uncompressed":
        bits, lam = 32, 1.0
    tail = m[-10:]
    vals = {
        "actual_bpd": sum(float(r["bpd"]) for r in tail) / 10,
        "r20_acc": float(m[19]["test_acc"]) * 100,
        "r40_acc": float(m[39]["test_acc"]) * 100,
        "r60_acc": float(m[59]["test_acc"]) * 100,
        "r80_acc": float(m[79]["test_acc"]) * 100,
        "tail10_acc": sum(float(r["test_acc"]) for r in tail) / 10 * 100,
        "tail10_loss": sum(float(r["test_loss"]) for r in tail) / 10,
        "tail10_nmse": sum(float(r["mean_nmse"]) for r in tail) / 10,
    }
    if method == "uncompressed":
        vals["actual_bpd"] = 32.0
    groups[(method, bits, lam)].append(vals)

order = {"uncompressed": 0, "raw": 1, "hadamard": 2, "fourier_kashin_balanced": 3}
rows = []
for (method, bits, lam), runs in groups.items():
    row = {"method": method, "bits": bits, "lam": lam, "n_seeds": len(runs)}
    for k in METRICS:
        v = [r[k] for r in runs]
        row[k + "_mean"] = round(statistics.mean(v), 4)
        row[k + "_std"] = round(statistics.stdev(v), 4) if len(v) > 1 else 0.0
    rows.append(row)
rows.sort(key=lambda r: (order.get(r["method"], 9), int(r["bits"]), float(r["lam"])))

with io.open(args.out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("runs:", n_runs, "| groups:", len(rows), "->", args.out)
