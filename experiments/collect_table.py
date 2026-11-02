# -*- coding: utf-8 -*-
"""汇总 results/<batch>/ 下所有 run 的关键指标到一张总表 CSV。
用法: python experiments/collect_table.py --root results/stage4_mid_full --out results/stage4_mid_full/master_table.csv
"""
import argparse, csv, io, json, os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

rows = []
for d in sorted(os.listdir(args.root)):
    mc = Path(args.root) / d / "metrics.csv"
    hp = Path(args.root) / d / "hparams.json"
    if not (mc.exists() and hp.exists()):
        continue
    with io.open(str(mc), encoding="utf-8") as f:
        m = list(csv.DictReader(f))
    with io.open(str(hp), encoding="utf-8-sig") as f:
        h = json.load(f)
    tail = m[-10:]
    rows.append({
        "run": d,
        "method": h.get("transform"),
        "bits": h.get("bits"),
        "lam": h.get("lam"),
        "stochastic": h.get("stochastic"),
        "seed": h.get("seed"),
        "actual_bpd": round(sum(float(r["bpd"]) for r in tail) / len(tail), 4),
        "r20_acc": round(float(m[19]["test_acc"]) * 100, 2) if len(m) >= 20 else "",
        "r40_acc": round(float(m[39]["test_acc"]) * 100, 2) if len(m) >= 40 else "",
        "r60_acc": round(float(m[59]["test_acc"]) * 100, 2) if len(m) >= 60 else "",
        "r80_acc": round(float(m[79]["test_acc"]) * 100, 2) if len(m) >= 80 else "",
        "tail10_acc": round(sum(float(r["test_acc"]) for r in tail) / len(tail) * 100, 2),
        "tail10_loss": round(sum(float(r["test_loss"]) for r in tail) / len(tail), 4),
        "tail10_nmse": round(sum(float(r["mean_nmse"]) for r in tail) / len(tail), 4),
    })

order = {"uncompressed": 0, "raw": 1, "hadamard": 2, "fourier_kashin_balanced": 3, "fourier_kashin": 3}
rows.sort(key=lambda r: (order.get(r["method"], 9), int(r["bits"]), float(r["lam"] or 0)))
with io.open(args.out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
    w.writeheader()
    w.writerows(rows)
print("rows:", len(rows), "->", args.out)
