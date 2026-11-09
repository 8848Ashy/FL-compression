# -*- coding: utf-8 -*-
"""CIFAR overnight orchestrator (runs on server under nohup).

Phases:
 1) Group C smoke (DRIVE lr=0.1, momentum=0.9) 20 rounds.
 2) Auto-pick LR group: C if unc_C@r20 > unc_A@r20 + 1 and > 18, else A.
 3) Pilot: winner LR x 7 configs x 3 seeds x 400 rounds.
 4) Full grid: 43 configs x 3 seeds x 200 rounds (same shape as EMNIST mid).
Writes decision log + final summary CSVs.
"""
import json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
os.chdir(ROOT)

COMMON = ["--task", "cifar_resnet9", "--partition", "dirichlet", "--dirichlet_alpha", "0.2",
          "--num_clients", "100", "--clients_per_round", "5",
          "--local_epochs", "3", "--server_lr", "1.0", "--weight_decay", "5e-4",
          "--scale_mode", "doscale"]

def run_batch(name, jobs, rounds, lr, mom, workers=6, gpus=2):
    common = list(COMMON) + [
        "--total_rounds", str(rounds), "--client_lr", str(lr), "--client_momentum", str(mom)]
    spec = {"batch": name, "output_dir": "results/" + name, "common": common, "jobs": jobs}
    jf = os.path.join("experiments", "_tmp_%s.json" % name)
    with open(jf, "w", encoding="utf-8") as f:
        json.dump(spec, f)
    print("===== %s start %s (lr=%s mom=%s, %d jobs x %d rounds) =====" %
          (name, time.ctime(), lr, mom, len(jobs), rounds), flush=True)
    r = subprocess.run([PY, "experiments/run_parallel.py", "--jobs", jf,
                        "--workers", str(workers), "--device", "auto", "--gpus", str(gpus)])
    print("===== %s done rc=%d %s =====" % (name, r.returncode, time.ctime()), flush=True)
    return r.returncode

def job(name, extra):
    return {"name": name, "args": extra + ["--exp_name", name], "threads": 4}

def seed_args(seed):
    return []

def acc_at(mc, rnd):
    try:
        with open(mc, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return float(rows[rnd - 1]["test_acc"]) * 100
    except Exception:
        return -1.0

import csv

log = open("results/cifar_overnight.log", "a", buffering=1)

# ---------------- Phase 1: group C smoke ----------------
c_jobs = [
    job("cifar_unc_lrC", ["--transform", "uncompressed", "--bits", "8", "--stochastic", "1"]),
    job("cifar_kb2_lrC", ["--transform", "fourier_kashin_balanced", "--bits", "2", "--lam", "1.3",
                          "--stochastic", "1", "--kashin_iters", "15", "--kashin_eta", "0.9"]),
]
jobs = []
for s in (42,):
    for j in c_jobs:
        name = "%s_s%d" % (j["name"], s)
        jj = dict(j); jj["name"] = name
        jj["args"] = j["args"] + ["--seed", str(s), "--exp_name", name]
        jobs.append(jj)
# note: run_experiment --seed must appear once; append before exp_name
jobs = []
for s in (42,):
    for j in c_jobs:
        name = "%s_s%d" % (j["name"], s)
        args = j["args"] + ["--seed", str(s), "--exp_name", name]
        jobs.append({"name": name, "args": args, "threads": 4})

rc = run_batch("cifar_smokeC", jobs, 20, 0.1, 0.9, workers=2, gpus=2)

# ---------------- Phase 2: auto decision ----------------
a20 = acc_at("results/cifar_smoke/cifar_unc_lrA/metrics.csv", 20)
c20 = acc_at("results/cifar_smokeC/cifar_unc_lrC_s42/metrics.csv", 20)
use_C = (c20 > a20 + 1.0) and (c20 > 18.0)
lr, mom = (0.1, 0.9) if use_C else (0.05, 0.0)
log.write("decision: unc_A@20=%.2f unc_C@20=%.2f -> use lr=%s mom=%s (C=%s)\n" %
          (a20, c20, lr, mom, use_C))

# ---------------- Phase 3: pilot 400 rounds ----------------
PILOT = [("unc", ["--transform", "uncompressed", "--bits", "8", "--stochastic", "1"]),
         ("raw_b1", ["--transform", "raw", "--bits", "1", "--stochastic", "0"]),
         ("had_b1", ["--transform", "hadamard", "--bits", "1", "--lam", "1.21059", "--stochastic", "1"]),
         ("kb_b1_lam1.0", ["--transform", "fourier_kashin_balanced", "--bits", "1", "--lam", "1.0",
                           "--stochastic", "1", "--kashin_iters", "15", "--kashin_eta", "0.9"]),
         ("kb_b1_lam1.21", ["--transform", "fourier_kashin_balanced", "--bits", "1", "--lam", "1.21059",
                            "--stochastic", "1", "--kashin_iters", "15", "--kashin_eta", "0.9"]),
         ("kb_b1_lam2.0", ["--transform", "fourier_kashin_balanced", "--bits", "1", "--lam", "2.0",
                           "--stochastic", "1", "--kashin_iters", "15", "--kashin_eta", "0.9"]),
         ("kb_b2_lam1.3", ["--transform", "fourier_kashin_balanced", "--bits", "2", "--lam", "1.3",
                           "--stochastic", "1", "--kashin_iters", "15", "--kashin_eta", "0.9"])]
jobs = []
for s in (42, 43, 44):
    for tag, extra in PILOT:
        name = "pilot_%s_s%d" % (tag, s)
        jobs.append({"name": name, "args": extra + ["--seed", str(s), "--exp_name", name],
                     "threads": 4})
run_batch("cifar_pilot", jobs, 400, lr, mom, workers=6, gpus=2)

# ---------------- Phase 4: full grid 200 rounds ----------------
GRID = [("unc", ["--transform", "uncompressed", "--bits", "8", "--stochastic", "1"])]
for b in range(1, 7):
    GRID.append(("raw_b%d" % b, ["--transform", "raw", "--bits", str(b), "--stochastic", "0"]))
for b in range(1, 6):
    GRID.append(("had_b%d" % b, ["--transform", "hadamard", "--bits", str(b), "--lam", "1.21059",
                                 "--stochastic", "1"]))
for b in range(1, 6):
    for lam in ("1.0", "1.1", "1.21059", "1.3", "1.5", "2.0"):
        GRID.append(("kb_b%d_lam%s" % (b, lam),
                     ["--transform", "fourier_kashin_balanced", "--bits", str(b), "--lam", lam,
                      "--stochastic", "1", "--kashin_iters", "15", "--kashin_eta", "0.9"]))
GRID.append(("kb_b1_lam1.6", ["--transform", "fourier_kashin_balanced", "--bits", "1", "--lam", "1.6",
                              "--stochastic", "1", "--kashin_iters", "15", "--kashin_eta", "0.9"]))
jobs = []
for s in (42, 43, 44):
    for tag, extra in GRID:
        name = "%s_s%d" % (tag, s)
        jobs.append({"name": name, "args": extra + ["--seed", str(s), "--exp_name", name],
                     "threads": 4})
run_batch("cifar_full", jobs, 200, lr, mom, workers=6, gpus=2)

# ---------------- Summary ----------------
for d in ("cifar_pilot", "cifar_full"):
    subprocess.run([PY, "experiments/collect_table.py", "--root", "results/" + d,
                    "--out", "results/%s/all_seeds.csv" % d])
    subprocess.run([PY, "experiments/make_summary.py", "--root", "results/" + d,
                    "--out", "results/%s/summary.csv" % d])
print("ALL DONE", time.ctime(), flush=True)
