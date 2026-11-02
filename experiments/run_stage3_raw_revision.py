# -*- coding: utf-8 -*-
"""Stage-3 raw, final protocol: LS alpha* calibration inside quantize_doscale.

No external server_lr table: all methods run --server_lr 1.0.
Two clean raw comparisons (deterministic rounding):
  stage3_raw_b{1,2,3}_det_max      : clip_quantile 0.0   (重尾 max 边界, 内格拥挤)
  stage3_raw_b{1,2,3}_det_clip995  : clip_quantile 0.995 (q99.5 边界 + 截断)
Expected invariants (to verify after the run):
  1. b1 det_max == det_clip995 (bit-exact, clip irrelevant at 1-bit);
  2. det_clip995 mean_nmse strictly decreasing in bits, test_acc increasing;
  3. everything at server_lr = 1.0.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
OUT = ROOT / "results" / "stage3_teacher_b123"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run_raw_final.log"

COMMON = [
    str(PY), str(ROOT / "run_experiment.py"),
    "--task", "emnist_cnn", "--partition", "dirichlet", "--dirichlet_alpha", "0.3",
    "--num_clients", "100", "--clients_per_round", "10",
    "--total_rounds", "100", "--seed", "42",
    "--scale_mode", "doscale", "--stochastic", "0",
    "--transform", "raw", "--server_lr", "1.0", "--output_dir", str(OUT),
]

jobs = []
for b in (1, 2, 3):
    jobs.append((b, "0.0"))
    jobs.append((b, "0.995"))

with open(str(LOG), "a", buffering=1) as log:
    for idx, (bits, clip_q) in enumerate(jobs, 1):
        name = "stage3_raw_b%d_det" % bits
        name += "_clip995" if clip_q != "0.0" else "_max"
        cmd = COMMON + ["--bits", str(bits), "--clip_quantile", clip_q, "--exp_name", name]
        log.write("===== [%d/%d] %s start %s\n" % (idx, len(jobs), name, time.ctime()))
        ret = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        log.write("===== [%d/%d] %s exit=%d %s\n" % (idx, len(jobs), name,
                  ret.returncode, time.ctime()))
print("RAW FINAL DONE ->", OUT)
