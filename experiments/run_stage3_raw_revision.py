# -*- coding: utf-8 -*-
"""Stage-3 raw revision: delta-calibrated (WAGE Sec.4.4) and optional 99.5% clip.

WAGE rule: keep eta * sigma(k) ~ const across bit-widths. Here the grid step is
2*bound/(2^b-1), so delta_b = delta_1 * (2^b-1) with delta_1 = 0.02.
Runs into the same results dir as stage3 so plot/compare tools see one family.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
OUT = ROOT / "results" / "stage3_teacher_b123"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run_raw_revision.log"

COMMON = [
    str(PY), str(ROOT / "run_experiment.py"),
    "--task", "emnist_cnn", "--partition", "dirichlet", "--dirichlet_alpha", "0.3",
    "--num_clients", "100", "--clients_per_round", "10",
    "--total_rounds", "100", "--seed", "42",
    "--scale_mode", "doscale", "--stochastic", "0",
    "--transform", "raw", "--output_dir", str(OUT),
]

DELTA = {1: "0.02", 2: "0.06", 3: "0.14"}   # delta_b = 0.02 * (2^b - 1)

jobs = []
for b in (1, 2, 3):
    jobs.append((b, "0.0"))        # dcal only
    jobs.append((b, "0.995"))      # dcal + clip

with open(str(LOG), "a", buffering=1) as log:
    for idx, (bits, clip_q) in enumerate(jobs, 1):
        name = "stage3_raw_b%d_det_dcal" % bits
        if clip_q != "0.0":
            name += "_clip"
        cmd = COMMON + ["--bits", str(bits), "--server_lr", DELTA[bits],
                        "--clip_quantile", clip_q, "--exp_name", name]
        log.write("===== [%d/%d] %s start %s\n" % (idx, len(jobs), name, time.ctime()))
        ret = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        log.write("===== [%d/%d] %s exit=%d %s\n" % (idx, len(jobs), name,
                  ret.returncode, time.ctime()))
print("RAW REVISION DONE ->", OUT)
