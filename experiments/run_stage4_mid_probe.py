# -*- coding: utf-8 -*-
"""Stage-4 mid-SNR probe: alpha=0.2, M=5, server_lr=1.25, E=3, 80 rounds, b=1.
lambda sweep {1.1, 1.3, 1.6, 2.0} with fourier_kashin_balanced (iters 15, eta 0.9).
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
OUT = ROOT / "results" / "stage4_mid_probe"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run_all.log"

COMMON = [
    str(PY), str(ROOT / "run_experiment.py"),
    "--task", "emnist_cnn", "--partition", "dirichlet", "--dirichlet_alpha", "0.2",
    "--num_clients", "100", "--clients_per_round", "5",
    "--total_rounds", "80", "--seed", "42", "--local_epochs", "3",
    "--server_lr", "1.25",
    "--scale_mode", "doscale", "--stochastic", "1", "--bits", "1",
    "--transform", "fourier_kashin_balanced",
    "--kashin_iters", "15", "--kashin_eta", "0.9",
    "--output_dir", str(OUT),
]

jobs = ["1.1", "1.3", "1.6", "2.0"]

with open(str(LOG), "a", buffering=1) as log:
    for idx, lam in enumerate(jobs, 1):
        name = "stage4mid_kashinbal_b1_lam%s" % lam
        cmd = COMMON + ["--lam", lam, "--exp_name", name]
        log.write("===== [%d/%d] %s start %s\n" % (idx, len(jobs), name, time.ctime()))
        ret = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        log.write("===== [%d/%d] %s exit=%d %s\n" % (idx, len(jobs), name,
                  ret.returncode, time.ctime()))
print("MID PROBE DONE ->", OUT)
