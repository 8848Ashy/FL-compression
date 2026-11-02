# -*- coding: utf-8 -*-
"""Stage-4 low-SNR verdict probe: strip the spatial-averaging umbrella.

M=2 (mean over 2 clients), alpha=0.1 (strong Non-IID), server_lr=1.5,
E=1 (E=5 showed no upd_norm growth), 80 rounds, seed 42.
Question: does 3x NMSE gap (15.8 vs 5.5) show up in accuracy when noise
averaging is removed?
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
OUT = ROOT / "results" / "stage4_lowsnr_probe"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run_all.log"

COMMON = [
    str(PY), str(ROOT / "run_experiment.py"),
    "--task", "emnist_cnn", "--partition", "dirichlet", "--dirichlet_alpha", "0.1",
    "--num_clients", "100", "--clients_per_round", "2",
    "--total_rounds", "80", "--seed", "42", "--local_epochs", "1",
    "--server_lr", "1.5",
    "--scale_mode", "doscale", "--stochastic", "1", "--bits", "1",
    "--transform", "fourier_kashin_balanced",
    "--kashin_iters", "15", "--kashin_eta", "0.9",
    "--output_dir", str(OUT),
]

jobs = ["1.1", "2.0"]

with open(str(LOG), "a", buffering=1) as log:
    for idx, lam in enumerate(jobs, 1):
        name = "stage4snr_kashinbal_b1_lam%s" % lam
        cmd = COMMON + ["--lam", lam, "--exp_name", name]
        log.write("===== [%d/%d] %s start %s\n" % (idx, len(jobs), name, time.ctime()))
        ret = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        log.write("===== [%d/%d] %s exit=%d %s\n" % (idx, len(jobs), name,
                  ret.returncode, time.ctime()))
print("LOW-SNR PROBE DONE ->", OUT)
