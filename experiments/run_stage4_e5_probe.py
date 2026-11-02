# -*- coding: utf-8 -*-
"""Stage-4 E=5 probe: only local_epochs 1->5, everything else identical to stage3.

Question: does E=5 (larger local updates / more distortion per round) make the
bpd-accuracy curve respond to lambda at b=1?
4 runs: kashin_balanced(lam 1.1/1.3/2.0, iters 15, eta 0.9) + hadamard b1.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
OUT = ROOT / "results" / "stage4_e5_probe"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run_all.log"

COMMON = [
    str(PY), str(ROOT / "run_experiment.py"),
    "--task", "emnist_cnn", "--partition", "dirichlet", "--dirichlet_alpha", "0.3",
    "--num_clients", "100", "--clients_per_round", "10",
    "--total_rounds", "50", "--seed", "42", "--local_epochs", "5",
    "--scale_mode", "doscale", "--stochastic", "1", "--bits", "1",
    "--output_dir", str(OUT),
]

jobs = [
    ("fourier_kashin_balanced", "1.1"),
    ("fourier_kashin_balanced", "1.3"),
    ("fourier_kashin_balanced", "2.0"),
    ("hadamard", None),  # lam 实际 = 131072/108271 = 1.21059
]

with open(str(LOG), "a", buffering=1) as log:
    for idx, (transform, lam) in enumerate(jobs, 1):
        name = "stage4e5_%s_b1" % transform.replace("fourier_", "")
        if lam:
            name += "_lam%s" % lam
        else:
            name += "_lam1.21059"
        cmd = COMMON + ["--transform", transform, "--exp_name", name]
        if lam:
            cmd += ["--lam", lam, "--kashin_iters", "15", "--kashin_eta", "0.9"]
        log.write("===== [%d/%d] %s start %s\n" % (idx, len(jobs), name, time.ctime()))
        ret = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        log.write("===== [%d/%d] %s exit=%d %s\n" % (idx, len(jobs), name,
                  ret.returncode, time.ctime()))
print("STAGE4 E5 PROBE DONE ->", OUT)
