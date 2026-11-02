# -*- coding: utf-8 -*-
"""Stage-4 mid probe extension: b=1,2 x lam={1.2,1.5,2.0}, same regime as run_stage4_mid_probe."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
OUT = ROOT / "results" / "stage4_mid_probe"
LOG = OUT / "run_all_ext.log"

COMMON = [
    str(PY), str(ROOT / "run_experiment.py"),
    "--task", "emnist_cnn", "--partition", "dirichlet", "--dirichlet_alpha", "0.2",
    "--num_clients", "100", "--clients_per_round", "5",
    "--total_rounds", "80", "--seed", "42", "--local_epochs", "3",
    "--server_lr", "1.25",
    "--scale_mode", "doscale", "--stochastic", "1",
    "--transform", "fourier_kashin_balanced",
    "--kashin_iters", "15", "--kashin_eta", "0.9",
    "--output_dir", str(OUT),
]

jobs = [(b, lam) for b in (1, 2) for lam in ("1.2", "1.5", "2.0")]

with open(str(LOG), "a", buffering=1) as log:
    for idx, (bits, lam) in enumerate(jobs, 1):
        name = "stage4mid_kashinbal_b%d_lam%s" % (bits, lam)
        cmd = COMMON + ["--bits", str(bits), "--lam", lam, "--exp_name", name]
        log.write("===== [%d/%d] %s start %s\n" % (idx, len(jobs), name, time.ctime()))
        ret = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        log.write("===== [%d/%d] %s exit=%d %s\n" % (idx, len(jobs), name,
                  ret.returncode, time.ctime()))
print("MID EXT DONE ->", OUT)
