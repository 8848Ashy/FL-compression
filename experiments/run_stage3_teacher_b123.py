# -*- coding: utf-8 -*-
"""Stage-3 (teacher protocol): b=1/2/3 only, within-method consistent quantizers.

- raw        : deterministic rounding (signSGD family), 1-bit = sign x max|x|
- hadamard   : stochastic rounding, lambda fixed by power-of-2 padding (131072/108271)
- kashin     : stochastic rounding, lambda in {1.0, 1.10, 1.21059, 1.30, 1.60, 2.00}
               (1.21059 -> D=131072, identical to hadamard's padded dimension)
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = Path(sys.executable)
OUT = ROOT / "results" / "stage3_teacher_b123"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run_all.log"

COMMON = [
    str(PY), str(ROOT / "run_experiment.py"),
    "--task", "emnist_cnn", "--partition", "dirichlet", "--dirichlet_alpha", "0.3",
    "--num_clients", "100", "--clients_per_round", "10",
    "--total_rounds", "100", "--seed", "42",
    "--scale_mode", "doscale", "--output_dir", str(OUT),
]

jobs = []
for b in (1, 2, 3):
    # raw: signSGD 的 delta（server_lr）。b=1 重建为 ±max|x|，范数被重尾 max 放大
    # ~max/RMS≈50 倍，delta≈0.02 把步长拉回真实梯度量级；全 bit 共用（方法级超参）。
    jobs.append(("raw", b, None, 0, "0.02"))
    jobs.append(("hadamard", b, None, 1, "1.0"))
    for lam in ("1.0", "1.10", "1.21059", "1.30", "1.60", "2.00"):
        jobs.append(("fourier_kashin", b, lam, 1, "1.0"))

with open(str(LOG), "a", buffering=1) as log:
    for idx, (transform, bits, lam, stochastic, server_lr) in enumerate(jobs, 1):
        name = "stage3_%s_b%d" % (transform, bits)
        if lam:
            name += "_lam%s" % lam
        if not stochastic:
            name += "_det"
        cmd = COMMON + ["--transform", transform, "--bits", str(bits),
                        "--stochastic", str(stochastic), "--server_lr", server_lr,
                        "--exp_name", name]
        if lam:
            cmd += ["--lam", lam]
        log.write("===== [%d/%d] %s start %s\n" % (idx, len(jobs), name, time.ctime()))
        ret = subprocess.run(cmd, cwd=str(ROOT), stdout=log,
                             stderr=subprocess.STDOUT)
        log.write("===== [%d/%d] %s exit=%d %s\n" % (idx, len(jobs), name,
                  ret.returncode, time.ctime()))
print("ALL DONE ->", OUT)
