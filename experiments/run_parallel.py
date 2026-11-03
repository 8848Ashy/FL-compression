# -*- coding: utf-8 -*-
"""Parallel batch runner for multi-config / multi-seed sweeps.

Usage:
  python experiments/run_parallel.py --jobs experiments/jobs_stage4_mid_3seeds.json \
      --workers 16 --device cpu

Job file format:
{
  "batch": "stage4_mid_3seeds",
  "output_dir": "results/stage4_mid_3seeds",
  "common": ["--task", "emnist_cnn", ...],          # shared CLI args
  "jobs": [{"name": "cfg_s42", "args": ["--seed", "42", ...], "threads": 2}]
}

Each job runs as an independent process with OMP_NUM_THREADS=threads
(default 2) so 40-core servers can run ~16-20 experiments concurrently
without thread thrash. Per-job logs: <output_dir>/logs/<name>.log
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_job(py, root, common, job, out_dir, device, log_dir):
    name = job["name"]
    args = list(common) + list(job["args"])
    if "--device" not in args:
        args += ["--device", device]
    threads = str(job.get("threads", 2))
    env = dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
    log_path = Path(log_dir) / (name + ".log")
    t0 = time.time()
    with open(str(log_path), "w", buffering=1) as log:
        ret = subprocess.run([str(py), str(root / "run_experiment.py")] + args,
                             cwd=str(root), stdout=log, stderr=subprocess.STDOUT, env=env)
    return name, ret.returncode, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip jobs whose <output_dir>/<exp_name>/metrics.csv already has >=80 rounds")
    args = ap.parse_args()

    spec = json.loads(args.jobs.read_text(encoding="utf-8"))
    root = args.root
    py = Path(sys.executable)
    out_dir = root / spec["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    common = list(spec["common"]) + ["--output_dir", str(out_dir)]

    def name_of(job):
        args_ = list(common) + list(job["args"])
        return args_[args_.index("--exp_name") + 1]

    def is_done(job):
        mc = out_dir / name_of(job) / "metrics.csv"
        if not mc.exists():
            return False
        try:
            with open(str(mc), encoding="utf-8") as f:
                return sum(1 for _ in csv.DictReader(f)) >= 80
        except Exception:
            return False

    import csv
    if args.skip_existing:
        todo = [j for j in spec["jobs"] if not is_done(j)]
        skipped = len(spec["jobs"]) - len(todo)
        if skipped:
            print("[%s] skip-existing: %d done, %d to run" % (spec["batch"], skipped, len(todo)), flush=True)
        spec["jobs"] = todo
        if not spec["jobs"]:
            print("[%s] all jobs already done" % spec["batch"], flush=True)
            return

    print("[%s] %d jobs, %d workers, device=%s" % (spec["batch"], len(spec["jobs"]), args.workers, args.device), flush=True)
    results = []
    t_all = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_job, py, root, common, j, out_dir, args.device, log_dir): j
                for j in spec["jobs"]}
        for fut in as_completed(futs):
            name, rc, el = fut.result()
            results.append((name, rc, el))
            print("  %-44s exit=%d  %.1fs" % (name, rc, el), flush=True)
    ok = sum(1 for _, rc, _ in results if rc == 0)
    summary = ("[%s] DONE %d/%d ok, total %.1f min -> %s"
               % (spec["batch"], ok, len(results), (time.time() - t_all) / 60, out_dir))
    print(summary, flush=True)
    with open(str(out_dir / "batch_summary.txt"), "a", encoding="utf-8") as f:
        f.write(summary + "\n")
        for name, rc, el in sorted(results):
            f.write("  %-44s exit=%d  %.1fs\n" % (name, rc, el))


if __name__ == "__main__":
    main()
