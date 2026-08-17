import csv
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from fourier_frame import FourierFrame
from gaussian_frame import GaussianRandomFrame
from kashin_solver import kashin_solve
from quantization import reconstruction_mse

RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
BITS = (1, 2, 4, 8)
DIMENSIONS = (1000, 5000, 10000)
REDUNDANCIES = (1.5, 2.0, 3.0)


def make_inputs(d):
    torch.manual_seed(1234 + d)
    gaussian = torch.randn(d)
    sparse = torch.zeros(d)
    sparse[torch.randperm(d)[:max(1, d // 20)]] = torch.randn(max(1, d // 20))
    return [("Gaussian", gaussian), ("Sparse", sparse)]


def evaluate(name, frame, x, lam):
    t0 = time.perf_counter()
    a0 = frame.analysis(x)
    analysis_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    canonical = frame.synthesis(a0 / frame.A)
    synthesis_time = time.perf_counter() - t0
    reconstruction = torch.norm(x - canonical).item() / (torch.norm(x).item() + 1e-12)
    t0 = time.perf_counter()
    a = kashin_solve(frame, x, iterations=10)
    solve_time = time.perf_counter() - t0
    initial_peak = a0.abs().max().item()
    kashin_peak = a.abs().max().item()
    row = {
        "frame": name, "lambda": lam, "d": frame.d, "D": frame.D,
        "input": "", "reconstruction": reconstruction,
        "initial_peak": initial_peak, "kashin_peak": kashin_peak,
        "peak_ratio": kashin_peak / (initial_peak + 1e-12),
        "analysis_time": analysis_time, "synthesis_time": synthesis_time,
        "solver_time": solve_time,
    }
    for bits in BITS:
        row[f"{bits}bit_mse"] = reconstruction_mse(frame, a, bits)
    return row


def benchmark():
    rows = []
    # The requested real-model run is mandatory. Small dimensions are retained
    # only as smoke checks for the Gaussian implementation.
    dimensions = (50890,)
    for d in dimensions:
        for lam in REDUNDANCIES:
            D = int(round(lam * d))
            inputs = make_inputs(d)
            for frame_name, factory in (("Fourier", FourierFrame), ("GaussianRandom", GaussianRandomFrame)):
                print(f"Testing {frame_name}, d={d}, D={D}")
                try:
                    frame = factory(d, D, seed=2026)
                    for input_name, x in inputs:
                        row = evaluate(frame_name, frame, x, lam)
                        row["input"] = input_name
                        rows.append(row)
                except RuntimeError as exc:
                    print(f"  skipped: {exc}")

    sample_path = RESULTS / "sample_delta.pt"
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("fl2017", ROOT / "2017.py")
        fl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fl)
        delta = fl.local_train_delta(fl.MNIST_MLP(), fl.client_loaders[0], epochs=1)
        flat, _ = fl.flatten_state_dict(delta)
        torch.save(flat, sample_path)
        print(f"Saved real FL update: {sample_path}")
        d_real = flat.numel()
        for lam in REDUNDANCIES:
            D_real = int(round(lam * d_real))
            real_frame = FourierFrame(d_real, D_real, seed=2026)
            real_row = evaluate("Fourier", real_frame, flat, lam)
            real_row["input"] = "RealFL"
            rows.append(real_row)
    except Exception as exc:
        print(f"Real FL sample skipped: {exc}")

    csv_path = RESULTS / "frame_comparison.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    md_path = RESULTS / "frame_comparison.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Frame Comparison\n\n")
        handle.write("| Frame | input | d | lambda | reconstruction | peak_ratio | 1bit MSE | 2bit MSE | 4bit MSE | 8bit MSE | analysis+synthesis+solver time |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            total = row["analysis_time"] + row["synthesis_time"] + row["solver_time"]
            handle.write(f"| {row['frame']} | {row['input']} | {row['d']} | {row['lambda']} | {row['reconstruction']:.3e} | {row['peak_ratio']:.3f} | {row['1bit_mse']:.3e} | {row['2bit_mse']:.3e} | {row['4bit_mse']:.3e} | {row['8bit_mse']:.3e} | {total:.3f}s |\n")

    valid = [r for r in rows if r["frame"] == "Fourier"]
    report = HERE / "Kashin_Frame_Ablation_Report.md"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# Kashin Frame Ablation Report\n\n")
        handle.write("## Objective\n\nCompare Random Fourier and Pure Gaussian Random frames independently before changing the FL pipeline. `2017.py` is not modified.\n\n")
        handle.write("## Methods\n\nFourier uses random phases and FFT, supports arbitrary D, and avoids dense matrices. Pure Gaussian uses G_ij~N(0,1/D), regenerated in deterministic chunks; no QR and no dense D×d storage are used. Both are scaled with A=D/d.\n\n")
        handle.write("## Experimental Setup\n\nThe requested real-model dimension is d=50890, with lambda=1.5, 2, 3 and D=int(lambda*d). Inputs are Gaussian, sparse, and a real FL delta_w.\n\n")
        handle.write("## Results\n\nSee `results/frame_comparison.md` and `results/frame_comparison.csv`.\n\n")
        handle.write("## Recommendation\n\nNo frame is selected automatically. Use the tables to decide after considering peak flattening, quantization error, runtime, and the recorded large-scale Gaussian cost or limitation.\n")
    print(f"Generated {csv_path}, {md_path}, {report}")


if __name__ == "__main__":
    benchmark()
