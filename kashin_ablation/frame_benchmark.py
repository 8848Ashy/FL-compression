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
from gaussian_qr_frame import GaussianQRFrame
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
    for d in DIMENSIONS:
        for lam in REDUNDANCIES:
            D = int(round(lam * d))
            inputs = make_inputs(d)
            for frame_name, factory in (("Fourier", FourierFrame), ("GaussianQR", GaussianQRFrame)):
                print(f"Testing {frame_name}, d={d}, D={D}")
                try:
                    frame = factory(d, D, seed=2026)
                    if getattr(frame, "Q", True) is None:
                        print("  skipped: Gaussian QR size limit")
                        continue
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
    best = min(valid, key=lambda r: (r["peak_ratio"], r["4bit_mse"])) if valid else None
    best_json = {
        "selected_frame": best["frame"] if best else "Fourier",
        "reason": {
            "peak_ratio": best["peak_ratio"] if best else None,
            "quantization_error": best["4bit_mse"] if best else None,
            "speed": "fast FFT, arbitrary D"
        }
    }
    with (RESULTS / "best_frame.json").open("w", encoding="utf-8") as handle:
        json.dump(best_json, handle, indent=2)
    report = HERE / "Kashin_Frame_Ablation_Report.md"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# Kashin Frame Ablation Report\n\n")
        handle.write("## Objective\n\nCompare alternative Kashin frames independently before changing the FL pipeline.\n\n")
        handle.write("## Methods\n\nFourier uses random phases and FFT, supports arbitrary D, and avoids dense matrices. Gaussian QR constructs a dense Gaussian matrix and is limited to small validation dimensions.\n\n")
        handle.write("## Results\n\nSee `results/frame_comparison.md` and `results/frame_comparison.csv`.\n\n")
        handle.write("## Recommendation\n\nFourier is recommended for the subsequent FL experiment because it supports arbitrary D and has FFT-scale computational cost. Gaussian QR is useful as a small-scale reference, but its dense construction is not practical for the real model dimension.\n")
    print(f"Generated {csv_path}, {md_path}, {report}")


if __name__ == "__main__":
    benchmark()
