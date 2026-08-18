"""Fourier Kashin-frame benchmark for one real FL update."""
import csv
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
from fourier_frame import FourierFrame
from kashin_solver import kashin_solve
from quantization import reconstruction_mse


def run():
    seed, d, redundancy, bits, iterations = 2026, 50890, 2.0, 2, 10
    D = int(redundancy * d)
    sample_path = RESULTS / "sample_delta.pt"
    if not sample_path.exists():
        raise FileNotFoundError(sample_path)
    x = torch.load(sample_path, map_location="cpu", weights_only=True).float().flatten()
    if x.numel() != d:
        raise ValueError(f"sample_delta.pt has {x.numel()} values; expected {d}")
    init = time.perf_counter(); frame = FourierFrame(d, D, seed=seed)
    init_time = time.perf_counter() - init
    start = time.perf_counter(); initial = frame.analysis(x)
    analysis_time = time.perf_counter() - start
    start = time.perf_counter(); kashin = kashin_solve(frame, x, iterations=iterations)
    solver_time = time.perf_counter() - start
    start = time.perf_counter(); reconstructed = frame.synthesis(kashin)
    synthesis_time = time.perf_counter() - start
    canonical = frame.synthesis(initial / frame.A)
    initial_peak = initial.abs().max().item()
    kashin_peak = kashin.abs().max().item()
    row = {
        "frame": "Fourier", "lambda": redundancy, "d": d, "D": D,
        "input": "RealFL", "bits": bits, "iterations": iterations,
        "initial_peak": initial_peak, "kashin_peak": kashin_peak,
        "peak_ratio": kashin_peak / initial_peak,
        "reconstruction_mse": torch.mean((canonical - x) ** 2).item(),
        "relative_reconstruction_error": (torch.norm(canonical - x) / torch.norm(x)).item(),
        "quantization_mse": reconstruction_mse(frame, kashin, bits),
        "relative_quantization_error": (torch.norm(reconstructed - x) / torch.norm(x)).item(),
        "frame_init_time": init_time, "analysis_time": analysis_time,
        "synthesis_time": synthesis_time, "kashin_solver_time": solver_time,
        "peak_memory_mb": 0.0, "seed": seed, "status": "completed",
    }
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "frame_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys()); writer.writeheader(); writer.writerow(row)
    with (RESULTS / "frame_comparison.md").open("w", encoding="utf-8") as f:
        f.write("# Fourier Kashin Frame Benchmark\n\nOnly the Fourier frame is retained.\n\n")
        f.write("| Metric | Value |\n|---|---:|\n")
        for key, value in row.items(): f.write(f"| {key} | {value} |\n")
    print(f"[PASS] Fourier benchmark completed: {RESULTS / 'frame_comparison.csv'}")
    print(f"d={d}, D={D}, bits={bits}, iterations={iterations}")
    print(f"peak_ratio={row['peak_ratio']:.6f}, quantization_mse={row['quantization_mse']:.6e}")


if __name__ == "__main__":
    run()
