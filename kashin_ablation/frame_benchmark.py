"""Fourier-Kashin redundancy (lambda) ablation on one real FL update."""
import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"
sys.path.insert(0, str(ROOT))
from fourier_frame import FourierFrame
from kashin_solver import kashin_solve
from quantization import uniform_quantize

LAMBDA_SETTINGS = [1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
BITS_SETTINGS = [1, 2, 4, 8]
D = 50890
SEED = 2026
ITERATIONS = 10


def load_real_update():
    path = RESULTS / "sample_delta.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    x = torch.load(path, map_location="cpu", weights_only=True).float().flatten()
    if x.numel() != D:
        raise ValueError(f"sample_delta.pt has {x.numel()} values; expected {D}")
    return x


def pareto_efficient(rows):
    efficient = []
    for candidate in rows:
        candidate_comm = float(candidate["total_communication_bits"])
        candidate_error = float(candidate["relative_reconstruction_error"])
        dominated = any(
            float(other["total_communication_bits"]) <= candidate_comm
            and float(other["relative_reconstruction_error"]) <= candidate_error
            and (
                float(other["total_communication_bits"]) < candidate_comm
                or float(other["relative_reconstruction_error"]) < candidate_error
            )
            for other in rows
        )
        if not dominated:
            efficient.append(candidate)
    return efficient


def make_plots(rows):
    PLOTS.mkdir(parents=True, exist_ok=True)
    by_lambda = {lam: [r for r in rows if r["lambda"] == lam] for lam in LAMBDA_SETTINGS}
    flatten = [by_lambda[lam][0] for lam in LAMBDA_SETTINGS]
    plt.figure(figsize=(8, 5))
    plt.plot([r["lambda"] for r in flatten], [r["peak_ratio"] for r in flatten], "o-")
    plt.xlabel("Redundancy ratio lambda = D / d"); plt.ylabel("Kashin peak ratio")
    plt.title("Fourier-Kashin Coefficient Flattening"); plt.grid(True, linestyle="--", alpha=.5)
    plt.tight_layout(); plt.savefig(PLOTS / "lambda_vs_peak_ratio.png", dpi=300); plt.close()

    plt.figure(figsize=(8, 5))
    for bits in BITS_SETTINGS:
        subset = [r for r in rows if r["bits"] == bits]
        subset.sort(key=lambda r: r["total_communication_bits"])
        plt.plot([r["total_communication_bits"] for r in subset],
                 [r["relative_reconstruction_error"] for r in subset], "o-", label=f"{bits}-bit")
    plt.xlabel("Total coefficient communication bits"); plt.ylabel("Relative reconstruction error")
    plt.title("Communication–Error Trade-off by Bitwidth"); plt.grid(True, linestyle="--", alpha=.5)
    plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "communication_vs_error.png", dpi=300); plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot([r["lambda"] for r in flatten], [r["total_time"] for r in flatten], "o-")
    plt.xlabel("Redundancy ratio lambda = D / d"); plt.ylabel("Total time (s)")
    plt.title("Fourier-Kashin Computation Cost"); plt.grid(True, linestyle="--", alpha=.5)
    plt.tight_layout(); plt.savefig(PLOTS / "lambda_vs_total_time.png", dpi=300); plt.close()


def write_report(rows, pareto):
    path = RESULTS / "Lambda_Ablation_Report.md"
    with path.open("w", encoding="utf-8") as f:
        f.write("# Fourier-Kashin Lambda Ablation\n\n")
        f.write("## 1. Objective\n\nStudy how `lambda = D / d` affects coefficient flattening, low-bit quantization, and communication. No lambda is assumed optimal in advance.\n\n")
        f.write("## 2. Experimental Setup\n\n")
        f.write(f"- d = {D}\n- lambda = {', '.join(map(str, LAMBDA_SETTINGS))}\n- bits = {', '.join(map(str, BITS_SETTINGS))}\n- input = RealFL\n- iterations = {ITERATIONS}\n- seed = {SEED}\n\n")
        f.write("## 3. Lambda vs D\n\n| lambda | D | D/d |\n|---:|---:|---:|\n")
        for lam in LAMBDA_SETTINGS:
            d_value = int(lam * D + 0.5)
            f.write(f"| {lam} | {d_value} | {d_value / D:.4f} |\n")
        f.write("\n## 4. Coefficient Flattening\n\n| lambda | initial peak | Kashin peak | peak ratio |\n|---:|---:|---:|---:|\n")
        for r in [next(x for x in rows if x["lambda"] == lam and x["bits"] == 1) for lam in LAMBDA_SETTINGS]:
            f.write(f"| {r['lambda']} | {r['initial_peak']:.6g} | {r['kashin_peak']:.6g} | {r['peak_ratio']:.6f} |\n")
        f.write("\n## 5. Quantization\n\n| lambda | 1-bit error | 2-bit error | 4-bit error | 8-bit error |\n|---:|---:|---:|---:|---:|\n")
        for lam in LAMBDA_SETTINGS:
            vals = [next(r["relative_reconstruction_error"] for r in rows if r["lambda"] == lam and r["bits"] == b) for b in BITS_SETTINGS]
            f.write(f"| {lam} | " + " | ".join(f"{v:.6g}" for v in vals) + " |\n")
        f.write("\n## 6. Communication\n\nFor each configuration, coefficient payload is `D * bits`; no additional metadata is counted because this ablation uses the shared fixed construction and the existing scalar quantizer protocol.\n\n")
        f.write("## 7. Computation Cost\n\nSee `lambda_vs_total_time.png` and the CSV for initialization, analysis, solver, synthesis, and total times.\n\n")
        f.write("## 8. Trade-off Analysis\n\nIncreasing lambda increases D and communication. The measured peak ratio and reconstruction error determine whether the additional coefficients are worthwhile. Comparisons across lambda are made separately for each bitwidth.\n\n")
        f.write("## 9. Pareto-efficient Configurations\n\n| lambda | bits | communication | relative error |\n|---:|---:|---:|---:|\n")
        for r in sorted(pareto, key=lambda x: (x["bits"], x["lambda"])):
            f.write(f"| {r['lambda']} | {r['bits']} | {r['total_communication_bits']:.0f} | {r['relative_reconstruction_error']:.6g} |\n")
        f.write("\n## 10. Recommendation\n\nThe next FL experiment should test the lambda values that are Pareto-efficient at the target communication budget. This report does not preselect lambda=2; the recommendation must follow the generated table and plots.\n")


def run():
    torch.manual_seed(SEED)
    x = load_real_update()
    rows = []
    for lam in LAMBDA_SETTINGS:
        D_current = int(lam * D + 0.5)
        total_start = time.perf_counter()
        init_start = time.perf_counter(); frame = FourierFrame(D, D_current, seed=SEED)
        init_time = time.perf_counter() - init_start
        analysis_start = time.perf_counter(); initial = frame.analysis(x)
        analysis_time = time.perf_counter() - analysis_start
        solver_start = time.perf_counter(); kashin = kashin_solve(frame, x, iterations=ITERATIONS)
        solver_time = time.perf_counter() - solver_start
        synthesis_start = time.perf_counter(); original_reconstruction = frame.synthesis(kashin)
        synthesis_time = time.perf_counter() - synthesis_start
        initial_l2, kashin_l2 = torch.norm(initial).item(), torch.norm(kashin).item()
        initial_peak, kashin_peak = initial.abs().max().item(), kashin.abs().max().item()
        for bits in BITS_SETTINGS:
            quantized = uniform_quantize(kashin, bits)
            reconstructed = frame.synthesis(quantized)
            error = reconstructed - x
            rows.append({
                "lambda": lam, "d": D, "D": D_current, "input_type": "RealFL",
                "bits": bits, "seed": SEED, "iterations": ITERATIONS,
                "initial_peak": initial_peak, "kashin_peak": kashin_peak,
                "peak_ratio": kashin_peak / initial_peak,
                "initial_l2": initial_l2, "kashin_l2": kashin_l2,
                "initial_linf": initial.abs().max().item(), "kashin_linf": kashin.abs().max().item(),
                "quantization_mse": torch.mean(error ** 2).item(),
                "relative_reconstruction_error": (torch.norm(error) / torch.norm(x)).item(),
                "relative_l2_error": (torch.norm(error) / torch.norm(x)).item(),
                "coefficient_bits": D_current * bits, "total_communication_bits": D_current * bits,
                "frame_initialization_time": init_time, "analysis_time": analysis_time,
                "kashin_solver_time": solver_time, "synthesis_time": synthesis_time,
                "total_time": time.perf_counter() - total_start, "peak_memory": "not_measured", "status": "completed",
            })
        print(f"lambda={lam} D={D_current} completed")
    fields = list(rows[0].keys()); csv_path = RESULTS / "lambda_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    efficient = pareto_efficient(rows); write_report(rows, efficient); make_plots(rows)
    print(f"[PASS] Wrote {csv_path}, report, and plots")


if __name__ == "__main__": run()
