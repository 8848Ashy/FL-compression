import csv
import copy
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.mnist_mlp import MNIST_MLP
from federated.local_training import local_train_delta
from federated.evaluation import evaluate_model
from federated.aggregation import federated_round_original_update
from compression.srk import federated_round_srk_update
from compression.kashin_frame import FourierKashinFrame
from compression.kashin_compressor import federated_round_kashin_update


def _crn_seed_list(seed, round_index, num_clients, salt):
    return [(seed + salt) * 1_000_000 + round_index * 100 + client_id for client_id in range(num_clients)]


def run_kashin_lowbit_experiment(client_loaders, test_loader, num_clients=10, rounds=8, seed=2026,
                                 save_outputs=True, crn_paired=True):
    root = Path(__file__).resolve().parents[1]; results = root / "results"; plots = root / "plots"
    results.mkdir(exist_ok=True); plots.mkdir(exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"low-bit experiment device: {device}")
    base = MNIST_MLP().to(device); d = sum(p.numel() for p in base.parameters()); srk_D = 2 ** int(np.ceil(np.log2(d)))
    lambda_values = (1.0, 1.5, 2.0, 2.5, 3.0)
    specs = [(f"SRK-{bits}bit", "SRK", bits, "") for bits in (2, 4, 8, 16)]
    specs += [(f"Kashin-{bits}bit-lambda{lam:g}", "Kashin", bits, lam)
              for bits in (2, 4, 8) for lam in lambda_values]
    specs += [("Original", "Original", 32, "")]
    frames = {lam: FourierKashinFrame(d, int(round(lam * d)), seed=seed) for lam in lambda_values}
    models = {s[0]: copy.deepcopy(base) for s in specs}; cumulative = {s[0]: 0.0 for s in specs}; rows = []
    def traffic(kind, bits, lam=""):
        if kind == "Original": return d * 32 + 64
        encoded_dim = srk_D if kind == "SRK" else frames[lam].D
        return encoded_dim * bits + 64
    print(f"SRK-2bit: {traffic('SRK', 2)} bits/client/round; Kashin-2bit-lambda1: {traffic('Kashin', 2, 1.0)} bits/client/round")
    if crn_paired and getattr(client_loaders[0], "generator", None) is None:
        print("crn_paired: loaders have no shuffle generator (built without paired_shuffle); only quantization noise is paired")
    for r in range(rounds):
        deltas = {name: [] for name, _, _, _ in specs}
        shuffle_seeds = _crn_seed_list(seed, r, num_clients, 0) if crn_paired else None
        for client_id in range(num_clients):
            if shuffle_seeds is not None:
                loader_generator = getattr(client_loaders[client_id], "generator", None)
                if loader_generator is not None: loader_generator.manual_seed(shuffle_seeds[client_id])
            for name in deltas: deltas[name].append(local_train_delta(models[name], client_loaders[client_id], epochs=2))
        quant_seeds = _crn_seed_list(seed, r, num_clients, 1) if crn_paired else None
        for name, kind, bits, lam in specs:
            k = 2 ** bits; relerrs = []
            if kind == "SRK": federated_round_srk_update(models[name], deltas[name], k, rotation_seed=seed * 1000 + bits * 100 + r,
                                                         quant_seeds=quant_seeds, relerr_out=relerrs)
            elif kind == "Original": federated_round_original_update(models[name], deltas[name]); relerrs = [0.0] * num_clients
            else: federated_round_kashin_update(models[name], deltas[name], k, frames[lam], iterations=10,
                                                quant_seeds=quant_seeds, relerr_out=relerrs)
            cumulative[name] += traffic(kind, bits, lam) * num_clients
            rows.append({"method": kind, "label": name, "bits": bits, "lambda": lam,
                         "round": r + 1, "accuracy": evaluate_model(models[name], test_loader),
                         "relerr2": float(np.mean(relerrs)), "relerr2_max": float(np.max(relerrs)),
                         "total_bits": cumulative[name], "communication_MB": cumulative[name] / 8 / 1024 / 1024,
                         "normalized_bits": traffic(kind, bits, lam) / d,
                         "cumulative_normalized_bits": traffic(kind, bits, lam) * (r + 1) / d})
        print(f"low-bit round {r + 1}/{rounds} complete")
    if not save_outputs:
        return rows
    fields = list(rows[0]);
    with (results / "kashin_lowbit_round_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    targets = [85, 87, 88, 89, 90]; target_rows = []
    for name, kind, bits, lam in specs:
        for target in targets:
            reached = [x for x in rows if x["label"] == name and x["accuracy"] * 100 >= target]
            if reached:
                x = min(reached, key=lambda v: v["round"]); target_rows.append({"method": name, "bits": bits, "target_accuracy": target, "communication_MB": x["communication_MB"], "round_reached": x["round"], "status": "REACHED"})
            else: target_rows.append({"method": name, "bits": bits, "target_accuracy": target, "communication_MB": "", "round_reached": "", "status": "NOT_REACHED"})
    with (results / "kashin_accuracy_saving.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(target_rows[0])); w.writeheader(); w.writerows(target_rows)
    def draw(xkey, filename, xlabel):
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, _, _, _ in specs:
            s = [x for x in rows if x["label"] == name]; ax.plot([x[xkey] for x in s], [x["accuracy"] * 100 for x in s], marker="o", label=name)
        ax.set_xlabel(xlabel); ax.set_ylabel("Test Accuracy (%)"); ax.grid(True, linestyle="--", alpha=.5); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(plots / filename, dpi=300); plt.close(fig)
    draw("communication_MB", f"kashin_lowbit_accuracy_communication_{run_stamp}.png", "Cumulative Total Communication (MB)")
    draw("cumulative_normalized_bits", f"kashin_lowbit_normalized_communication_{run_stamp}.png", "Cumulative Bits per Original Dimension per Client")
    fig, ax = plt.subplots(figsize=(10, 6)); colors = {2: "tab:blue", 4: "tab:orange", 8: "tab:green", 16: "tab:red"}; markers = {2: "x", 4: "s", 8: "^", 16: "D"}
    for bits in (2, 4, 8, 16):
        srk = [x for x in rows if x["method"] == "SRK" and x["bits"] == bits and x["round"] == rounds]
        if srk:
            ax.scatter(srk[0]["normalized_bits"], srk[0]["accuracy"] * 100, color="black", marker=markers[bits], s=75, label=f"SRK b={bits}")
        k = sorted([x for x in rows if x["method"] == "Kashin" and x["bits"] == bits and x["round"] == rounds], key=lambda x: x["lambda"])
        if k:
            ax.plot([x["normalized_bits"] for x in k], [x["accuracy"] * 100 for x in k], marker="o", color=colors[bits], label=f"Kashin b={bits}")
            for x in k:
                ax.annotate(f"λ={x['lambda']:g}", (x["normalized_bits"], x["accuracy"] * 100), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Per-round Communication (bit/dim/client)"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Accuracy–Communication Trade-off: SRK vs Kashin λ Sweep")
    ax.grid(True, linestyle="--", alpha=.5); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(plots / f"kashin_lambda_tradeoff_per_dim_{run_stamp}.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, _, _, _ in specs:
        s = [x for x in target_rows if x["method"] == name and x["status"] == "REACHED"]; ax.plot([x["target_accuracy"] for x in s], [float(x["communication_MB"]) for x in s], marker="o", label=name)
    ax.set_xlabel("Target Accuracy (%)"); ax.set_ylabel("Communication (MB)"); ax.grid(True, linestyle="--", alpha=.5); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(plots / f"kashin_lowbit_target_accuracy_{run_stamp}.png", dpi=300); plt.close(fig)
    relerr_summary = []
    for name, kind, bits, lam in specs:
        values = [x["relerr2"] for x in rows if x["label"] == name]
        relerr_summary.append({"label": name, "method": kind, "bits": bits, "lambda": lam,
                               "normalized_bits": traffic(kind, bits, lam) / d, "relerr2": float(np.mean(values))})
    with (results / "kashin_relerr2_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(relerr_summary[0])); w.writeheader(); w.writerows(relerr_summary)
    fig, ax = plt.subplots(figsize=(10, 6))
    for bits in (2, 4, 8, 16):
        srk = [x for x in relerr_summary if x["method"] == "SRK" and x["bits"] == bits]
        if srk:
            ax.scatter(srk[0]["normalized_bits"], srk[0]["relerr2"], color="black", marker=markers[bits], s=75, label=f"SRK b={bits}")
        kline = sorted([x for x in relerr_summary if x["method"] == "Kashin" and x["bits"] == bits], key=lambda x: x["lambda"])
        if kline:
            ax.plot([x["normalized_bits"] for x in kline], [x["relerr2"] for x in kline], marker="o", color=colors[bits], label=f"Kashin b={bits}")
            for x in kline:
                ax.annotate(f"λ={x['lambda']:g}", (x["normalized_bits"], x["relerr2"]), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_yscale("log"); ax.set_xlabel("Per-round Communication (bit/dim/client)")
    ax.set_ylabel("Mean relative squared error  ||delta_hat - delta||^2 / ||delta||^2")
    ax.set_title("Compression Distortion vs Communication: SRK vs Kashin λ Sweep")
    ax.grid(True, linestyle="--", alpha=.5); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(plots / f"kashin_relerr2_tradeoff_{run_stamp}.png", dpi=300); plt.close(fig)
    print("mean relative squared compression error per configuration:")
    for x in relerr_summary: print(f"  {x['label']:28s} {x['relerr2']:.6e}")
    print("[PASS] four low-bit figures + relerr2 summary saved")
    return rows
