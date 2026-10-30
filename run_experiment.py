"""Single entry point for reproducible federated compression experiments."""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from compression.encoders import ENCODERS, VectorEncoder
from data.mnist_federated import build_mnist_federated_data
from federated.aggregation import run_federated_round
from federated.evaluation import evaluate_model_metrics
from models.mnist_mlp import MNIST_CNN, MNIST_MLP


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("mnist_mlp", "mnist_cnn", "emnist_cnn"), default="mnist_mlp")
    parser.add_argument("--partition", choices=("iid", "dirichlet"), default="iid")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5)
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--clients_per_round", type=int, default=10)
    parser.add_argument("--images_per_client", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_rounds", type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--client_lr", type=float, default=0.05)
    parser.add_argument("--server_lr", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--transform", choices=ENCODERS, default="uncompressed")
    parser.add_argument("--scale_mode", choices=("doscale", "noscale"), default="doscale")
    parser.add_argument("--scale_type", choices=("l1_mean", "l2_norm", "max"), default="l1_mean")
    parser.add_argument("--stochastic", type=int, choices=(0, 1), default=1,
                        help="1=stochastic rounding (hadamard/kashin family), 0=deterministic (signSGD family)")
    parser.add_argument("--clip_quantile", type=float, default=0.0,
                        help="if in (0,1), doscale bound = quantile(|x|) with clipping")
    parser.add_argument("--bits", type=int, choices=(1, 2, 3, 4, 8), default=1)
    parser.add_argument("--lam", type=float, default=1.3)
    parser.add_argument("--clip_val", type=float, default=0.25,
                        help="reference L2 norm for noscale dimension_scaled bounds")
    parser.add_argument("--noscale_bound_mode", choices=("dimension_scaled", "raw_fixed"),
                        default="dimension_scaled")
    parser.add_argument("--kashin_iters", type=int, default=5)
    parser.add_argument("--kashin_eta", type=float, default=0.9)
    parser.add_argument("--frame_seed_mode", choices=("independent", "shared"), default="independent")
    parser.add_argument("--output_dir", type=Path, default=Path("./results/default"))
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--emnist_split", choices=("balanced", "byclass"), default="balanced")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.transform in ("drive", "drive_plus"):
        raise NotImplementedError("DRIVE and DRIVE+ are deferred in this refactor")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    dataset = "emnist" if args.task == "emnist_cnn" else "mnist"
    clients, test_loader = build_mnist_federated_data(
        root="./data", num_clients=args.num_clients,
        images_per_client=args.images_per_client, batch_size_train=args.batch_size,
        batch_size_test=256, partition=args.partition,
        dirichlet_alpha=args.dirichlet_alpha, dataset=dataset,
        seed=args.seed, emnist_split=args.emnist_split,
    )
    if args.task == "mnist_mlp":
        model = MNIST_MLP()
    elif args.task == "mnist_cnn":
        model = MNIST_CNN(num_classes=10)
    else:
        model = MNIST_CNN(num_classes=47 if args.emnist_split == "balanced" else 62)
    model.to(device)
    encoder = VectorEncoder(
        transform=args.transform, scale_mode=args.scale_mode, bits=args.bits,
        lam=args.lam, clip_val=args.clip_val, noscale_bound_mode=args.noscale_bound_mode,
        scale_type=args.scale_type,
        kashin_iters=args.kashin_iters, kashin_eta=args.kashin_eta,
        stochastic=bool(args.stochastic),
        clip_quantile=args.clip_quantile,
    )
    exp_name = args.exp_name or (
        f"{args.transform}_{args.scale_mode}_b{args.bits}_lam{args.lam:.2f}_s{args.seed}"
    )
    out = args.output_dir / exp_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "hparams.json").write_text(json.dumps(vars(args) | {"device": str(device),
        "exp_name": exp_name}, indent=2, default=str), encoding="utf-8")
    columns = ("round", "train_loss", "test_loss", "test_acc", "mean_nmse", "mean_mse",
               "bpd", "actual_lam", "cumulative_bpd", "elapsed_sec")
    cumulative_bpd = 0.0
    started = time.perf_counter()
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for round_idx in range(args.total_rounds):
            stats = run_federated_round(
                model, clients, encoder, clients_per_round=args.clients_per_round,
                round_idx=round_idx, base_seed=args.seed,
                frame_seed_mode=args.frame_seed_mode, local_epochs=args.local_epochs,
                client_lr=args.client_lr, server_lr=args.server_lr,
            )
            cumulative_bpd += stats["bpd"]
            test_metrics = evaluate_model_metrics(model, test_loader)
            writer.writerow({
                "round": round_idx + 1, "train_loss": stats["train_loss"],
                "test_loss": test_metrics["test_loss"], "test_acc": test_metrics["test_acc"],
                "mean_nmse": stats["mean_nmse"], "mean_mse": stats["mean_mse"],
                "bpd": stats["bpd"], "actual_lam": stats["actual_lam"],
                "cumulative_bpd": cumulative_bpd,
                "elapsed_sec": time.perf_counter() - started,
            })
            stream.flush()
            print(f"round={round_idx + 1}/{args.total_rounds} acc={test_metrics['test_acc']:.4f} "
                  f"nmse={stats['mean_nmse']:.5g} bpd={stats['bpd']:.3g}", flush=True)
    print(f"Saved run to {out}")


if __name__ == "__main__":
    main()
