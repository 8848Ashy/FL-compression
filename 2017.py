"""Thin orchestration entry point for the modular MNIST federated project.

Algorithm implementations live in models/, data/, compression/, federated/ and
utils/. This file intentionally contains no frame, quantizer, solver, or plot
mathematics.
"""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch

import config
from data.mnist_federated import build_mnist_federated_data
from models.mnist_mlp import MNIST_MLP
from federated.local_training import local_train_delta
from federated.evaluation import evaluate_model
from federated.aggregation import federated_round_original_update
from compression.srk import federated_round_srk_update
from compression.kashin_frame import FourierKashinFrame
from compression.kashin_compressor import federated_round_kashin_update
from utils.communication import (quantized_payload_bits, normalized_bits_per_dim,
                                  total_communication_bits, total_communication_mb,
                                  srk_encoded_dimension, kashin_encoded_dimension)
from experiments.lowbit import run_kashin_lowbit_experiment


def verify_communication_formulas(d=50890):
    srk_d = srk_encoded_dimension(d)
    assert d * 32 == 32 * d
    for bits in (1, 2, 4, 8):
        assert quantized_payload_bits(srk_d, bits) == srk_d * bits + 64
        for lam in (2.0, 2.5, 3.0):
            D = kashin_encoded_dimension(d, lam)
            assert quantized_payload_bits(D, bits) == D * bits + 64
    assert total_communication_bits(100, 10, 8) == 8000
    print("[PASS] centralized communication formulas")


def verify_fourier_frames(d=50890):
    for D in (65536, 101780):
        frame = FourierKashinFrame(d, D, seed=config.EXPERIMENT_SEED)
        assert not hasattr(frame, "use_fwht")
        assert frame.fft_phase is not None
        print(f"D={D}: Fourier/FFT backend")


def verify_original_delta_path():
    model = MNIST_MLP()
    zero_delta = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
    assert federated_round_original_update(model, [zero_delta, zero_delta]) == 32.0


def verify_srk_and_kashin_shapes():
    model = MNIST_MLP(); d = sum(p.numel() for p in model.parameters())
    zero_delta = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
    srk_bits = federated_round_srk_update(copy.deepcopy(model), [zero_delta, zero_delta], 4, rotation_seed=2026)
    frame = FourierKashinFrame(d, kashin_encoded_dimension(d, config.KASHIN_LAMBDA), seed=2026)
    kashin_bits = federated_round_kashin_update(copy.deepcopy(model), [zero_delta, zero_delta], 4, frame, iterations=2)
    assert math.isfinite(srk_bits) and math.isfinite(kashin_bits)


def main():
    torch.manual_seed(config.MODEL_SEED); np.random.seed(config.MODEL_SEED)
    verify_communication_formulas()
    verify_fourier_frames()
    verify_original_delta_path()
    verify_srk_and_kashin_shapes()
    if not config.RUN_FULL_EXPERIMENT:
        print("Lightweight validation mode; no full FL experiment requested.")
        return
    if config.RUN_LOWBIT_EXPERIMENT:
        client_loaders, test_loader = build_mnist_federated_data(num_clients=config.NUM_CLIENTS, images_per_client=config.IMAGES_PER_CLIENT)
        run_kashin_lowbit_experiment(client_loaders, test_loader, config.NUM_CLIENTS, config.NUM_ROUNDS_FOCUS, config.EXPERIMENT_SEED)
    else:
        print(f"Modular project is ready. Configured Kashin lambda={config.KASHIN_LAMBDA}.")


if __name__ == "__main__":
    main()
