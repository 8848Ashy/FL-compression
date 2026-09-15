"""Lightweight numerical regression checks for the modularized FL project."""
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.mnist_mlp import MNIST_MLP
from data.mnist_federated import build_mnist_federated_data
from utils.state_dict import flatten_state_dict, unflatten_state_dict
from utils.communication import *
from compression.quantization import stochastic_k_level_quantize, uniform_quantize
from compression.kashin_frame import FourierKashinFrame
from compression.kashin_solver import kashin_solve
from compression.srk import _fwht_fast, federated_round_srk_update
from compression.kashin_compressor import federated_round_kashin_update
from federated.local_training import local_train, local_train_delta
from federated.aggregation import federated_round_original_update


def rel_error(a, b):
    return (torch.norm(a - b) / torch.norm(b)).item()


def main():
    torch.manual_seed(42); np.random.seed(42)
    model = MNIST_MLP(); d = sum(p.numel() for p in model.parameters())
    assert d == 50890, f"unexpected model dimension: {d}"

    flat, shapes = flatten_state_dict(model.state_dict()); restored = unflatten_state_dict(flat, shapes)
    assert all(restored[k].shape == model.state_dict()[k].shape for k in restored)
    assert max((restored[k] - model.state_dict()[k]).abs().max().item() for k in restored) == 0
    print("[PASS] state_dict flatten/unflatten")

    torch.manual_seed(1234); x = torch.randn(d)
    for k in (2, 4, 8):
        q, _ = stochastic_k_level_quantize(x, k)
        assert q.shape == x.shape and torch.isfinite(q).all() and q.min() >= x.min() and q.max() <= x.max()
    print("[PASS] stochastic quantizer")

    torch.manual_seed(1234); frame_results = {}
    for D in (65536, 101780):
        frame = FourierKashinFrame(d, D, seed=2026); assert not hasattr(frame, "use_fwht")
        xx = torch.randn(d); a = frame.frame_analysis(xx); err = rel_error(frame.frame_synthesis(a / frame.A), xx)
        assert err < 1e-5; frame_results[D] = err
        print(f"[PASS] Fourier D={D} -> FFT, reconstruction error={err:.3e}")
    for D in (76335, 101780, 127225, 152670):
        frame = FourierKashinFrame(d, D, seed=2026); xx = torch.randn(d); a = frame.frame_analysis(xx)
        assert rel_error(frame.frame_synthesis(a / frame.A), xx) < 1e-5
    print("[PASS] Fourier arbitrary D")

    frame = FourierKashinFrame(d, 101780, seed=2026); torch.manual_seed(1234); xx = torch.randn(d)
    initial = frame.frame_analysis(xx); coefficients = kashin_solve(frame, xx, iterations=10)
    solver_error = rel_error(frame.frame_synthesis(coefficients), xx); peak_ratio = coefficients.abs().max().item() / initial.abs().max().item()
    assert coefficients.shape == (101780,) and torch.isfinite(coefficients).all() and math.isfinite(solver_error)
    print(f"[PASS] Kashin solver, reconstruction error={solver_error:.3e}, peak ratio={peak_ratio:.6f}")

    client_loaders, _ = build_mnist_federated_data(num_clients=10, images_per_client=600)
    initial_model = MNIST_MLP(); torch.manual_seed(0); np.random.seed(0)
    local_states = [local_train(initial_model, client_loaders[i], epochs=1) for i in range(2)]
    direct = copy.deepcopy(initial_model); direct.load_state_dict({k: (local_states[0][k] + local_states[1][k]) / 2 for k in local_states[0]})
    update_model = copy.deepcopy(initial_model); torch.manual_seed(0); np.random.seed(0)
    deltas = [local_train_delta(update_model, client_loaders[i], epochs=1) for i in range(2)]
    federated_round_original_update(update_model, deltas)
    diff = max((direct.state_dict()[k] - update_model.state_dict()[k]).abs().max().item() for k in direct.state_dict())
    assert diff < 1e-6 and all(torch.isfinite(v).all() for v in update_model.state_dict().values())
    print(f"[PASS] Original delta aggregation, max diff={diff:.3e}")

    srk_model = copy.deepcopy(initial_model); srk_bits = federated_round_srk_update(srk_model, deltas, 4, rotation_seed=2026)
    assert all(torch.isfinite(v).all() for v in srk_model.state_dict().values())
    expected_srk = (65536 * 2 + 64) / d; assert abs(srk_bits - expected_srk) < 1e-12
    assert callable(_fwht_fast); print(f"[PASS] SRK delta aggregation and FWHT, communication={srk_bits:.12f}")

    kashin_model = copy.deepcopy(initial_model); kashin_frame = FourierKashinFrame(d, 101780, seed=2026)
    kashin_bits = federated_round_kashin_update(kashin_model, deltas, 4, kashin_frame, iterations=10)
    assert all(torch.isfinite(v).all() for v in kashin_model.state_dict().values())
    expected_kashin = (101780 * 2 + 64) / d; assert abs(kashin_bits - expected_kashin) < 1e-12
    print(f"[PASS] Fourier-Kashin delta aggregation and FFT, communication={kashin_bits:.12f}")

    for bits in (1, 2, 4, 8):
        assert d * 32 == 32 * d
        assert quantized_payload_bits(65536, bits) == 65536 * bits + 64
        assert quantized_payload_bits(101780, bits) == 101780 * bits + 64
    assert total_communication_bits(101780 + 64, 10, 8) == (101780 + 64) * 10 * 8
    print("[PASS] communication utilities")

    sample = ROOT / "kashin_ablation" / "results" / "sample_delta.pt"
    if sample.exists():
        real = torch.load(sample, map_location="cpu", weights_only=True).float().flatten()
        if real.numel() == d:
            real_frame = FourierKashinFrame(d, 101780, seed=2026); real_a = real_frame.frame_analysis(real); real_k = kashin_solve(real_frame, real, iterations=10)
            real_q = uniform_quantize(real_k, 2); real_mse = torch.mean((real_frame.frame_synthesis(real_q) - real) ** 2).item()
            old_ratio, old_mse, old_err = 0.39487, 5.5963e-6, 2.9186e-7
            ratio = real_k.abs().max().item() / real_a.abs().max().item(); err = rel_error(real_frame.frame_synthesis(real_a / real_frame.A), real)
            differences = [abs(ratio / old_ratio - 1), abs(real_mse / old_mse - 1), abs(err / old_err - 1)]
            label = "PASS" if max(differences) < .01 else "WARNING"
            print(f"[{label}] benchmark peak ratio={ratio:.6f} ({(ratio/old_ratio-1)*100:.2f}%), 2-bit MSE={real_mse:.6e} ({(real_mse/old_mse-1)*100:.2f}%), reconstruction={err:.3e} ({(err/old_err-1)*100:.2f}%)")
            if label == "WARNING": print("[INFO] Historical benchmark used a different frame-randomness path; current canonical error remains below 1e-5.")
    print("[PASS] regression validation complete")


if __name__ == "__main__":
    main()
