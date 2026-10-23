"""Single-vector compression probes independent of model/state-dict code."""
from __future__ import annotations

import math

import torch

from compression.kashin_frame import FourierKashinFrame
from compression.kashin_solver import kashin_solve, kashin_solve_balanced
from compression.quantization import stochastic_k_level_quantize
from compression.srk import _fwht_fast


METHODS = (
    "hadamard_srk",
    "fourier_srk",
    "fourier_kashin_historical",
    "fourier_kashin_balanced",
)


def _validate_vector(x: torch.Tensor, bits: int) -> None:
    if not isinstance(x, torch.Tensor) or x.ndim != 1 or x.numel() == 0:
        raise ValueError("x must be a nonempty 1D torch.Tensor")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("x must contain finite floating-point values")
    if x.norm().item() == 0:
        raise ValueError("x must have nonzero norm")
    if not isinstance(bits, int) or bits < 1:
        raise ValueError("bits must be a positive integer")


def prepare_single_vector_compression(
    x: torch.Tensor,
    method: str,
    lam: float,
    frame_seed: int = 0,
    *,
    frame: FourierKashinFrame | None = None,
) -> dict:
    """Prepare the fixed transform and coefficients once for repeated bit trials."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not math.isfinite(lam) or lam < 1:
        raise ValueError("lam must be finite and >= 1")
    d = x.numel()
    if method == "hadamard_srk":
        D = 1 << (d - 1).bit_length()
        padded = torch.zeros(D, dtype=x.dtype, device=x.device)
        padded[:d] = x
        signs_generator = torch.Generator(device=x.device)
        signs_generator.manual_seed(frame_seed)
        signs = torch.where(
            torch.rand(D, generator=signs_generator, device=x.device) < 0.5,
            -torch.ones((), dtype=x.dtype, device=x.device),
            torch.ones((), dtype=x.dtype, device=x.device),
        )
        coefficients = _fwht_fast(padded * signs)
        reconstruct = lambda values: (_fwht_fast(values) * signs)[:d]
        return {"D": D, "actual_lambda": D / d, "coefficients": coefficients,
                "reconstruct": reconstruct}

    D = max(d, int(round(lam * d)))
    if frame is None:
        frame = FourierKashinFrame(d, D, seed=frame_seed)
    elif frame.d != d or frame.D != D:
        raise ValueError("provided frame dimensions do not match x and lam")
    direct = frame.frame_analysis(x) / frame.A
    if method == "fourier_srk":
        coefficients = direct
    elif method == "fourier_kashin_historical":
        coefficients = kashin_solve(frame, x, iterations=10)
    else:
        coefficients = kashin_solve_balanced(frame, x, iterations=20, decay=0.9)
    return {"D": D, "actual_lambda": D / d, "coefficients": coefficients,
            "reconstruct": frame.frame_synthesis}


def evaluate_prepared_vector_compression(
    x: torch.Tensor,
    prepared: dict,
    bits: int,
    quant_seed: int,
    *,
    metadata_bits: int = 64,
) -> dict:
    _validate_vector(x, bits)
    coefficients = prepared["coefficients"]
    if metadata_bits < 0:
        raise ValueError("metadata_bits must be nonnegative")
    generator = torch.Generator(device=coefficients.device)
    generator.manual_seed(int(quant_seed))
    quantized, _ = stochastic_k_level_quantize(
        coefficients, k_levels=2**bits, generator=generator
    )
    reconstructed_unquantized = prepared["reconstruct"](coefficients)
    reconstructed_quantized = prepared["reconstruct"](quantized)
    denominator = x.square().sum()
    algorithm_error = reconstructed_unquantized - x
    quantization_error = reconstructed_quantized - reconstructed_unquantized
    total_error = reconstructed_quantized - x
    nmse = lambda error: float(error.square().sum().div(denominator).item())
    return {
        "actual_bits": int(prepared["D"] * bits + metadata_bits),
        "D": int(prepared["D"]),
        "actual_lambda": float(prepared["actual_lambda"]),
        "kappa_inf": float(
            math.sqrt(prepared["D"]) * coefficients.abs().max().item()
            / x.norm().item()
        ),
        "alg_nmse": nmse(algorithm_error),
        "quant_nmse": nmse(quantization_error),
        "cross_term": float(
            (2 * torch.dot(algorithm_error, quantization_error) / denominator).item()
        ),
        "total_nmse": nmse(total_error),
    }


def evaluate_single_vector_compression(
    x: torch.Tensor,
    method: str,
    lam: float,
    bits: int,
    quant_seed: int,
    frame_seed: int = 0,
) -> dict:
    """Compress and reconstruct one vector, returning bit and error metrics.

    ``frame_seed`` fixes the randomized transform; ``quant_seed`` controls only
    stochastic rounding. For Hadamard, ``lam`` is accepted for a common API but
    actual redundancy is set by padding to the next power of two.
    """
    _validate_vector(x, bits)
    prepared = prepare_single_vector_compression(x, method, lam, frame_seed)
    return evaluate_prepared_vector_compression(x, prepared, bits, quant_seed)
