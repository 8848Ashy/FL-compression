"""Unified vector encoder registry for centralized federated experiments."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from compression.kashin_frame import FourierKashinFrame
from compression.kashin_solver import kashin_solve, kashin_solve_balanced
from compression.quantization import quantize_doscale, quantize_static_noscale
from compression.srk import hadamard_forward, hadamard_inverse


ENCODERS = (
    "uncompressed", "raw", "hadamard", "drive", "drive_plus",
    "fourier_kashin", "fourier_kashin_balanced",
)


@dataclass
class VectorEncoder:
    transform: str = "uncompressed"
    scale_mode: str = "doscale"
    bits: int = 1
    lam: float = 1.3
    clip_val: float = 0.02
    scale_type: str = "l1_mean"
    kashin_iters: int = 5
    kashin_eta: float = 0.9
    stochastic: bool = True

    def __post_init__(self):
        if self.transform not in ENCODERS:
            raise ValueError(f"transform must be one of {ENCODERS}")
        if self.scale_mode not in ("doscale", "noscale"):
            raise ValueError("scale_mode must be doscale or noscale")
        if self.bits not in (1, 2, 3, 4, 8):
            raise ValueError("bits must be one of 1, 2, 3, 4, 8")
        if self.lam < 1 or not math.isfinite(self.lam):
            raise ValueError("lam must be finite and >= 1")
        if self.kashin_iters < 1 or not 0 < self.kashin_eta < 1:
            raise ValueError("kashin_iters must be positive and kashin_eta in (0, 1)")

    def encode_decode(self, flat_delta: torch.Tensor, seed: int) -> tuple[torch.Tensor, dict]:
        if flat_delta.ndim != 1 or not flat_delta.numel() or not flat_delta.is_floating_point():
            raise ValueError("flat_delta must be a nonempty 1D floating tensor")
        if not torch.isfinite(flat_delta).all():
            raise ValueError("flat_delta must be finite")
        if self.transform in ("drive", "drive_plus"):
            raise NotImplementedError("DRIVE encoders are intentionally deferred")

        d = flat_delta.numel()
        device = flat_delta.device
        qgen = torch.Generator(device=device).manual_seed(int(seed))
        inverse = lambda value: value
        if self.transform == "uncompressed":
            reconstructed = flat_delta.clone()
            actual_lam, bpd = 1.0, 32.0
        else:
            if self.transform == "hadamard":
                D = 1 << (d - 1).bit_length()
                tgen = torch.Generator(device=device).manual_seed(int(seed))
                signs = torch.where(
                    torch.rand(D, generator=tgen, device=device) < 0.5,
                    -torch.ones(D, dtype=flat_delta.dtype, device=device),
                    torch.ones(D, dtype=flat_delta.dtype, device=device),
                )
                coefficients = hadamard_forward(flat_delta, signs)
                inverse = lambda value: hadamard_inverse(value, d, signs)
            elif self.transform in ("fourier_kashin", "fourier_kashin_balanced"):
                D = max(d, int(round(self.lam * d)))
                frame = FourierKashinFrame(d, D, seed=int(seed))
                if self.transform == "fourier_kashin_balanced":
                    coefficients = kashin_solve_balanced(
                        frame, flat_delta, iterations=self.kashin_iters, decay=self.kashin_eta
                    )
                else:
                    coefficients = kashin_solve(frame, flat_delta, iterations=self.kashin_iters)
                inverse = frame.frame_synthesis
            else:  # raw / SignSGD
                D, coefficients = d, flat_delta

            if self.scale_mode == "doscale":
                quantized = quantize_doscale(
                    coefficients, bits=self.bits, scale_type=self.scale_type,
                    stochastic=self.stochastic, generator=qgen,
                )
            else:
                quantized = quantize_static_noscale(
                    coefficients, bits=self.bits, clip_val=self.clip_val,
                    stochastic=self.stochastic, generator=qgen,
                )
            reconstructed = inverse(quantized)
            actual_lam = D / d
            bpd = self.bits * actual_lam

        error = reconstructed - flat_delta
        mse = error.square().mean().item()
        nmse = (error.square().sum() / flat_delta.square().sum().clamp_min(1e-12)).item()
        return reconstructed, {"bpd": float(bpd), "actual_lam": float(actual_lam),
                               "mse": float(mse), "nmse": float(nmse)}


ENCODER_REGISTRY = {name: VectorEncoder for name in ENCODERS}


def build_encoder(name: str, **kwargs) -> VectorEncoder:
    """Registry-style constructor kept as a convenience for CLI callers."""
    try:
        encoder_type = ENCODER_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown encoder {name!r}; choose from {ENCODERS}") from exc
    return encoder_type(transform=name, **kwargs)
