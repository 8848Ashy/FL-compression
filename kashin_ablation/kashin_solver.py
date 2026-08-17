import math

import torch


def kashin_solve(frame, x, iterations=10, clipping_level=None):
    """Frame-independent iterative clipped Kashin coefficient solver."""
    if clipping_level is None:
        clipping_level = 2.0 * torch.norm(x) / math.sqrt(frame.D)
    coefficient = torch.zeros(frame.D, dtype=x.dtype, device=x.device)
    residual = x.clone()
    for _ in range(iterations):
        candidate = frame.analysis(residual) / frame.A
        clipped = torch.clamp(candidate, -clipping_level, clipping_level)
        coefficient += clipped
        residual -= frame.synthesis(clipped)
    return coefficient

