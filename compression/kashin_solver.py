import math
import torch


def kashin_solve(frame, x, iterations=10, clipping_level=None):
    if clipping_level is None: clipping_level = 2.0 * torch.norm(x) / math.sqrt(frame.D)
    coefficient = torch.zeros(frame.D, dtype=x.dtype, device=x.device); residual = x.clone()
    for _ in range(iterations):
        candidate = frame.frame_analysis(residual) / frame.A
        clipped = torch.clamp(candidate, -clipping_level, clipping_level)
        coefficient += clipped; residual -= frame.frame_synthesis(clipped)
    return coefficient


def kashin_solve_balanced(frame, x, iterations=20, decay=0.9):
    """Shrinking clipping with an exact residual correction.

    Retain the smallest-range feasible candidate, including direct analysis.
    Range is the relevant scale for our min/max stochastic quantizer. This is
    a measured solver variant, not a claim of a Kashin bound for this frame.
    The historical solver above remains available for an explicit ablation.
    """
    if iterations < 1 or not 0 < decay < 1:
        raise ValueError("iterations must be positive and decay in (0, 1)")
    best = frame.frame_analysis(x) / frame.A
    if x.norm() == 0:
        return best
    best_range = (best.max() - best.min()).item()
    coefficient = torch.zeros_like(best)
    residual = x.clone()
    threshold = 2 * x.norm() / math.sqrt(frame.A * frame.D)
    for _ in range(iterations):
        candidate = frame.frame_analysis(residual) / frame.A
        clipped = candidate.clamp(-threshold, threshold)
        coefficient += clipped
        residual = x - frame.frame_synthesis(coefficient)
        corrected = coefficient + frame.frame_analysis(residual) / frame.A
        width = (corrected.max() - corrected.min()).item()
        if width < best_range:
            best, best_range = corrected.clone(), width
        threshold *= decay
    return best
