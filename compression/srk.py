import math
import torch


def _fwht_fast(tensor):
    """快速沃尔什-哈达玛变换 (Fast Walsh-Hadamard Transform, FWHT)。
    用于实现随机正交旋转，将向量的离群值（Outliers）均匀摊平到各个维度。
    """
    n = tensor.shape[0]; result = tensor.clone(); h = 1
    while h < n:
        blocks = result.view(-1, 2 * h); left, right = blocks[:, :h].clone(), blocks[:, h:].clone()
        blocks[:, :h], blocks[:, h:] = left + right, left - right; h *= 2
    return result.view(n) / math.sqrt(n)


def hadamard_forward(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """Pad a vector and apply a normalized randomized Hadamard transform."""
    if x.ndim != 1 or signs.ndim != 1 or signs.numel() < x.numel():
        raise ValueError("x and signs must be 1D and signs must cover x")
    if signs.numel() & (signs.numel() - 1):
        raise ValueError("signs length must be a power of two")
    padded = torch.zeros_like(signs, dtype=x.dtype, device=x.device)
    padded[:x.numel()] = x
    return _fwht_fast(padded * signs.to(device=x.device, dtype=x.dtype))


def hadamard_inverse(y: torch.Tensor, orig_dim: int, signs: torch.Tensor) -> torch.Tensor:
    """Invert a normalized Hadamard transform and crop to the input dimension."""
    if y.ndim != 1 or signs.ndim != 1 or y.numel() != signs.numel():
        raise ValueError("y and signs must be 1D vectors of equal length")
    if not 0 <= orig_dim <= y.numel():
        raise ValueError("orig_dim is outside the encoded dimension")
    return (_fwht_fast(y) * signs.to(device=y.device, dtype=y.dtype))[:orig_dim]