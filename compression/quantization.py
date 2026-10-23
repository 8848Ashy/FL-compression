import math
import torch


def stochastic_k_level_quantize(X_tensor, k_levels, generator=None):
    if not X_tensor.is_floating_point() or not X_tensor.numel() or not torch.isfinite(X_tensor).all():
        raise ValueError('Expected nonempty finite floating-point coefficients')
    if not isinstance(k_levels, int) or k_levels < 2:
        raise ValueError('k_levels must be an integer >= 2')
    X_max, X_min = torch.max(X_tensor), torch.min(X_tensor)
    if X_max == X_min:
        indices = torch.zeros_like(X_tensor, dtype=torch.long)
        return torch.full_like(X_tensor, X_min), indices
    # Double intermediates also handle subnormal ranges and wide finite inputs.
    scale = X_max.double() - X_min.double()
    normalized = (X_tensor.double() - X_min.double()) / scale * (k_levels - 1)
    r = torch.floor(normalized).long(); r = torch.clamp(r, 0, k_levels - 2)
    prob = normalized - r
    if generator is None:
        is_upper = (torch.rand_like(X_tensor) < prob).float()
    else:
        # generator must live on the same device as X_tensor
        is_upper = (torch.rand(X_tensor.shape, generator=generator, dtype=X_tensor.dtype, device=X_tensor.device) < prob).float()
    quantized_r = r.float() + is_upper
    decoded = (X_min.double() + quantized_r.double() / (k_levels - 1) * scale).to(X_tensor.dtype)
    return decoded, quantized_r.long()


def uniform_quantize(x, bits):
    levels = 2 ** bits; xmin, xmax = x.min(), x.max()
    if (xmax - xmin).abs() < 1e-12: return x.clone()
    step = (xmax - xmin) / (levels - 1)
    indices = torch.round((x - xmin) / step).clamp(0, levels - 1)
    return xmin + indices * step


def lloyd_max_codebook(values, k_levels, iterations=20):
    xmin, xmax = values.min(), values.max()
    centers = torch.linspace(xmin, xmax, k_levels, device=values.device, dtype=values.dtype)
    for _ in range(iterations):
        boundaries = (centers[:-1] + centers[1:]) / 2
        indices = torch.bucketize(values, boundaries)
        updated = torch.stack([values[indices == i].mean() if torch.any(indices == i) else centers[i] for i in range(k_levels)])
        if torch.max(torch.abs(updated - centers)) < 1e-8: break
        centers = updated
    return centers


def quantize_with_codebook(values, centers):
    indices = torch.argmin(torch.abs(values[:, None] - centers[None, :]), dim=1)
    return centers[indices], indices


def lloyd_max_quantize(values, k_levels, iterations=20):
    centers = lloyd_max_codebook(values, k_levels, iterations)
    return quantize_with_codebook(values, centers), centers


def quantize_doscale(x, bits=1, scale_type="l1_mean", stochastic=True, generator=None):
    """Quantize with a per-vector dynamic scale (the scale is side metadata)."""
    if x.ndim == 0 or not x.numel() or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("x must be a nonempty finite floating tensor")
    if bits < 1:
        raise ValueError("bits must be >= 1")
    if scale_type == "l1_mean":
        scale = x.abs().mean()
    elif scale_type == "l2_norm":
        scale = x.norm() / math.sqrt(x.numel())
    elif scale_type == "max":
        scale = x.abs().max()
    else:
        raise ValueError("scale_type must be l1_mean, l2_norm, or max")
    if scale.item() == 0:
        return torch.zeros_like(x)
    scale = scale.clamp_min(torch.finfo(x.dtype).tiny)
    if bits == 1:
        return scale * torch.where(x >= 0, 1.0, -1.0)
    # The selected statistic is the dynamic clipping radius for multibit coding.
    return _quantize_symmetric(x.clamp(-scale, scale), scale, bits, stochastic, generator)


def quantize_static_noscale(x, bits=1, clip_val=0.05, stochastic=True, generator=None):
    """Quantize with a fixed public range and no transmitted scale metadata."""
    if x.ndim == 0 or not x.numel() or not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("x must be a nonempty finite floating tensor")
    if bits < 1 or not math.isfinite(float(clip_val)) or clip_val <= 0:
        raise ValueError("bits must be >= 1 and clip_val must be positive")
    bound = torch.as_tensor(clip_val, dtype=x.dtype, device=x.device)
    clipped = x.clamp(-bound, bound)
    if bits == 1:
        return torch.where(clipped >= 0, bound, -bound)
    return _quantize_symmetric(clipped, bound, bits, stochastic, generator)


def _quantize_symmetric(x, bound, bits, stochastic, generator):
    levels = 2 ** bits
    # Use equally spaced levels including both endpoints.
    step = 2 * bound / (levels - 1)
    position = ((x + bound) / step).clamp(0, levels - 1)
    if stochastic:
        lower = position.floor()
        probability = position - lower
        draw = torch.rand(position.shape, device=x.device, dtype=x.dtype, generator=generator)
        indices = lower + (draw < probability).to(x.dtype)
    else:
        indices = position.round()
    return -bound + indices * step
