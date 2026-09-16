import torch


def stochastic_k_level_quantize(X_tensor, k_levels, generator=None):
    X_max, X_min = torch.max(X_tensor), torch.min(X_tensor)
    scale = X_max - X_min
    if scale < 1e-8:
        return X_tensor, torch.zeros_like(X_tensor, dtype=torch.long)
    normalized = (X_tensor - X_min) / scale * (k_levels - 1)
    r = torch.floor(normalized).long(); r = torch.clamp(r, 0, k_levels - 2)
    prob = normalized - r
    if generator is None:
        is_upper = (torch.rand_like(X_tensor) < prob).float()
    else:
        # generator must live on the same device as X_tensor
        is_upper = (torch.rand(X_tensor.shape, generator=generator, dtype=X_tensor.dtype, device=X_tensor.device) < prob).float()
    quantized_r = r.float() + is_upper
    return X_min + quantized_r * scale / (k_levels - 1), quantized_r.long()


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
