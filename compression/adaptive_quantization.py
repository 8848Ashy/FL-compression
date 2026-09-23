"""Per-vector empirical MSE fitting; deterministic and generally biased.

Uniform: alternating nearest assignment and least-squares offset/spacing.
Lloyd-Max: nearest assignment and centroid updates, initialized from uniform.
These are local optimizers, not certificates of globally minimum distortion.
No test labels, assumed Gaussian distribution, or persistent fitted state.
"""
import torch


def _assign(x, centers):
    return torch.bucketize(x.contiguous(), (centers[:-1] + centers[1:]) / 2)


@torch.no_grad()
def optimized_uniform_codebook(values, bits, iterations=30):
    if bits not in (1, 2):
        raise ValueError('This controlled experiment supports 1 and 2 bits')
    x = values.flatten()
    if not x.numel() or not torch.isfinite(x).all():
        raise ValueError('Expected a nonempty finite vector')
    levels = 2 ** bits
    grid = torch.arange(levels, device=x.device, dtype=x.dtype)
    if x.min() == x.max():
        return x[0].repeat(levels)
    # Multiple starts mitigate poor local minima caused by outliers.
    starts = [(x.min(), x.max())]
    for tail in (.001, .01, .05, .1, .2):
        bounds = torch.quantile(x, x.new_tensor([tail, 1-tail]))
        starts.append((bounds[0], bounds[1]))
    best, best_loss = None, float('inf')
    for lo, hi in starts:
        centers = lo + grid * ((hi - lo) / (levels - 1))
        for _ in range(iterations):
            idx = _assign(x, centers)
            loss = (centers[idx] - x).square().mean().item()
            if loss < best_loss:
                best, best_loss = centers.clone(), loss
            k = idx.to(x.dtype)
            var = (k - k.mean()).square().mean()
            if var <= 0:
                break
            spacing = ((k-k.mean()) * (x-x.mean())).mean() / var
            offset = x.mean() - spacing * k.mean()
            updated = offset + spacing * grid
            if torch.equal(updated, centers):
                break
            centers = updated
        loss = (centers[_assign(x, centers)]-x).square().mean().item()
        if loss < best_loss:
            best, best_loss = centers.clone(), loss
    return best


@torch.no_grad()
def adaptive_quantize(values, bits, method='optimized-uniform', iterations=30):
    """Return decoded coefficients, indices, float32 metadata, ideal wire bits.

    Uniform sends two endpoints. Lloyd-Max sends all 2**bits centers.
    Decode uses only indices and transmitted metadata; no original values.
    """
    if method not in ('optimized-uniform', 'lloyd-max'):
        raise ValueError(method)
    x = values.flatten().float()
    centers = optimized_uniform_codebook(x, bits, iterations)
    # At 1 bit both families contain exactly the same two-value quantizers.
    if method == 'lloyd-max' and bits > 1:
        best_loss = (centers[_assign(x, centers)]-x).square().mean()
        for _ in range(iterations):
            idx = _assign(x, centers)
            count = torch.bincount(idx, minlength=len(centers))
            sums = torch.zeros_like(centers).scatter_add_(0, idx, x)
            updated = torch.where(count > 0, sums/count.clamp_min(1), centers)
            loss = (updated[_assign(x, updated)]-x).square().mean()
            if loss > best_loss or torch.equal(updated, centers):
                break
            centers, best_loss = updated, loss
    metadata = centers[[0, -1]].clone() if method == 'optimized-uniform' else centers.clone()
    if method == 'optimized-uniform':
        centers = metadata[0] + torch.arange(2**bits, device=x.device) * ((metadata[1]-metadata[0])/(2**bits-1))
    indices = _assign(x, centers).reshape(values.shape)
    decoded = decode_adaptive(indices, metadata, bits, method)
    return decoded, indices, metadata, values.numel()*bits + metadata.numel()*32


def decode_adaptive(indices, metadata, bits, method):
    if method == 'optimized-uniform':
        return metadata[0] + indices.to(metadata.dtype) * ((metadata[1]-metadata[0])/(2**bits-1))
    if method == 'lloyd-max':
        return metadata[indices]
    raise ValueError(method)
