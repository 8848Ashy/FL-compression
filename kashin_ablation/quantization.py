import torch


def uniform_quantize(x, bits):
    levels = 2 ** bits
    xmin, xmax = x.min(), x.max()
    if (xmax - xmin).abs() < 1e-12:
        return x.clone()
    step = (xmax - xmin) / (levels - 1)
    indices = torch.round((x - xmin) / step).clamp(0, levels - 1)
    return xmin + indices * step


def reconstruction_mse(frame, coefficient, bits):
    quantized = uniform_quantize(coefficient, bits)
    reconstructed = frame.synthesis(quantized)
    return torch.mean((reconstructed - frame.synthesis(coefficient)) ** 2).item()

