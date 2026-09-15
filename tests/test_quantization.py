import torch
from compression.quantization import stochastic_k_level_quantize, lloyd_max_quantize


def test_quantizers_finite():
    values = torch.linspace(-1, 1, 100)
    quantized, _ = stochastic_k_level_quantize(values, 4)
    lm, _ = lloyd_max_quantize(values, 4)
    lm = lm[0]
    assert torch.isfinite(quantized).all() and torch.isfinite(lm).all()
