import torch
from compression.quantization import stochastic_k_level_quantize, lloyd_max_quantize
from compression.srk import _fwht_fast


def test_quantizers_finite():
    values = torch.linspace(-1, 1, 100)
    quantized, _ = stochastic_k_level_quantize(values, 4)
    lm, _ = lloyd_max_quantize(values, 4)
    lm = lm[0]
    assert torch.isfinite(quantized).all() and torch.isfinite(lm).all()


def test_stochastic_quantizer_respects_levels_and_bounds():
    values = torch.linspace(-2.0, 3.0, 129)
    for bits in (1, 2, 4):
        generator = torch.Generator().manual_seed(100 + bits)
        quantized, indices = stochastic_k_level_quantize(
            values, 2**bits, generator=generator
        )
        assert quantized.shape == values.shape
        assert indices.min() >= 0 and indices.max() < 2**bits
        assert quantized.min() >= values.min() and quantized.max() <= values.max()
        assert torch.isfinite(quantized).all()


def test_normalized_fwht_is_orthogonal_and_self_inverse():
    values = torch.randn(1024, generator=torch.Generator().manual_seed(7))
    reconstructed = _fwht_fast(_fwht_fast(values))
    assert torch.allclose(reconstructed, values, atol=2e-6, rtol=2e-6)
    assert torch.allclose(_fwht_fast(values).norm(), values.norm(), atol=2e-5, rtol=2e-5)
