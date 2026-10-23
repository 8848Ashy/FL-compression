import pytest
import torch

from compression.encoders import ENCODERS, VectorEncoder
from compression.quantization import quantize_doscale, quantize_static_noscale


@pytest.mark.parametrize("transform", [x for x in ENCODERS if x not in ("drive", "drive_plus")])
@pytest.mark.parametrize("scale_mode", ("doscale", "noscale"))
def test_encoder_round_trip_shape_and_metrics(transform, scale_mode):
    x = torch.linspace(-0.2, 0.3, 17)
    encoder = VectorEncoder(transform=transform, scale_mode=scale_mode, bits=2,
                            lam=1.3, clip_val=0.25, kashin_iters=3,
                            stochastic=False)
    y, metrics = encoder.encode_decode(x, seed=17)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    if transform == "uncompressed":
        assert metrics["bpd"] == 32.0
    else:
        expected = 2 * metrics["actual_lam"]
        if scale_mode == "doscale":
            expected += 32 / x.numel()
        assert metrics["bpd"] == pytest.approx(expected)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_multibit_dynamic_quantizer_uses_full_range_and_returns_metadata():
    x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
    result = quantize_doscale(x, bits=2, stochastic=False)
    assert set(result) == {"values", "bpc", "scale"}
    assert result["scale"] == 10.0
    assert result["bpc"] == 2.0
    assert result["values"].min() == -10.0
    assert result["values"].max() == 10.0


def test_static_quantizer_returns_metadata_and_dimensional_bounds():
    x = torch.linspace(-1, 1, 17)
    result = quantize_static_noscale(x, bits=2, clip_val=0.2, stochastic=False)
    assert set(result) == {"values", "bpc", "scale"}
    assert result["bpc"] == 2.0
    assert result["scale"] == 0.2
    raw = VectorEncoder(transform="raw", scale_mode="noscale", clip_val=0.25)
    had = VectorEncoder(transform="hadamard", scale_mode="noscale", clip_val=0.25)
    assert raw._effective_clip(16, 16) == pytest.approx(0.25)
    assert had._effective_clip(16, 16) < raw._effective_clip(16, 16)
