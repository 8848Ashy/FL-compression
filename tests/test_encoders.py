import pytest
import torch

from compression.encoders import ENCODERS, VectorEncoder


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
    assert metrics["bpd"] == (32.0 if transform == "uncompressed" else 2 * metrics["actual_lam"])
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
