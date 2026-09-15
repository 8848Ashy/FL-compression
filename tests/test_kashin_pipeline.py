import torch
from compression.kashin_frame import FourierKashinFrame
from compression.kashin_solver import kashin_solve


def test_fourier_frame_arbitrary_dimensions():
    x = torch.randn(64)
    for D in (97, 128):
        frame = FourierKashinFrame(64, D, seed=2026)
        assert frame._forward_transform.__qualname__.startswith("FourierKashinFrame")
        a = frame.frame_analysis(x); assert torch.isfinite(a).all()
        assert torch.norm(frame.frame_synthesis(a / frame.A) - x) / torch.norm(x) < 1e-5
        assert torch.isfinite(kashin_solve(frame, x, iterations=2)).all()

