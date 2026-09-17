"""Numerical guarantees needed for same-input compression comparisons."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from compression.kashin_frame import FourierKashinFrame
from compression.kashin_solver import kashin_solve_balanced
from compression.quantization import stochastic_k_level_quantize
from compression.srk import _fwht_fast


def main():
    torch.set_num_threads(2)
    torch.manual_seed(31)
    for d, D in ((64, 64), (64, 97), (64, 128), (50890, 65536)):
        frame = FourierKashinFrame(d, D, seed=11)
        x = torch.randn(d)
        direct = frame.frame_analysis(x) / frame.A
        balanced = kashin_solve_balanced(frame, x)
        assert torch.norm(frame.frame_synthesis(balanced) - x) / x.norm() < 2e-6
        assert balanced.max() - balanced.min() <= direct.max() - direct.min() + 1e-6
        assert kashin_solve_balanced(frame, torch.zeros(d)).abs().max() == 0
        if d == D:
            assert torch.allclose(balanced, direct, atol=2e-6)
    x = torch.randn(128)
    assert torch.allclose(_fwht_fast(_fwht_fast(x)), x, atol=1e-6)
    # Empirical unbiasedness with independent rounding, and actual code bounds.
    x = torch.linspace(-1, 1, 64)
    g = torch.Generator().manual_seed(77)
    draws = []
    for _ in range(2000):
        q, codes = stochastic_k_level_quantize(x, 4, generator=g)
        assert codes.min() >= 0 and codes.max() < 4
        draws.append(q)
    assert (torch.stack(draws).mean(0) - x).abs().max() < .035
    print('[PASS] reconstruction, coefficient range, square-frame control, FWHT, quantization')


if __name__ == '__main__':
    main()
