import math
import time
import warnings

import torch

from frame_base import BaseFrame


class GaussianQRFrame(BaseFrame):
    """Dense Gaussian QR tight frame for small-scale validation only."""

    def __init__(self, d, D, seed=2026, max_matrix_entries=2_000_000):
        super().__init__(d, D)
        if d * D > max_matrix_entries:
            warnings.warn(
                "Gaussian QR frame is only intended for small-scale frame validation.",
                RuntimeWarning)
        generator = torch.Generator()
        generator.manual_seed(seed)
        # Do not crash on large settings; benchmark code can skip construction.
        if d * D > max_matrix_entries:
            self.Q = None
            return
        G = torch.randn(D, d, generator=generator)
        Q, _ = torch.linalg.qr(G, mode="reduced")
        self.Q = Q

    def analysis(self, x):
        if self.Q is None:
            raise RuntimeError("Gaussian QR frame skipped because it exceeds the size limit")
        return math.sqrt(self.A) * (self.Q @ x)

    def synthesis(self, a):
        if self.Q is None:
            raise RuntimeError("Gaussian QR frame skipped because it exceeds the size limit")
        return math.sqrt(self.A) * (self.Q.T @ a)

