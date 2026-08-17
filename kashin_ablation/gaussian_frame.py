import math
import time
import warnings

import torch

from frame_base import BaseFrame


class GaussianRandomFrame(BaseFrame):
    """Pure Gaussian random tight-frame approximation without QR or dense storage.

    Let G_ij ~ N(0, 1/D). We use U = sqrt(D/d) G, so
    E[U^T U] = (D/d) I and the Kashin solver's frame bound is A=D/d.
    Blocks are regenerated deterministically from (seed, block index), so
    analysis and synthesis use exactly the same random matrix without storing it.
    """

    def __init__(self, d, D, seed=2026, block_size=512):
        super().__init__(d, D)
        self.seed = int(seed)
        self.block_size = int(block_size)
        self.init_time = 0.0
        self._block_seeds = [self.seed + 1000003 * index
                             for index in range((self.D + self.block_size - 1) // self.block_size)]

    def _block(self, start, end, device):
        block_index = start // self.block_size
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._block_seeds[block_index])
        rows = end - start
        matrix = torch.randn(rows, self.d, generator=generator, dtype=torch.float32)
        matrix /= math.sqrt(self.D)
        return matrix.to(device)

    def analysis(self, x):
        values = []
        for start in range(0, self.D, self.block_size):
            end = min(start + self.block_size, self.D)
            values.append(self._block(start, end, x.device) @ x)
        return math.sqrt(self.A) * torch.cat(values)

    def synthesis(self, a):
        result = torch.zeros(self.d, dtype=a.dtype, device=a.device)
        for start in range(0, self.D, self.block_size):
            end = min(start + self.block_size, self.D)
            result += self._block(start, end, a.device).T @ a[start:end]
        return math.sqrt(self.A) * result

