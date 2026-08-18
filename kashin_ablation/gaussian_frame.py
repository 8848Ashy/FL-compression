import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from frame_base import BaseFrame


class GaussianRandomFrame(BaseFrame):
    """Pure Gaussian random tight-frame approximation without QR or dense storage.

    Let G_ij ~ N(0, 1/D). We use U = sqrt(D/d) G, so
    E[U^T U] = (D/d) I and the Kashin solver's frame bound is A=D/d.
    Blocks are regenerated deterministically from (seed, block index), so
    analysis and synthesis use exactly the same random matrix without storing it.
    """

    def __init__(self, d, D, seed=2026, block_size=1024, cache_dir=None):
        super().__init__(d, D)
        init_start = time.perf_counter()
        self.seed = int(seed)
        self.block_size = int(block_size)
        self.init_time = 0.0
        self._block_seeds = [self.seed + 1000003 * index
                             for index in range((self.D + self.block_size - 1) // self.block_size)]
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._prepare_cache()
        self.init_time = time.perf_counter() - init_start

    def _prepare_cache(self):
        """Generate each Gaussian block once; never regenerate during matvecs."""
        for block_index, start in enumerate(range(0, self.D, self.block_size)):
            end = min(start + self.block_size, self.D)
            path = self.cache_dir / f"gaussian_block_{block_index:05d}.bin"
            expected_bytes = (end - start) * self.d * 4
            if path.exists() and path.stat().st_size == expected_bytes:
                continue
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._block_seeds[block_index])
            matrix = torch.randn(end - start, self.d, generator=generator, dtype=torch.float32)
            matrix /= math.sqrt(self.D)
            matrix.numpy().tofile(path)

    def _block(self, start, end, device):
        block_index = start // self.block_size
        if self.cache_dir is not None:
            path = self.cache_dir / f"gaussian_block_{block_index:05d}.bin"
            array = np.memmap(path, dtype="float32", mode="r", shape=(end - start, self.d))
            return torch.from_numpy(np.array(array, copy=True)).to(device)
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
