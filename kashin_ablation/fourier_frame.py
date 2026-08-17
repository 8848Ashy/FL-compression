import math
import torch

from frame_base import BaseFrame


class FourierFrame(BaseFrame):
    """Random-phase Fourier tight frame, supporting arbitrary D without dense matrices."""

    def __init__(self, d, D, seed=2026):
        super().__init__(d, D)
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.indices = torch.randperm(D, generator=generator)[:d]
        angles = 2 * math.pi * torch.rand(D // 2 + 1, generator=generator)
        angles[0] = 0.0
        if D % 2 == 0:
            angles[-1] = 0.0
        self.phase = torch.polar(torch.ones_like(angles), angles)
        self.signs = torch.where(
            torch.rand(D, generator=generator) < 0.5,
            torch.tensor(-1.0), torch.tensor(1.0))

    def _forward(self, x):
        spectrum = torch.fft.rfft(x)
        return torch.fft.irfft(spectrum * self.phase.to(x.device), n=self.D)

    def _transpose(self, x):
        spectrum = torch.fft.rfft(x)
        return torch.fft.irfft(spectrum * self.phase.conj().to(x.device), n=self.D)

    def analysis(self, x):
        padded = torch.zeros(self.D, dtype=x.dtype, device=x.device)
        padded[self.indices.to(x.device)] = x
        return math.sqrt(self.A) * self.signs.to(x.device) * self._transpose(padded)

    def synthesis(self, a):
        transformed = self._forward(self.signs.to(a.device) * a)
        return math.sqrt(self.A) * transformed[self.indices.to(a.device)]

