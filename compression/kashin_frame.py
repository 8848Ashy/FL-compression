import math
import torch


class FourierKashinFrame:
    """Random-phase Fourier tight frame. Always FFT; never FWHT."""
    def __init__(self, d, D, seed=2026):
        self.d, self.D, self.A = d, D, D / d
        generator = torch.Generator(); generator.manual_seed(seed)
        self.indices = torch.randperm(D, generator=generator)[:d]
        self.column_signs = (torch.rand(D, generator=generator) < 0.5).float() * 2 - 1
        angles = 2 * math.pi * torch.rand(D // 2 + 1, generator=generator)
        angles[0] = 0
        if D % 2 == 0: angles[-1] = 0
        self.fft_phase = torch.polar(torch.ones_like(angles), angles)

    def _forward_transform(self, x):
        spectrum = torch.fft.rfft(x)
        return torch.fft.irfft(spectrum * self.fft_phase.to(spectrum.device), n=self.D)

    def _transpose_transform(self, x):
        spectrum = torch.fft.rfft(x)
        return torch.fft.irfft(spectrum * self.fft_phase.conj().to(spectrum.device), n=self.D)

    def frame_analysis(self, x):
        padded = torch.zeros(self.D, dtype=x.dtype, device=x.device)
        padded[self.indices.to(x.device)] = x
        return math.sqrt(self.A) * self.column_signs.to(x.device) * self._transpose_transform(padded)

    def frame_synthesis(self, a):
        transformed = self._forward_transform(self.column_signs.to(a.device) * a)
        return math.sqrt(self.A) * transformed[self.indices.to(a.device)]

