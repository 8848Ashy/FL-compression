import math
import torch


class FourierKashinFrame:
    """Random-phase Fourier tight frame. Always FFT; never FWHT."""
    def __init__(self, d, D, seed=2026, generator=None, device="cpu"):
        self.d, self.D, self.A = d, D, D / d
        device = torch.device(device)
        if generator is None:
            generator = torch.Generator(device=device); generator.manual_seed(seed)
        elif not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        elif torch.device(generator.device) != device:
            raise ValueError("generator and frame device must match")
        self.device = device
        self.indices = torch.randperm(D, generator=generator, device=device)[:d]
        self.column_signs = (torch.rand(D, generator=generator, device=device) < 0.5).float() * 2 - 1
        angles = 2 * math.pi * torch.rand(D // 2 + 1, generator=generator, device=device)
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
