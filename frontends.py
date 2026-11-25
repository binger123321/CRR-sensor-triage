"""Learnable first-layer front-ends for MPA-SincNet comparisons.

All front-ends map (B, 10000) -> (B, 40, 9750) so the rest of the network is unchanged.
  sinc    : vectorised copy of the authors' SincConv1d (same parameters, same output)
  gabor   : LEAF-style real Gabor filters (learnable centre frequency and bandwidth)
  wavelet : WaveletKernelNet-style Laplace wavelets (learnable frequency and damping)
  cnn     : unconstrained Conv1d with 40 x 251 free weights (control)
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_FILT, L, FS = 40, 251, 10000


def mel_init(n=N_FILT, fs=FS):
    """Identical to the authors' initialisation: returns lower edges b1 and bandwidths (Hz)."""
    high = 2595 * np.log10(1 + (fs / 2) / 700)
    mel = np.linspace(80, high, n)
    f_cos = 700 * (10 ** (mel / 2595) - 1)
    b1, b2 = np.roll(f_cos, 1), np.roll(f_cos, -1)
    b1[0], b2[-1] = 30, fs / 2 - 100
    return b1, b2 - b1


class SincConv1dFast(nn.Module):
    """Numerically equivalent to network/sincnet.py::SincConv1d, without the per-filter Python loop."""
    def __init__(self, N_filt=N_FILT, Filt_dim=L, fs=FS):
        super().__init__()
        self.N_filt, self.Filt_dim, self.fs = N_filt, Filt_dim, fs
        b1, bw = mel_init(N_filt, fs)
        self.freq_scale = fs * 1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1 / self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy(bw / self.freq_scale))
        n = torch.linspace(0, Filt_dim, steps=Filt_dim)
        self.register_buffer("window", (0.54 - 0.46 * torch.cos(2 * math.pi * n / Filt_dim)).float(), persistent=False)
        self.register_buffer("t_right", torch.linspace(1, (Filt_dim - 1) / 2, steps=int((Filt_dim - 1) / 2)) / fs, persistent=False)

    def _sinc(self, band):                      # band: (N,1) in Hz
        x = 2 * math.pi * band * self.t_right   # (N, half)
        yr = torch.sin(x) / x
        return torch.cat([torch.flip(yr, [1]), torch.ones(band.shape[0], 1, device=band.device), yr], 1)

    def filters(self):
        beg = torch.abs(self.filt_b1).float() + 50.0 / self.freq_scale
        end = beg + (torch.abs(self.filt_band).float() + 50.0 / self.freq_scale)
        lp1 = 2 * beg[:, None] * self._sinc(beg[:, None] * self.freq_scale)
        lp2 = 2 * end[:, None] * self._sinc(end[:, None] * self.freq_scale)
        bp = lp2 - lp1
        bp = bp / bp.max(dim=1, keepdim=True).values
        return bp * self.window

    def forward(self, x):
        return F.conv1d(x.unsqueeze(1).float(), self.filters().unsqueeze(1))


class GaborConv1d(nn.Module):
    """Real Gabor filterbank (LEAF-style filters): exp(-t^2 / 2 s^2) * cos(2 pi f t)."""
    def __init__(self, N_filt=N_FILT, Filt_dim=L, fs=FS):
        super().__init__()
        b1, bw = mel_init(N_filt, fs)
        centre = b1 + bw / 2
        sigma_t = np.sqrt(2 * np.log(2)) / (np.pi * np.maximum(bw, 50.0))       # FWHM bandwidth -> temporal std (s)
        sigma_t = np.minimum(sigma_t, (Filt_dim / fs) / 6)                          # keep the envelope inside the kernel
        self.fs = fs
        self.f = nn.Parameter(torch.tensor(centre / fs, dtype=torch.float32))      # normalised centre frequency
        self.log_sigma = nn.Parameter(torch.tensor(np.log(sigma_t * fs), dtype=torch.float32))  # std in samples (log)
        half = (Filt_dim - 1) // 2
        self.register_buffer("t", torch.arange(-half, half + 1, dtype=torch.float32), persistent=False)

    def filters(self):
        f = self.f.abs().clamp(1e-4, 0.5)[:, None]
        s = self.log_sigma.exp().clamp(1.0, self.t.numel() / 4)[:, None]
        return torch.exp(-0.5 * (self.t / s) ** 2) * torch.cos(2 * math.pi * f * self.t)

    def forward(self, x):
        return F.conv1d(x.unsqueeze(1).float(), self.filters().unsqueeze(1))


class LaplaceWaveletConv1d(nn.Module):
    """WaveletKernelNet-style Laplace wavelets: exp(-zeta/sqrt(1-zeta^2) * w t) * sin(w t), t >= 0."""
    def __init__(self, N_filt=N_FILT, Filt_dim=L, fs=FS):
        super().__init__()
        b1, bw = mel_init(N_filt, fs)
        centre = b1 + bw / 2
        self.f = nn.Parameter(torch.tensor(centre / fs, dtype=torch.float32))
        # damping chosen so the envelope decays to ~5% within the kernel for every filter (clipped to a valid range)
        w = 2 * np.pi * centre / fs
        zeta = np.clip(3.0 / (w * Filt_dim), 0.01, 0.6)
        self.logit_zeta = nn.Parameter(torch.tensor(np.log(zeta / (1 - zeta)), dtype=torch.float32))
        self.register_buffer("t", torch.arange(Filt_dim, dtype=torch.float32), persistent=False)

    def filters(self):
        w = 2 * math.pi * self.f.abs().clamp(1e-4, 0.5)[:, None]
        z = torch.sigmoid(self.logit_zeta).clamp(0.005, 0.95)[:, None]
        k = torch.exp(-z / torch.sqrt(1 - z ** 2) * w * self.t) * torch.sin(w * self.t)
        return k / (k.abs().max(dim=1, keepdim=True).values + 1e-8)

    def forward(self, x):
        return F.conv1d(x.unsqueeze(1).float(), self.filters().unsqueeze(1))


class FreeConv1d(nn.Module):
    def __init__(self, N_filt=N_FILT, Filt_dim=L, fs=FS):
        super().__init__()
        self.conv = nn.Conv1d(1, N_filt, Filt_dim, bias=False)

    def forward(self, x):
        return self.conv(x.unsqueeze(1))


FRONTENDS = {"sinc": SincConv1dFast, "gabor": GaborConv1d, "wavelet": LaplaceWaveletConv1d, "cnn": FreeConv1d}
