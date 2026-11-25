import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv1d(nn.Module):
    def __init__(self, N_filt, Filt_dim, fs):
        super(SincConv1d, self).__init__()
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

        # Mel Initialization of the filterbanks
        low_freq_mel = 80
        high_freq_mel = (2595 * np.log10(1 + (self.fs / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.N_filt)  # Equally spaced in Mel scale
        f_cos = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (self.fs / 2) - 100
        self.freq_scale = self.fs * 1.0

        self.filt_b1 = nn.Parameter(torch.from_numpy(b1 / self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2 - b1) / self.freq_scale))

    def forward(self, x):
        device = x.device
        filters = torch.zeros((self.N_filt, self.Filt_dim)).to(device)

        # Get beginning and end frequencies of the filters.
        min_freq = 50.0
        min_band = 50.0

        # Filter window (hamming).
        n = torch.linspace(0, self.Filt_dim, steps=self.Filt_dim)
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / self.Filt_dim)
        window = window.float().to(device)

        t_right = (torch.linspace(1, (self.Filt_dim - 1) / 2, steps=int((self.Filt_dim - 1) / 2)) / self.fs).to(device)
        filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + min_band / self.freq_scale)

        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float() * self.sinc(filt_beg_freq[i].float() * self.freq_scale, t_right)
            low_pass2 = 2 * filt_end_freq[i].float() * self.sinc(filt_end_freq[i].float() * self.freq_scale, t_right)
            band_pass = (low_pass2 - low_pass1)
            band_pass = band_pass / torch.max(band_pass)
            filters[i, :] = band_pass.to(device) * window

        out = F.conv1d(x.unsqueeze(1), filters.unsqueeze(1))  # x为单通道

        return out

    def sinc(self, band, t_right):
        y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band * t_right)
        y_left = torch.flip(y_right, [0])
        y = torch.cat([y_left, torch.ones(1).to(y_right.device), y_right])
        return y


if __name__ == '__main__':
    # 使用示例
    N_filt = 40
    Filt_dim = 251
    fs = 10000

    model = SincConv1d(N_filt, Filt_dim, fs)
    x = torch.randn([8, 10000])
    out = model(x)
    print(out.shape)
