"""MPA-SincNet+ : the authors' network with three optional additions.

  env   : an envelope branch. Log short-time energy (window 100 samples, hop 50) of each of the 4 channels
          goes through a small 1-D CNN; it sees how energy rises and decays in time, which the
          sinc branch summarises away by global average pooling.
  feat  : 88 hand-crafted features (log band energies + kurtosis, crest factor, duration, late/early energy),
          standardised with training-split statistics, through a 2-layer MLP.
  logits: the classifier returns logits (no softmax), so CrossEntropyLoss is applied once.

With env=False, feat=False the sinc/MPA part is the authors' architecture (same layer names).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.ippa import AAP, AMP
from frontends import FRONTENDS


class EnvBranch(nn.Module):
    def __init__(self, out=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(32, out, 3, padding=1), nn.BatchNorm1d(out), nn.GELU())

    def forward(self, x):                                   # x: (B*4, 10000)
        e = F.avg_pool1d(x.float().unsqueeze(1) ** 2, 100, 50)       # short-time energy, (B*4,1,199)
        e = torch.log10(e + 1e-20)
        e = e - e.amax(-1, keepdim=True)                    # gain-invariant: 0 dB at the peak
        h = self.net(e)
        return torch.cat([h.mean(-1), h.amax(-1)], 1)       # (B*4, 2*out)


class SpecBranch(nn.Module):
    """Multi-resolution log spectrogram (n_fft 128 / 512 / 2048) -> 3-channel 64x64 image per sensor -> small 2-D CNN."""
    def __init__(self, out=64, nffts=(128, 512, 2048)):
        super().__init__()
        self.nffts = nffts
        for n in nffts: self.register_buffer(f"win{n}", torch.hann_window(n), persistent=False)
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU(), nn.MaxPool2d(2))
        self.net = nn.Sequential(blk(len(nffts), 16), blk(16, 32), blk(32, out))

    def forward(self, x):                                   # x: (B*4, 10000)
        ims = []
        with torch.autocast(x.device.type, enabled=False):
            xf = x.float()
            for n in self.nffts:
                S = torch.stft(xf, n, hop_length=n // 4, window=getattr(self, f"win{n}"), return_complex=True).abs() ** 2
                S = 10 * torch.log10(S + 1e-20)
                S = (S - S.amax((1, 2), keepdim=True)).clamp(min=-80) / 80 + 1    # 0 dB at the max, floor at -80 dB
                ims.append(F.adaptive_avg_pool2d(S.unsqueeze(1), (64, 64)))
        h = self.net(torch.cat(ims, 1))                     # (B*4, out, 8, 8)
        return torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1)


class MultiScaleShrink(nn.Module):
    """DRSN-style soft thresholding (Zhao et al., IEEE TII 2020) with MPA-style multi-scale context.
    tau_c = alpha_c * mean_t |h_c|, alpha_c in (0,1) from avg-pooled |h| at 1/2/4 segments; h <- sign(h) * max(|h| - tau, 0)."""
    def __init__(self, c=64, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c * 7, c * 7 // r), nn.ReLU(inplace=True), nn.Linear(c * 7 // r, c), nn.Sigmoid())

    def forward(self, h):
        a = h.abs(); b, c, _ = a.shape
        z = torch.cat([F.adaptive_avg_pool1d(a, k).reshape(b, -1) for k in (1, 2, 4)], 1)
        tau = self.fc(z).unsqueeze(-1) * a.mean(-1, keepdim=True)
        return torch.sign(h) * F.relu(a - tau)


class SensorGraph(nn.Module):
    """Graph attention over the 4 sensors. Nodes: per-sensor MPA features (mean+max over time) + position embedding.
    Edges: attention logits q_i.k_j plus a learned bias from the relative position (dx, dy, dz, distance)."""
    def __init__(self, c=64, d=64, layers=2):
        super().__init__()
        self.inp = nn.Linear(2 * c, d); self.pos = nn.Linear(3, d)
        self.edge = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, layers))
        self.q = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)]); self.k = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.v = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)]); self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.d = d

    def forward(self, h, xyz):                      # h: (B, 4, 64, T), xyz: (B, 4, 3)
        x = self.inp(torch.cat([h.mean(-1), h.amax(-1)], -1).float()) + self.pos(xyz)
        rel = xyz[:, :, None] - xyz[:, None]        # (B, 4, 4, 3)
        eb = self.edge(torch.cat([rel, rel.norm(dim=-1, keepdim=True)], -1))   # (B, 4, 4, layers)
        for l in range(len(self.q)):
            att = (self.q[l](x) @ self.k[l](x).transpose(1, 2)) / self.d ** 0.5 + eb[..., l]
            x = self.ln[l](x + att.softmax(-1) @ self.v[l](x))
        return torch.cat([x.mean(1), x.amax(1)], -1)  # (B, 2d)


class SincNetPlus(nn.Module):
    def __init__(self, frontend="sinc", n_ch=4, n_cls=3, mpa=True, env=False, feat=0, logits=True, drop=0.3, spec=False, bilstm=False, shrink=False, graph=False):
        super().__init__()
        self.n_ch, self.mpa, self.use_env, self.n_feat, self.logits = n_ch, mpa, env, feat, logits
        self.sincnet = nn.Sequential(FRONTENDS[frontend](), nn.MaxPool1d(3), nn.LayerNorm(3250), nn.LeakyReLU(0.2))
        self.feature_extraction1 = nn.Sequential(
            nn.Conv1d(40, 64, 5, stride=1, padding=0), nn.MaxPool1d(3), nn.LayerNorm(1082), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 5, stride=1, padding=1), nn.MaxPool1d(3), nn.LayerNorm(360), nn.LeakyReLU(0.2))
        self.aap = AAP(in_channel=64, channel=64, reduction=16)
        self.amp = AMP(in_channel=64, channel=64, reduction=16)
        self.bilstm = bilstm
        if bilstm:   # temporal aggregation with a bidirectional LSTM instead of global average pooling
            self.fc = nn.Sequential(nn.Conv1d(256, 256, 1), nn.MaxPool1d(3), nn.LayerNorm(120), nn.LeakyReLU(0.2))
            self.lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        else:
            self.fc = nn.Sequential(nn.Conv1d(256, 256, 1), nn.MaxPool1d(3), nn.LayerNorm(120), nn.LeakyReLU(0.2),
                                    nn.AdaptiveAvgPool1d(1))
        d = 256
        self.use_spec = spec; self.use_graph = graph
        self.shrink = MultiScaleShrink(64) if shrink else None
        if graph:
            self.graph = SensorGraph(64, 64); d += 128
        if spec:
            self.spec = SpecBranch(64); d += n_ch * 128
        if env:
            self.env = EnvBranch(32); d += n_ch * 64
        if feat:
            self.feat_mlp = nn.Sequential(nn.Linear(feat, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.GELU())
            d += 64
        self.drop = nn.Dropout(drop) if (env or feat or spec or bilstm or graph or shrink) else nn.Identity()
        self.classifer = nn.Sequential(nn.Linear(d, n_cls))

    def forward(self, x, f=None):                           # x: (B, 4, 10000), f: (B, n_feat)
        B = x.shape[0]
        xr = x.reshape(B * self.n_ch, -1)
        h = self.feature_extraction1(self.sincnet(xr))
        if self.use_graph: f, xyz = f[:, :-12], f[:, -12:].reshape(B, self.n_ch, 3).float()
        if self.mpa: h = self.amp(self.aap(h))
        if self.shrink is not None: h = self.shrink(h)
        g = self.graph(h.reshape(B, self.n_ch, h.shape[1], -1), xyz) if self.use_graph else None
        h = self.fc(h.reshape(B, self.n_ch * 64, -1))
        if self.bilstm:
            o, _ = self.lstm(h.transpose(1, 2).float())          # (B, 120, 256)
            h = 0.5 * (o.mean(1) + o.amax(1))
        else:
            h = h.flatten(1)
        parts = [h]
        if self.use_spec: parts.append(self.spec(xr).reshape(B, -1))
        if self.use_env: parts.append(self.env(xr).reshape(B, -1))
        if self.n_feat: parts.append(self.feat_mlp(f))
        if g is not None: parts.append(g)
        z = self.classifer(self.drop(torch.cat(parts, 1)))
        return z if self.logits else F.softmax(z, dim=-1)
