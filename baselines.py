"""Baseline classifiers trained under exactly the same protocol as MS2-SincNet (forward(x, f) -> logits; f ignored).

resnet1d : 1-D ResNet on the 4-channel waveform (stem stride 4, four stages of two basic blocks)
crnn     : CNN + bidirectional LSTM + attention pooling (common CNN-LSTM microseismic classifier)
transformer : convolutional patch embedding (80 samples / token -> 125 tokens) + 4-layer transformer encoder
speccnn  : 2-D CNN on the multi-resolution spectrogram only (the spectrogram stream of MS2-SincNet on its own)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_plus import SpecBranch


def _norm(x):  # per-event gain normalisation, same as the SincNet input scaling
    return x / (x.flatten(1).std(1)[:, None, None] + 1e-8)


class Block1d(nn.Module):
    def __init__(self, i, o, s):
        super().__init__()
        self.c1 = nn.Conv1d(i, o, 7, s, 3, bias=False); self.b1 = nn.BatchNorm1d(o)
        self.c2 = nn.Conv1d(o, o, 7, 1, 3, bias=False); self.b2 = nn.BatchNorm1d(o)
        self.sc = nn.Sequential() if (i == o and s == 1) else nn.Sequential(nn.Conv1d(i, o, 1, s, bias=False), nn.BatchNorm1d(o))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sc(x))


class ResNet1D(nn.Module):
    def __init__(self, n_cls=3, w=(32, 64, 96, 128)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(4, w[0], 15, 4, 7, bias=False), nn.BatchNorm1d(w[0]), nn.ReLU(), nn.MaxPool1d(3, 2, 1))
        layers, c = [], w[0]
        for k, o in enumerate(w):
            layers += [Block1d(c, o, 1 if k == 0 else 2), Block1d(o, o, 1)]; c = o
        self.body = nn.Sequential(*layers); self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * c, n_cls))

    def forward(self, x, f=None):
        h = self.body(self.stem(_norm(x)))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], 1))


class CRNN(nn.Module):
    def __init__(self, n_cls=3, hid=64):
        super().__init__()
        def blk(i, o, p): return nn.Sequential(nn.Conv1d(i, o, 9, padding=4), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(p))
        self.cnn = nn.Sequential(blk(4, 32, 4), blk(32, 64, 4), blk(64, 64, 4), blk(64, 64, 4))   # 10000 -> 39 steps
        self.rnn = nn.LSTM(64, hid, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hid, 1); self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(2 * hid, n_cls))

    def forward(self, x, f=None):
        h = self.cnn(_norm(x)).transpose(1, 2)
        with torch.autocast(x.device.type, enabled=False):
            h, _ = self.rnn(h.float())
        a = self.att(h).softmax(1)
        return self.head((a * h).sum(1))


class TransformerCls(nn.Module):
    def __init__(self, n_cls=3, d=96, layers=4, heads=4, patch=80):
        super().__init__()
        self.embed = nn.Sequential(nn.Conv1d(4, d, 9, padding=4), nn.GELU(), nn.Conv1d(d, d, patch, patch))
        n_tok = 10000 // patch
        self.cls = nn.Parameter(torch.zeros(1, 1, d)); self.pos = nn.Parameter(torch.randn(1, n_tok + 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers); self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, n_cls)

    def forward(self, x, f=None):
        t = self.embed(_norm(x)).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        return self.head(self.norm(self.enc(t))[:, 0])


class SpecCNN(nn.Module):
    def __init__(self, n_cls=3):
        super().__init__()
        self.spec = SpecBranch(64); self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * 128, n_cls))

    def forward(self, x, f=None):
        B = x.shape[0]
        return self.head(self.spec(x.reshape(B * 4, -1)).reshape(B, -1))


BASELINES = {"resnet1d": ResNet1D, "crnn": CRNN, "transformer": TransformerCls, "speccnn": SpecCNN}
