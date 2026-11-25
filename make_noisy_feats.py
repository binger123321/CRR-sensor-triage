"""Noisy copies of the 1500 training events for noise-aware training, with matching descriptors.

For realisation r = 0..R-1, every (event, sensor) gets white noise at an SNR drawn uniformly from 0-30 dB relative to the
channel's RMS. noise_bank(X, snr, r) regenerates the noise deterministically, so only the SNRs and the descriptors are
stored: resources/noisy_feats.npz with snr (R, 1500, 4), f94 (R, 1500, 94), fps (R, 1500, 4, 47).
usage: python -I make_noisy_feats.py [R=3]
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_io import load_pkl
from feats94 import feats94
from feats_ps import feats_ps


def snr_draw(R, n, c=4):
    return np.random.default_rng(777).uniform(0, 30, (R, n, c)).astype(np.float32)


def noise_bank(X, snr_r, r):
    """White noise for realisation r; X (n, 4, T) float32, snr_r (n, 4) in dB."""
    rng = np.random.default_rng(500 + r)
    rms = np.sqrt((X.astype(np.float64) ** 2).mean(-1, keepdims=True)).astype(np.float32)
    return rng.standard_normal(X.shape, dtype=np.float32) * rms * (10 ** (-snr_r[..., None] / 20)).astype(np.float32)


if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    X = np.stack([np.array(e, dtype=np.float32) for e, _ in load_pkl("resources/signal_selected_train_data.pkl")])
    snr = snr_draw(R, len(X)); f94, fps = [], []
    for r in range(R):
        Xn = X + noise_bank(X, snr[r], r)
        f94.append(np.concatenate([feats94(Xn[i:i + 250]) for i in range(0, len(Xn), 250)]))
        fps.append(np.concatenate([feats_ps(Xn[i:i + 250]) for i in range(0, len(Xn), 250)])); print("realisation", r, flush=True)
    np.savez_compressed("resources/noisy_feats.npz", snr=snr, f94=np.stack(f94), fps=np.stack(fps))
    print("saved resources/noisy_feats.npz")
