"""Per-sensor event descriptors for the sensor-set model (47 per sensor).

feats_ps(X) with X of shape (n, 4, 10000) at 10 kHz returns (n, 4, 47):
  40 log band energies (40 log-spaced bands, 10 Hz - 5 kHz, gain-normalised per sensor)
   4 time-shape features (log kurtosis, log crest factor, duration above 30 % of peak envelope, log late/early ratio)
   3 envelope/multifractal features (log rise time, log post-peak decay rate, MFDFA width)
These are the same quantities as feats94, but kept per sensor instead of being reduced to a mean and s.d. over the
four sensors, so a dead sensor can simply be left out of the set. Dead (all-zero) sensors get an all-zero row.
usage: python -I feats_ps.py   -> writes resources/feats_ps.npz (tr: (1500,4,47), pr: (578,4,47))
"""
import os, sys, numpy as np
from scipy.stats import kurtosis
from scipy.signal import hilbert
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats94 import _edges, _f, _env_feats, _mf_width


def feats_ps(X):
    X = np.asarray(X, dtype=np.float32); n, c, _ = X.shape
    F = np.abs(np.fft.rfft(X - X.mean(-1, keepdims=True), axis=-1)) ** 2
    B = np.stack([F[..., (_f >= a) & (_f < b)].sum(-1) for a, b in zip(_edges[:-1], _edges[1:])], -1)
    B = np.log10(B + 1e-12); B = B - B.mean(-1, keepdims=True)                       # (n, c, 40)
    env = np.abs(hilbert(X, axis=-1)); e = env / (env.max(-1, keepdims=True) + 1e-12)
    with np.errstate(all="ignore"):
        k = np.nan_to_num(kurtosis(X, axis=-1), nan=-3.0)
    crest = np.abs(X).max(-1) / (X.std(-1) + 1e-12)
    dur = (e > 0.3).mean(-1); late = e[..., 5000:].mean(-1) / (e[..., :5000].mean(-1) + 1e-12)
    T = np.stack([np.log1p(np.abs(k)), np.log(crest + 1e-12), dur, np.log(late + 1e-6)], -1)   # (n, c, 4)
    S = X.reshape(n * c, -1).astype(np.float64); out = []
    for i in range(0, len(S), 1000):
        r, d = _env_feats(S[i:i + 1000]); w = _mf_width(S[i:i + 1000]); out.append(np.stack([r, d, w], -1))
    E = np.concatenate(out).reshape(n, c, 3)
    Fps = np.nan_to_num(np.concatenate([B, T, E], -1), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Fps[np.abs(X).max(-1) == 0] = 0.0
    return Fps


if __name__ == "__main__":
    from safe_io import load_pkl
    out = {}
    for k, p in [("tr", "resources/signal_selected_train_data.pkl"), ("pr", "resources/signal_selected_predict_data.pkl")]:
        X = np.stack([np.array(e, dtype=np.float32) for e, _ in load_pkl(p)])
        out[k] = np.concatenate([feats_ps(X[i:i + 250]) for i in range(0, len(X), 250)]); print(k, out[k].shape, flush=True)
    np.savez_compressed("resources/feats_ps.npz", **out); print("saved resources/feats_ps.npz")
