"""The 94 hand-crafted event descriptors used by the physical stream, as one function of the waveforms.

feats94(X) with X of shape (n, 4, 10000) at 10 kHz returns (n, 94):
  80 spectral-shape features (log band energies in 40 log-spaced bands from 10 Hz to 5 kHz, gain-normalised; mean and s.d. over sensors)
   8 time-shape features (kurtosis, crest factor, duration above 30 % of peak envelope, late/early envelope ratio; mean and s.d.)
   6 envelope/multifractal features (rise time, post-peak decay rate, MFDFA width; mean and s.d.)
Identical to resources/handcrafted_feats_v2.npz (checked by `python feats94.py --check`).
A missing (dead) sensor is represented by an all-zero channel; all statistics are made finite in that case.
"""
import sys, numpy as np
from scipy.stats import kurtosis
from scipy.signal import hilbert

FS = 10000
_edges = np.geomspace(10, 5000, 41); _f = np.fft.rfftfreq(10000, 1 / FS)


def spec_feats(X):
    F = np.abs(np.fft.rfft(X - X.mean(-1, keepdims=True), axis=-1)) ** 2
    B = np.stack([F[..., (_f >= a) & (_f < b)].sum(-1) for a, b in zip(_edges[:-1], _edges[1:])], -1)
    B = np.log10(B + 1e-12); B = B - B.mean(-1, keepdims=True)
    return np.concatenate([B.mean(1), B.std(1)], -1)


def time_feats(X):
    env = np.abs(hilbert(X, axis=-1)); e = env / (env.max(-1, keepdims=True) + 1e-12)
    with np.errstate(all="ignore"):
        k = np.nan_to_num(kurtosis(X, axis=-1), nan=-3.0)
    crest = np.abs(X).max(-1) / (X.std(-1) + 1e-12)
    dur = (e > 0.3).mean(-1); late = e[..., 5000:].mean(-1) / (e[..., :5000].mean(-1) + 1e-12)
    T = np.stack([np.log1p(np.abs(k)), np.log(crest + 1e-12), dur, np.log(late + 1e-6)], -1)
    return np.concatenate([T.mean(1), T.std(1)], -1)


def _env_feats(S):
    e = np.abs(hilbert(S - S.mean(-1, keepdims=True), axis=-1))
    k = np.ones(50) / 50
    e = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, e)
    pk = e.argmax(1); pv = e[np.arange(len(e)), pk]
    rise = np.zeros(len(e)); dec = np.zeros(len(e))
    for i in range(len(e)):
        if pv[i] <= 0: continue
        below = np.flatnonzero(e[i, :pk[i] + 1] < 0.1 * pv[i])
        rise[i] = (pk[i] - (below[-1] if len(below) else 0)) / FS * 1000
        seg = e[i, pk[i]:pk[i] + 500]
        if len(seg) < 50: seg = e[i, max(0, pk[i] - 500):pk[i] + 1][::-1]
        t = np.arange(len(seg)) / FS * 1000
        dec[i] = -np.polyfit(t, np.log(seg / pv[i] + 1e-6), 1)[0]
    return np.log1p(rise), np.log1p(np.clip(dec, 0, None))


_QS = np.array([-5, -3, -2, -1, 1, 2, 3, 5], float); _SC = np.array([16, 32, 64, 128, 256, 512])


def _mf_width(S):
    Y = np.cumsum(S - S.mean(-1, keepdims=True), -1)
    logF = np.empty((len(S), len(_QS), len(_SC)))
    for j, s in enumerate(_SC):
        n = Y.shape[1] // s; seg = Y[:, :n * s].reshape(len(S), n, s)
        t = np.arange(s) - (s - 1) / 2
        slope = (seg * t).sum(-1, keepdims=True) / (t ** 2).sum(); icpt = seg.mean(-1, keepdims=True)
        F2 = ((seg - icpt - slope * t) ** 2).mean(-1) + 1e-30
        for k, q in enumerate(_QS):
            logF[:, k, j] = np.log((F2 ** (q / 2)).mean(-1)) / q
    ls = np.log(_SC); ls = ls - ls.mean()
    h = (logF * ls).sum(-1) / (ls ** 2).sum()
    tau = _QS * h - 1; alpha = np.gradient(tau, _QS, axis=1)
    w = alpha.max(1) - alpha.min(1)
    w[np.abs(S).max(-1) == 0] = 0.0
    return w


def extra_feats(X):
    n = len(X); S = X.reshape(n * X.shape[1], -1).astype(np.float64); out = []
    for i in range(0, len(S), 1000):
        r, d = _env_feats(S[i:i + 1000]); w = _mf_width(S[i:i + 1000]); out.append(np.stack([r, d, w], -1))
    E = np.concatenate(out).reshape(n, X.shape[1], 3)
    return np.concatenate([E.mean(1), E.std(1)], -1)


def feats94(X):
    X = np.asarray(X, dtype=np.float32)
    F = np.hstack([spec_feats(X), time_feats(X), extra_feats(X)]).astype(np.float32)
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


if __name__ == "__main__" and "--check" in sys.argv:
    from safe_io import load_pkl
    X = np.array([np.array(e, dtype=np.float32) for e, _ in load_pkl("resources/signal_selected_train_data.pkl")[:60]])
    ref = np.load("resources/handcrafted_feats_v2.npz")["tr"][:60]
    got = feats94(X)
    print("max abs diff", np.abs(got - ref).max(), "per block", [float(np.abs(got - ref)[:, a:b].max()) for a, b in [(0, 80), (80, 88), (88, 94)]])
