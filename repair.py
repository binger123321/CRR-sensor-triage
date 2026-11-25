"""Check-repair-remove (CRR) sensor triage: signal tests on every sensor, type-specific repair by plain signal processing,
removal when nothing can be recovered.

Every repair is conditional on its own test, whose threshold is the 99.5th percentile of the test on clean validation channels
(see calibrate), so a healthy channel is left untouched. No fault labels are used.
  spikes   isolated one-sample excursions far above the signal  -> replaced by interpolation from the neighbouring samples
  drift    share of power below 3 Hz too high                    -> slow trend removed, zero-phase high-pass at 5 Hz
  hum      power at 50/150 Hz far above the neighbouring bands   -> zero-phase notches at 50, 100, 150 Hz
  gap      runs of exact zeros: up to 60 % of the record         -> filled with noise at the channel's own noise floor
           more than 60 %, with another live sensor left        -> channel removed (set to zero = not alive)
A flat (dead) channel is removed by the set model itself. Clipping, a noise-only channel and loss of gain are not repaired:
the first two cannot be undone from one channel, the last does not need it.

    thr = calibrate(X_clean_validation)        # (n, 4, 10000) float32, 10 kHz
    X_repaired, actions = repair(X, thr)       # actions: bitmask per sensor, see repair()
"""
import numpy as np
from scipy.signal import medfilt, butter, sosfiltfilt, iirnotch, tf2sos
FS = 10000
_f = np.fft.rfftfreq(10000, 1 / FS)
_HP = butter(4, 5, "highpass", fs=FS, output="sos")
_NOTCH = [tf2sos(*iirnotch(f0, 20, fs=FS)) for f0 in (50, 100, 150)]


def _spikes(x):
    """One-sample excursions: far above the bulk of the record (3 x the 98th percentile of |x|) and standing alone (the three samples
    on either side all below 35 % of it). A seismic arrival is never one sample wide at 10 kHz; on clean records this marks at most
    a handful of samples in about 5 % of the channels."""
    ax = np.abs(x); p98 = np.quantile(ax, 0.98, axis=-1, keepdims=True) + 1e-12; nb = np.zeros_like(ax)
    for d in (1, 2, 3): nb = np.maximum(nb, np.maximum(np.roll(ax, d, -1), np.roll(ax, -d, -1)))
    return (ax > 3 * p98) & (nb < 0.35 * ax)


def _contrast(D):
    """Each detector relative to the median of the other sensors of the same event: a fault sits on one sensor, site noise on all."""
    S = D.shape[1]; return D - np.stack([np.median(np.delete(D, s, 1), 1) for s in range(S)], 1)


def detectors(X):
    """(n, S, 4): spike count (log), low-frequency share (log10), hum ratio (log10), fraction of the record in zero runs."""
    X = np.asarray(X, np.float32); ax = np.abs(X); out = np.zeros(X.shape[:2] + (4,), np.float32)
    out[..., 0] = np.log1p(_spikes(X).sum(-1))
    P = np.abs(np.fft.rfft(X - X.mean(-1, keepdims=True), axis=-1)) ** 2          # a constant offset is not drift
    out[..., 1] = np.log10(P[..., _f < 3].sum(-1) / (P.sum(-1) + 1e-20) + 1e-12)
    hum = P[..., (np.abs(_f - 50) <= 2) | (np.abs(_f - 150) <= 2)].mean(-1)
    nb = P[..., ((_f >= 30) & (_f <= 44)) | ((_f >= 56) & (_f <= 70)) | ((_f >= 130) & (_f <= 144)) | ((_f >= 156) & (_f <= 170))].mean(-1)
    out[..., 2] = np.log10(hum / (nb + 1e-20) + 1e-12)
    z = (X == 0); run = z & np.roll(z, 1, -1) & np.roll(z, -1, -1)                    # samples inside runs of at least three zeros
    out[..., 3] = run.mean(-1)
    out[ax.max(-1) == 0] = 0
    return out


def calibrate(Xclean, q=0.995):
    """Thresholds from clean validation events: cross-sensor contrast (spikes, hum) and absolute level (spikes, drift, hum)."""
    D = detectors(Xclean); thr = np.quantile(_contrast(D)[..., :3].reshape(-1, 3), q, axis=0)
    thr[1] = np.inf                           # drift: judged by its absolute level only (slow site noise is common on single sensors)
    return np.r_[thr, np.quantile(D[..., :3].reshape(-1, 3), q, axis=0)]      # [contrast x 3, absolute x 3]


def repair(X, thr, only=None, skip=()):
    """Returns (repaired X, actions (n, S) int bitmask: 1 despiked, 2 high-passed, 4 notched, 8 removed, 16 gaps filled).
    only: optional (n, S) boolean; when given, repair is attempted on those channels only.
    skip: steps to leave out, any of "remove", "gap", "spike", "drift", "hum" (for ablation)."""
    X = np.asarray(X, np.float32).copy(); n, S, L = X.shape; D = detectors(X); Cn = _contrast(D); act = np.zeros((n, S), int)
    thr = np.asarray(thr, float); thr = np.r_[thr, thr] if len(thr) == 3 else thr
    # a detector fires when it stands out from the other sensors (a fault on one sensor) OR when it is high in absolute terms
    # (the same disturbance on the whole array, where no sensor stands out)
    fire = [(Cn[..., j] > thr[j]) | (D[..., j] > thr[3 + j]) for j in range(3)]
    ok = np.ones((n, S), bool) if only is None else only
    live = np.abs(X).max(-1) > 0
    m = ok & live & (D[..., 3] > 0.60) & ("remove" not in skip)                                               # mostly gaps: nothing to recover
    m[(live & ~m).sum(1) == 0] = False                                               # never remove the last live sensors: fill them instead
    X[m] = 0; act[m] |= 8; live &= ~m
    m = ok & live & (D[..., 3] > 0.005) & ("gap" not in skip)                                              # shorter gaps: filled with noise at the channel's own floor,
    if m.any():                                                                      # so that the edge of the gap does not look like an arrival
        x = X[m]; g = np.random.default_rng(12345)
        for i in range(len(x)):
            z = x[i] == 0; z = z & (np.roll(z, 1) | np.roll(z, -1)); v = x[i][~z]
            w = v[: len(v) // 50 * 50].reshape(50, -1).std(1); floor = np.sort(w)[:5].mean()          # quietest tenth of the live part
            x[i][z] = g.standard_normal(int(z.sum())).astype(np.float32) * floor
        X[m] = x; act[m] |= 16
    m = ok & live & (D[..., 0] > 0) & ("spike" not in skip)                                                  # spikes: always removed (the criterion is specific enough)
    if m.any():
        x = X[m]
        for _ in range(4):                                                           # repeated: removing the largest spikes exposes the rest
            bad = _spikes(x)
            if not bad.any(): break
            x[bad] = (0.5 * (np.roll(x, 2, -1) + np.roll(x, -2, -1)))[bad]
        X[m] = x; act[m] |= 1
    m = ok & live & fire[1] & ("drift" not in skip)                                            # drift: slow trend removed by a polynomial fit, then high-pass
    if m.any():
        x = X[m].astype(np.float64); tq = np.linspace(-1, 1, L); xd = x.reshape(len(x), 100, -1).mean(-1); td = tq.reshape(100, -1).mean(-1)
        co = np.polynomial.legendre.legfit(td, xd.T, 8); x = x - np.polynomial.legendre.legval(tq, co)
        X[m] = sosfiltfilt(_HP, x, axis=-1, padtype="even", padlen=3000).astype(np.float32); act[m] |= 2
    m = ok & live & fire[2] & ("hum" not in skip)                                            # hum
    if m.any():
        x = X[m]
        for sos in _NOTCH: x = sosfiltfilt(sos, x, axis=-1)
        X[m] = x.astype(np.float32); act[m] |= 4
    return X, act
