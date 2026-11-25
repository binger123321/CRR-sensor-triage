"""Sensor faults with graded severity, used for (i) severity curves on the independent set and (ii) fault-aware training.

Seven fault types on one sensor; level 1 (mild) .. 5 (severe):
  clip    clipped at {50, 30, 20, 10, 5} % of the channel peak                      (amplifier / ADC saturation)
  spike   {5, 10, 20, 50, 100} one-sample spikes of 3x the channel peak             (electrical interference)
  drift   0.3-1 Hz sinusoid of {0.5, 1, 2, 5, 10} x the channel peak                (baseline drift)
  noisy   white noise at {10, 0, -10, -20, -inf} dB relative to the channel RMS     (decoupled sensor; -inf = noise only)
  gap     {10, 30, 50, 70, 90} % of the record zeroed in 2-5 blocks                 (packet loss)          [never used in training]
  hum     50 Hz + 150 Hz mains pickup of {0.2, 0.5, 1, 2, 5} x the channel RMS      (power-line pickup)    [never used in training]
  gain    channel multiplied by {0.3, 0.1, 0.03, 0.01, 0.003}                       (failing pre-amplifier) [never used in training]
TRAIN_KINDS are the types a model may see in fault-aware training; HELD_OUT types are evaluated only.
"""
import numpy as np
FS = 10000; _t = np.arange(10000) / FS
LEVELS = dict(clip=(0.5, 0.3, 0.2, 0.1, 0.05), spike=(5, 10, 20, 50, 100), drift=(0.5, 1, 2, 5, 10), noisy=(10, 0, -10, -20, None),
              gap=(0.1, 0.3, 0.5, 0.7, 0.9), hum=(0.2, 0.5, 1, 2, 5), gain=(0.3, 0.1, 0.03, 0.01, 0.003))
TRAIN_KINDS = ("clip", "spike", "drift", "noisy"); HELD_OUT = ("gap", "hum", "gain"); KINDS = TRAIN_KINDS + HELD_OUT


def fault_ch(x, kind, v, rng):
    """x: (n, 10000) one channel per event; v: severity value (scalar or (n,) array). Returns the faulty channel."""
    x = x.astype(np.float32).copy(); n = len(x); v = np.broadcast_to(np.asarray(v, dtype=object), (n,))
    pk = np.abs(x).max(-1); rms = np.sqrt((x ** 2).mean(-1))
    for i in range(n):
        vi = v[i]
        if pk[i] == 0: continue
        if kind == "clip": x[i] = np.clip(x[i], -vi * pk[i], vi * pk[i])
        elif kind == "spike":
            j = rng.choice(x.shape[-1], int(vi), replace=False); x[i, j] += 3 * pk[i] * rng.choice([-1.0, 1.0], int(vi))
        elif kind == "drift": x[i] += vi * pk[i] * np.sin(2 * np.pi * rng.uniform(0.3, 1.0) * _t + rng.uniform(0, 2 * np.pi))
        elif kind == "noisy":
            w = rng.standard_normal(x.shape[-1]).astype(np.float32) * rms[i]
            x[i] = w if vi is None else x[i] + w * 10 ** (-vi / 20)
        elif kind == "gap":
            nb = rng.integers(2, 6); L = int(vi * x.shape[-1] / nb)
            for s in rng.integers(0, x.shape[-1] - L, nb): x[i, s:s + L] = 0
        elif kind == "hum":
            ph = rng.uniform(0, 2 * np.pi, 2); x[i] += vi * rms[i] * (np.sin(2 * np.pi * 50 * _t + ph[0]) + 0.5 * np.sin(2 * np.pi * 150 * _t + ph[1]))
        elif kind == "gain": x[i] *= vi
        else: raise ValueError(kind)
    return x


def fault_event(X, kind, level, rng):
    """One random sensor per event gets the fault at the given level (1..5). Returns (faulty X, index of the faulty sensor)."""
    X = X.copy(); s = rng.integers(0, X.shape[1], len(X)); idx = np.arange(len(X))
    X[idx, s] = fault_ch(X[idx, s], kind, LEVELS[kind][level - 1], rng)
    return X, s


def train_bank(X, copies=3, seed=7):
    """Faulty copies of the training events for fault-aware training: for each copy and event one random sensor gets one fault of
    TRAIN_KINDS at a random level. Deterministic. Returns sensor (copies, n), kind index, level, faulty channel waveforms (copies, n, 10000)."""
    rng = np.random.default_rng(seed); n = len(X); S, K, L, W = [], [], [], []
    for r in range(copies):
        s = rng.integers(0, X.shape[1], n); k = rng.integers(0, len(TRAIN_KINDS), n); l = rng.integers(1, 6, n)
        w = X[np.arange(n), s].astype(np.float32).copy()
        for ki, kind in enumerate(TRAIN_KINDS):
            for lv in range(1, 6):
                m = (k == ki) & (l == lv)
                if m.any(): w[m] = fault_ch(w[m], kind, LEVELS[kind][lv - 1], rng)
        S.append(s); K.append(k); L.append(l); W.append(w)
    return np.stack(S), np.stack(K), np.stack(L), np.stack(W)
