"""Sensitivity scan: which disturbances actually hurt the classifiers? (no retraining)

Single-sensor faults turned out to be nearly harmless (severity_eval.py). This scan looks for what is not, with disturbances that
hit the whole array or the structure of the record, each at five levels, identical inputs for all models, descriptors recomputed:
  all_clip    every sensor clipped at {50, 30, 20, 10, 5} % of its peak                 (a strong event saturating the array)
  all_spike   {5, 10, 20, 50, 100} spikes on every sensor                              (site-wide electrical interference)
  all_drift   drift of {0.5, 1, 2, 5, 10} x peak on every sensor
  all_hum     mains hum of {0.2, 0.5, 1, 2, 5} x RMS on every sensor
  all_gap     the same {10, 30, 50, 70, 90} % of the record missing on every sensor     (acquisition outage)
  white       white noise at {0, -5, -10, -15, -20} dB on every sensor
  minenoise   a real noise-class record (training set) added at {20, 10, 5, 0, -5} dB   (machinery running during the event)
  overlap     a second real event of another class added at {20, 10, 5, 0, -5} dB below / at the level of the first
  late        record shifted so that the first {10, 20, 30, 40, 50} % is empty and the end is cut off (late trigger)
  early       the last {10, 30, 50, 70, 90} % of the record cut off (zeros)              (truncated record)
  clock       time axis stretched by {-20, -10, -5, +10, +20} %                           (sampling-rate error)
  shuffle     sensor order: reversed, rotated by 1, 2, 3, random                         (SNR ranking wrong)
  one_good    only the weakest sensor is clean; the other three get white noise at {10, 5, 0, -5, -10} dB
Reported: ensemble accuracy and per-class recall.  Writes outputs/paper/scan_results.json.   usage: python -I scan_eval.py [TAG ...]
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, torch
from evalcommon import *
from feats94 import feats94
from feats_ps import feats_ps
from faults2 import fault_ch, LEVELS

TAGS = [t for t in sys.argv[1:] if not t.startswith("-")] or ["MS2-SincNet", "F4cdN", "DSN", "SETN", "RST", "RSTF", "SETF"]
rng = np.random.default_rng(57721); n = len(Xp); L = Xp.shape[-1]; idx = np.arange(n)
rms = lambda x: np.sqrt((x ** 2).mean(-1, keepdims=True)) + 1e-12
NOISE_POOL = Xt[itr][yt[itr] == 2]                                     # real noise-class records of the training split
POOL = {c: Xt[itr][yt[itr] == c] for c in range(3)}


def every(kind, lv):
    X = Xp.copy()
    for s in range(4): X[:, s] = fault_ch(Xp[:, s], kind, LEVELS[kind][lv], rng)
    return X


def all_gap(fr):
    X = Xp.copy()
    for i in range(n):
        nb = rng.integers(2, 6); ln = int(fr * L / nb)
        for st in rng.integers(0, L - ln, nb): X[i, :, st:st + ln] = 0
    return X


def add(other, snr_db):
    """Add another 4-sensor record, sensor by sensor, at snr_db relative to each channel's RMS."""
    return (Xp + other / rms(other) * rms(Xp) * 10 ** (-snr_db / 20)).astype(np.float32)


def overlap(snr_db):
    cls = [rng.choice([c for c in range(3) if c != yp[i]]) for i in range(n)]
    other = np.stack([POOL[c][rng.integers(0, len(POOL[c]))] for c in cls])
    return add(other, snr_db)


def late(fr):
    k = int(fr * L); X = np.zeros_like(Xp); X[..., k:] = Xp[..., :L - k]; return X


def early(fr):
    X = Xp.copy(); X[..., L - int(fr * L):] = 0; return X


def clock(pc):
    t = np.arange(L) * (1 + pc / 100.0); X = np.zeros_like(Xp)
    for i in range(n):
        for s in range(4): X[i, s] = np.interp(t, np.arange(L), Xp[i, s], right=0.0)
    return X


def one_good(snr_db):
    X = Xp.copy(); w = rng.standard_normal(X[:, :3].shape).astype(np.float32); X[:, :3] += w * rms(Xp[:, :3]) * 10 ** (-snr_db / 20); return X


def conds():
    yield "clean", "-", Xp
    for kind in ("clip", "spike", "drift", "hum"):
        for lv in range(5): yield f"all_{kind}", str(LEVELS[kind][lv]), every(kind, lv)
    for fr in (0.1, 0.3, 0.5, 0.7, 0.9): yield "all_gap", str(fr), all_gap(fr)
    for db in (0, -5, -10, -15, -20): yield "white", str(db), (Xp + rng.standard_normal(Xp.shape).astype(np.float32) * rms(Xp) * 10 ** (-db / 20)).astype(np.float32)
    for db in (20, 10, 5, 0, -5): yield "minenoise", str(db), add(NOISE_POOL[rng.integers(0, len(NOISE_POOL), n)], db)
    for db in (20, 10, 5, 0, -5): yield "overlap", str(db), overlap(db)
    for fr in (0.1, 0.2, 0.3, 0.4, 0.5): yield "late", str(fr), late(fr)
    for fr in (0.1, 0.3, 0.5, 0.7, 0.9): yield "early", str(fr), early(fr)
    for pc in (-20, -10, -5, 10, 20): yield "clock", str(pc), clock(pc)
    for nm, order in (("reversed", [3, 2, 1, 0]), ("rot1", [1, 2, 3, 0]), ("rot2", [2, 3, 0, 1]), ("rot3", [3, 0, 1, 2])): yield "shuffle", nm, Xp[:, order]
    yield "shuffle", "random", np.stack([Xp[i, rng.permutation(4)] for i in range(n)])
    for db in (10, 5, 0, -5, -10): yield "one_good", str(db), one_good(db)


nets = load_models(TAGS); assert nets, "no checkpoints found"
R = {m: {} for m in nets}
for name, lv, X in conds():
    X = np.ascontiguousarray(X, dtype=np.float32); F94, FPS = feats94(X), feats_ps(X)
    for m, (ms, a) in nets.items():
        P = np.mean([predict(mm, X, feats_for(a, X, F94, FPS)) for mm in ms], 0); pr = P.argmax(1)
        R[m][f"{name}:{lv}"] = dict(acc=float((pr == yp).mean()), recall=[float((pr[yp == c] == c).mean()) for c in range(3)],
                                    pred_share=[float((pr == c).mean()) for c in range(3)])
    print(name, lv, {m: round(R[m][f"{name}:{lv}"]["acc"] * 100, 1) for m in R}, flush=True)
json.dump(R, open("outputs/paper/scan_results.json", "w"), indent=1); print("saved outputs/paper/scan_results.json")
