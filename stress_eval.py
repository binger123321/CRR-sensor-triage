"""Compound stress test and an imputation baseline (no retraining). Conditions were fixed before any model was evaluated on them.

  dead+noise10   one dead sensor (each of 4) + white noise at 10 dB on the other three
  dead+noise5    one dead sensor + white noise at 5 dB on the others
  dead+noisy     sensor i dead and sensor i+1 replaced by noise of its own RMS (4 pairs)
  noisy+noise10  sensor i replaced by noise + white noise at 10 dB on the others
  spike+drift    spikes on sensor i and low-frequency drift on sensor i+1 (4 pairs)
  dead:copy      (fixed-channel models only) the dead channel is replaced by a copy of the best live sensor, the simplest imputation
Descriptors are recomputed on every corrupted record. Writes outputs/paper/stress_results.json.  usage: python -I stress_eval.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np
from evalcommon import *
from feats94 import feats94
from feats_ps import feats_ps
from make_faults import fault

rng = np.random.default_rng(31415)
def add_noise(X, snr, skip=()):
    n = rng.standard_normal(X.shape).astype(np.float32) * np.sqrt((X ** 2).mean(-1, keepdims=True)) * 10 ** (-snr / 20)
    for s in skip: n[:, s] = 0
    return X + n

def conds():
    for i in range(4):
        j = (i + 1) % 4
        X = add_noise(Xp, 10); X[:, i] = 0; yield "dead+noise10", X
        X = add_noise(Xp, 5); X[:, i] = 0; yield "dead+noise5", X
        X = fault(Xp, j, "noise", rng); X[:, i] = 0; yield "dead+noisy", X
        X = add_noise(fault(Xp, i, "noise", rng), 10, skip=(i,)); yield "noisy+noise10", X
        X = fault(fault(Xp, i, "spike", rng), j, "drift", rng); yield "spike+drift", X
        X = Xp.copy(); X[:, i] = Xp[:, 0 if i else 1]; yield "dead:copy", X

nets = load_models()
R = {n: {} for n in nets}
for name, X in conds():
    F94, FPS = feats94(X), feats_ps(X)
    for n, (ms, a) in nets.items():
        if name == "dead:copy" and a.get("arch") in SETS: continue
        R[n].setdefault(name, []).append(ens_acc(ms, X, feats_for(a, X, F94, FPS)))
    print(name, {n: round(R[n][name][-1] * 100, 1) for n in R if name in R[n]}, flush=True)
S = {n: {k: dict(mean=float(np.mean(v)), worst=float(np.min(v))) for k, v in R[n].items()} for n in R}
json.dump(S, open("outputs/paper/stress_results.json", "w"), indent=1); print("saved outputs/paper/stress_results.json")
