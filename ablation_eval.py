"""Ablation of sensor triage (no retraining).

Conditions: clean records; the field-like mixture of final_eval.py (three realisations); and, on all four sensors, 20 spikes,
drift of 2 x peak, hum of 5 x RMS, half the record missing.
Arms:  recorded | triage (thresholds at the 99.5th percentile of clean validation channels) | thresholds at the 99th / 99.9th
       percentile | triage without one step (spike, drift, hum, gap filling, removal) |
       always-on conditioning: 5-point median filter, 5 Hz high-pass and 50/100/150 Hz notches on every live channel, no checks.
Writes outputs/paper/ablation_results.json.   usage: python -I ablation_eval.py [TAG ...]
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, torch
from scipy.signal import medfilt, sosfiltfilt
from evalcommon import *
from feats94 import feats94
from feats_ps import feats_ps
from faults2 import fault_ch, LEVELS
import repair as RP

TAGS = [t for t in sys.argv[1:] if not t.startswith("-")] or ["published", "MS2-SincNet", "F4cdN", "SETN", "SETF"]
n = len(Xp); THR = {q: RP.calibrate(Xt[ite], q=q) for q in (0.99, 0.995, 0.999)}
nets = load_models([t for t in TAGS if t != "published"]); pub = None
if "published" in TAGS and os.path.exists("resources/checkpoint.pth"):
    from network.model import SincNetModel
    from safe_io import load_weights
    pub = SincNetModel(4).to(dev); pub.load_state_dict(load_weights("resources/checkpoint.pth")); pub.eval()
NAMES = (["published"] if pub is not None else []) + list(nets); assert NAMES, "no models"


@torch.no_grad()
def accs(X):
    X = np.ascontiguousarray(X, dtype=np.float32); F94, FPS = feats94(X), feats_ps(X); out = {}
    if pub is not None: out["published"] = float((np.concatenate([pub(torch.tensor(X[i:i + 64], device=dev)).float().cpu().numpy() for i in range(0, n, 64)]).argmax(1) == yp).mean())
    for m, (ms, a) in nets.items(): out[m] = ens_acc(ms, X, feats_for(a, X, F94, FPS))
    return out


def always_on(X):
    X = X.copy(); live = np.abs(X).max(-1) > 0; x = X[live]
    x = medfilt(x, [1, 5]).astype(np.float64); x = sosfiltfilt(RP._HP, x, axis=-1, padtype="even", padlen=3000)
    for sos in RP._NOTCH: x = sosfiltfilt(sos, x, axis=-1)
    X[live] = x.astype(np.float32); return X


def mixture(rep):
    KIND = ["healthy", "dead", "spike", "drift", "hum", "gap", "clip"]; PROB = [0.55, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05]
    rng = np.random.default_rng(1000 + rep); st = rng.choice(len(KIND), (n, 4), p=PROB)
    for i in np.flatnonzero((st == 1).all(1)): st[i, rng.integers(4)] = 0
    X = Xp.copy()
    for k, kind in enumerate(KIND):
        ii, ss = np.nonzero(st == k)
        if kind == "healthy" or len(ii) == 0: continue
        if kind == "dead": X[ii, ss] = 0; continue
        lv = rng.integers(0, 5, len(ii))
        for l in range(5):
            m = lv == l
            if m.any(): X[ii[m], ss[m]] = fault_ch(Xp[ii[m], ss[m]], kind, LEVELS[kind][l], rng)
    return X


def allsens(kind, v, seed):
    rng = np.random.default_rng(seed); X = Xp.copy()
    for s in range(4): X[:, s] = fault_ch(Xp[:, s], kind, v, rng)
    return X


CONDS = {"clean": lambda: Xp.copy(), "mixture:0": lambda: mixture(0), "mixture:1": lambda: mixture(1), "mixture:2": lambda: mixture(2),
         "all_spike:20": lambda: allsens("spike", 20, 1), "all_drift:2": lambda: allsens("drift", 2, 2), "all_hum:5": lambda: allsens("hum", 5, 3),
         "all_gap:0.5": lambda: allsens("gap", 0.5, 4)}
if os.environ.get("EVAL_QUICK"): CONDS = {k: CONDS[k] for k in ("clean", "mixture:0", "all_spike:20")}
ARMS = {"recorded": lambda X: X, "triage": lambda X: RP.repair(X, THR[0.995])[0], "thr_q99": lambda X: RP.repair(X, THR[0.99])[0], "thr_q99.9": lambda X: RP.repair(X, THR[0.999])[0],
        "always_on": always_on}
for st in ("spike", "drift", "hum", "gap", "remove"): ARMS["no_" + st] = (lambda X, st=st: RP.repair(X, THR[0.995], skip=(st,))[0])
R = {}
for cn, make in CONDS.items():
    X = make(); R[cn] = {}
    for an, f in ARMS.items(): R[cn][an] = accs(f(X))
    print(cn, {an: {m: round(v * 100, 1) for m, v in r.items()} for an, r in R[cn].items()}, flush=True)
    json.dump(R, open("outputs/paper/ablation_results.json", "w"), indent=1)
print("saved outputs/paper/ablation_results.json")
