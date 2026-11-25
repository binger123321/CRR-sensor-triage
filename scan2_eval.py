"""Array-wide disturbances with and without signal repair (no retraining).

scan_eval.py showed that what a single faulty sensor cannot do, the same disturbance on every sensor does: spikes, drift, gaps or hum
on the whole array break every classifier, because no healthy sensor is left to carry the decision. Here the same conditions are
classified as they are (none) and after repair.py (repair: always-on de-spiking, drift removal, mains notches, gaps filled with noise at
the channel's own floor), for every model, with descriptors recomputed on the repaired records.
Conditions (five levels each, all four sensors): all_spike, all_drift, all_hum, all_gap, all_clip, late (empty start), early (cut end),
mixed (spikes + drift + hum together, levels 1-5), and clean (cost of repair on healthy data).
Writes outputs/paper/scan2_results.json.   usage: python -I scan2_eval.py [TAG ...]
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, torch
from evalcommon import *
from feats94 import feats94
from feats_ps import feats_ps
from faults2 import fault_ch, LEVELS
import repair as RP

TAGS = [t for t in sys.argv[1:] if not t.startswith("-")] or ["published", "MS2-SincNet", "F4cdN", "DSN", "SETN", "RST", "RSTF", "SETF"]
rng = np.random.default_rng(141421); n = len(Xp); L = Xp.shape[-1]
THR = RP.calibrate(Xt[ite])


def every(kinds, lv, X0=None):
    X = (Xp if X0 is None else X0).copy()
    for kind in kinds:
        for s in range(4): X[:, s] = fault_ch(X[:, s], kind, LEVELS[kind][lv], rng)
    return X


def all_gap(fr):
    X = Xp.copy()
    for i in range(n):
        nb = rng.integers(2, 6); ln = int(fr * L / nb)
        for st in rng.integers(0, L - ln, nb): X[i, :, st:st + ln] = 0
    return X


def late(fr):
    k = int(fr * L); X = np.zeros_like(Xp); X[..., k:] = Xp[..., :L - k]; return X


def early(fr):
    X = Xp.copy(); X[..., L - int(fr * L):] = 0; return X


def conds():
    yield "clean", "-", Xp
    for kind in ("spike", "drift", "hum", "clip"):
        for lv in range(5): yield f"all_{kind}", str(LEVELS[kind][lv]), every([kind], lv)
    for lv in range(5): yield "mixed", str(lv + 1), every(["drift", "hum", "spike"], lv)
    for fr in (0.1, 0.3, 0.5, 0.7, 0.9): yield "all_gap", str(fr), all_gap(fr)
    for fr in (0.1, 0.2, 0.3, 0.4, 0.5): yield "late", str(fr), late(fr)
    for fr in (0.1, 0.3, 0.5, 0.7, 0.9): yield "early", str(fr), early(fr)


nets = load_models([t for t in TAGS if t != "published"]); assert nets, "no checkpoints found"
pub = None
if "published" in TAGS and os.path.exists("resources/checkpoint.pth"):
    from network.model import SincNetModel
    from safe_io import load_weights
    pub = SincNetModel(4).to(dev); pub.load_state_dict(load_weights("resources/checkpoint.pth")); pub.eval()


@torch.no_grad()
def pub_acc(X): return float((np.concatenate([pub(torch.tensor(X[i:i + 64], device=dev)).float().cpu().numpy() for i in range(0, len(X), 64)]).argmax(1) == yp).mean())


R = {m: {} for m in (["published"] if pub is not None else []) + list(nets)}; INFO = {}
for name, lv, X in conds():
    X = np.ascontiguousarray(X, dtype=np.float32); Xr, act = RP.repair(X, THR); key = f"{name}:{lv}"
    den = np.sqrt((Xp ** 2).mean(-1)) + 1e-12; live = np.abs(Xr).max(-1) > 0
    INFO[key] = dict(touched=float((act > 0).mean()), removed=float(((act & 8) > 0).mean()),
                     err_before=float(np.median(np.sqrt(((X - Xp) ** 2).mean(-1)) / den)), err_after=float(np.median((np.sqrt(((Xr - Xp) ** 2).mean(-1)) / den)[live])))
    for arm, x in (("none", X), ("repair", Xr)):
        F94, FPS = feats94(x), feats_ps(x)
        if pub is not None: R["published"].setdefault(key, {})[arm] = pub_acc(x)
        for m, (ms, a) in nets.items():
            P = np.mean([predict(mm, x, feats_for(a, x, F94, FPS)) for mm in ms], 0); pr = P.argmax(1)
            R[m].setdefault(key, {})[arm] = float((pr == yp).mean()); R[m][key][arm + "_recall"] = [float((pr[yp == c] == c).mean()) for c in range(3)]
    print(name, lv, json.dumps({k: round(v, 3) for k, v in INFO[key].items()}), {m: (round(R[m][key]["none"] * 100, 1), round(R[m][key]["repair"] * 100, 1)) for m in R}, flush=True)
R["_info"] = INFO
json.dump(R, open("outputs/paper/scan2_results.json", "w"), indent=1); print("saved outputs/paper/scan2_results.json")
