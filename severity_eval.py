"""Accuracy and fault localisation against fault severity (no retraining).

For each of the seven fault types of faults2.py and each of five severity levels, one random sensor of every independent
event gets the fault (fixed seed, identical inputs for all models; descriptors recomputed on the faulty records).
Reported per model: ensemble accuracy. For ReSeT runs also: how often the sensor with the largest cross-sensor-prediction
residual, and the sensor with the lowest attention weight, is the faulty one.
Fault types gap, hum and gain are never used in training by any model.
Writes outputs/paper/severity_results.json.   usage: python -I severity_eval.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, torch
from evalcommon import *
from feats94 import feats94
from feats_ps import feats_ps
from faults2 import KINDS, LEVELS, fault_event

@torch.no_grad()
def run(ms, a, X, F, bs=64):
    """Ensemble probabilities; mean residual and attention weight per sensor when the model provides them."""
    P, E, W = [], [], []
    for m in ms:
        p_, e_, w_ = [], [], []
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs], device=dev); fb = torch.tensor(F[i:i + bs], device=dev)
            if a.get("arch") == "reset":
                z, aux = m(xb, fb, aux=True); w_.append(aux["w"].float().cpu().numpy())
                if aux["e"] is not None: e_.append(aux["e"].float().cpu().numpy())
            else: z = m(xb, fb)
            p_.append(z.float().softmax(1).cpu().numpy())
        P.append(np.concatenate(p_))
        if e_: E.append(np.concatenate(e_))
        if w_: W.append(np.concatenate(w_))
    return np.mean(P, 0), (np.mean(E, 0) if E else None), (np.mean(W, 0) if W else None)

nets = load_models()
R = {n: {} for n in nets}
rng = np.random.default_rng(271828)
for kind in KINDS:
    for lv in range(1, 6):
        X, s = fault_event(Xp, kind, lv, rng); F94, FPS = feats94(X), feats_ps(X)
        for n, (ms, a) in nets.items():
            P, E, W = run(ms, a, X, feats_for(a, X, F94, FPS))
            r = dict(acc=float((P.argmax(1) == yp).mean()))
            if E is not None: r["csp_top1"] = float((E.argmax(1) == s).mean())
            if W is not None: r["att_top1"] = float((W.argmin(1) == s).mean()); r["att_faulty"] = float(W[np.arange(len(s)), s].mean())
            R[n][f"{kind}:{lv}"] = r
        print(kind, lv, LEVELS[kind][lv - 1], {n: round(R[n][f"{kind}:{lv}"]["acc"] * 100, 1) for n in R}, flush=True)
json.dump(dict(levels={k: [str(v) for v in LEVELS[k]] for k in KINDS}, results=R), open("outputs/paper/severity_results.json", "w"), indent=1)
print("saved outputs/paper/severity_results.json")
