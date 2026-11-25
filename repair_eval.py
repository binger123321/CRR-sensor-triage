"""Detect - repair - remove: does treating a faulty sensor raise the classification accuracy? (no retraining of the classifiers)

Arms, on identical faulty records of the 578 independent events:
  none            the record is classified as it is
  remove          the fault head (quality checks + model scores, trained on synthetic faults of the validation split) names faulty
                  sensors; they are removed from the set
  repair          type-specific signal repair (repair.py: de-spiking, drift removal, mains notches; a channel that is mostly gaps is
                  removed), triggered by its own detectors; descriptors recomputed
  repair+remove   repair first, then the fault head removes what still looks faulty
  oracle          the faulty sensor and the fault type are known: repaired if repairable (spike, drift, hum, gaps), removed if nothing can be
                  recovered (noise only), kept otherwise (clipping, gain)
Fixed-channel models have no exact removal, so only none and repair are reported for them.
Conditions: clean; each of the seven fault types at five levels on one random sensor; "scarce": two sensors dead and a fault (level 3) on
one of the two left; "double": two different faults (level 3) on two sensors.  Writes outputs/paper/repair_results.json.
usage: python -I repair_eval.py [TAG ...]   (default: SETN RST RSTF SETF F4cdN MS2-SincNet)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, torch
from sklearn.ensemble import HistGradientBoostingClassifier
from diaglib import *
from feats94 import feats94
from feats_ps import feats_ps
from faults2 import KINDS, TRAIN_KINDS, LEVELS, fault_event, fault_ch
import repair as RP

TAGS = [t for t in sys.argv[1:] if not t.startswith("-")] or ["SETN", "RST", "RSTF", "SETF", "F4cdN", "MS2-SincNet"]
REPAIRABLE, LOST = ("spike", "drift", "hum", "gap"), ("noisy", "zero")
rng = np.random.default_rng(4242)
Xv = Xt[ite]; ntr = 200 if not Q else min(40, len(Xv) // 2)
THR = RP.calibrate(Xv)


@torch.no_grad()
def probs_keep(ms, a, X, F, keep, bs=96):
    """Ensemble probabilities of a set model with an explicit keep mask (n, 4) on top of the data-derived alive mask."""
    reset = a.get("arch") == "reset"; P = []
    for m in ms:
        p_ = []
        for i in range(0, len(X), bs):
            x = torch.tensor(X[i:i + bs], device=dev); f = torch.tensor(F[i:i + bs], device=dev)
            alive = (x.abs().amax(-1) > 0) & torch.tensor(keep[i:i + bs], device=dev)
            p_.append(fuse(m, content(m, x, f, x.abs().amax(-1) > 0), alive, reset)[0].cpu().numpy())
        P.append(np.concatenate(p_))
    return np.mean(P, 0)


def conditions():
    """(name, X, faulty: list of (sensor index array, kind) )"""
    n = len(Xp); idx = np.arange(n)
    yield "clean", Xp.copy(), []
    for k in KINDS:
        for lv in range(1, 6):
            X, s = fault_event(Xp, k, lv, rng); yield f"{k}:{lv}", X, [(s, k)]
    for k in ("clip", "spike", "drift", "noisy", "gap", "hum"):                  # two sensors dead, a fault on one of the two left
        perm = np.stack([rng.permutation(4) for _ in range(n)]); X = Xp.copy(); X[idx, perm[:, 0]] = 0; X[idx, perm[:, 1]] = 0
        s = perm[:, 2]; X[idx, s] = fault_ch(X[idx, s], k, LEVELS[k][2], rng); yield f"scarce:{k}", X, [(s, k)]
    for k1, k2 in (("drift", "spike"), ("hum", "noisy"), ("clip", "gap"), ("spike", "hum"), ("drift", "noisy")):
        perm = np.stack([rng.permutation(4) for _ in range(n)]); X = Xp.copy(); s1, s2 = perm[:, 0], perm[:, 1]
        X[idx, s1] = fault_ch(X[idx, s1], k1, LEVELS[k1][2], rng); X[idx, s2] = fault_ch(X[idx, s2], k2, LEVELS[k2][2], rng)
        yield f"double:{k1}+{k2}", X, [(s1, k1), (s2, k2)]


def oracle(X, faulty):
    """Repair what is repairable on the known faulty sensors, mark for removal what is lost."""
    X = X.copy(); keep = np.ones(X.shape[:2], bool); idx = np.arange(len(X))
    for s, k in faulty:
        if k in REPAIRABLE:
            only = np.zeros(X.shape[:2], bool); only[idx, s] = True; X, _ = RP.repair(X, np.full(3, -1e9), only=only)
        elif k in LOST: keep[idx, s] = False
    return X, keep


nets = load_models(TAGS); assert nets, "no checkpoints found"
setn = {n: v for n, v in nets.items() if v[1].get("arch") in ("set", "reset") and not v[1].get("impute")}
# ---- fault heads (one per set-model family), trained on synthetic faults of the validation split, raw and repaired
TRAIN = [("clean", Xv[:ntr], np.full(ntr, -1))] + [(k,) + fault_event(Xv[:ntr], k, l, rng) for k in TRAIN_KINDS for l in range(1, 6)]
TRAIN += [(k, RP.repair(X, THR)[0], s) for k, X, s in TRAIN[1:]]               # the head also sees what repaired records look like
CAL = Xv[ntr:]; prep = lambda X: (qc(X), feats_ps(X))
TRq = [prep(c[1]) for c in TRAIN]; CAq = prep(CAL); print("fault-head training data ready", flush=True)
HEAD = {}
for name, (ms, a) in setn.items():
    Ftr, Ltr = [], []
    for (kind, X, s), (q, fps) in zip(TRAIN, TRq):
        S, _, _ = scores(ms, a, X, feats_for(a, X, FPS=fps)); f_ = feats(q, S); lab = np.zeros((len(X), 4), int)
        if kind != "clean":
            still = np.abs(X[np.arange(len(X)), s] - Xv[:ntr][np.arange(len(X)), s]).max(-1) > 0.05 * np.abs(Xv[:ntr][np.arange(len(X)), s]).max(-1)
            lab[np.arange(len(X))[still], s[still]] = 1                           # a sensor repaired back to the truth is no longer faulty
        Ftr.append(f_.reshape(-1, f_.shape[-1])); Ltr.append(lab.reshape(-1))
    h = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=4, class_weight="balanced", random_state=0).fit(np.concatenate(Ftr), np.concatenate(Ltr))
    Sc, _, _ = scores(ms, a, CAL, feats_for(a, CAL, FPS=CAq[1])); Fc = feats(CAq[0], Sc)
    pc = h.predict_proba(Fc.reshape(-1, Fc.shape[-1]))[:, 1].reshape(-1, 4)
    HEAD[name] = (h, float(np.quantile(pc, 0.99)))                                # per-sensor threshold: 1 % of clean sensors flagged


def flagged(name, ms, a, X, fps):
    h, thr = HEAD[name]; S, _, _ = scores(ms, a, X, feats_for(a, X, FPS=fps)); f_ = feats(qc(X), S)
    pr = h.predict_proba(f_.reshape(-1, f_.shape[-1]))[:, 1].reshape(-1, 4); fl = pr > thr
    fl[fl.sum(1) >= (np.abs(X).max(-1) > 0).sum(1)] = False                       # never remove every live sensor
    return fl


acc = lambda P: float((P.argmax(1) == yp).mean())
REALGAP = (RP.detectors(Xp)[..., 3] > 0.05).any(1); print("independent events with a zero-filled stretch (> 5 % of a channel):", int(REALGAP.sum()), flush=True)
R = {n: {} for n in nets}
for cname, X, faulty in conditions():
    Xr, act = RP.repair(X, THR); Xo, keep_o = oracle(X, faulty)
    D = {k: dict(fps=feats_ps(x), f94=feats94(x)) for k, x in (("raw", X), ("rep", Xr), ("ora", Xo))}
    ones = np.ones(X.shape[:2], bool); info = dict(repaired=float((act > 0).any(1).mean()))
    if faulty:
        idx = np.arange(len(X)); info["repaired_faulty"] = float(np.mean([(act[idx, s] > 0).mean() for s, _ in faulty]))
    for n, (ms, a) in nets.items():
        if n in setn:
            F0, F1, F2 = (feats_for(a, x, FPS=D[k]["fps"]) for k, x in (("raw", X), ("rep", Xr), ("ora", Xo)))
            fl0 = flagged(n, ms, a, X, D["raw"]["fps"]); fl1 = flagged(n, ms, a, Xr, D["rep"]["fps"])
            PA = dict(none=probs_keep(ms, a, X, F0, ones), remove=probs_keep(ms, a, X, F0, ~fl0), repair=probs_keep(ms, a, Xr, F1, ones),
                      repair_remove=probs_keep(ms, a, Xr, F1, ~fl1), oracle=probs_keep(ms, a, Xo, F2, keep_o))
            r = {k: acc(v) for k, v in PA.items()}; r.update(flag_rate=float(fl0.any(1).mean()), flag_rate_after_repair=float(fl1.any(1).mean()))
            if faulty: r["flag_hit"] = float(np.mean([fl0[np.arange(len(X)), s].mean() for s, _ in faulty]))
        else:
            PA = dict(none=np.mean([predict(m, X, feats_for(a, X, D["raw"]["f94"], D["raw"]["fps"])) for m in ms], 0),
                      repair=np.mean([predict(m, Xr, feats_for(a, Xr, D["rep"]["f94"], D["rep"]["fps"])) for m in ms], 0))
            r = {k: acc(v) for k, v in PA.items()}
        if cname == "clean":                                    # the records that already contain zero-filled stretches in the public data
            r["real_gap_n"] = int(REALGAP.sum())
            for k, v in PA.items(): r["real_gap_" + k] = float((v.argmax(1) == yp)[REALGAP].mean()) if REALGAP.any() else float("nan")
            r["errors_" + "none"] = int((PA["none"].argmax(1) != yp).sum()); r["errors_repair"] = int((PA["repair"].argmax(1) != yp).sum())
        R[n][cname] = r
    R.setdefault("_info", {})[cname] = info
    print(cname, json.dumps(info), {n: {k: round(v * 100, 1) for k, v in R[n][cname].items() if k in ("none", "remove", "repair", "repair_remove", "oracle")} for n in nets}, flush=True)
json.dump(R, open("outputs/paper/repair_results.json", "w"), indent=1); print("saved outputs/paper/repair_results.json")
