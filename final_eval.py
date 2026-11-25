"""Three closing tests of sensor triage (no retraining).

1. mixture   Field-like mixture: every sensor of every independent event is, independently, healthy (55 %), dead (10 %), or carries
             spikes (10 %), drift (10 %), hum (5 %), gaps (5 %) or clipping (5 %) at a random severity (at least one sensor stays live).
             Three random realisations. Arms: as recorded (dead channels are zeros) and after triage (repair.py; the set models then
             remove what is dead). All models.
2. variants  Fault variants that were NOT considered when the tests and repairs were designed, on all four sensors:
             small spikes (1.5 x peak), huge spikes (10 x peak), two-sample spikes, baseline steps, a linear trend, a random-walk
             baseline, drift at 3 Hz, 60 Hz mains (the notches are at 50/100/150 Hz), many short dropouts. As recorded / repaired.
3. timing    Cost per event of the check-and-repair step, of the descriptors, and of the classifier.
Writes outputs/paper/final_results.json.   usage: python -I final_eval.py [TAG ...]
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, time, numpy as np, torch
from evalcommon import *
from feats94 import feats94
from feats_ps import feats_ps
from faults2 import fault_ch, LEVELS
import repair as RP

TAGS = [t for t in sys.argv[1:] if not t.startswith("-")] or ["published", "MS2-SincNet", "F4cdN", "DSN", "SETN", "SETF", "RST", "RSTF"]
n, L = len(Xp), Xp.shape[-1]; tt = np.arange(L) / 1e4; THR = RP.calibrate(Xt[ite])
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


def both(X):
    Xr, act = RP.repair(X, THR); return dict(recorded=accs(X), triage=accs(Xr), touched=float((act > 0).mean()), removed=float(((act & 8) > 0).mean()))


R = {}
# ------------------------------------------------------------------ 1 mixture
KIND = ["healthy", "dead", "spike", "drift", "hum", "gap", "clip"]; PROB = [0.55, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05]
R["mixture"] = []
for rep in range(3):
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
    r = both(X); r["share"] = {kd: float((st == k).mean()) for k, kd in enumerate(KIND)}; r["events_all_healthy"] = float((st == 0).all(1).mean())
    R["mixture"].append(r); print("mixture", rep, {m: (round(r["recorded"][m] * 100, 1), round(r["triage"][m] * 100, 1)) for m in NAMES}, flush=True)

# ------------------------------------------------------------------ 2 variants never used in design
def spikes(x, rng, amp, k=20, width=1):
    x = x.copy(); pk = np.abs(x).max(-1)
    for i in range(len(x)):
        j = rng.choice(L - width, k, replace=False); sg = rng.choice([-1.0, 1.0], k)
        for w in range(width): x[i, j + w] += amp * pk[i] * sg
    return x


def variant(name, rng):
    X = Xp.copy()
    for s in range(4):
        x = Xp[:, s]; pk = np.abs(x).max(-1, keepdims=True); rms = np.sqrt((x ** 2).mean(-1, keepdims=True))
        if name == "spikes 1.5 x peak": y = spikes(x, rng, 1.5)
        elif name == "spikes 10 x peak": y = spikes(x, rng, 10)
        elif name == "two-sample spikes": y = spikes(x, rng, 3, width=2)
        elif name == "baseline steps":
            y = x.copy()
            for i in range(n):
                for j in np.sort(rng.integers(500, L - 500, 3)): y[i, j:] += rng.choice([-1, 1]) * pk[i, 0]
        elif name == "linear trend": y = x + 3 * pk * (tt[None] - 0.5) * rng.choice([-1, 1], (n, 1))
        elif name == "random-walk baseline":
            w = np.cumsum(rng.standard_normal((n, L)), -1); y = x + 2 * pk * w / np.abs(w).max(-1, keepdims=True)
        elif name == "drift at 3 Hz": y = x + pk * np.sin(2 * np.pi * 3 * tt[None] + rng.uniform(0, 6.28, (n, 1)))
        elif name == "60 Hz mains": y = x + 2 * rms * (np.sin(2 * np.pi * 60 * tt[None] + rng.uniform(0, 6.28, (n, 1))) + 0.5 * np.sin(2 * np.pi * 180 * tt[None]))
        elif name == "30 short dropouts":
            y = x.copy()
            for i in range(n):
                for j in rng.integers(0, L - 60, 30): y[i, j:j + 60] = 0
        X[:, s] = y
    return X.astype(np.float32)


R["variants"] = {}
for name in ("spikes 1.5 x peak", "spikes 10 x peak", "two-sample spikes", "baseline steps", "linear trend", "random-walk baseline", "drift at 3 Hz", "60 Hz mains", "30 short dropouts"):
    r = both(variant(name, np.random.default_rng(abs(hash(name)) % 2 ** 31 if False else sum(map(ord, name))))); R["variants"][name] = r
    print("variant", name, {m: (round(r["recorded"][m] * 100, 1), round(r["triage"][m] * 100, 1)) for m in NAMES}, flush=True)

# ------------------------------------------------------------------ 3 timing
Xs = Xp[:128].copy(); T = {}
t0 = time.perf_counter(); RP.repair(Xs, THR); T["check_repair_ms_per_event_healthy"] = (time.perf_counter() - t0) / len(Xs) * 1e3
Xf = Xs.copy()
for s in range(4): Xf[:, s] = fault_ch(fault_ch(Xs[:, s], "spike", 20, np.random.default_rng(1)), "drift", 2, np.random.default_rng(2))
t0 = time.perf_counter(); RP.repair(Xf, THR); T["check_repair_ms_per_event_all_faulty"] = (time.perf_counter() - t0) / len(Xs) * 1e3
t0 = time.perf_counter(); FPS = feats_ps(Xs); T["descriptors_per_sensor_ms_per_event"] = (time.perf_counter() - t0) / len(Xs) * 1e3
t0 = time.perf_counter(); F94 = feats94(Xs); T["descriptors_array_ms_per_event"] = (time.perf_counter() - t0) / len(Xs) * 1e3
for m, (ms, a) in nets.items():
    F = feats_for(a, Xs, F94, FPS); predict(ms[0], Xs[:32], F[:32])
    if dev.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter(); predict(ms[0], Xs, F)
    if dev.type == "cuda": torch.cuda.synchronize()
    T[f"classifier_ms_per_event:{m}"] = (time.perf_counter() - t0) / len(Xs) * 1e3; T[f"parameters:{m}"] = int(sum(p.numel() for p in ms[0].parameters()))
T["device"] = str(dev) + (" " + torch.cuda.get_device_name(0) if dev.type == "cuda" else ""); R["timing"] = T; print("timing", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in T.items()}), flush=True)
json.dump(R, open("outputs/paper/final_results.json", "w"), indent=1); print("saved outputs/paper/final_results.json")
