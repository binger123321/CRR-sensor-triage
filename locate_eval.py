"""Which sensor is faulty? Localisation of a faulty sensor against fault type and severity (no fault labels from the field).

Per sensor of every event we compute scores that need no fault labels:
  qc         five signal-quality checks (flat, clipping, spikes, drift, envelope incoherent with the other sensors)
  disagree   JS divergence between the class probabilities from this sensor alone and from the other sensors
  outlier    distance of the sensor's token from the mean token of the others
  lowatt     minus the attention-pooling weight of the sensor
  csp        cross-sensor-prediction residual (ReSeT only)
Each score is also expressed relative to the other sensors of the same event. A small classifier (the "fault head") is then
trained to name the faulty sensor from these scores, using SYNTHETIC faults (faults2.TRAIN_KINDS, random level) applied to 200 of
the 300 validation events; the other 100 clean validation events set the alarm threshold (5 % false alarms per event).
Test: the 578 independent events with each of the seven fault types at five levels on one random sensor per event, and clean.
The types gap, hum and gain are never seen by the networks or by the fault head.
Reported per model family and score set: top-1 localisation, detection rate, detect-and-localise rate, clean false alarms, and the
classification accuracy with the flagged sensor removed.  Writes outputs/paper/locate_results.json.
usage: python -I locate_eval.py [TAG ...]   (default: SETN RST RSTF SETF)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, torch, torch.nn.functional as Fn
from scipy.signal import medfilt, hilbert
from sklearn.ensemble import HistGradientBoostingClassifier
from evalcommon import *
from feats_ps import feats_ps
from faults2 import KINDS, TRAIN_KINDS, LEVELS, fault_event

TAGS = sys.argv[1:] or ["SETN", "RST", "RSTF", "SETF"]


def qc(X):
    """(n, 4, 5) model-free quality scores; larger = more suspicious."""
    n = len(X); out = np.zeros((n, 4, 5), np.float32); ax = np.abs(X); mx = ax.max(-1)
    out[..., 0] = (mx == 0)
    out[..., 1] = (ax >= 0.995 * mx[..., None]).mean(-1) * (mx > 0)
    r = X - medfilt(X.reshape(-1, X.shape[-1]), [1, 5]).reshape(X.shape); p995 = np.quantile(ax, 0.995, axis=-1, keepdims=True) + 1e-12
    out[..., 2] = np.log1p(((np.abs(r) > 0.6 * ax) & (ax > 2.5 * p995)).sum(-1))
    P = np.abs(np.fft.rfft(X, axis=-1)) ** 2; f = np.fft.rfftfreq(X.shape[-1], 1e-4)
    out[..., 3] = np.log10(P[..., f < 3].sum(-1) / (P.sum(-1) + 1e-20) + 1e-12)
    env = np.log(np.abs(hilbert(X, axis=-1)).reshape(n, 4, 1000, 10).mean(-1) + 1e-12)
    env = (env - env.mean(-1, keepdims=True)) / (env.std(-1, keepdims=True) + 1e-9); E = np.fft.rfft(env, 2048, axis=-1); L = 300
    C = np.zeros((n, 4, 4), np.float32)
    for i in range(4):
        for j in range(i + 1, 4):
            cc = np.fft.irfft(E[:, i] * np.conj(E[:, j]), 2048, axis=-1) / 1000
            C[:, i, j] = C[:, j, i] = np.concatenate([cc[:, :L], cc[:, -L:]], 1).max(1)
    out[..., 4] = -np.stack([np.median(np.delete(C[:, s], s, 1), 1) for s in range(4)], 1)
    out[mx == 0, 1:] = 0
    return out


def js(p, q):
    m_ = 0.5 * (p + q); kl = lambda a, b: (a * (torch.log(a + 1e-9) - torch.log(b + 1e-9))).sum(-1)
    return 0.5 * kl(p, m_) + 0.5 * kl(q, m_)


@torch.no_grad()
def content(m, x, f, alive):
    B, S, _ = x.shape; emb = torch.zeros(B * S, 256, device=dev); fl = alive.reshape(-1)
    if fl.any(): emb[fl] = m.encode(x.reshape(B * S, -1)[fl]).float()
    return m.tok(torch.cat([emb.reshape(B, S, 256), m.desc(f.float())], -1)).float()


@torch.no_grad()
def fuse(m, c, alive, reset):
    """Class probabilities, attention-pooling weights and (ReSeT) CSP residuals for a given alive mask, from pre-rank tokens c."""
    a = alive.clone(); a[a.sum(1) == 0, 0] = True; B, S, _ = c.shape; t = c + m.rank.weight[:S][None]; e = None
    if reset:
        bias = c.new_zeros(B, S); na = a.sum(1, keepdim=True)
        if m.use_csp:
            e = 1 - Fn.cosine_similarity(m.predict_tokens(c, a).float(), c, dim=-1)
            if m.use_gate:
                ed = e * a; z = ed / (ed.sum(1, keepdim=True) / na + 1e-6) - 1; bias = -Fn.softplus(m.gate_a) * z * a * (na > 1)
        key = bias.masked_fill(~a, float("-inf")); am = key[:, None, None, :].expand(B, m.heads, S, S).reshape(B * m.heads, S, S)
        u = m.fuse(t, mask=am); w = (m.pool(u).squeeze(-1) + key).softmax(1)
    else:
        u = m.fuse(t, src_key_padding_mask=~a); w = m.pool(u).squeeze(-1).masked_fill(~a, float("-inf")).softmax(1)
    z_ = m.classifer(torch.cat([(w[..., None] * u).sum(1), (u * a[..., None]).sum(1) / a.sum(1, keepdim=True)], 1))
    return z_.softmax(1), w, e


@torch.no_grad()
def scores(ms, a, X, F, drop=None, bs=96):
    """Model scores (n, 4, k) averaged over seeds [disagree, outlier, lowatt, (csp)], ensemble probabilities, and ensemble
    probabilities with sensor drop[j] removed (drop = -1 keeps all)."""
    reset = a.get("arch") == "reset"; n = len(X); SC, P, PD = [], [], []
    for m in ms:
        sc, p_, pd_ = [], [], []
        for i in range(0, n, bs):
            x = torch.tensor(X[i:i + bs], device=dev); f = torch.tensor(F[i:i + bs], device=dev); alive = x.abs().amax(-1) > 0
            c = content(m, x, f, alive); p, w, e = fuse(m, c, alive, reset); cols = []
            dis = torch.zeros_like(w); out = torch.zeros_like(w)
            for s in range(4):
                only = torch.zeros_like(alive); only[:, s] = alive[:, s]; wo = alive.clone(); wo[:, s] = False
                dis[:, s] = js(fuse(m, c, only, reset)[0], fuse(m, c, wo, reset)[0])
                o = wo.float()[..., None]; out[:, s] = (c[:, s] - (c * o).sum(1) / o.sum(1).clamp(min=1)).norm(dim=-1)
            cols = [dis, out, -w * alive.sum(1, keepdim=True)] + ([e] if e is not None else [])
            sc.append(torch.stack(cols, -1).cpu().numpy()); p_.append(p.cpu().numpy())
            if drop is not None:
                d = torch.tensor(drop[i:i + bs], device=dev); al2 = alive.clone(); hit = d >= 0; al2[hit, d[hit]] = False
                pd_.append(fuse(m, c, al2, reset)[0].cpu().numpy())
        SC.append(np.concatenate(sc)); P.append(np.concatenate(p_))
        if drop is not None: PD.append(np.concatenate(pd_))
    return np.mean(SC, 0), np.mean(P, 0), (np.mean(PD, 0) if PD else None)


def feats(Q, S):
    """Per-sensor feature rows from qc (n,4,5) and model scores (n,4,k): raw values and contrast with the other sensors of the event."""
    A = np.concatenate([Q, S], -1); other = (A.sum(1, keepdims=True) - A) / 3.0
    return np.concatenate([A, A - other], -1)                      # (n, 4, 2*(5+k))


def cols(nq, k, which):
    base = dict(qc=list(range(5)), model=list(range(5, 5 + k)), all=list(range(5 + k)))[which] if which in ("qc", "model", "all") else [5 + which]
    return base + [b + 5 + k for b in base]


nets = {n: v for n, v in load_models(TAGS).items() if v[1].get("arch") in ("set", "reset") and not v[1].get("impute")}
assert nets, "no set / ReSeT checkpoints found"
Xv, yv = Xt[ite], yt[ite]; ntr = 200 if not Q else min(40, len(Xv) // 2)
rng = np.random.default_rng(1618)
# ---- synthetic faults on the validation split (for the fault head) and on the independent set (test); shared by all models
TRAIN = [("clean", 0, Xv[:ntr], np.full(ntr, -1))] + [(k, l) + fault_event(Xv[:ntr], k, l, rng) for k in TRAIN_KINDS for l in range(1, 6)]
CAL = Xv[ntr:]
TEST = [("clean", 0, Xp, np.full(len(Xp), -1))] + [(k, l) + fault_event(Xp, k, l, rng) for k in KINDS for l in range(1, 6)]
prep = lambda X: (qc(X), feats_ps(X))
TRq = [prep(c[2]) for c in TRAIN]; CAq = prep(CAL); TEq = [prep(c[2]) for c in TEST]; print("faulty records and descriptors ready", flush=True)
R = {}
for name, (ms, a) in nets.items():
    k = 4 if (a.get("arch") == "reset" and not a.get("no_csp")) else 3
    snames = ["disagree", "outlier", "lowatt", "csp"][:k]
    ff = lambda X, FPS: feats_for(a, X, FPS=FPS)
    # fault head: training rows
    Ftr, Ltr = [], []
    for (kind, lv, X, s), (q, fps) in zip(TRAIN, TRq):
        S, _, _ = scores(ms, a, X, ff(X, fps)); f_ = feats(q, S); lab = np.zeros((len(X), 4), int)
        if kind != "clean": lab[np.arange(len(X)), s] = 1
        Ftr.append(f_.reshape(-1, f_.shape[-1])); Ltr.append(lab.reshape(-1))
    Ftr, Ltr = np.concatenate(Ftr), np.concatenate(Ltr)
    Sc, _, _ = scores(ms, a, CAL, ff(CAL, CAq[1])); Fc = feats(CAq[0], Sc)
    heads = {}
    for which in ("qc", "model", "all"):
        cc = cols(5, k, which); h = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=4, class_weight="balanced", random_state=0).fit(Ftr[:, cc], Ltr)
        pc = h.predict_proba(Fc.reshape(-1, Fc.shape[-1])[:, cc])[:, 1].reshape(-1, 4)
        heads[which] = (h, cc, float(np.quantile(pc.max(1), 0.95)))
    R[name] = dict(scores=snames)
    for (kind, lv, X, s), (q, fps) in zip(TEST, TEq):
        F = ff(X, fps); S, P, _ = scores(ms, a, X, F); f_ = feats(q, S); r = dict(acc=float((P.argmax(1) == yp).mean()))
        for which, (h, cc, thr) in heads.items():
            pr = h.predict_proba(f_.reshape(-1, f_.shape[-1])[:, cc])[:, 1].reshape(-1, 4); flag = pr.max(1) > thr
            if kind == "clean": r[which] = dict(false_alarm=float(flag.mean()))
            else: r[which] = dict(top1=float((pr.argmax(1) == s).mean()), detect=float(flag.mean()), localise=float((flag & (pr.argmax(1) == s)).mean()))
            if which == "all":
                drop = np.where(flag, pr.argmax(1), -1); _, _, PD = scores(ms, a, X, F, drop=drop); r["acc_remove"] = float((PD.argmax(1) == yp).mean())
                if kind != "clean": _, _, PO = scores(ms, a, X, F, drop=s); r["acc_oracle"] = float((PO.argmax(1) == yp).mean())
        if kind != "clean":
            for j, sn in enumerate(snames): r["raw_" + sn] = float(((S[..., j] - (S[..., j].sum(1, keepdims=True) - S[..., j]) / 3).argmax(1) == s).mean())
        R[name][f"{kind}:{lv}"] = r
        print(name, kind, lv, json.dumps({k_: (round(v, 3) if isinstance(v, float) else {a_: round(b_, 3) for a_, b_ in v.items()}) for k_, v in r.items()}), flush=True)
json.dump(dict(levels={k: [str(v) for v in LEVELS[k]] for k in KINDS}, train_kinds=TRAIN_KINDS, results=R), open("outputs/paper/locate_results.json", "w"), indent=1)
print("saved outputs/paper/locate_results.json")
