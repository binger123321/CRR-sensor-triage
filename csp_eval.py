"""What the cross-sensor prediction residual of ReSeT knows about a faulty sensor (no fault labels, no retraining).

For each ReSeT run in outputs/paper (five seeds; residuals averaged over seeds) and each of the 20 single-sensor faults of
make_faults.py: is the sensor with the largest relative residual the faulty one (top-1), is the event flagged (largest z above
the 95th percentile of clean validation events), and how much attention-pooling weight does the faulty sensor get.
Writes outputs/paper/csp_results.json.  usage: python -I csp_eval.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, torch
from evalcommon import *
from make_faults import fault_inputs
FZ = np.load("resources/faults.npz")

@torch.no_grad()
def resid(ms, X, F, bs=64):
    E, W = [], []
    for m in ms:
        e_, w_ = [], []
        for i in range(0, len(X), bs):
            _, aux = m(torch.tensor(X[i:i + bs], device=dev), torch.tensor(F[i:i + bs], device=dev), aux=True)
            e_.append(aux["e"].float().cpu().numpy()); w_.append(aux["w"].float().cpu().numpy())
        E.append(np.concatenate(e_)); W.append(np.concatenate(w_))
    return np.mean(E, 0), np.mean(W, 0)

rel = lambda E, X: E / (E * (np.abs(X).max(-1) > 0)).sum(1, keepdims=True).clip(1e-9) * (np.abs(X).max(-1) > 0).sum(1, keepdims=True)
nets = {n: v for n, v in load_models().items() if v[1].get("arch") == "reset" and not v[1].get("no_csp")}
R = {}
for n, (ms, a) in nets.items():
    Ev, _ = resid(ms, Xt[ite], feats_for(a, Xt[ite], FPS=Fps["tr"][ite])); Zv = rel(Ev, Xt[ite])
    mu_, sd_ = Zv.mean(0), Zv.std(0) + 1e-6; thr = float(np.quantile(((Zv - mu_) / sd_).max(1), 0.95))
    Ec, Wc = resid(ms, Xp, feats_for(a, Xp, FPS=Fps["pr"][: len(Xp)]))
    R[n] = dict(clean_false_alarm=float(((((rel(Ec, Xp) - mu_) / sd_).max(1)) > thr).mean()), clean_att=Wc.mean(0).tolist())
    for k, (kind, s, Xf) in enumerate(fault_inputs(np.stack([np.array(e, np.float32) for e, _ in Dp]))):
        Xf = Xf[: len(Xp)]; Ef, Wf = resid(ms, Xf, feats_for(a, Xf, FPS=FZ["FPS"][k][: len(Xp)]))
        Z = (rel(Ef, Xf) - mu_) / sd_
        if kind == "zero": Z[:, s] = Z.max() + 1                   # a dead channel is known from the data itself
        R[n][f"{kind}:{s + 1}"] = dict(top1=float((Z.argmax(1) == s).mean()), detect=float((Z.max(1) > thr).mean()),
                                        localise=float(((Z.max(1) > thr) & (Z.argmax(1) == s)).mean()),
                                        att_faulty=float(Wf[:, s].mean()), att_others=float(np.delete(Wf, s, 1).mean()))
    for kind in ("noise", "clip", "spike", "drift"):
        R[n][kind] = {q: float(np.mean([R[n][f"{kind}:{s}"][q] for s in range(1, 5)])) for q in ("top1", "detect", "localise", "att_faulty", "att_others")}
        print(n, kind, {q: round(v, 3) for q, v in R[n][kind].items()}, flush=True)
json.dump(R, open("outputs/paper/csp_results.json", "w"), indent=1); print("saved outputs/paper/csp_results.json")
