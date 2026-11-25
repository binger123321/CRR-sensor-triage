"""Self-diagnosis of a faulty sensor with the sensor-set model (no fault labels, no retraining).

Because the set model accepts any subset of sensors, the encoder runs once per sensor and the fusion is re-run with
different alive masks. For each event and sensor i we compute
  disagree  JS divergence between the prediction from sensor i alone and the prediction without sensor i
  outlier   distance of sensor i's token from the mean token of the other sensors (pre-fusion, 128-d)
  lowatt    -(attention-pooling weight of sensor i) x number of sensors
Each score is standardised per SNR-rank position with the clean validation split (300 events of the training file), and
the three z-scores are averaged (combined). A sensor is flagged when its combined z exceeds the threshold that gives 5 %
false alarms (any sensor) on clean validation events. Detect-and-remove: the flagged sensor is dropped (alive = 0).

Test: the 578 independent events, clean and with each of the 5 fault types (make_faults.py) on each sensor position.
Five seeds of a set model: scores are averaged over seeds; the class probabilities of the five models are averaged.
Writes outputs/paper/diag_results.json and outputs/paper/diag_scores.npz.
usage: python -I diag_eval.py [tag=SETN] ; env DIAG_QUICK=N uses the first N test events
"""
import sys, os, json, glob, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_io import load_pkl, load_weights
from model_set import SensorSetSincNet
from make_faults import fault_inputs
from sklearn.model_selection import train_test_split

TAG = sys.argv[1] if len(sys.argv) > 1 else "SETN"
CKDIR = os.environ.get("DIAG_CKDIR", "outputs/paper")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Dp = load_pkl("resources/signal_selected_predict_data.pkl"); Xp = np.stack([np.array(e, np.float32) for e, _ in Dp]); yp = np.array([int(l) for _, l in Dp])
Dt = load_pkl("resources/signal_selected_train_data.pkl"); Xt = np.stack([np.array(e, np.float32) for e, _ in Dt]); yt = np.array([int(l) for _, l in Dt])
itr, ite = train_test_split(np.arange(len(yt)), test_size=0.2, random_state=42)
Fps = np.load("resources/feats_ps.npz"); live = Fps["tr"][itr].reshape(-1, 47); live = live[np.abs(live).sum(1) > 0]
mu, sd = live.mean(0), live.std(0) + 1e-6
FZ = np.load("resources/faults.npz")
Q = int(os.environ.get("DIAG_QUICK", "0"))
if Q: Xp, yp = Xp[:Q], yp[:Q]
norm = lambda F, X: ((F - mu) / sd).astype(np.float32) * (np.abs(X).max(-1) > 0)[..., None]

ck = sorted(glob.glob(f"{CKDIR}/{TAG}_s[0-9]_best.pth")); assert ck, "no checkpoints"
nets = []
for p in ck:
    m = SensorSetSincNet(n_desc=47).to(dev); m.load_state_dict(load_weights(p)); m.eval(); nets.append(m)


@torch.no_grad()
def tokens(m, X, F):
    """Pre-fusion tokens (B, 4, 128) and the data-derived alive mask (B, 4)."""
    x = torch.tensor(X, device=dev); f = torch.tensor(F, device=dev); B, S, _ = x.shape
    alive = x.abs().amax(-1) > 0
    emb = torch.zeros(B * S, 256, device=dev); fl = alive.reshape(-1)
    if fl.any(): emb[fl] = m.encode(x.reshape(B * S, -1)[fl]).float()
    t = m.tok(torch.cat([emb.reshape(B, S, 256), m.desc(f)], -1)) + m.rank.weight[:S][None]
    return t, alive


@torch.no_grad()
def fuse(m, t, alive):
    """Fusion and classification for a given alive mask; returns probabilities and attention-pooling weights."""
    a = alive.clone(); a[a.sum(1) == 0, 0] = True; pad = ~a
    u = m.fuse(t, src_key_padding_mask=pad)
    w = m.pool(u).squeeze(-1).masked_fill(pad, float("-inf")).softmax(1)
    att = (w[..., None] * u).sum(1); mean = (u * a[..., None]).sum(1) / a.sum(1, keepdim=True)
    return m.classifer(torch.cat([att, mean], 1)).softmax(1), w


def js(p, q):
    m_ = 0.5 * (p + q); kl = lambda a, b: (a * (np.log(a + 1e-9) - np.log(b + 1e-9))).sum(-1)
    return 0.5 * kl(p, m_) + 0.5 * kl(q, m_)


def analyse(X, F, bs=96):
    """Scores (n, 4, 3), ensemble probabilities of the full set (n, 3), attention weights (n, 4), per-model tokens skipped."""
    n = len(X); SC = np.zeros((len(nets), n, 4, 3), np.float32); PF = np.zeros((len(nets), n, 3), np.float32); W = np.zeros((len(nets), n, 4), np.float32)
    for k, m in enumerate(nets):
        for i in range(0, n, bs):
            t, alive = tokens(m, X[i:i + bs], F[i:i + bs]); B = t.shape[0]
            p_full, w = fuse(m, t, alive); PF[k, i:i + B] = p_full.cpu().numpy(); W[k, i:i + B] = w.cpu().numpy()
            for s in range(4):
                only = torch.zeros_like(alive); only[:, s] = alive[:, s]
                wo = alive.clone(); wo[:, s] = False
                p_only, _ = fuse(m, t, only); p_wo, _ = fuse(m, t, wo)
                d = js(p_only.cpu().numpy(), p_wo.cpu().numpy())
                others = wo.float()[..., None]
                mean_o = (t * others).sum(1) / others.sum(1).clamp(min=1)
                o = (t[:, s] - mean_o).norm(dim=-1).cpu().numpy()
                SC[k, i:i + B, s] = np.stack([d, o, -w[:, s].cpu().numpy() * alive.sum(1).cpu().numpy()], -1)
    return SC.mean(0), PF.mean(0), W.mean(0)


@torch.no_grad()
def classify_removed(X, F, drop):
    """Ensemble probabilities with sensor drop[j] (or none if -1) removed for event j."""
    P = []
    for m in nets:
        out = []
        for i in range(0, len(X), 96):
            t, alive = tokens(m, X[i:i + 96], F[i:i + 96]); d = torch.tensor(drop[i:i + 96], device=dev)
            hit = d >= 0; alive[hit, d[hit]] = False
            out.append(fuse(m, t, alive)[0].cpu().numpy())
        P.append(np.concatenate(out))
    return np.mean(P, 0)


# ---- signal quality checks per sensor (no learning): flat, clipping, spikes, low-frequency drift, missing event
from scipy.signal import medfilt, hilbert
def qc(X):
    """(n, 4, 5) quality scores; larger = more suspicious."""
    n = len(X); out = np.zeros((n, 4, 5), np.float32)
    ax = np.abs(X); mx = ax.max(-1)
    out[..., 0] = (mx == 0).astype(np.float32)                                       # flat (identically zero)
    out[..., 1] = (ax >= 0.995 * mx[..., None]).mean(-1) * (mx > 0)                   # fraction of samples at the peak (clipping)
    r = X - medfilt(X.reshape(-1, X.shape[-1]), [1, 5]).reshape(X.shape)              # residual after a 5-point median filter
    p995 = np.quantile(ax, 0.995, axis=-1, keepdims=True) + 1e-12
    out[..., 2] = np.log1p(((np.abs(r) > 0.6 * ax) & (ax > 2.5 * p995)).sum(-1))    # one-sample spikes far above the signal
    P = np.abs(np.fft.rfft(X, axis=-1)) ** 2; f = np.fft.rfftfreq(X.shape[-1], 1e-4)
    out[..., 3] = np.log10(P[..., f < 3].sum(-1) / (P.sum(-1) + 1e-20) + 1e-12)      # share of power below 3 Hz (drift)
    env = np.abs(hilbert(X, axis=-1)).reshape(n, 4, 1000, 10).mean(-1)                # 1 kHz envelope
    env = np.log(env + 1e-12); env = (env - env.mean(-1, keepdims=True)) / (env.std(-1, keepdims=True) + 1e-9)
    E = np.fft.rfft(env, 2048, axis=-1); L = 300
    C = np.zeros((n, 4, 4), np.float32)
    for i in range(4):
        for j in range(4):
            if i < j:
                cc = np.fft.irfft(E[:, i] * np.conj(E[:, j]), 2048, axis=-1) / 1000
                C[:, i, j] = C[:, j, i] = np.concatenate([cc[:, :L], cc[:, -L:]], 1).max(1)
    out[..., 4] = -np.stack([np.median(np.delete(C[:, s], s, 1), 1) for s in range(4)], 1)   # envelope incoherent with the others
    out[mx == 0, 1:] = 0
    return out
QCN = ["flat", "clip", "spike", "drift", "incoherent", "ignored_by_attention"]

# ---- calibration on the clean validation split (never the test set)
SCv, _, _ = analyse(Xt[ite], norm(Fps["tr"][ite], Xt[ite]))
CMU, CSD = SCv.mean(0), SCv.std(0) + 1e-6                     # per sensor position and score type: (4, 3)
z = lambda SC: ((SC - CMU) / CSD)                              # (n, 4, 3)
zc = lambda SC: z(SC).mean(-1)                                 # combined
names = ["disagree", "outlier", "lowatt", "combined"]
def score_set(SC): Z = z(SC); return {nm: (Z[..., j] if j < 3 else Z.mean(-1)) for j, nm in enumerate(names)}
val = score_set(SCv)
THR = {nm: float(np.quantile(val[nm].max(1), 0.95)) for nm in names}   # 5 % false alarm on clean validation events

QV = qc(Xt[ite]); QMU, QSD = QV.reshape(-1, 5).mean(0), QV.reshape(-1, 5).std(0) + 1e-6
QZ = lambda Q_: (Q_ - QMU) / QSD
QTHR = np.quantile(QZ(QV).reshape(-1, 5), 0.999, axis=0); QTHR[0] = 0.5        # 0.1 % false alarms per sensor and check
ATT_THR = float(np.quantile(z(SCv)[..., 2], 0.999))                            # learned check: sensor ignored by attention
def qc_flag(Xf, SCf=None):
    Qf = qc(Xf); Z = QZ(Qf); Z[..., 0] = Qf[..., 0]
    if SCf is not None: Z = np.concatenate([Z, z(SCf)[..., 2:3]], -1)
    T = QTHR if Z.shape[-1] == 5 else np.r_[QTHR, ATT_THR]
    ex = (Z - T) / (np.abs(T) + 1)                                                # exceedance of each check
    worst = ex.max(-1)                                                            # (n, 4)
    return worst.max(1) > 0, worst.argmax(1), ex.argmax(-1)
R = dict(tag=TAG, n_models=len(nets), thr=THR, qc_thr=QTHR.tolist())
SCc, PFc, Wc = analyse(Xp, norm(Fps["pr"][: len(Xp)], Xp))
sc = score_set(SCc)
R["clean"] = dict(acc=float((PFc.argmax(1) == yp).mean()), false_alarm={nm: float((sc[nm].max(1) > THR[nm]).mean()) for nm in names},
                  att_mean=Wc.mean(0).tolist())
dropc = np.where(sc["combined"].max(1) > THR["combined"], sc["combined"].argmax(1), -1)
fl, sen, _ = qc_flag(Xp, SCc); R["clean"]["qc_false_alarm"] = float(fl.mean())
R["clean"]["acc_qc_remove"] = float((classify_removed(Xp, norm(Fps["pr"][: len(Xp)], Xp), np.where(fl, sen, -1)).argmax(1) == yp).mean())
R["clean"]["acc_detect_remove"] = float((classify_removed(Xp, norm(Fps["pr"][: len(Xp)], Xp), dropc).argmax(1) == yp).mean())
store = {"clean_combined": sc["combined"], "clean_att": Wc}
print("clean", json.dumps(R["clean"]), flush=True)
for k, (kind, s, Xf) in enumerate(fault_inputs(np.stack([np.array(e, np.float32) for e, _ in Dp]))):
    Xf = Xf[: len(Xp)]; Ff = norm(FZ["FPS"][k][: len(Xp)], Xf)
    SCf, PFf, Wf = analyse(Xf, Ff); scf = score_set(SCf)
    det = {nm: float((scf[nm].max(1) > THR[nm]).mean()) for nm in names}
    loc = {nm: float(((scf[nm].max(1) > THR[nm]) & (scf[nm].argmax(1) == s)).mean()) for nm in names}
    top1 = {nm: float((scf[nm].argmax(1) == s).mean()) for nm in names}
    drop = np.where(scf["combined"].max(1) > THR["combined"], scf["combined"].argmax(1), -1)
    acc_dr = float((classify_removed(Xf, Ff, drop).argmax(1) == yp).mean())
    oracle = float((classify_removed(Xf, Ff, np.full(len(Xf), s)).argmax(1) == yp).mean())
    att_f = float(Wf[:, s].mean()); att_o = float(np.delete(Wf, s, 1).mean())
    fl, sen, chk = qc_flag(Xf, SCf)
    qc_det = float(fl.mean()); qc_loc = float((fl & (sen == s)).mean())
    qc_check = {QCN[c]: float(((chk[np.arange(len(Xf)), s] == c) & fl & (sen == s)).mean()) for c in range(6)}
    acc_qc = float((classify_removed(Xf, Ff, np.where(fl, sen, -1)).argmax(1) == yp).mean())
    R[f"{kind}:{s + 1}"] = dict(acc=float((PFf.argmax(1) == yp).mean()), acc_detect_remove=acc_dr, acc_oracle_remove=oracle,
                                detect=det, localise=loc, top1=top1, att_faulty=att_f, att_others=att_o,
                                qc_detect=qc_det, qc_localise=qc_loc, qc_check=qc_check, acc_qc_remove=acc_qc)
    store[f"{kind}_{s + 1}_combined"] = scf["combined"]; store[f"{kind}_{s + 1}_att"] = Wf
    print(kind, s + 1, json.dumps(R[f"{kind}:{s + 1}"]), flush=True)
os.makedirs("outputs/paper", exist_ok=True)
json.dump(R, open(f"outputs/paper/diag_results_{TAG}.json", "w"), indent=1)
np.savez_compressed(f"outputs/paper/diag_scores_{TAG}.npz", **store)
print("saved", f"outputs/paper/diag_results_{TAG}.json")
