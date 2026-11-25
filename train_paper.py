"""Train MPA-SincNet+ variants. Same data and the same 80/20 split (random_state=42) as the authors.

Model selection uses ONLY the 300-event split (lowest validation loss); the 578 independent events are
never used for any choice. Independent-event probabilities of the selected epoch are saved for ensembling.

usage: python -I train_paper.py --tag F4 --seed 0 --aug --feat --spec --bilstm [--chdrop 0.5] [--label_frac 0.25] [--arch resnet1d]

Paper version of train_plus.py. Additions:
  --arch        baseline architecture from baselines.py (resnet1d / crnn / transformer / speccnn); default = MS2-SincNet family
  --chdrop p    sensor dropout: with probability p a training event loses 1 or 2 of its 4 sensors (channels zeroed and
                the 94 descriptors replaced by those computed with the same sensors missing, from feats94_masks.npz)
  --label_frac  train on a stratified fraction of the 1,200 training events (validation and independent sets unchanged)
Outputs go to outputs/paper/.
"""
import sys, os, time, json, math, argparse, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_io import load_pkl
from model_plus import SincNetPlus
from baselines import BASELINES
from model_set import SensorSetSincNet, ReSeT, DeepSetSincNet
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wd", type=float, default=1e-2); ap.add_argument("--ls", type=float, default=0.05)
ap.add_argument("--warmup", type=int, default=5); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--softmax", action="store_true", help="keep the authors' softmax-before-CE")
ap.add_argument("--aug", action="store_true"); ap.add_argument("--env", action="store_true")
ap.add_argument("--feat", action="store_true"); ap.add_argument("--no-mpa", action="store_true")
ap.add_argument("--spec", action="store_true"); ap.add_argument("--bilstm", action="store_true")
ap.add_argument("--shrink", action="store_true", help="DRSN-style multi-scale soft-threshold shrinkage after MPA")
ap.add_argument("--graph", action="store_true", help="sensor-geometry graph stream (needs --cv_fold: coordinates exist for the training set only)")
ap.add_argument("--cv_fold", type=int, default=-1, help="0-4: 5-fold stratified CV on the 1,500 training events")
ap.add_argument("--coords", default="resources/sensor_coords_train.npz")
ap.add_argument("--frontend", default="sinc"); ap.add_argument("--max_batches", type=int, default=0)
ap.add_argument("--train", default="resources/signal_selected_train_data.pkl")
ap.add_argument("--pred", default="resources/signal_selected_predict_data.pkl")
ap.add_argument("--feats", default="resources/handcrafted_feats_v2.npz")
ap.add_argument("--arch", default="plus", choices=["plus", "set", "reset", "deepset"] + list(BASELINES))
ap.add_argument("--no_csp", action="store_true"); ap.add_argument("--no_gate", action="store_true"); ap.add_argument("--impute", action="store_true")
ap.add_argument("--scd", type=float, default=0.0, help="weight of subset-consistency distillation (clean full view -> corrupted view)")
ap.add_argument("--csp_w", type=float, default=0.5, help="weight of the cross-sensor prediction loss")
ap.add_argument("--ps_feats", default="resources/feats_ps.npz", help="per-sensor descriptors for --arch set")
ap.add_argument("--drop_max", type=int, default=2, help="sensor dropout removes 1..drop_max sensors")
ap.add_argument("--chdrop", type=float, default=0.0)
ap.add_argument("--masks", default="resources/feats94_masks.npz")
ap.add_argument("--label_frac", type=float, default=1.0)
ap.add_argument("--noisy_feats", default="", help="resources/noisy_feats.npz: noisy copies of the training events with matching descriptors")
ap.add_argument("--fault_p", type=float, default=0.0, help="probability that one sensor of a training event is replaced by a faulty copy (clip / spike / drift / noisy, random severity); set models only")
ap.add_argument("--fault_feats", default="resources/fault_feats.npz")
ap.add_argument("--noise_p", type=float, default=0.0, help="probability that a training event is replaced by a noisy copy (0-30 dB per sensor)")
a = ap.parse_args()
os.makedirs("outputs/paper", exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = dev.type == "cuda"
torch.backends.cudnn.benchmark = True

def arr(d): return torch.tensor(np.array([np.array(e, dtype=np.float32) for e, _ in d])), torch.tensor([int(l) for _, l in d])
X, Y = arr(load_pkl(a.train)); Xp, Yp = arr(load_pkl(a.pred))
Fz = np.load(a.ps_feats if a.arch in ("set", "reset", "deepset") else a.feats); Ftr_all, Fp = Fz["tr"].astype(np.float32), Fz["pr"].astype(np.float32)
if a.cv_fold >= 0:      # held-out fold plays the role of the independent set; 15 % of the rest is the validation split
    from sklearn.model_selection import StratifiedKFold
    rest, test = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(Y)), Y.numpy()))[a.cv_fold]
    itr, ite = train_test_split(rest, test_size=0.15, stratify=Y.numpy()[rest], random_state=a.cv_fold)
    Xp, Yp, Fp = X[test], Y[test], Ftr_all[test]
else:
    itr, ite = train_test_split(np.arange(len(Y)), test_size=0.2, random_state=42)     # authors' split
    assert not a.graph, "--graph needs --cv_fold (no sensor coordinates for the predict set)"
    if a.label_frac < 1.0:      # stratified subset of the training split; seed-dependent so each seed sees a different subset
        itr, _ = train_test_split(itr, train_size=a.label_frac, stratify=Y.numpy()[itr], random_state=100 + a.seed)
if a.arch in ("set", "reset", "deepset"):     # per-sensor descriptors: one set of statistics shared by all sensors (live sensors of the training split)
    live = Ftr_all[itr].reshape(-1, Ftr_all.shape[-1]); live = live[np.abs(live).sum(1) > 0]
    mu, sd = live.mean(0), live.std(0) + 1e-6
else:
    mu, sd = Ftr_all[itr].mean(0), Ftr_all[itr].std(0) + 1e-6                       # training-split statistics only
if a.chdrop > 0:
    Mz = np.load(a.masks); MASKS = torch.tensor(Mz["masks"]); assert MASKS[0].all()
    DROP_IDX = torch.tensor([k for k, m in enumerate(Mz["masks"]) if 4 - a.drop_max <= m.sum() <= 3])   # lose 1..drop_max sensors
    FM = torch.tensor(((Mz["tr"] - mu) / sd).astype(np.float32)) if a.arch not in ("set", "reset", "deepset") else None           # (15, 1500, 94)
Ftr_all, Fp = (Ftr_all - mu) / sd, (Fp - mu) / sd
n_feat = Ftr_all.shape[-1] if a.feat else 0
if not a.feat: Ftr_all, Fp = Ftr_all[..., :0], Fp[..., :0]
if a.graph:             # sensor positions relative to the centroid of the 4 sensors, in units of 300 m
    G = np.load(a.coords)["xyz"]; G = ((G - G.mean(1, keepdims=True)) / 300.0).reshape(len(G), -1).astype(np.float32)
    Ftr_all, Fp = np.hstack([Ftr_all, G]), np.hstack([Fp, G[test]])
if a.noise_p > 0:       # noisy copies of the training events (noise regenerated with the stored seeds) + matching descriptors
    from make_noisy_feats import noise_bank
    NZ = np.load(a.noisy_feats); X_np = X.numpy()
    NB = torch.tensor(np.stack([noise_bank(X_np, NZ["snr"][r], r) for r in range(len(NZ["snr"]))]), dtype=torch.float16)
    NF = NZ["fps" if a.arch in ("set", "reset", "deepset") else "f94"].astype(np.float32); NF = torch.tensor((NF - mu) / sd)[..., :n_feat]
if a.fault_p > 0:       # faulty copies of one sensor per event (regenerated from the seed) + descriptors of that faulty sensor
    assert a.arch in ("set", "reset", "deepset"), "--fault_p needs per-sensor descriptors"
    from faults2 import train_bank
    FZ_ = np.load(a.fault_feats); S_, K_, L_, W_ = train_bank(X.numpy()); assert (S_ == FZ_["sensor"]).all() and (K_ == FZ_["kind"]).all()
    FW = torch.tensor(W_); FSn = torch.tensor(S_); FFt = torch.tensor(((FZ_["fps"] - mu) / sd).astype(np.float32))[..., :n_feat]
Ftr_all, Fp = torch.tensor(Ftr_all), torch.tensor(Fp)
T = lambda t: t.to(dev)
Xtr, Ytr, Ftr = T(X[itr]), T(Y[itr]), T(Ftr_all[itr]); Xte, Yte, Fte = T(X[ite]), T(Y[ite]), T(Ftr_all[ite])
Xp, Yp, Fp = T(Xp), T(Yp), T(Fp)

if a.arch == "reset":
    m = ReSeT(n_desc=n_feat, csp=not a.no_csp, gate=not a.no_gate, impute=a.impute).to(dev)
elif a.arch == "deepset":
    m = DeepSetSincNet(n_desc=n_feat).to(dev)
elif a.arch == "set":
    m = SensorSetSincNet(n_desc=n_feat).to(dev)
elif a.arch == "plus":
    m = SincNetPlus(a.frontend, mpa=not a.no_mpa, env=a.env, feat=n_feat, logits=not a.softmax,
                    spec=a.spec, bilstm=a.bilstm, shrink=a.shrink, graph=a.graph).to(dev)
else:
    m = BASELINES[a.arch]().to(dev)
itr_t = torch.tensor(itr, device=dev)
if a.noise_p > 0: NB, NF = NB.to(dev), NF.to(dev)
if a.fault_p > 0: FW, FSn, FFt = FW.to(dev), FSn.to(dev), FFt.to(dev)

def apply_fault(x, f, idx, skip):
    """Replace one sensor of some events by its faulty copy, with the descriptors of that faulty copy (events in `skip` are left alone)."""
    hit = (torch.rand(len(idx), device=dev) < a.fault_p) & ~skip
    if hit.any():
        rows = hit.nonzero().squeeze(1); rr = torch.randint(FW.shape[0], (len(rows),), device=dev); oi = itr_t[idx[rows]]; ss = FSn[rr, oi]
        x = x.clone(); f = f.clone(); x[rows, ss] = FW[rr, oi]
        if n_feat: f[rows, ss] = FFt[rr, oi]
    return x, f
if a.chdrop > 0: MASKS, DROP_IDX = MASKS.to(dev), DROP_IDX.to(dev); FM = FM.to(dev) if FM is not None else None
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd)
steps_per_ep = math.ceil(len(Xtr) / a.bs); total = a.epochs * steps_per_ep; warm = a.warmup * steps_per_ep
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

def augment(x):
    """Random circular shift (same for the 4 channels), polarity flip, gain, and white noise at 20-40 dB SNR."""
    B = x.shape[0]
    sh = torch.randint(-1500, 1501, (B,), device=x.device)
    idx = (torch.arange(x.shape[-1], device=x.device)[None, :] - sh[:, None]) % x.shape[-1]
    x = torch.gather(x, 2, idx[:, None, :].expand(-1, x.shape[1], -1))
    x = x * torch.where(torch.rand(B, 1, 1, device=x.device) < 0.5, -1.0, 1.0)
    x = x * torch.exp(torch.empty(B, 1, 1, device=x.device).uniform_(-0.7, 0.7))
    snr = torch.empty(B, x.shape[1], 1, device=x.device).uniform_(20, 40)
    noise = torch.randn_like(x) * x.std(-1, keepdim=True) * 10 ** (-snr / 20)
    return x + noise

def augment_pair(xc, xn):
    """Same shift, polarity and gain for the clean view xc and the corrupted view xn; small independent noise on each."""
    B = xc.shape[0]
    sh = torch.randint(-1500, 1501, (B,), device=xc.device)
    idx = (torch.arange(xc.shape[-1], device=xc.device)[None, :] - sh[:, None]) % xc.shape[-1]
    idx = idx[:, None, :].expand(-1, xc.shape[1], -1)
    g = torch.where(torch.rand(B, 1, 1, device=xc.device) < 0.5, -1.0, 1.0) * torch.exp(torch.empty(B, 1, 1, device=xc.device).uniform_(-0.7, 0.7))
    out = []
    for x in (xc, xn):
        x = torch.gather(x, 2, idx) * g
        snr = torch.empty(B, x.shape[1], 1, device=x.device).uniform_(20, 40)
        out.append(x + torch.randn_like(x) * x.std(-1, keepdim=True) * 10 ** (-snr / 20))
    return out

def lossf(out, y, ls):
    if a.softmax: return F.cross_entropy(out, y)               # authors: CE on softmax outputs
    return F.cross_entropy(out, y, label_smoothing=ls)

@torch.no_grad()
def evaluate(Xs, Ys, Fs, bs=64):
    m.eval(); P = []
    for i in range(0, len(Xs), bs):
        with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
            P.append(m(Xs[i:i + bs], Fs[i:i + bs]).float())
    P = torch.cat(P)
    prob = P if a.softmax else P.softmax(1)
    nll = F.nll_loss(torch.log(prob.clamp_min(1e-8)), Ys).item()
    return (prob.argmax(1) == Ys).float().mean().item(), nll, prob

tag = f"{a.tag}_f{a.cv_fold}_s{a.seed}" if a.cv_fold >= 0 else f"{a.tag}_s{a.seed}"; logf = open(f"outputs/paper/{tag}.jsonl", "w"); best = 1e9
meta = dict(vars(a), device=str(dev), n_train=len(itr), n_params=sum(p.numel() for p in m.parameters()))
print(json.dumps(meta), flush=True)
json.dump(meta, open(f"outputs/paper/{tag}_args.json", "w"))
for ep in range(1, a.epochs + 1):
    t0 = time.time(); m.train(); perm = torch.randperm(len(Xtr), device=dev); L = c = n = 0
    for bi, i in enumerate(range(0, len(Xtr), a.bs)):
        idx = perm[i:i + a.bs]; xb = Xtr[idx]; fb = Ftr[idx]
        if a.arch == "reset":       # two views: clean full set (teacher) and corrupted set (noisy copy and/or sensors removed)
            xn, fn = xb, fb; hitn = torch.zeros(len(idx), dtype=torch.bool, device=dev)
            if a.noise_p > 0:
                hitn = torch.rand(len(idx), device=dev) < a.noise_p
                if hitn.any():
                    rr = torch.randint(NB.shape[0], (int(hitn.sum()),), device=dev); oi = itr_t[idx[hitn]]
                    xn = xn.clone(); fn = fn.clone(); xn[hitn] = xn[hitn] + NB[rr, oi].float(); fn[hitn] = NF[rr, oi]
            if a.fault_p > 0: xn, fn = apply_fault(xn, fn, idx, hitn)
            xc, xn = augment_pair(xb, xn) if a.aug else (xb, xn)
            if a.chdrop > 0:
                hit = torch.rand(len(idx), device=dev) < a.chdrop
                k = torch.where(hit, DROP_IDX[torch.randint(len(DROP_IDX), (len(idx),), device=dev)], torch.zeros_like(idx))
                xn = xn * MASKS[k][:, :, None].float(); fn = fn * MASKS[k][:, :, None].float()
            B_ = len(idx)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                z2, aux = m(torch.cat([xc, xn]), torch.cat([fb, fn]), aux=True)
            z2 = z2.float(); zc, zn = z2[:B_], z2[B_:]; yb = Ytr[idx]
            loss = 0.5 * (lossf(zc, yb, a.ls) + lossf(zn, yb, a.ls))
            if a.scd > 0:
                Tk = 2.0
                loss = loss + a.scd * Tk * Tk * F.kl_div(F.log_softmax(zn / Tk, 1), F.softmax(zc.detach() / Tk, 1), reduction="batchmean")
            if aux["e"] is not None:
                al = aux["alive"].float(); ok = (al.sum(1, keepdim=True) > 1).float() * al
                loss = loss + a.csp_w * (aux["e"].float() * ok).sum() / ok.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            L += loss.item() * B_; c += (zc.argmax(1) == yb).sum().item(); n += B_
            if a.max_batches and bi + 1 >= a.max_batches: break
            continue
        hitn = torch.zeros(len(idx), dtype=torch.bool, device=dev)
        if a.noise_p > 0:           # replace some events by a noisy copy, with descriptors computed on that noisy copy
            hitn = torch.rand(len(idx), device=dev) < a.noise_p
            if hitn.any():
                rr = torch.randint(NB.shape[0], (int(hitn.sum()),), device=dev); oi = itr_t[idx[hitn]]
                xb = xb.clone(); fb = fb.clone()
                xb[hitn] = xb[hitn] + NB[rr, oi].float(); fb[hitn] = NF[rr, oi]
        if a.fault_p > 0: xb, fb = apply_fault(xb, fb, idx, hitn)
        if a.aug: xb = augment(xb)
        if a.chdrop > 0:            # sensor dropout, applied after augmentation so dead channels stay exactly zero
            hit = torch.rand(len(idx), device=dev) < a.chdrop
            if a.arch not in ("set", "reset", "deepset"): hit = hit & ~hitn      # fixed-channel model: no array descriptors exist for noisy copies with sensors removed
            k = torch.where(hit, DROP_IDX[torch.randint(len(DROP_IDX), (len(idx),), device=dev)], torch.zeros_like(idx))
            xb = xb * MASKS[k][:, :, None].float()
            if n_feat and FM is not None: fb = FM[k, itr_t[idx]]
            elif n_feat: fb = fb * MASKS[k][:, :, None].float()
        with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
            out = m(xb, fb); loss = lossf(out.float(), Ytr[idx], a.ls)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        L += loss.item() * len(idx); c += (out.argmax(1) == Ytr[idx]).sum().item(); n += len(idx)
        if a.max_batches and bi + 1 >= a.max_batches: break
    va, vl, vprob = evaluate(Xte, Yte, Fte); pr, pl, pprob = evaluate(Xp, Yp, Fp)
    row = dict(epoch=ep, train_loss=L / n, train_acc=c / n, val_acc=va, val_loss=vl, pred_acc=pr, pred_loss=pl,
               lr=opt.param_groups[0]["lr"], sec=time.time() - t0)
    logf.write(json.dumps(row) + "\n"); logf.flush(); print(json.dumps(row), flush=True)
    if vl < best:                                              # selection on the validation split only
        best = vl; sel = dict(epoch=ep, val_acc=va, val_loss=vl, pred_acc=pr)
        np.save(f"outputs/paper/{tag}_pred_probs.npy", pprob.cpu().numpy()); np.save(f"outputs/paper/{tag}_val_probs.npy", vprob.cpu().numpy())
        torch.save(m.state_dict(), f"outputs/paper/{tag}_best.pth")
np.save(f"outputs/paper/{tag}_pred_probs_last.npy", pprob.cpu().numpy())
json.dump(sel, open(f"outputs/paper/{tag}_selected.json", "w"))
print("done", tag, json.dumps(sel), flush=True)
