"""Train MPA-SincNet+ variants. Same data and the same 80/20 split (random_state=42) as the authors.

Model selection uses ONLY the 300-event split (lowest validation loss); the 578 independent events are
never used for any choice. Independent-event probabilities of the selected epoch are saved for ensembling.

usage: python -I train_plus.py --tag C3 --seed 0 [--softmax] [--aug] [--env] [--feat] [--lr 3e-4] [--epochs 80]
"""
import sys, os, time, json, math, argparse, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_io import load_pkl
from model_plus import SincNetPlus
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
ap.add_argument("--feats", default="resources/handcrafted_feats.npz")
a = ap.parse_args()
os.makedirs("outputs/plus", exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = dev.type == "cuda"
torch.backends.cudnn.benchmark = True

def arr(d): return torch.tensor(np.array([np.array(e, dtype=np.float32) for e, _ in d])), torch.tensor([int(l) for _, l in d])
X, Y = arr(load_pkl(a.train)); Xp, Yp = arr(load_pkl(a.pred))
Fz = np.load(a.feats); Ftr_all, Fp = Fz["tr"].astype(np.float32), Fz["pr"].astype(np.float32)
if a.cv_fold >= 0:      # held-out fold plays the role of the independent set; 15 % of the rest is the validation split
    from sklearn.model_selection import StratifiedKFold
    rest, test = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(len(Y)), Y.numpy()))[a.cv_fold]
    itr, ite = train_test_split(rest, test_size=0.15, stratify=Y.numpy()[rest], random_state=a.cv_fold)
    Xp, Yp, Fp = X[test], Y[test], Ftr_all[test]
else:
    itr, ite = train_test_split(np.arange(len(Y)), test_size=0.2, random_state=42)     # authors' split
    assert not a.graph, "--graph needs --cv_fold (no sensor coordinates for the predict set)"
mu, sd = Ftr_all[itr].mean(0), Ftr_all[itr].std(0) + 1e-6                           # training-split statistics only
Ftr_all, Fp = (Ftr_all - mu) / sd, (Fp - mu) / sd
n_feat = Ftr_all.shape[1] if a.feat else 0
if not a.feat: Ftr_all, Fp = Ftr_all[:, :0], Fp[:, :0]
if a.graph:             # sensor positions relative to the centroid of the 4 sensors, in units of 300 m
    G = np.load(a.coords)["xyz"]; G = ((G - G.mean(1, keepdims=True)) / 300.0).reshape(len(G), -1).astype(np.float32)
    Ftr_all, Fp = np.hstack([Ftr_all, G]), np.hstack([Fp, G[test]])
Ftr_all, Fp = torch.tensor(Ftr_all), torch.tensor(Fp)
T = lambda t: t.to(dev)
Xtr, Ytr, Ftr = T(X[itr]), T(Y[itr]), T(Ftr_all[itr]); Xte, Yte, Fte = T(X[ite]), T(Y[ite]), T(Ftr_all[ite])
Xp, Yp, Fp = T(Xp), T(Yp), T(Fp)

m = SincNetPlus(a.frontend, mpa=not a.no_mpa, env=a.env, feat=n_feat, logits=not a.softmax,
                spec=a.spec, bilstm=a.bilstm, shrink=a.shrink, graph=a.graph).to(dev)
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

tag = f"{a.tag}_f{a.cv_fold}_s{a.seed}" if a.cv_fold >= 0 else f"{a.tag}_s{a.seed}"; logf = open(f"outputs/plus/{tag}.jsonl", "w"); best = 1e9
print(json.dumps(dict(vars(a), device=str(dev), n_params=sum(p.numel() for p in m.parameters()))), flush=True)
for ep in range(1, a.epochs + 1):
    t0 = time.time(); m.train(); perm = torch.randperm(len(Xtr), device=dev); L = c = n = 0
    for bi, i in enumerate(range(0, len(Xtr), a.bs)):
        idx = perm[i:i + a.bs]; xb = Xtr[idx]
        if a.aug: xb = augment(xb)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
            out = m(xb, Ftr[idx]); loss = lossf(out.float(), Ytr[idx], a.ls)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        L += loss.item() * len(idx); c += (out.argmax(1) == Ytr[idx]).sum().item(); n += len(idx)
        if a.max_batches and bi + 1 >= a.max_batches: break
    va, vl, vprob = evaluate(Xte, Yte, Fte); pr, pl, pprob = evaluate(Xp, Yp, Fp)
    row = dict(epoch=ep, train_loss=L / n, train_acc=c / n, val_acc=va, val_loss=vl, pred_acc=pr, pred_loss=pl,
               lr=opt.param_groups[0]["lr"], sec=time.time() - t0)
    logf.write(json.dumps(row) + "\n"); logf.flush(); print(json.dumps(row), flush=True)
    if vl < best:                                              # selection on the validation split only
        best = vl; sel = dict(epoch=ep, val_acc=va, val_loss=vl, pred_acc=pr)
        np.save(f"outputs/plus/{tag}_pred_probs.npy", pprob.cpu().numpy()); np.save(f"outputs/plus/{tag}_val_probs.npy", vprob.cpu().numpy())
        torch.save(m.state_dict(), f"outputs/plus/{tag}_best.pth")
np.save(f"outputs/plus/{tag}_pred_probs_last.npy", pprob.cpu().numpy())
json.dump(sel, open(f"outputs/plus/{tag}_selected.json", "w"))
print("done", tag, json.dumps(sel), flush=True)
