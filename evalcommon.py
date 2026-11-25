"""Shared loading code for the evaluation scripts: data, descriptor statistics, and five-seed ensembles of every model family."""
import sys, os, json, glob, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_io import load_pkl, load_weights
from model_plus import SincNetPlus
from model_set import SensorSetSincNet, ReSeT, DeepSetSincNet
from sklearn.model_selection import train_test_split

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Dp = load_pkl("resources/signal_selected_predict_data.pkl")
Xp = np.stack([np.array(e, np.float32) for e, _ in Dp]); yp = np.array([int(l) for _, l in Dp])
Dt = load_pkl("resources/signal_selected_train_data.pkl")
Xt = np.stack([np.array(e, np.float32) for e, _ in Dt]); yt = np.array([int(l) for _, l in Dt])
itr, ite = train_test_split(np.arange(len(yt)), test_size=0.2, random_state=42)
Fv2 = np.load("resources/handcrafted_feats_v2.npz"); mu94, sd94 = Fv2["tr"][itr].mean(0), Fv2["tr"][itr].std(0) + 1e-6
Fps = np.load("resources/feats_ps.npz"); _l = Fps["tr"][itr].reshape(-1, 47); _l = _l[np.abs(_l).sum(1) > 0]
mups, sdps = _l.mean(0), _l.std(0) + 1e-6
Q = int(os.environ.get("EVAL_QUICK", "0"))
if Q: Xp, yp = Xp[:Q], yp[:Q]
SETS = ("set", "reset", "deepset")


def build(a):
    f = 47 if a.get("feat") else 0
    if a.get("arch") == "reset": return ReSeT(n_desc=f, csp=not a.get("no_csp"), gate=not a.get("no_gate"), impute=a.get("impute", False))
    if a.get("arch") == "deepset": return DeepSetSincNet(n_desc=f)
    if a.get("arch") == "set": return SensorSetSincNet(n_desc=f)
    return SincNetPlus("sinc", feat=94 if a.get("feat") else 0, spec=a.get("spec", False), bilstm=a.get("bilstm", False))


def load_models(tags=None):
    """{name: (list of nets, args)} for the older runs and every run in outputs/paper (full-label runs only)."""
    M = [("B_augmented", "outputs/plus", "B_aug", dict()), ("MS2-SincNet", "outputs/plus", "F4_pi", dict(feat=True, spec=True, bilstm=True))]
    for f in sorted(glob.glob("outputs/paper/*_s0_args.json")):
        a = json.load(open(f)); tag = os.path.basename(f)[:-len("_s0_args.json")]
        if a.get("label_frac", 1.0) < 1.0 or a.get("arch", "plus") not in ("plus",) + SETS: continue
        M.append((tag, "outputs/paper", tag, a))
    nets = {}
    for name, d, tag, a in M:
        if tags and name not in tags: continue
        ck = sorted(glob.glob(f"{d}/{tag}_s[0-9]_best.pth"))
        if not ck: continue
        ms = []
        for p in ck:
            m = build(a).to(dev); m.load_state_dict(load_weights(p)); m.eval(); ms.append(m)
        nets[name] = (ms, a)
    return nets


def feats_for(a, X, F94=None, FPS=None):
    """Standardised descriptors for a model family from raw descriptor arrays (computed on X)."""
    if a.get("arch") in SETS:
        F = ((FPS - mups) / sdps).astype(np.float32) * (np.abs(X).max(-1) > 0)[..., None]
        return F if a.get("feat") else F[..., :0]
    return ((F94 - mu94) / sd94).astype(np.float32)[:, :94 if a.get("feat") else 0]


@torch.no_grad()
def predict(m, X, F, bs=64):
    P = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], device=dev); fb = torch.tensor(F[i:i + bs], device=dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            P.append(m(xb, fb).float().softmax(1).cpu().numpy())
    return np.concatenate(P)


def ens_acc(ms, X, F): return float((np.mean([predict(m, X, F) for m in ms], 0).argmax(1) == yp).mean())
