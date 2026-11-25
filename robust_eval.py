"""Robustness of trained models on the 578 independent events (no retraining; nothing here is used for model selection).

Conditions
  clean          : original data
  drop:<mask>    : dead sensors -- the channels not in <mask> are zeroed and the 94 descriptors are recomputed
                   with those sensors missing (14 masks: lose 1, 2 or 3 of the 4 sensors)
  snr:<dB>       : white Gaussian noise added to every channel at 20 / 10 / 5 / 0 dB relative to the channel's RMS
                   (fixed random seed, identical for all models); descriptors recomputed on the noisy waveforms

Models: the published MPA-SincNet weights plus every run listed in MODELS (five seeds each, probabilities averaged).
Writes outputs/paper/robust_results.json and outputs/paper/robust_probs.npz.
usage: python -I robust_eval.py
"""
import sys, os, json, glob, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_io import load_pkl, load_weights
from model_plus import SincNetPlus
from baselines import BASELINES
from feats94 import feats94
from feats_ps import feats_ps
from model_set import SensorSetSincNet, ReSeT, DeepSetSincNet

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs/paper", exist_ok=True)
D = [(np.array(e, dtype=np.float32), int(l)) for e, l in load_pkl("resources/signal_selected_predict_data.pkl")]
Xp = np.stack([e for e, _ in D]); yp = np.array([l for _, l in D])
QUICK = int(os.environ.get("ROBUST_QUICK", "0"))          # smoke test only: use the first N events
if QUICK: Xp, yp = Xp[:QUICK], yp[:QUICK]
Ytr = np.array([int(l) for _, l in load_pkl("resources/signal_selected_train_data.pkl")])
from sklearn.model_selection import train_test_split
itr_full, _ = train_test_split(np.arange(len(Ytr)), test_size=0.2, random_state=42)
Fv2 = np.load("resources/handcrafted_feats_v2.npz"); Mz = np.load("resources/feats94_masks.npz")

# ---------------- perturbed inputs (waveforms + descriptors) ----------------
PRF = Fv2["pr"][:len(yp)]; MPR = Mz["pr"][:, :len(yp)]
HAVE_PS = os.path.exists("resources/feats_ps.npz")
if HAVE_PS:     # per-sensor descriptors for the sensor-set model; a dead sensor's row is simply zeroed (and ignored by the model)
    Fps = np.load("resources/feats_ps.npz"); PSP = Fps["pr"][:len(yp)]
    live = Fps["tr"][itr_full].reshape(-1, PSP.shape[-1]); live = live[np.abs(live).sum(1) > 0]
    PS_MU, PS_SD = live.mean(0), live.std(0) + 1e-6
COND = {"clean": (Xp, PRF)}; COND_PS = {"clean": PSP} if HAVE_PS else {}
for k, m in enumerate(Mz["masks"]):
    if m.all(): continue
    c = "drop:" + "".join(str(int(v)) for v in m)
    COND[c] = (Xp * m[None, :, None].astype(np.float32), MPR[k])
    if HAVE_PS: COND_PS[c] = PSP * m[None, :, None]
for snr in (20, 10, 5, 0):
    rng = np.random.default_rng(1000 + snr)
    noise = rng.standard_normal(Xp.shape).astype(np.float32) * np.sqrt((Xp ** 2).mean(-1, keepdims=True)) * 10 ** (-snr / 20)
    Xn = Xp + noise; COND[f"snr:{snr}"] = (Xn, feats94(Xn))
    if HAVE_PS: COND_PS[f"snr:{snr}"] = feats_ps(Xn)
    print("prepared", f"snr:{snr}", flush=True)


def build(args):
    if args.get("arch") == "reset": return ReSeT(n_desc=47 if args.get("feat") else 0, csp=not args.get("no_csp"), gate=not args.get("no_gate"), impute=args.get("impute", False))
    if args.get("arch") == "deepset": return DeepSetSincNet(n_desc=47 if args.get("feat") else 0)
    if args.get("arch") == "set": return SensorSetSincNet(n_desc=47 if args.get("feat") else 0)
    if args.get("arch", "plus") != "plus": return BASELINES[args["arch"]]()
    nf = 94 if args.get("feat") else 0
    return SincNetPlus(args.get("frontend", "sinc"), mpa=not args.get("no_mpa", False), env=args.get("env", False), feat=nf,
                       logits=not args.get("softmax", False), spec=args.get("spec", False), bilstm=args.get("bilstm", False),
                       shrink=args.get("shrink", False))


@torch.no_grad()
def predict(m, X, F, softmax_out=False, bs=64):
    m.eval(); P = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], device=dev); fb = torch.tensor(F[i:i + bs], device=dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            o = m(xb) if fb is None else m(xb, fb)
        o = o.float(); P.append((o if softmax_out else o.softmax(1)).cpu().numpy())
    return np.concatenate(P)


# models: (name, directory, tag, args); args for the older runs written out explicitly
OLD = {"A_logits": dict(), "F4_pi": dict(feat=True, spec=True, bilstm=True)}
MODELS = [("A_training_fixed", "outputs/plus", "A_logits", OLD["A_logits"]),
          ("B_augmented", "outputs/plus", "B_aug", OLD["A_logits"]),
          ("MS2-SincNet", "outputs/plus", "F4_pi", OLD["F4_pi"])]
for f in sorted(glob.glob("outputs/paper/*_s0_args.json")):
    a = json.load(open(f)); tag = os.path.basename(f)[:-len("_s0_args.json")]
    if a.get("label_frac", 1.0) < 1.0: continue                     # few-label runs are evaluated clean only (in the summary script)
    MODELS.append((tag, "outputs/paper", tag, a))

R, PROBS = {}, {}
# published weights (single model, softmax output, no descriptors)
if os.path.exists("resources/checkpoint.pth"):
    from network.model import SincNetModel
    pub = SincNetModel(4).to(dev); pub.load_state_dict(load_weights("resources/checkpoint.pth")); pub.eval()
    R["published"] = {}
    for c, (X, F) in COND.items():
        with torch.no_grad():
            P = np.concatenate([pub(torch.tensor(X[i:i + 64], device=dev)).float().cpu().numpy() for i in range(0, len(X), 64)])
        R["published"][c] = dict(single=[float((P.argmax(1) == yp).mean())], ens=float((P.argmax(1) == yp).mean())); PROBS[f"published|{c}"] = P
    print("published", R["published"]["clean"], flush=True)

for name, d, tag, args in MODELS:
    ck = sorted(glob.glob(f"{d}/{tag}_s[0-9]_best.pth"))
    if not ck: print("skip", name, "(no checkpoints)"); continue
    is_set = args.get("arch") in ("set", "reset", "deepset")
    if is_set and not HAVE_PS: print("skip", name, "(no resources/feats_ps.npz)"); continue
    nf = 94 if (args.get("feat") and args.get("arch", "plus") == "plus") else 0
    mu = Fv2["tr"][itr_full].mean(0); sd = Fv2["tr"][itr_full].std(0) + 1e-6       # same statistics as training (full label set)
    nets = []
    for p in ck:
        m = build(args).to(dev); m.load_state_dict(load_weights(p)); nets.append(m)
    R[name] = {}
    for c, (X, F) in COND.items():
        if is_set:
            Fs = ((COND_PS[c] - PS_MU) / PS_SD).astype(np.float32) * (np.abs(X).max(-1) > 0)[..., None]
            if not args.get("feat"): Fs = Fs[..., :0]
        else:
            Fs = ((F - mu) / sd).astype(np.float32)[:, :nf]
        Ps = [predict(m, X, Fs) for m in nets]; E = np.mean(Ps, 0)
        R[name][c] = dict(single=[float((P.argmax(1) == yp).mean()) for P in Ps], ens=float((E.argmax(1) == yp).mean()))
        PROBS[f"{name}|{c}"] = E
    print(name, len(nets), "clean", R[name]["clean"]["ens"], "drop1", np.mean([v["ens"] for k, v in R[name].items() if k.startswith("drop:") and k.count("1") == 3]), flush=True)

# summary: mean over masks with the same number of lost sensors
for name in R:
    for lost in (1, 2, 3):
        ks = [k for k in R[name] if k.startswith("drop:") and k[5:].count("0") == lost]
        R[name][f"lost{lost}_mean"] = float(np.mean([R[name][k]["ens"] for k in ks]))
        R[name][f"lost{lost}_worst"] = float(np.min([R[name][k]["ens"] for k in ks]))
json.dump(R, open("outputs/paper/robust_results.json", "w"), indent=1)
np.savez_compressed("outputs/paper/robust_probs.npz", **{k.replace(":", "_").replace("|", "__"): v for k, v in PROBS.items()})
print("saved outputs/paper/robust_results.json")
