"""Accuracy under realistic single-sensor faults (no retraining). Needs resources/faults.npz from make_faults.py.

For every fault type (zero, noise, clip, spike, drift) and faulty sensor (1-4), the faulty waveforms are regenerated
with the same seed as make_faults.py and fed to each model unflagged, i.e. the model is not told which sensor is faulty.
If the acquisition system flags the faulty sensor, the fixed-channel models can zero it and the sensor-set models can
drop it; both cases equal the 'zero' fault, so 'zero' is also the 'flagged' result for every fault type.
Writes outputs/paper/fault_results.json.  usage: python -I fault_eval.py
"""
import sys, os, json, glob, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_io import load_pkl, load_weights
from model_plus import SincNetPlus
from model_set import SensorSetSincNet, ReSeT, DeepSetSincNet
from make_faults import fault_inputs
from sklearn.model_selection import train_test_split

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D = [(np.array(e, dtype=np.float32), int(l)) for e, l in load_pkl("resources/signal_selected_predict_data.pkl")]
Xp = np.stack([e for e, _ in D]); yp = np.array([l for _, l in D])
Ytr = np.array([int(l) for _, l in load_pkl("resources/signal_selected_train_data.pkl")])
itr, _ = train_test_split(np.arange(len(Ytr)), test_size=0.2, random_state=42)
Fv2 = np.load("resources/handcrafted_feats_v2.npz"); mu, sd = Fv2["tr"][itr].mean(0), Fv2["tr"][itr].std(0) + 1e-6
Fps = np.load("resources/feats_ps.npz"); live = Fps["tr"][itr].reshape(-1, 47); live = live[np.abs(live).sum(1) > 0]
pmu, psd = live.mean(0), live.std(0) + 1e-6
FZ = np.load("resources/faults.npz")

MODELS = [("B_augmented", "outputs/plus", "B_aug", dict()), ("MS2-SincNet", "outputs/plus", "F4_pi", dict(feat=True, spec=True, bilstm=True))]
for tag in ("F4cd", "SET0", "SET", "SET3", "F4cdN", "SETN", "RST", "RST_nogate", "RST_noscd", "RST_imp", "RST_base", "DSN", "RSTF", "SETF"):
    f = f"outputs/paper/{tag}_s0_args.json"
    if os.path.exists(f): MODELS.append((tag, "outputs/paper", tag, json.load(open(f))))


def build(a):
    if a.get("arch") == "reset": return ReSeT(n_desc=47 if a.get("feat") else 0, csp=not a.get("no_csp"), gate=not a.get("no_gate"), impute=a.get("impute", False))
    if a.get("arch") == "deepset": return DeepSetSincNet(n_desc=47 if a.get("feat") else 0)
    if a.get("arch") == "set": return SensorSetSincNet(n_desc=47 if a.get("feat") else 0)
    return SincNetPlus("sinc", feat=94 if a.get("feat") else 0, spec=a.get("spec", False), bilstm=a.get("bilstm", False))


@torch.no_grad()
def predict(m, X, F, bs=64):
    m.eval(); P = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], device=dev); fb = torch.tensor(F[i:i + bs], device=dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            P.append(m(xb, fb).float().softmax(1).cpu().numpy())
    return np.concatenate(P)


nets = {}
for name, d, tag, a in MODELS:
    ck = sorted(glob.glob(f"{d}/{tag}_s[0-9]_best.pth"))
    if not ck: print("skip", name); continue
    ms = []
    for p in ck:
        m = build(a).to(dev); m.load_state_dict(load_weights(p)); ms.append(m)
    nets[name] = (ms, a)
from network.model import SincNetModel
pub = SincNetModel(4).to(dev); pub.load_state_dict(load_weights("resources/checkpoint.pth")); pub.eval()

R = {n: {} for n in ["published"] + list(nets)}
for k, (kind, s, Xf) in enumerate(fault_inputs(Xp)):
    assert FZ["names"][k] == kind and FZ["sensor"][k] == s
    c = f"{kind}:{s + 1}"
    with torch.no_grad():
        P = np.concatenate([pub(torch.tensor(Xf[i:i + 64], device=dev)).float().cpu().numpy() for i in range(0, len(Xf), 64)])
    R["published"][c] = float((P.argmax(1) == yp).mean())
    for name, (ms, a) in nets.items():
        if a.get("arch") in ("set", "reset", "deepset"):
            F = ((FZ["FPS"][k] - pmu) / psd).astype(np.float32) * (np.abs(Xf).max(-1) > 0)[..., None]
            if not a.get("feat"): F = F[..., :0]
        else:
            F = ((FZ["F94"][k] - mu) / sd).astype(np.float32)[:, :94 if a.get("feat") else 0]
        E = np.mean([predict(m, Xf, F) for m in ms], 0)
        R[name][c] = float((E.argmax(1) == yp).mean())
    print(c, {n: round(R[n][c] * 100, 1) for n in R}, flush=True)
for n in R:
    for kind in ("zero", "noise", "clip", "spike", "drift"):
        v = [R[n][f"{kind}:{s}"] for s in range(1, 5)]
        R[n][f"{kind}_mean"] = float(np.mean(v)); R[n][f"{kind}_worst"] = float(np.min(v))
os.makedirs("outputs/paper", exist_ok=True)
json.dump(R, open("outputs/paper/fault_results.json", "w"), indent=1)
print("saved outputs/paper/fault_results.json")
