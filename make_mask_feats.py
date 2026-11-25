"""Features with dead sensors: for each of the 15 non-empty subsets of the 4 channels, zero the others and recompute feats94.
Output resources/feats94_masks.npz: masks (15,4) bool, tr (15,1500,94), pr (15,578,94)."""
import sys, itertools, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from safe_io import load_pkl
from feats94 import feats94
X = np.array([np.array(e, dtype=np.float32) for e, _ in load_pkl("resources/signal_selected_train_data.pkl")])
Xp = np.array([np.array(e, dtype=np.float32) for e, _ in load_pkl("resources/signal_selected_predict_data.pkl")])
masks = np.array([m for m in itertools.product([0, 1], repeat=4) if sum(m)][::-1], bool)   # full set first
tr = np.zeros((len(masks), len(X), 94), np.float32); pr = np.zeros((len(masks), len(Xp), 94), np.float32)
for k, m in enumerate(masks):
    t = time.time(); w = m[None, :, None].astype(np.float32)
    tr[k] = feats94(X * w); pr[k] = feats94(Xp * w); print(k, m.astype(int), f"{time.time()-t:.0f}s", flush=True)
np.savez("resources/feats94_masks.npz", masks=masks, tr=tr, pr=pr)
ref = np.load("resources/handcrafted_feats_v2.npz"); print("full-set check", np.abs(tr[0] - ref["tr"]).max(), np.abs(pr[0] - ref["pr"]).max())
