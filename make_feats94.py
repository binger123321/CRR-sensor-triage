"""Array-level descriptors (94 per event) of the training and independent events, for the fixed-channel models.
Writes resources/handcrafted_feats_v2.npz (already provided; this script regenerates it from the waveforms)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from safe_io import load_pkl
from feats94 import feats94

arr = lambda D: np.stack([np.array(e, np.float32) for e, _ in D])
tr = feats94(arr(load_pkl("resources/signal_selected_train_data.pkl"))); pr = feats94(arr(load_pkl("resources/signal_selected_predict_data.pkl")))
out = "resources/handcrafted_feats_v2.npz"
if os.path.exists(out):
    ref = np.load(out); print("max abs difference from the provided file:", float(np.abs(tr - ref["tr"]).max()), float(np.abs(pr - ref["pr"]).max()))
np.savez(out, tr=tr.astype(np.float32), pr=pr.astype(np.float32)); print("saved", out)
