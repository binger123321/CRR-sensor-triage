"""Descriptors of the faulty copies used in fault-aware training (descriptor-consistent, as for the noisy copies).
Only the faulty sensor changes, so only its 47 per-sensor descriptors are recomputed; the other sensors keep theirs.
Writes resources/fault_feats.npz: sensor, kind, level (copies, 1500) and fps (copies, 1500, 47) of the faulty sensor.
The faulty waveforms are regenerated from the seed by faults2.train_bank.   usage: python -I make_fault_feats.py
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_io import load_pkl
from feats_ps import feats_ps
from faults2 import train_bank
X = np.stack([np.array(e, np.float32) for e, _ in load_pkl("resources/signal_selected_train_data.pkl")])
S, K, L, W = train_bank(X)
fps = np.stack([feats_ps(W[r][:, None, :])[:, 0] for r in range(len(W))]).astype(np.float32)
np.savez_compressed("resources/fault_feats.npz", sensor=S, kind=K, level=L, fps=fps)
print("saved resources/fault_feats.npz", fps.shape)
