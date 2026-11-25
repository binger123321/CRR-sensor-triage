"""Realistic single-sensor faults on the 578 independent events, with descriptors recomputed.

Fault types, applied to one sensor at a time (4 positions):
  zero    channel identically zero (disconnected / unpowered; the case used in training)
  noise   channel replaced by white noise with the channel's original RMS (sensor dead, electronics alive)
  clip    channel clipped at 10 % of its peak amplitude (saturated amplifier / ADC)
  spike   20 random spikes of 3x the channel peak added (electrical interference)
  drift   0.3-1 Hz sinusoid with 2x the channel peak added (baseline drift / low-frequency pickup)
Fixed random seed; identical inputs for all models. fault_inputs() regenerates the faulty waveforms deterministically
(same seed, same order), so only descriptors are stored: resources/faults.npz with F94 (n_cond, 578, 94),
FPS (n_cond, 578, 4, 47), names, sensor.
usage: python -I make_faults.py
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safe_io import load_pkl
from feats94 import feats94
from feats_ps import feats_ps

FS = 10000
t = np.arange(10000) / FS
KINDS = ("zero", "noise", "clip", "spike", "drift")


def fault(X, s, kind, rng):
    X = X.copy(); x = X[:, s]; pk = np.abs(x).max(-1, keepdims=True); rms = np.sqrt((x ** 2).mean(-1, keepdims=True))
    if kind == "zero": x[:] = 0
    elif kind == "noise": x[:] = rng.standard_normal(x.shape).astype(np.float32) * rms
    elif kind == "clip": x[:] = np.clip(x, -0.1 * pk, 0.1 * pk)
    elif kind == "spike":
        for i in range(len(x)):
            j = rng.choice(x.shape[-1], 20, replace=False); x[i, j] += 3 * pk[i, 0] * rng.choice([-1, 1], 20)
    elif kind == "drift":
        f = rng.uniform(0.3, 1.0, (len(x), 1)); ph = rng.uniform(0, 2 * np.pi, (len(x), 1))
        x[:] = x + 2 * pk * np.sin(2 * np.pi * f * t + ph)
    X[:, s] = x
    return X


def fault_inputs(Xp):
    """Yields (kind, sensor, faulty waveforms) in a fixed order with a fixed seed."""
    rng = np.random.default_rng(2026)
    for kind in KINDS:
        for s in range(4):
            yield kind, s, fault(Xp, s, kind, rng)


if __name__ == "__main__":
    Xp = np.stack([np.array(e, dtype=np.float32) for e, _ in load_pkl("resources/signal_selected_predict_data.pkl")])
    names, sens, F94, FPS = [], [], [], []
    for kind, s, Xf in fault_inputs(Xp):
        names.append(kind); sens.append(s); F94.append(feats94(Xf)); FPS.append(feats_ps(Xf)); print(kind, s, flush=True)
    np.savez_compressed("resources/faults.npz", F94=np.stack(F94).astype(np.float32), FPS=np.stack(FPS).astype(np.float32),
                        names=np.array(names), sensor=np.array(sens))
    print("saved resources/faults.npz")
