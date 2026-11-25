"""Quick check of the released models and of sensor triage on the 578 independent events (no training, CPU is enough).

Prints the ensemble accuracy of SensorSet-SincNet (tag SETN) and MS2-SincNet for
  1. complete records, 2. the highest-SNR sensor dead (set to zero),
  3. twenty spikes on every sensor, as recorded and after check-repair-remove triage.
Measured with the released models: SETN 96.5 / 96.4 / 55.7 -> 97.1 %;  MS2-SincNet 97.9 / 56.6 / 38.6 -> 97.2 %.
About ten minutes on a CPU.
usage: python quick_check.py        (EVAL_QUICK=100 python quick_check.py  for the first 100 events only)
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from evalcommon import Xp, Xt, ite, load_models, feats_for, ens_acc
from feats94 import feats94
from feats_ps import feats_ps
from faults2 import fault_ch
import repair as RP

nets = load_models(["SETN", "MS2-SincNet"]); assert nets, "no checkpoints found in outputs/"
thr = RP.calibrate(Xt[ite])                                   # thresholds from the clean validation split, no fault labels
rng = np.random.default_rng(0)
dead = Xp.copy(); dead[:, 0] = 0
spk = Xp.copy()
for s in range(4): spk[:, s] = fault_ch(spk[:, s], "spike", 20, rng)
spk_r, actions = RP.repair(spk, thr)
print("sensors repaired: %.0f %%, removed: %.0f %%" % (100 * (actions > 0).mean(), 100 * ((actions & 8) > 0).mean()))
for label, X in (("complete records", Xp), ("sensor 1 dead", dead), ("20 spikes on every sensor, as recorded", spk), ("20 spikes on every sensor, after triage", spk_r)):
    X = np.ascontiguousarray(X, np.float32); F94, FPS = feats94(X), feats_ps(X)
    print("%-42s" % label, "  ".join("%s %.1f %%" % (m, 100 * ens_acc(ms, X, feats_for(a, X, F94, FPS))) for m, (ms, a) in nets.items()), flush=True)
