# CRR: check–repair–remove sensor triage for fault-tolerant microseismic event classification

Code, trained models and fault benchmark for the paper

> J. Zhang, Q. Zhang, Y. Fan, J. Zhang, Z. Wu, Y. Wang.
> *CRR: a check–repair–remove sensor triage framework for fault-tolerant microseismic event classification.*
> Submitted to Computers & Geosciences, 2026.

Mine microseismic classifiers label each event (microseismic, blast, noise) from a few sensors. This repository shows what dead and faulty sensors do to such classifiers, and provides a remedy that needs no fault labels and no retraining:

1. **Check** every sensor with four signal tests (spikes, baseline drift, mains hum, missing data). Thresholds are the 99.5th percentile of each test on clean validation records.
2. **Pass, repair or remove.** A healthy sensor passes unchanged; a recoverable one is repaired (de-spiking, de-trending and 5 Hz high-pass, 50/100/150 Hz notches, gaps filled at the sensor's own noise floor); a sensor with nothing to recover is removed from the set, never replaced by zeros.
3. **Classify** with SensorSet-SincNet, a multi-stream network over a *set* of sensors in which a removed sensor has exactly no influence on the output.

## Contents

| File | Purpose |
|---|---|
| `repair.py` | Sensor triage: `calibrate`, `detectors`, `repair` |
| `model_set.py` | SensorSet-SincNet, ReSeT and Deep Sets fusion |
| `model_plus.py`, `frontends.py`, `baselines.py` | MS²-SincNet (fixed-channel reference) and baseline classifiers |
| `network/` | Published MPA-SincNet network (third party, see `NOTICE.md`) |
| `feats94.py`, `feats_ps.py` | Event descriptors: 94 per array, 47 per sensor |
| `faults2.py`, `make_faults.py` | Fault benchmark: seven fault types at five severities |
| `train_paper.py`, `train_plus.py` | Training |
| `*_eval.py` | Evaluation scripts, one per experiment (table below) |
| `quick_check.py` | Short check of the released models and of triage |
| `outputs/paper/`, `outputs/plus/` | Trained models (five seeds each) and the result files behind the tables |
| `resources/` | Descriptor files and the published checkpoint |

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt
```

A GPU is used when available. All evaluation scripts also run on a CPU; training was done on one NVIDIA RTX 3070 Ti Laptop GPU (8 GB).

## Data

The waveforms are the public release of Peng et al. (Measurement 270, 2026): https://doi.org/10.5281/zenodo.15654539 (CC BY 4.0). Download `Data&Network_MPA_SincNet.zip` and copy two files into `resources/`:

```
resources/signal_selected_train_data.pkl      1500 events (1200 training / 300 validation, random_state=42)
resources/signal_selected_predict_data.pkl    578 independent events
```

Each event holds the four sensors with the highest signal-to-noise ratio, 10 000 samples at 10 kHz, in decreasing SNR order. Labels: 0 microseismic, 1 blast, 2 noise.

## Quick check

```
python quick_check.py
```

loads the released ensembles and prints the accuracy on the 578 independent events for complete records, for the highest-SNR sensor dead, and for twenty spikes on every sensor as recorded and after triage. Measured with the released models (about ten minutes on a CPU, much less on a GPU). The spike positions are drawn with a different random seed from the paper, so these two rows differ from Table 5 by a few tenths of a point:

| Condition | SensorSet-SincNet | MS²-SincNet |
|---|---|---|
| Complete records | 96.5 % | 97.9 % |
| Sensor 1 dead | 96.4 % | 56.6 % |
| 20 spikes on every sensor, as recorded | 55.7 % | 38.6 % |
| 20 spikes on every sensor, after triage | 97.1 % | 97.2 % |

## Using triage on your own records

```python
import numpy as np, repair

# X: (n_events, n_sensors, 10000) float32 at 10 kHz
thr = repair.calibrate(X_clean_validation)      # once, on records known to be healthy
X_out, actions = repair.repair(X, thr)

# actions[i, s] is a bit mask: 1 de-spiked, 2 drift removed, 4 hum notched, 8 removed, 16 gaps filled
alive = np.abs(X_out).max(-1) > 0               # removed sensors are identically zero; the set model skips them
```

The repair step can be placed in front of any classifier. Removal is safe only with a classifier that tolerates a missing sensor, such as the set model here.

## Reproducing the paper

Descriptor files that are too large for the repository are regenerated first (a few minutes each):

```
python make_mask_feats.py      # resources/feats94_masks.npz, descriptors for every dead-sensor subset
python make_faults.py          # resources/faults.npz, single-sensor fault benchmark
```

Evaluation with the released models (no training):

| Script | Result | In the paper |
|---|---|---|
| `robust_eval.py` | Complete records, all 14 dead-sensor subsets, added noise | Tables 2–4, Figs. 6–9 |
| `fault_eval.py`, `severity_eval.py` | One faulty sensor, seven fault types, five severities | Section 5.5 |
| `scan_eval.py`, `scan2_eval.py` | The same disturbance on all sensors, as recorded and repaired | Table 5, Fig. 10 |
| `repair_eval.py` | Repair against removal, scarce sensors, real zero-padded events | Section 5.7 |
| `final_eval.py` | Field-like mixture, unseen fault variants, timing | Section 5.7, Supplement S4 |
| `ablation_eval.py` | Triage against always-on filtering, each step left out, thresholds | Table 6 |
| `stress_eval.py`, `csp_eval.py`, `locate_eval.py`, `diag_eval.py SETN` | Learned alternatives | Section 5.8, Supplement S1–S2 |

Each script writes a JSON file to `outputs/paper/`; the files produced for the paper are included.

Training from scratch (five seeds, `--seed 0` to `4`; about 15 minutes per run on the GPU above):

```
# SensorSet-SincNet, base model
python train_paper.py --tag SET0 --arch set --aug --feat --seed 0
# SensorSet-SincNet, final model: sensor removal and noise-aware training
python make_noisy_feats.py
python train_paper.py --tag SETN --arch set --aug --feat --chdrop 0.5 --noise_p 0.5 --noisy_feats resources/noisy_feats.npz --seed 0
# MS2-SincNet, fixed-channel reference
python train_plus.py --tag F4_pi --aug --feat --spec --bilstm --feats resources/handcrafted_feats_v2.npz --seed 0
# MS2-SincNet with sensor removal and noise-aware training
python train_paper.py --tag F4cdN --aug --feat --spec --bilstm --chdrop 0.5 --noise_p 0.5 --noisy_feats resources/noisy_feats.npz --seed 0
# Deep Sets fusion, ReSeT, fault-aware training
python train_paper.py --tag DSN --arch deepset --aug --feat --chdrop 0.5 --drop_max 3 --noise_p 0.5 --noisy_feats resources/noisy_feats.npz --seed 0
python train_paper.py --tag RST --arch reset --aug --feat --chdrop 0.5 --drop_max 3 --noise_p 0.5 --noisy_feats resources/noisy_feats.npz --scd 1.0 --seed 0
python make_fault_feats.py
python train_paper.py --tag SETF --arch set --aug --feat --chdrop 0.5 --drop_max 3 --noise_p 0.5 --noisy_feats resources/noisy_feats.npz --fault_p 0.3 --seed 0
# baselines: crnn, resnet1d, speccnn, transformer
python train_paper.py --tag BL_crnn --arch crnn --aug --seed 0
```

Protocol: the model of each run is the epoch with the lowest validation loss; the 578 independent events are never used for model selection; reported accuracies are five-model ensembles; paired comparisons use exact McNemar tests.

## Licence

The code written for this repository is released under the MIT licence (`LICENSE`). The files in `network/` and `resources/checkpoint.pth` come from the MPA-SincNet release and keep its CC BY 4.0 licence (`NOTICE.md`).

## Citation

Please cite the paper above, and the data source: Peng et al., *Multi-sensor microseismic event classification with SNR-based sensor selection and MPA-SincNet model*, Measurement 270 (2026) 120934.
