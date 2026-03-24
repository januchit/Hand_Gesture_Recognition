# CNN-ResBiGRU-SE: Deep Residual Network with Channel Attention for Hand Gesture Recognition via sEMG

[![IEEE Access](https://img.shields.io/badge/Published-IEEE%20Access-blue)](https://doi.org/10.1109/ACCESS.2024.0429000)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green)](https://www.python.org/)
[![TensorFlow 2.17](https://img.shields.io/badge/TensorFlow-2.17.1-orange)](https://www.tensorflow.org/)
[![Google Colab](https://img.shields.io/badge/Run%20in-Google%20Colab-yellow)](https://colab.research.google.com/)

---

## Overview

This repository provides the official implementation of **CNN-ResBiGRU-SE**, a deep learning framework for surface electromyography (sEMG)-based hand gesture recognition (HGR) introduced in:

> **S. Mekruksavanich, N. Hnoohom, and A. Jitpattanakul,**
> *"Deep Residual Network with Channel Attention for Improving Hand Gesture Recognition with Surface Electromyography Signal,"*
> **IEEE Access**, vol. 11, 2024. DOI: [10.1109/ACCESS.2024.0429000](https://doi.org/10.1109/ACCESS.2024.0429000)

The CNN-ResBiGRU-SE architecture integrates three complementary modules:

| Module | Description | Role |
|---|---|---|
| **ConvB Block × 2** | Conv1D → BatchNorm → ReLU (×3, strided) | Spatial feature extraction from multi-channel sEMG |
| **ResBiGRU Block** | BiGRU + residual connection + LayerNorm | Temporal dependency modeling (forward & backward context) |
| **SE Block** | Squeeze-and-Excitation (channel attention) | Adaptive channel reweighting |
| **GAP + Dense** | Global Average Pooling + SoftMax | Classification head |

### Reported Results

| Dataset | Gesture Set | Accuracy (%) | F1-Score (%) |
|---|---|---|---|
| NinaPro-DB1 | Exercise A (12 gestures) | **91.73** | 91.67 |
| NinaPro-DB1 | Exercise B (17 gestures) | **89.72** | 89.75 |
| NinaPro-DB1 | Exercise C (23 gestures) | **85.18** | 85.15 |
| NinaPro-DB1 | A+B+C (52 gestures) | **89.14** | 89.17 |
| NinaPro-DB5 | Exercise A (12 gestures) | **96.93** | 96.93 |
| NinaPro-DB5 | Exercise B (17 gestures) | **95.58** | 95.59 |
| NinaPro-DB5 | Exercise C (23 gestures) | **89.67** | 80.70 |
| NinaPro-DB5 | A+B+C (52 gestures) | **92.59** | 92.60 |

---

## Repository Contents

```
CNN-ResBiGRU-SE-sEMG-HGR/
├── CNN_ResBiGRU_SE_NinaPro_DB1.ipynb   # Notebook for NinaPro-DB1
├── CNN_ResBiGRU_SE_NinaPro_DB5.ipynb   # Notebook for NinaPro-DB5
└── README.md                            # This file
```

---

## Datasets

### NinaPro-DB1

| Property | Value |
|---|---|
| Subjects | 27 non-disabled |
| Device | OttoBock MyoBock 13E200-50 |
| Channels | 10 |
| Sampling rate | 100 Hz |
| Gestures | 52 (+ rest) across 3 exercise groups |
| Repetitions | 10 per gesture |

**Download instructions:**
1. Visit the official NinaPro website: **https://ninapro.hevs.ch/instructions/DB1.html**
2. Register for a free account and request access.
3. Download all subject `.mat` files for Exercises 1, 2, and 3.
4. Organise the files using the structure below:

```
DB1/
├── S1/
│   ├── S1_A1_E1.mat    ← Exercise 1 (12 basic finger movements)
│   ├── S1_A1_E2.mat    ← Exercise 2 (17 hand/wrist movements)
│   └── S1_A1_E3.mat    ← Exercise 3 (23 functional/grasping movements)
├── S2/
│   ├── S2_A1_E1.mat
│   ...
└── S27/
    └── S27_A1_E3.mat
```

> **Key `.mat` variables used in this code:**
> - `emg` — (N × 10) sEMG amplitude signals (OttoBock hardware-processed, 100 Hz)
> - `restimulus` — (N × 1) gesture class labels (0 = rest, 1–52 = gestures)
> - `rerepetition` — (N × 1) repetition number (1–10)

**Paper:** Atzori et al., *"Characterization of a benchmark database for myoelectric movement classification,"* IEEE Trans. Neural Syst. Rehabil. Eng., 23(1):73–83, 2015. [DOI](https://doi.org/10.1109/TNSRE.2014.2328495)

---

### NinaPro-DB5

| Property | Value |
|---|---|
| Subjects | 10 non-disabled |
| Device | 2 × Thalmic Myo armbands |
| Channels | 16 (8 channels per armband) |
| Sampling rate | 200 Hz |
| Gestures | 52 (+ rest) across 3 exercise groups |
| Repetitions | 6 per gesture |

**Armband placement:**
- Armband 1: near the elbow at the radiohumeral joint
- Armband 2: rotated 22.5° toward the wrist from Armband 1

**Download instructions:**
1. Visit the official NinaPro website: **https://ninapro.hevs.ch/instructions/DB5.html**
2. Register for a free account and request access.
3. Download all subject `.mat` files for Exercises 1, 2, and 3.
4. Organise the files as shown:

```
DB5/
├── S1_E1_A1.mat    ← Subject 1, Exercise 1 (12 basic finger movements)
├── S1_E2_A1.mat    ← Subject 1, Exercise 2 (17 hand/wrist movements)
├── S1_E3_A1.mat    ← Subject 1, Exercise 3 (23 functional/grasping movements)
├── S2_E1_A1.mat
...
└── S10_E3_A1.mat
```

> **Key `.mat` variables used in this code:**
> - `emg` — (N × 16) raw differential sEMG (first 8 cols: proximal armband; last 8 cols: distal armband)
> - `restimulus` — (N × 1) gesture class labels (0 = rest, 1–52 = gestures)
> - `rerepetition` — (N × 1) repetition number (1–6)

**Paper:** Pizzolato et al., *"Comparison of six electromyography acquisition setups on hand movement classification tasks,"* PLOS ONE, 12(10):1–17, 2017. [DOI](https://doi.org/10.1371/journal.pone.0186132)

---

## Quick Start (Google Colab)

### Step 1 — Upload dataset to Google Drive
After downloading (see above), upload the `DB1/` or `DB5/` folder to your Google Drive.

### Step 2 — Open the notebook in Colab
Click the badge at the top of this page, or open the `.ipynb` file directly in [Google Colab](https://colab.research.google.com/).

### Step 3 — Set the dataset path
In **Cell 4** of each notebook, update the path variable:

```python
# For DB1:
DB1_PATH = '/content/drive/MyDrive/NinaPro/DB1'   # ← update to your path

# For DB5:
DB5_PATH = '/content/drive/MyDrive/NinaPro/DB5'   # ← update to your path
```

### Step 4 — Configure experiment (optional)
In **Cell 3**, set the exercise subset you wish to run:
```python
EXERCISE = 'E1+E2+E3'   # Full 52-gesture set (default)
# Options: 'E1'  (12 gestures, Exercise A)
#          'E2'  (17 gestures, Exercise B)
#          'E3'  (23 gestures, Exercise C)
#          'E1+E2+E3'  (52 gestures, all combined)
```

### Step 5 — Run all cells
`Runtime → Run all` (or `Ctrl+F9`)

---

## Preprocessing Pipeline

### NinaPro-DB1
The OttoBock MyoBock 13E200-50 sensor applies **internal hardware conditioning** (analog band-pass amplification, full-wave rectification, and RMS smoothing) before the 100 Hz digital output. Accordingly, only the following software steps are applied:
```
raw sEMG (100 Hz) → Zero-mean, unit-variance normalisation (per channel)
```

### NinaPro-DB5
The Thalmic Myo armbands deliver raw differential sEMG at 200 Hz. Software preprocessing:
```
raw sEMG (200 Hz)
  → 4th-order Butterworth bandpass: 20–90 Hz  (Nyquist limit = 100 Hz)
  → IIR notch filter at 50 Hz  (power-line interference suppression)
  → Zero-mean, unit-variance normalisation (per channel)
```
> **Note:** The upper cutoff of 90 Hz is constrained to be below the Nyquist frequency (fs/2 = 100 Hz) as required by the sampling theorem.

---

## Segmentation

A sliding-window approach is used to segment continuous sEMG recordings:

| Dataset | Window size | Stride | Duration | Step |
|---|---|---|---|---|
| NinaPro-DB1 | 20 samples | 5 samples | 200 ms | 50 ms |
| NinaPro-DB5 | 40 samples | 10 samples | 200 ms | 50 ms |

Only windows containing a single, uniform gesture label (no rest frames, no boundary overlap) are retained.

---

## Training Protocol

| Parameter | Value |
|---|---|
| Cross-validation | 5-fold, **repetition-level** partitioning within each subject |
| Optimizer | Adam |
| Learning rate | 0.001 |
| β₁ / β₂ / ε | 0.9 / 0.999 / 1×10⁻⁸ |
| Loss | Categorical cross-entropy + L2 regularisation |
| Max epochs | 200 |
| Early stopping | patience = 20 (monitors `val_accuracy`) |
| Batch size | 32 |
| Platform | Google Colab Pro+ (NVIDIA Tesla L4 GPU) |

> **Cross-validation note:** Folds are assigned at the **gesture repetition level** before windowing. All overlapping windows derived from a given repetition belong exclusively to either the training partition or the test partition, preventing data leakage from temporally correlated adjacent windows.

---

## Environment

```
Python         3.10
TensorFlow     2.17.1
Keras          2.5.0
NumPy          1.26.4
Pandas         2.2.2
SciPy          1.11+
scikit-learn   1.5.2
Matplotlib     3.8+
Seaborn        0.13+
```

To install all dependencies in Colab, the first cell of each notebook runs:
```bash
pip install scipy scikit-learn seaborn tensorflow --quiet
```

---

## Citation

If you use this code or the CNN-ResBiGRU-SE model in your research, please cite:

```bibtex
@article{mekruksavanich2024cnn,
  title   = {Deep Residual Network with Channel Attention for Improving
             Hand Gesture Recognition with Surface Electromyography Signal},
  author  = {Mekruksavanich, Sakorn and Hnoohom, Narit and Jitpattanakul, Anuchit},
  journal = {IEEE Access},
  volume  = {11},
  year    = {2024},
  doi     = {10.1109/ACCESS.2024.0429000}
}
```

---

## License

This code is released for academic and research purposes.
The NinaPro datasets are subject to their own terms of use; please refer to the [NinaPro website](https://ninapro.hevs.ch/) for details.

---

## Contact

**Corresponding author:** Anuchit Jitpattanakul
Department of Mathematics, Faculty of Applied Science, King Mongkut's University of Technology North Bangkok
Email: anuchit.j@sci.kmutnb.ac.th
