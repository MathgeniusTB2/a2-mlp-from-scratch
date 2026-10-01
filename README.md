# A2: Studying Loss Functions and the Training Criterion (NumPy)

This repository contains the code and results for Assessment Task 2 of the subject
*Advanced Data Analytics Algorithms, Machine Learning*. The object of study is the
**training criterion (loss function)** of a Multi-Layer Perceptron (MLP), implemented entirely
from scratch in NumPy, with no machine-learning framework used for the algorithm itself. The
empirical task is handwritten-digit classification on MNIST.

Five criteria are compared under one fixed network: **cross-entropy**, **squared error**,
**absolute error**, **label smoothing**, and a hand-designed **expected-cost** criterion built
from a $10\times10$ reward matrix.

## Files

| Path | Description |
|---|---|
| `a2_mlp_numpy.ipynb` | Self-contained notebook: setup, data loading, the criteria and their gradients, training, cross-validation, per-epoch training/validation learning curves, final test evaluation and figures. |
| `figures/` | Figures produced by the notebook. |
| `metrics.json` | Machine-readable summary of every headline number. |
| `Daniel_James_Martirosov_24948933_2026_UTS_ML_Journal.pdf` / `journal.pdf` | Written report and implementation log (A2 submission). |
| `slides.pdf` / `a3_slides.pdf` | Presentation slides for A3 (`slides.tex`). |

## Running

The notebook is self-contained. It needs only NumPy and Matplotlib (both preinstalled on Colab),
downloads the four MNIST `idx` files on demand, normalises the data, trains the models and writes
the figures and `metrics.json`. Runtime is about 10 minutes (5 criteria x 5 folds + refits).

Locally, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run jupyter nbconvert --to notebook --execute --inplace a2_mlp_numpy.ipynb
```

## Headline results

Protocol: 5-fold cross-validation on the 60k training set to compare criteria, then each
criterion refit on all 60k and scored once on the official t10k test set, over three seeds.

| Quantity | Value |
|---|---|
| Architecture | 784-128-128-10, tanh, LeCun-normal init |
| Parameters | 118,282 |
| Training | mini-batch 128, Adam, lr 1e-3, 30 epochs |
| Cross-validation accuracy (range) | 0.9720 (my reward) -- 0.9786 (cross-entropy) |
| Test accuracy | label smoothing 0.9809, cross-entropy 0.9799, squared error 0.9785, absolute error 0.9766, my reward 0.9751 |
| Test cross-entropy | squared error 0.0826 (lowest), cross-entropy 0.0927, label smoothing 0.1558 |
| Calibration (ECE) | squared error 0.011 (best), cross-entropy 0.014, label smoothing 0.089 (worst) |
| Gradient norm on the mistakes | cross-entropy 1.24, label smoothing 1.00, squared 0.19, absolute 0.15, my reward 0.06 |
| Majority-class baseline | 0.1135 |
| Loss-objective gap | lowest test CE (squared error) is not the most accurate (label smoothing); label smoothing is most accurate but worst calibrated |
| Confusion difference vs seed noise | max cell 15.7 vs seed-to-seed baseline 15.0 (within noise) |

## Reproducibility

All randomness is seeded. The full protocol (cross-validation, three-seed refit) is reproducible
from the notebook; the dataset downloads automatically when `data/` is absent.

## AI assistance

AI assistance was used to scaffold parts of the code and to help draft and edit the written
report. A convolutional extension suggested earlier was removed because it could not be
defended under questioning. The design of the experiments, the reward matrix, the mathematical
derivations and the interpretation of the results are the author's; the implementation log in
the report PDF documents how the assistance was used and reviewed.
