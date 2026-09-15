# Uncertainty-Aware Distance Predictor

A vision model that looks at an image, predicts how far away an object is,
and — unlike a typical regressor — reports **how confident it is** in that
prediction. Four different models implement that second part in four
different ways, so they can be trained and compared side by side.

This README covers what each model does, how to run them, and why the key
parameters are set the way they are. For the full build log — every bug
hit, every fix tried, what worked and what didn't — see
[SESSION_NOTES.md](SESSION_NOTES.md). That's where the detail lives; this
file is meant to stay short.

## Setup

```bash
pip install -r requirements.txt
python main.py   # trains the default model, see below
```

`torch`/`torchvision` need a CUDA build to use a GPU — install from
PyTorch's own index rather than plain PyPI if you have one:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## The task

Given a 224×224 grayscale image containing one shape, predict a distance
in `[1, 500]`. `dataset.py` generates these on the fly: one fixed,
irregular shape, randomly rotated/scaled/positioned per sample, where
**scale encodes distance** (closer = bigger), same as monocular size cues
in real vision. The shape's radius is clamped to a pixel range, so past a
certain distance it stops shrinking — meaning two genuinely different true
distances can render as visually identical images. That's not a bug, it's
what makes "how confident are you?" a real, non-trivial question for this
task rather than a purely academic one.

Every model predicts uncertainty split into two kinds:

- **Aleatoric** — the image itself is ambiguous (like the radius-clamp
  case above). No amount of extra training removes this.
- **Epistemic** — the model hasn't seen enough like this yet. More/better
  training data *can* fix this.

## The four models

| Model | How it gets "how sure" | Cost per prediction |
|---|---|---|
| `DistanceNetMobileNet` (`dropout`) | Randomly switch off some neurons and ask 30 times; spread of the answers = uncertainty. | 30 forward passes |
| `DistanceNetMobileNetVariational` (`variational`) | Every connection in the head is a learned *range* of values, not one fixed number (hand-rolled Bayes by Backprop). Ask 30 times, sampling fresh values each time. | 30 forward passes |
| `DistanceNetMobileNetBlitz` (`blitz`) | Same idea as above, via the [BLiTZ](https://github.com/piEsposito/blitz-bayesian-deep-learning) library instead of hand-rolled code. | 30 forward passes |
| `DistanceNetMobileNetDER` (`der`) | Predicts a built-in confidence report directly in one shot (Deep Evidential Regression, via [TorchUncertainty](https://github.com/ENSTA-U2IS-AI/torch-uncertainty)) — nothing to sample. | 1 forward pass |

All four share the same frozen, ImageNet-pretrained MobileNetV3-Small
backbone (`torchvision`) — only the head and the training loss differ.
The backbone is frozen (not fine-tuned) because a few thousand synthetic
images isn't enough to usefully retrain it, and ImageNet features already
capture the general shape/edge detectors this task needs.

```bash
python main.py --model dropout       # default
python main.py --model variational
python main.py --model blitz
python main.py --model der

python inference.py                  # loads distance_net.pt, saves predictions.png
python compare_uncertainty.py        # compares every trained checkpoint on the same images
```

Each `--model` saves to its own checkpoint file (`distance_net.pt`,
`distance_net_variational.pt`, etc.), so training one doesn't overwrite
another — train any subset and `compare_uncertainty.py` picks up whichever
checkpoints exist.

### `dropout` — why the parameters are set the way they are

- **`Dropout(0.2)`** in the head does double duty: an ordinary
  regularizer during training, and the source of randomness for the
  30-pass uncertainty estimate at inference (kept active via `model.train()`
  in `inference.py`, even though we're not "training").
- **Trained with Beta-NLL, not plain MSE or plain Gaussian NLL.** Plain
  NLL lets the model cheat by inflating its variance instead of improving
  its mean — a well-documented failure mode. Beta-NLL removes that
  shortcut structurally. (Full derivation: SESSION_NOTES.md, Obstacles A-B.)
- **`log_var` is bounded with a smooth `tanh`, not `torch.clamp`.** A hard
  clamp has exactly zero gradient once a value hits the boundary, so a
  saturated prediction can never recover. This happened in practice
  (SESSION_NOTES.md, Obstacle C) before the fix.
- **`MC_SAMPLES = 30`** (`inference.py`) is just how many stochastic
  passes get averaged — more is smoother but costs more compute at
  prediction time.

**Status: working well.** Both uncertainty types vary sensibly across
images.

### `variational` — why the parameters are set the way they are

- **`prior_sigma = 1.0`.** Controls how strongly every weight gets pulled
  back toward zero. Started at `0.1`, which stalled the mean's ability to
  learn a real function of the input for dozens of epochs — loosening it
  fixed that.
- **`kl_weight = 0.1`** scales the KL-divergence regularization term
  relative to the data-fit loss.
- **`grad_clip_norm = 5.0`** was needed to stop a single large gradient
  step from throwing training off course — added after the mean branch
  showed real instability.

**Status: mean and epistemic uncertainty both work; aleatoric uncertainty
is currently broken** (collapses to maximum confidence regardless of the
image). Three different fixes were tried and ruled out — see
SESSION_NOTES.md, Obstacles D-E, for the full investigation and what's
likely to actually fix it.

### `blitz` — why the parameters are set the way they are

- Uses BLiTZ's own default prior/init hyperparameters
  (`prior_sigma_1=0.1`, `prior_sigma_2=0.4`, `prior_pi=1`,
  `posterior_rho_init=-7.0`) rather than tuned values — an attempt to
  loosen them to fix a known issue (below) made things considerably
  worse, so they're left at the library's defaults. See
  SESSION_NOTES.md, Obstacle F.
- **`elbo_sample_nbr = 3`**: BLiTZ's `sample_elbo()` averages the loss
  over this many independent weight samples per training step, instead
  of one. This is *why* this model doesn't have the same broken aleatoric
  branch as `variational` above — direct, if indirect, confirmation of
  what was causing that collapse.

**Status: mean and aleatoric uncertainty both work; epistemic uncertainty
is under-informative** (near-zero regardless of the image) with BLiTZ's
current defaults.

### `der` — why the parameters are set the way they are

- No dropout, no weight sampling — the head predicts four numbers
  (`loc`, `lambda`, `alpha`, `beta`) in one deterministic pass, and both
  uncertainty types come out as closed-form formulas on those numbers.
- **`min_alpha = 0.05`, `min_lmbda = 0.05`** (raised from the library's
  default of `1e-6`). Both uncertainty formulas divide by these values,
  and they naturally get pushed toward their floor during training — a
  floor of `1e-6` turned that into astronomically large, useless variance
  estimates. Raising the floor doesn't stop the pushing, it bounds how
  bad the result can be.
- **`reg_weight`** controls how strongly the model is penalized for being
  confident while wrong. Despite first appearances, this parameter did
  **not** turn out to control whether uncertainty collapses to its floor
  — see SESSION_NOTES.md, Obstacles G-H, before assuming it's the lever
  to tune here.

**Status: best point predictions of the four, uncertainty still not
right.** Trains/predicts noticeably faster than the other three (one pass
instead of thirty). `alpha`/`beta` now vary per image instead of one
constant value — real progress — but `lmbda`, the parameter epistemic
uncertainty specifically depends on, still collapses to its floor for
nearly every image, and it's not yet confirmed whether the resulting
uncertainty actually correlates with how wrong a given prediction is.
See SESSION_NOTES.md, Obstacles G-H and Part 7, for the full story
including a read of the raw numbers that corrected an earlier
too-optimistic conclusion.

## Project structure

| File | Role |
|---|---|
| `dataset.py` | Generates synthetic (image, true distance) pairs on the fly |
| `model.py` | All four model classes, sharing one frozen MobileNetV3-Small backbone |
| `variational.py` | Hand-rolled `BayesianLinear` (Bayes by Backprop) layer, used by the `variational` model |
| `train.py` | Training loop; branches per model on which loss/training scheme it needs |
| `main.py` | Entry point — `python main.py --model {dropout,variational,blitz,der}` |
| `inference.py` | Loads a checkpoint and runs uncertainty-aware predictions |
| `compare_uncertainty.py` | Runs any trained subset of the four models on the same images, side by side |

## Further reading

[SESSION_NOTES.md](SESSION_NOTES.md) has the complete story: every
obstacle hit, the reasoning behind each fix (including the ones that
didn't work), the full uncertainty-decomposition math, and the
alternatives considered and rejected (deep ensembles, quantile
regression, conformal prediction) with why.
