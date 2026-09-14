# Uncertainty-Aware Distance Predictor

A vision model that looks at an image, predicts how far away an object is,
and — unlike a typical regressor — reports **how confident it is** in that
prediction, split into two distinct kinds of uncertainty.

This document explains how the pieces fit together, exactly how the
uncertainty numbers are calculated, and why this particular approach was
chosen over the alternatives. For the blow-by-blow story of the bugs hit
while building it, see [SESSION_NOTES.md](SESSION_NOTES.md).

## Contents

- [Overview](#overview)
- [The dataset](#the-dataset)
- [The models](#the-models)
- [Training: the loss function](#training-the-loss-function)
- [How to run it](#how-to-run-it)
- [Getting the uncertainty, step by step](#getting-the-uncertainty-step-by-step)
- [Full variational layers (Bayes by Backprop)](#full-variational-layers-bayes-by-backprop)
- [Why this approach works](#why-this-approach-works)
- [Alternatives, and why they're more or less suitable](#alternatives-and-why-theyre-more-or-less-suitable)
- [Current status and limitations](#current-status-and-limitations)

## Overview

| File | Role |
|---|---|
| `dataset.py` | Generates synthetic (image, true distance) pairs on the fly |
| `model.py` | `DistanceNet` (small CNN) and `DistanceNetMobileNet` (frozen pretrained backbone + head), both outputting `(mean, log_var)` |
| `train.py` | Training loop + `BetaNLLLoss` |
| `main.py` | Entry point: trains a model (`--model dropout` or `--model variational`) and saves a checkpoint |
| `inference.py` | Loads a checkpoint and runs uncertainty-aware predictions, independent of training |
| `variational.py` | `BayesianLinear`: a full variational (Bayes by Backprop) layer, the alternative epistemic mechanism to MC Dropout |
| `compare_uncertainty.py` | Runs both trained checkpoints on the same images and prints their uncertainty estimates side by side |

The task: given a 224×224 grayscale image containing one shape, predict a
scalar distance in `[1, 500]`. Instead of one number, the model predicts
**two** — a mean and a variance — and combines that with multiple stochastic
forward passes at inference time to produce a full uncertainty breakdown.

## The dataset

`dataset.py` generates images on the fly rather than loading a real
photo dataset — free, unlimited, exactly-labeled data, at the cost of
realism.

- **One fixed, irregular polygon** (`BASE_SHAPE`) is used for every sample.
  Each sample places a rotated, scaled, translated copy of that *same*
  shape onto a noisy gray background. Rotation and position are random;
  **scale encodes distance** — closer objects render bigger, exactly like
  monocular size cues in real vision.
- Because the shape is fixed and only its pose varies, the model must learn
  to recognize *this specific object* under arbitrary 2D transforms, not
  just "any blob." That's also a harder problem for a CNN specifically: a
  `Conv2d` is translation-invariant by construction (a shifted input
  produces a shifted, otherwise-identical output) but **not**
  rotation-invariant — the network has to learn that a rotated silhouette
  is still the same object.
- Size falls off as `1/sqrt(distance)` rather than the true `1/distance`.
  The true physical projection would shrink far-away objects to
  sub-pixel size well before distance 500, making them literally
  unrecoverable from the image. The gentler falloff is a deliberate
  simplification that keeps the *entire* 1–500 range visually
  distinguishable, in exchange for not being physically exact.
- Radius is clamped to `[MIN_RADIUS, MAX_RADIUS]` pixels, which is *why*
  aleatoric uncertainty is real and unavoidable here, not just theoretical:
  past a certain distance the object saturates at `MIN_RADIUS` and stops
  shrinking, so two different true distances near the far end of the range
  can produce visually near-identical images. No amount of training data
  or model capacity removes that ambiguity — it's baked into the task.

## The models

Both models share the same output contract: `forward(x) -> (mean, log_var)`.

**`DistanceNet`** — a from-scratch baseline: three
`Conv2d → ReLU → MaxPool2d` blocks (channels 16 → 32 → 64), then
`AdaptiveAvgPool2d(1) → Linear(64,64) → ReLU → Dropout(0.2) → Linear(64,2)`.
The adaptive pool (rather than a hardcoded flatten size) means the head
works at any input resolution. Stacking multiple conv layers lets the
network build a hierarchy — layer 1 learns edges, layer 2 combines edges
into outlines, layer 3 combines outlines into "there's a solid blob, and
here's roughly how big it is" — which a single conv layer can't do in one
step.

**`DistanceNetMobileNet`** (used by `main.py`) — transfer learning on
`torchvision`'s MobileNetV3-Small, pretrained on ImageNet:

- The convolutional feature extractor (`backbone.features`) is **frozen**
  (`requires_grad = False`) and used purely as a fixed feature extractor.
  Only a small head (`Linear(576,64) → ReLU → Dropout(0.2) → Linear(64,2)`)
  is trained.
- `train()` is overridden so that even when the whole module is put in
  training mode, `self.backbone.eval()` is force-called — this keeps the
  backbone's `BatchNorm` layers using their frozen ImageNet running
  statistics instead of drifting toward statistics computed from our small
  batch of noisy synthetic grayscale images.
- Grayscale input is repeated to 3 channels and normalized with ImageNet's
  mean/std inside `forward()`, so callers can keep feeding plain `[0, 1]`
  grayscale tensors.
- This is the architecture real "distance from a photo" systems actually
  tend to use: a backbone that already knows general visual features
  transfers better than training a small CNN from random weights on only a
  few thousand images.

Both heads end in `Linear(_, 2)`, unbound into `pred_mean` and a raw
`log_var` that then passes through `_soft_clamp_log_var`.

### Why predict `log_var` instead of `var` (or `std`) directly

Variance must be positive. Predicting `log_var` and exponentiating it
(`var = exp(log_var)`) gets positivity for free from an unconstrained
linear output — no constrained activation function needed on that branch.

### Why a smooth `tanh` bound instead of `torch.clamp`

`log_var` is still bounded to `[LOG_VAR_MIN, LOG_VAR_MAX]` to avoid
`exp()` overflow/underflow, but via:

```python
LOG_VAR_MIN + 0.5 * (LOG_VAR_MAX - LOG_VAR_MIN) * (tanh(raw) + 1.0)
```

instead of `torch.clamp`. A hard clamp has **exactly zero gradient** once a
value saturates at the boundary — if a prediction ever drifts past the
ceiling, no gradient can ever pull it back, no matter how much more
training happens. This is not a theoretical concern: it's exactly what
happened during development (see Obstacle C in
[SESSION_NOTES.md](SESSION_NOTES.md)) — every prediction reported
*identical* maximum variance because `log_var` had saturated at its clamp
and could never recover. `tanh` squashes into the same range but always
has a small, nonzero gradient, so a saturated value stays correctable.

`LOG_VAR_MAX` is calibrated from the data itself — two log-units above
`log((MAX_DISTANCE - MIN_DISTANCE)^2)`, the worst-case squared error a
prediction could ever have on this task — rather than an arbitrary fixed
number, so the ceiling doesn't bind for ordinary (if inaccurate)
predictions.

## Training: the loss function

`train.py` trains with **Beta-NLL** (Seitzer et al., 2022), not MSE and not
plain Gaussian NLL.

**Why not plain MSE?** MSE only supervises the mean. There is no `log_var`
target to train against, so a model trained with MSE has no signal at all
for how confident to be — you'd have no aleatoric estimate to report.

**Why not plain Gaussian NLL?** The Gaussian negative log-likelihood is:

```
NLL = 0.5 * log(var) + (target - mean)^2 / (2 * var)
```

This lets the model minimize loss two ways: make `mean` more accurate, or
make `var` bigger (which shrinks the second term). Early in training, when
`mean` is still close to random, the second route is *cheaper* — inflate
`var` instead of doing the actual work of improving `mean`. This is
self-reinforcing: the gradient into `mean` is proportional to `1/var`, so
once `var` explodes, that same gradient collapses toward zero and `mean`
gets permanently stuck. This is a well-documented failure mode of
heteroscedastic regression, and it's exactly the collapse this project hit
twice — once from a cold start, and again at the handoff from an MSE
warm-up phase (see Obstacles A and B in SESSION_NOTES.md).

**Beta-NLL fixes it structurally.** `BetaNLLLoss` multiplies the
per-sample NLL by `var.detach() ** beta` — a term that affects the loss's
*magnitude* but, because it's detached (`.detach()`, stop-gradient), never
contributes to the backward pass:

```python
weight = var.detach() ** beta
loss = (weight * nll).mean()
```

With `beta = 1`, the gradient into `mean` becomes mathematically identical
in shape to plain MSE's gradient — completely decoupled from whatever
variance the model currently predicts, so inflating `var` no longer helps
shrink the loss along that axis. Meanwhile `log_var` still receives a
normal, well-behaved gradient pushing it toward the true residual scale.
This removes the exploit at its root instead of just delaying it (which is
what the earlier MSE-warm-up band-aid did), so there's no fragile
before/after transition left for training to break at.

## How to run it

```bash
# Train DistanceNetMobileNet (MC Dropout) and save a checkpoint (distance_net.pt)
python main.py

# Train DistanceNetMobileNetVariational (Bayes by Backprop) instead,
# saved separately as distance_net_variational.pt
python main.py --model variational

# Load a checkpoint (distance_net.pt by default) and run uncertainty-aware
# predictions, saving an annotated grid to predictions.png
python inference.py

# With both checkpoints trained, compare their epistemic estimates on the
# same images
python compare_uncertainty.py
```

Training and inference are deliberately separate scripts: `main.py` trains
and checkpoints; `inference.py` loads a checkpoint and predicts, without
ever retraining. The checkpoint records which model class it belongs to
(`MODEL_REGISTRY`), so `inference.py` can reconstruct the right
architecture without the caller having to specify it.

## Getting the uncertainty, step by step

Call `predict_with_uncertainty(model, device, image)` in `inference.py`.
For one input image, it:

1. Puts the model in `.train()` mode (`model.train()`), which re-enables
   the `Dropout` layer in the head — but the frozen `DistanceNetMobileNet`
   backbone stays in `eval()` regardless, because of its `train()`
   override, so only the head becomes stochastic.
2. Runs the **same image** through the model `MC_SAMPLES = 30` times.
   Because dropout randomly zeroes different head neurons each pass, each
   pass yields a slightly different `(mean_i, log_var_i)`.
3. Converts each pass's `log_var_i` to `var_i = exp(log_var_i)`.
4. Computes, across the 30 passes:

```python
epistemic_var = variance_of(means)         # spread of the point estimates
aleatoric_var  = mean(variances)           # average of the model's own per-pass uncertainty
total_var      = epistemic_var + aleatoric_var
```

5. Reports `mean` (the average of the 30 means) plus
   `epistemic_std`, `aleatoric_std`, `total_std` — square roots of the
   above, in the same units as distance.

This is the standard decomposition from Kendall & Gal, *"What Uncertainties
Do We Need in Bayesian Deep Learning for Computer Vision?"* (2017):

- **Aleatoric uncertainty** — irreducible noise/ambiguity *in the input
  itself*. This is what a single forward pass's `log_var` head estimates
  directly, and it's real by construction in this dataset: past a certain
  distance the rendered object saturates at `MIN_RADIUS`, so genuinely
  different true distances can render as visually near-identical images.
  More training data does **not** remove this.
- **Epistemic uncertainty** — uncertainty *about the model's own weights*:
  "I haven't seen enough inputs like this to be sure." This is what the
  spread of `mean_i` across the 30 stochastic passes estimates — if the
  head's weights were well-pinned-down by training for this kind of input,
  dropping different random subsets of neurons wouldn't move the
  prediction much; if they're not, it will. This kind of uncertainty *is*
  reducible with more or better training data.
- Summing the variances (not the standard deviations) to combine them
  follows directly from both being modeled as independent Gaussian
  contributions to the total predictive variance.

## Full variational layers (Bayes by Backprop)

`variational.py` implements the alternative flagged in the alternatives
table above: instead of approximating epistemic uncertainty with dropout
masks, `DistanceNetMobileNetVariational` (in `model.py`) gives every
weight in the head an actual learned Gaussian *distribution*, following
Blundell et al., 2015, *"Weight Uncertainty in Neural Networks."*

### How it works

**MC Dropout's epistemic estimate is a proxy.** A dropout mask randomly
zeroes out neurons; this happens to behave, approximately, like sampling
from an implicit distribution over network configurations, but that
distribution was never explicitly learned or chosen — dropout's rate
(0.2 here) is a fixed regularization hyperparameter, not a fitted spread.

**Bayes by Backprop makes the distribution the actual thing being
learned.** `BayesianLinear` (a drop-in replacement for `nn.Linear`) stores
two numbers per weight instead of one:

- `mu` — the distribution's mean, initialized the same way an ordinary
  `Linear` layer's weights would be.
- `rho` — an unconstrained number that, passed through `softplus`,
  becomes `sigma`, the distribution's spread. `softplus` (rather than
  e.g. clamping a raw value to a positive floor) is used for the same
  reason `log_var` uses a soft `tanh` bound elsewhere in this project: it
  keeps a smooth, never-exactly-zero gradient, so a parameter can't get
  permanently stuck the way `log_var` did when it was clamped with
  `torch.clamp` (see the model.py docstring and Obstacle C in
  SESSION_NOTES.md).

Every forward pass then draws a *fresh* weight via the **reparameterization
trick**:

```
w = mu + sigma * eps,   eps ~ N(0, 1)
```

`eps` is pure randomness with no learnable parameters in it, so
backpropagation can still flow into `mu` and `sigma` even though `w`
itself is a random sample — this is what makes the distribution
trainable via ordinary gradient descent rather than needing something
like MCMC. Because sampling happens on *every* call, this layer is
stochastic in both training and eval mode — unlike `nn.Dropout`, which
only randomizes in training mode.

### The KL term, and why sigma needs one

With nothing else in the loss, `sigma` has a one-directional incentive:
shrink toward zero, since an (effectively) deterministic layer fits
training data at least as well as a noisy one, every time. Bayes by
Backprop counters this with a KL-divergence penalty pulling each weight's
learned distribution `q(w) = N(mu, sigma^2)` back toward a fixed prior
`p(w) = N(0, prior_sigma^2)`:

```
KL(q || p) = log(prior_sigma / sigma) + (sigma^2 + mu^2) / (2 * prior_sigma^2) - 0.5
```

(closed-form because both distributions are Gaussian). `BayesianLinear.
kl_divergence()` sums this over every weight and bias; the model's own
`kl_divergence()` sums it again over every `BayesianLinear` layer in the
head. `train.py` adds `kl_weight * kl / len(train_loader)` to the loss —
dividing by the number of minibatches per epoch so that, summed over one
full epoch, the KL term's total contribution matches a single
`kl_divergence()` call rather than being counted once per minibatch
(the standard Blundell et al. weighting, simplified from their original
per-batch schedule to a flat average).

The overall objective is the standard variational free energy /
evidence-lower-bound (ELBO) tradeoff: fit the data (the NLL term) while
staying as close as the data allows to a simple prior (the KL term) —
conceptually the same fit-vs-regularize tension as any weight-decay
penalty, except here it regularizes an entire *distribution* per weight,
not just its mean.

**A subtlety this project already learned the hard way with `log_var`:**
initial *scale mismatches* between two competing loss terms can dominate
training before it even starts. `BayesianLinear` initializes `sigma` to
start **equal to `prior_sigma`**, not to some small arbitrary value. Since
`KL` contains a `log(prior_sigma / sigma)` term, starting with `sigma`
far below the prior would create a large, entirely artificial KL penalty
from the very first step — before a single gradient update reflects
anything about the actual data — which would then dominate and fight the
data-fit term early in training. Starting `sigma` at the prior's scale
keeps that ratio at zero, so the model starts KL-neutral and only the
data decides which weights need to move away from the prior, and by how
much.

### Getting uncertainty out of it

`inference.py`'s `predict_with_uncertainty()` already handles both model
types through the same interface: it checks `hasattr(model,
"kl_divergence")` to tell a variational model from a dropout-based one,
calls `model.eval()` for a variational model (its `BayesianLinear` layers
sample regardless of mode, so `eval()` only needs to keep `BatchNorm`/the
frozen backbone deterministic) versus `model.train()` for a dropout model,
and then runs the same `MC_SAMPLES`-pass loop either way — the
mean/aleatoric/epistemic decomposition described earlier is identical for
both, only the *source* of the per-pass randomness differs (weight
sampling vs. dropout masks).

### Comparing the two

Train both, then compare them on the same images:

```bash
python main.py                      # -> distance_net.pt (MC Dropout)
python main.py --model variational  # -> distance_net_variational.pt (Bayes by Backprop)
python compare_uncertainty.py       # prints both models' mean/epistemic/aleatoric side by side
```

What to look for: whether the two methods agree on *which* images are
epistemically uncertain (a sign the epistemic signal is measuring
something real about the input, not an artifact of one specific
mechanism), and whether the variational model's epistemic estimates shift
more sensibly as more training data is added — since, unlike a fixed
0.2 dropout rate, `sigma` per weight is something the model actually
fits.

### What actually happened when this was run: a real, reproducible collapse

This is worth documenting in the same spirit as the three obstacles in
SESSION_NOTES.md: not a bug in the usual sense, but a genuine failure mode
of this technique that's worth understanding rather than papering over.

**Symptom.** The first training run (`prior_sigma=0.1`, `kl_weight=1.0`,
`beta=1.0`, 50 epochs) produced a model whose mean prediction barely moved
off the dataset's average distance (`val_mae` crept from 250 to only 230
over 41 epochs), and whose predictions clustered in a narrow band far from
the true values regardless of input.

**Diagnosis.** Reading the *raw*, pre-soft-clamp `log_var` output directly
(rather than inferring from the final `aleatoric_std` numbers) showed it
sitting at a mean of **-180**, ranging as far as **-390**, on inputs where
the model's actual error was in the hundreds. `tanh` is exactly `+-1.0` in
float32 past `|x| ≈ 20` — so this wasn't "very confident," it was the
*exact same failure* the soft `tanh` bound was built to prevent
(Obstacle C in SESSION_NOTES.md), just reached by drifting through the
boundary via many small steps instead of hitting a hard clamp directly.
The likely mechanism: this branch's variance first inflated to explain
away early bad predictions (the classic Obstacle-A shortcut), and the
correction back down, driven by Adam's momentum, overshot straight through
the soft clamp's shrinking-but-technically-nonzero gradient region and
into genuine numerical dead space.

**What was tried, one change at a time, and what each one actually did:**

| Change | Effect |
|---|---|
| `prior_sigma` 0.1 → 1.0 (looser prior) | Freed the mean to move off the collapsed low band — real progress — but the log_var branch now collapsed to the *opposite* extreme: pinned near `LOG_VAR_MIN` (`aleatoric_std ≈ 0.01`) on almost every image, while still being wrong by 100+. |
| `kl_weight` 1.0 → 0.1 (weaker KL pull) | No change to the log_var collapse — still pinned at the floor for every image. |
| Gradient clipping (`clip_grad_norm_`, added to `train.py`) | **Fixed the mean branch** — it now tracks `true` about as well as the dropout model. Confirms the mean's problem really was large, destabilizing single steps. Did **not** fix log_var, because that branch's drift isn't one big step — it's many small, individually-reasonable-looking steps walking steadily in one direction. |
| `tanh(raw / 20)` — widen the soft clamp's linear region | Made things *worse*, not better: it also flattens the gradient by the same factor of 20 everywhere else on the curve (including near `raw=0`, where this branch normally operates), which slowed its self-correction down enough to let the original variance-inflation dynamic dominate for much longer. Mean collapsed low again, `aleatoric_std` swung to the other extreme (400+). Reverted. |
| Live per-epoch logging of raw `log_var` (`train.py`, `model.last_raw_log_var`) | Diagnostic, not a fix — but confirmed the drift is visible from epoch 1 and grows monotonically across training, rather than appearing suddenly at some later epoch. |

**Where it landed:** with gradient clipping + `prior_sigma=1.0` +
`kl_weight=0.1`, the **mean branch now works well** — `DistanceNetMobileNetVariational`'s
point predictions are competitive with `DistanceNetMobileNet`'s. The
**aleatoric branch does not**: `raw_log_var` still finishes training deeply
negative (mean ≈ -202, min ≈ -461 in the last full run), so
`aleatoric_std` reports `≈0.01` — maximum confidence — regardless of the
image, which is the opposite of trustworthy: the model is *confidently
wrong*, not honestly uncertain.

**Why this looks structural, not a tuning problem.** Three independent
levers (`prior_sigma`, `kl_weight`, the clamp's own gradient shape) were
each varied in isolation and none fixed it; one measurably made it worse.
The dropout model's head has the same shared-hidden-layer architecture
feeding both outputs and never shows this collapse, which points at what's
actually different: **per-sample weight resampling**. Every forward pass
through a `BayesianLinear` layer draws fresh weights, adding a source of
gradient noise that a dropout mask (also random, but from a much simpler,
bounded 0/1 distribution) doesn't inject to the same degree. That noise
appears sufficient to let this branch's parameters random-walk into
numerically dead territory over the course of many epochs, in a way
plain SGD noise on a deterministic head does not.

**Not yet tried, and the more likely real fixes:**
- Give `log_var` its own separate (non-shared) first `BayesianLinear`
  layer instead of sharing one with `mean`, so instability in one branch
  can't propagate through shared weights.
- Average the loss over 2-3 weight samples per training step instead of
  one, directly reducing the gradient noise that's letting this branch
  walk uncorrected.
- Add weight decay or a smaller learning rate specifically for the
  log_var output's parameters via a separate optimizer param group.

## Why this approach works

Three things had to be true simultaneously for the uncertainty numbers to
be trustworthy, and each maps to one design decision above:

1. **The mean must actually keep learning even while the variance branch is
   also being trained.** Beta-NLL decouples the mean's gradient from the
   variance's current value, so the two branches don't fight each other.
2. **The variance must be free to move to wherever it needs to be, in
   either direction, for the model's actual error scale on this task** —
   which for a 1–500 range with an initially-inaccurate mean can mean
   squared errors in the hundreds of thousands. The soft `tanh` bound
   (calibrated from the task's real worst case) keeps that headroom
   available and keeps a saturated prediction correctable.
3. **The two uncertainty types must come from mechanisms that actually
   target what they claim to measure.** Aleatoric noise is a property of a
   single input, so it's estimated by a direct per-input model output
   (`log_var`), trained with a proper scoring rule (NLL) so it's
   incentivized to match the true residual distribution rather than just
   be small. Epistemic uncertainty is a property of the model's confidence
   in its own weights, so it's estimated by literally perturbing the
   weights (via dropout) and observing how much the answer changes.

The empirical result (documented in SESSION_NOTES.md) matches this design:
`aleatoric_std` varies meaningfully across images (not pinned to one
constant), and the *lowest* uncertainty landed on the closest, largest,
least ambiguous object in a test batch — the direction you'd actually want
if you were going to trust these numbers for anything.

## Alternatives, and why they're more or less suitable

### For the point estimate (architecture)

| Approach | Verdict here |
|---|---|
| Plain MLP on flattened pixels | Rejected outright — no notion of spatial locality, so it would have to relearn "what an edge looks like" separately at every possible position, and the object's position is random in every sample. |
| Small CNN from scratch (`DistanceNet`) | Works, kept as the simple baseline; needs more data/epochs to reach the same accuracy as a pretrained backbone. |
| **Frozen pretrained backbone + small head (used: `DistanceNetMobileNet`)** | Chosen for `main.py`. ImageNet features already encode general shape/edge/texture detectors, so only a small head has to be learned from our few thousand synthetic images — closer to how real-world distance-from-photo systems are built in practice. |
| Deeper ResNet-style net / Vision Transformer trained from scratch | Considered and set aside — more parameters to fit from a small synthetic dataset with no pretraining benefit, higher compute cost, no clear accuracy win for a single-object regression task this constrained. |
| Classical (non-deep) CV, e.g. contour detection + calibrated size-to-distance formula | Would work *unusually well* here, precisely because the task's structure (one shape, size ∝ known function of distance) is simple enough to hand-engineer. Rejected because the point of the project is learning PyTorch / uncertainty estimation, not because it's a bad idea for this exact narrow task. |

### For the uncertainty estimate

| Approach | Trade-off vs. what's used here |
|---|---|
| **No uncertainty (bare point regression)** | Simplest, but gives no way to know when to distrust a prediction — the entire motivation for this project. |
| **Heteroscedastic NLL head only (aleatoric, used here)** | Cheap — one extra output, one forward pass. But alone it can't distinguish "this input is inherently ambiguous" from "the model just hasn't learned this region of input space yet," which matters if you want to know whether more training data would help. |
| **MC Dropout only (epistemic, used here)** | Also cheap — dropout is already a normal regularizer, just left on at inference. But alone it says nothing about irreducible per-input ambiguity, and would report *low* uncertainty on a confidently-wrong-because-the-image-is-genuinely-ambiguous case. |
| **Both combined (used here, Kendall & Gal 2017)** | Gets both signals from one trained model at the cost of `MC_SAMPLES` forward passes instead of one at inference time — no extra training cost or extra models. This is why it was chosen: best signal per unit of engineering and compute effort for a project also trying to keep training stable (fewer moving parts than an ensemble). |
| **Deep Ensembles** (Lakshminarayanan et al., 2017): train N independently-initialized models, use their disagreement as epistemic uncertainty | Generally gives a *better-calibrated* epistemic estimate than MC Dropout, since independently-trained models explore genuinely different solutions rather than the same model's local dropout perturbations. Rejected here mainly for cost: N× the training time, N× the checkpoints, N× the memory — steep for a learning project already iterating on a single model, and the frozen-backbone setup means most of that cost would be spent retraining near-identical heads anyway. |
| **Full variational / Bayes-by-Backprop layers** (implemented — see [Full variational layers](#full-variational-layers-bayes-by-backprop) below) | The "real" version of what MC Dropout approximates: every weight gets a learned distribution instead of a 0/1 dropout mask. More principled, at the cost of doubled parameter count per layer and a KL regularization term that needs its own tuning. |
| **Quantile regression** (predict e.g. the 10th/50th/90th percentile directly via pinball loss) | Makes no Gaussian assumption, so it's more robust if the true error distribution is skewed or heavy-tailed. Rejected here because it doesn't naturally decompose into aleatoric vs. epistemic components — you get an interval, not an explanation of *why* it's wide — which was a specific goal of this project. |
| **Conformal prediction** (post-hoc calibration on a held-out set to produce intervals with a guaranteed coverage rate) | Attractive because it's model-agnostic and gives a statistical coverage guarantee the Gaussian-NLL approach doesn't. Rejected as the primary mechanism because it's a wrapper around whatever point/uncertainty estimator you already have, not a replacement for one — and it still wouldn't decompose the *why* into aleatoric vs. epistemic. Worth layering on top later if calibrated coverage guarantees become the goal, rather than as a replacement for the current mechanism. |

### The common thread

Every alternative that was set aside was set aside for one of two reasons:
it doesn't produce the aleatoric/epistemic **decomposition** this project
specifically wants, or it multiplies training/engineering cost (more
models, more parameters, less stable training) for a gain that doesn't
clearly pay for itself at this project's scale. MC Dropout + a
heteroscedastic Beta-NLL head is the combination that gets a real,
decomposed, per-input uncertainty estimate out of a single trained model.

## Current status and limitations

- Mean accuracy is still improving with more training — a genuinely hard
  task given a 500-unit range, a backbone that's never seen synthetic
  noisy blobs, and only a few thousand training images.
- The uncertainty *mechanism* is verified working correctly: `aleatoric_std`
  varies meaningfully across images instead of collapsing to one constant,
  and lower uncertainty correctly lands on less ambiguous (closer, larger)
  objects.
- Because the backbone is frozen, MC Dropout's epistemic signal only
  reflects uncertainty in the small head's weights, not in the backbone's
  features — a real limitation of combining frozen transfer learning with
  MC Dropout specifically, since most of the network can never express "I
  haven't seen this before."
- A full variational (Bayes by Backprop) alternative to MC Dropout is
  implemented (`variational.py`, `DistanceNetMobileNetVariational`,
  `compare_uncertainty.py`) and trained/compared against the dropout
  model — see [What actually happened when this was run](#what-actually-happened-when-this-was-run-a-real-reproducible-collapse)
  for the full account. Its **mean prediction** is now competitive with
  the dropout model's (after adding gradient clipping to `train.py`), and
  its **epistemic estimate** (weight-sampling spread) is working. Its
  **aleatoric estimate** is not: `log_var` collapses to maximum confidence
  (`aleatoric_std ≈ 0.01`) regardless of the input, a real and reproducible
  failure mode of combining per-sample weight resampling with this
  heteroscedastic head, not fixed by three different hyperparameter/clamp
  interventions that were tried and ruled out.
- Natural next steps: train longer / with more data to improve mean
  accuracy; consider unfreezing (or partially fine-tuning) the backbone;
  fix the variational model's aleatoric collapse via a non-shared log_var
  head or multi-sample training steps (see above); or add a deep-ensemble
  variant to compare against both.
