# Project Journal: Uncertainty-Aware Distance Predictor

A record of what we built, in what order, and — especially — the three real bugs
we hit while adding uncertainty estimation, why each one happened, and how we
fixed it. Written to be read back later, not just skimmed once.

## The goal

Learn PyTorch by building a vision model that looks at an image and predicts
how far away an object is — and, more ambitiously, has it report *how
confident* it is in that prediction, using Bayesian-style uncertainty
estimation, rather than just spitting out a bare number.

---

## Part 1 — Building the baseline

### 1.1 A synthetic dataset instead of a real one

Real distance-labeled photo datasets are large downloads and messy to start
with, so instead `dataset.py` generates images on the fly: a shape on a noisy
background, where the shape's *size* encodes distance (closer = bigger),
exactly like how your own eyes judge distance from an object's apparent size.
This gives exact, free, unlimited ground-truth labels.

### 1.2 A small CNN, and why it has multiple layers

The first model (`DistanceNet`) is a few `Conv2d` + `ReLU` + `MaxPool2d`
blocks, growing in channels (16 → 32 → 64) while shrinking in spatial size.

**Why convolution instead of a plain fully-connected network (MLP)?** An MLP
flattens the image into a list of numbers and connects every pixel to every
neuron — it has no idea that two pixels are *neighbors*, so it would have to
re-learn "what an edge looks like" separately for every possible position in
the image. A `Conv2d` layer slides one small filter across the whole image, so
it learns a pattern once and reuses it everywhere. This matters a lot for us
because our object appears at a random position every time.

**Why stack several conv layers instead of one?** Each layer builds on the
last: layer 1 can only see raw pixels, so it learns primitive things (edges,
blobs). Layer 2 combines those into curved outlines. Layer 3 combines *those*
into "there's a solid blob, and here's its size." It's a ladder from pixels to
concepts — one layer can't jump straight from pixels to "object size."

### 1.3 Exploring alternatives

Before settling on an approach, we surveyed other options: global average
pooling instead of a big flatten+linear head, strided convolutions, deeper
ResNet-style nets, Vision Transformers, and even classical (non-deep-learning)
computer vision. We picked **transfer learning with a pretrained backbone**
next, since that's how most real-world "distance from a photo" systems are
actually built.

### 1.4 Switching to a pretrained MobileNetV3-Small backbone

`DistanceNetMobileNet` uses `torchvision`'s MobileNetV3-Small, pretrained on
ImageNet, with its convolutional feature extractor **frozen** (used purely as
a fixed feature extractor) and a small trainable head on top mapping its
output to our distance prediction. Only the head's weights get updated during
training — the backbone's `BatchNorm` layers are also explicitly kept in
`eval()` mode (see the `train()` override in `model.py`) so their running
statistics don't drift away from what they learned on real ImageNet photos.

### 1.5 Obstacle: training was running on CPU despite having a GPU

**What happened:** `torch.cuda.is_available()` returned `False` even though
the machine has an RTX 2060.

**Why:** The installed PyTorch build was the `+cpu` variant — PyTorch ships
separate builds per CUDA version, and pip doesn't auto-detect your GPU at
install time. The CPU build simply has no CUDA support compiled in at all.

**Fix:** Uninstalled and reinstalled `torch`/`torchvision` from PyTorch's
`cu126` index instead of the default PyPI one. Confirmed afterwards with
`torch.__version__` showing a `+cu126` suffix and `cuda.is_available() ==
True`.

### 1.6 Splitting training and inference into separate scripts

Originally `main.py` trained the model *and* immediately ran predictions in
one script. We split this into:

- `main.py` — trains and saves a checkpoint (`distance_net.pt`), which also
  records *which* model class it belongs to, so...
- `inference.py` — loads a saved checkpoint and runs predictions completely
  independently, without retraining.

### 1.7 Making the dataset harder and more realistic

Three changes, done together:

- **Image size** 64×64 → 224×224 (also conveniently MobileNet's native
  ImageNet resolution).
- **Distance range** 1–10 → 1–500. The *true* physical projection (apparent
  size ∝ 1/distance) would shrink far objects to sub-pixel size long before
  500, making them literally unrecoverable from the image — so we use a
  gentler `1/sqrt(distance)` falloff instead, a deliberate simplification to
  keep the whole range visually learnable.
- **Shape**: at first, a *new random* irregular polygon per sample. Then
  changed (per your request) to **one fixed asymmetric shape**, transformed
  each sample by a random rotation, scale (= distance), and position. This
  makes it a more honest "recognize this one object regardless of pose" task
  — and a harder one for a CNN specifically, since convolution is
  translation-invariant but *not* rotation-invariant by construction.

This also forced a small architecture fix: `DistanceNet`'s head had a
hardcoded flatten size (`64 * 8 * 8`) that assumed 64px input. Switched it to
`AdaptiveAvgPool2d(1)`, which works at any input resolution.

---

## Part 2 — Adding uncertainty

### 2.1 The concept, before writing any code

Two different kinds of "not sure," needing different fixes:

- **Aleatoric uncertainty** — irreducible noise/ambiguity *in the data
  itself*. In our dataset this is real by design: past ~400m, apparent size
  barely changes, so two very different true distances can look nearly
  identical. No amount of training removes this.
- **Epistemic uncertainty** — uncertainty *about the model's own weights*,
  i.e. "I haven't seen enough like this." This kind *is* reducible with more/
  better data.

**How to get each:**
- Aleatoric → change the model's output from one number to **two**: a mean
  and a `log_var` (log-variance), trained with **Gaussian negative
  log-likelihood (NLL)** instead of MSE, so the model can say "here's my
  guess, and here's how unsure I am about it," differently per input.
- Epistemic → **Monte Carlo Dropout**: keep a `Dropout` layer active even at
  inference time, run the same image through the network many times (each
  pass randomly drops different neurons), and use the *spread* of the
  resulting predictions as the uncertainty estimate.
- Combine both (Kendall & Gal, 2017): run N stochastic passes, average the
  per-pass variances for the aleatoric estimate, take the variance *of* the
  per-pass means for the epistemic estimate, and add them for a total.

### 2.2 Implementing it

- `model.py`: both models now output `(mean, log_var)` instead of one number.
- `train.py`: loss switched from `MSELoss` to Gaussian NLL.
- `inference.py`: added `predict_with_uncertainty()`, running 30 stochastic
  forward passes and reporting `aleatoric_std`, `epistemic_std`, and
  `total_std` alongside the point estimate.

---

## Part 3 — Three obstacles, and why each one happened

This is the part worth understanding properly, because all three are
well-known, documented failure modes of this exact technique (heteroscedastic
regression via NLL) — not one-off bugs in our code.

### Obstacle A: predictions all collapsed to the same value, with huge variance everywhere

**Symptom:** After the first NLL training run, predictions clustered around
7-10 regardless of the true distance, with a variance of roughly ±150
everywhere.

**Why this happens:** The NLL loss is:

```
loss = 0.5 * log(var) + (prediction_error)^2 / (2 * var)
```

Early in training, the mean prediction is basically random, so the error term
starts out large. The network discovers a cheap shortcut: **inflate `var`**
instead of actually improving the mean. Dividing a big error by a huge
variance shrinks that term fast, while the `log(var)` penalty only grows
slowly — so the loss drops quickly without the prediction ever getting more
accurate.

Worse, it's self-reinforcing: the gradient that would normally push the mean
to improve is proportional to `1 / var`. Once `var` explodes, that gradient
collapses toward zero too — so the mean prediction gets *permanently stuck*
wherever it happened to be, while `var` keeps growing to "explain away" the
now-frozen error.

**First fix tried:** train with plain MSE for the first several epochs
("warm-up"), so the mean gets a real head start before the variance branch is
allowed to do anything.

### Obstacle B: training would still freeze, right at the warm-up → NLL handoff

**Symptom:** MSE/MAE would improve during warm-up, then basically stop
changing the moment NLL kicked in.

**Why this happens:** During warm-up, the `log_var` output branch is never
touched by the loss, so its weights stay close to their random
initialization. The instant we switch to NLL, predicted variance is wildly
mismatched with the actual (still large) error — and that mismatch triggers
the *exact same* collapse from Obstacle A, just delayed to the transition
point instead of happening from epoch 1.

**Real fix:** replace the warm-up band-aid with **Beta-NLL** (Seitzer et al.,
2022). It multiplies the loss by a *detached* (stop-gradient) `var^beta`
weighting term. With `beta = 1`, this makes the gradient into the mean
**mathematically identical in shape to plain MSE** — completely decoupled
from whatever variance the model currently predicts — while `log_var` still
gets a normal, well-behaved gradient pushing it toward the true error scale.
This fixes the problem structurally instead of just delaying it, so there's
no fragile "before/after" moment left for it to break at.

### Obstacle C: variance was still identical for every single image

**Symptom:** Even after Beta-NLL, `aleatoric_std` was exactly `148.4` for
every prediction, no matter what the image showed, and predictions still
clustered near the dataset's average distance.

**Why this happens (two compounding causes):**

1. `log_var` was bounded with `torch.clamp(log_var, -10, 10)`. Hard clamps
   have **exactly zero gradient** once a value saturates at the boundary — so
   once a prediction drifted past the ceiling, no gradient could ever pull it
   back down again, no matter how much more training happened. `sqrt(exp(10))
   ≈ 148.4` — matching the observed number exactly confirmed this was the
   cause, not a coincidence.
2. The ceiling itself (`exp(10) ≈ 22,026`) was miscalibrated for this task.
   With distances up to 500, a still-inaccurate mean can easily have errors
   of 300-400, meaning squared errors in the *hundreds of thousands* — far
   above what the clamp allowed. The model wasn't "playing it safe"; it
   genuinely had nowhere higher to report.

**Fix:**
- Replaced the hard `torch.clamp` with a smooth `tanh`-based soft bound,
  which maps any raw output into the same range but always has a
  (small but never exactly zero) gradient, so a saturated value can still be
  pulled back.
- Recalibrated the ceiling based on the dataset's actual distance range
  (worst-case squared error), instead of an arbitrary fixed number.
- Increased training length (15 → 40 epochs), since the variance branch,
  now correctly decoupled from the mean, still needs its own gradient steps
  to learn *per-sample* differences rather than just the dataset-wide average
  error scale.

**Result after all three fixes:** `aleatoric_std` varies meaningfully across
images (roughly 93 to 157 in one test batch) instead of being pinned to one
number, and the smallest uncertainty landed on the closest, largest, least
ambiguous object in the batch — the direction you'd actually want.

---

## Where things stand

- Mean accuracy is still improving with more training (not yet converged) —
  this is a genuinely hard task: a 500m range, a frozen backbone that's never
  seen synthetic noisy blobs, and only 2000 training images.
- The uncertainty *mechanism* itself (aleatoric via NLL head, epistemic via
  MC Dropout) is now working correctly and producing per-image-dependent
  estimates rather than a collapsed or frozen output.
- Natural next steps: train longer / with more data to bring mean accuracy
  down further; and optionally swap MC Dropout for genuine variational
  (Bayes-by-Backprop) layers if you want the "real" Bayesian neural network
  experience rather than this well-established approximation of it.

---

## Part 4 — Building and debugging the "real" Bayesian alternative

Took the last suggestion above and built it: `variational.py`
(`BayesianLinear`, a Bayes by Backprop layer -- every weight gets a
learned Gaussian distribution instead of one number, sampled fresh via
the reparameterization trick on every forward pass) and
`DistanceNetMobileNetVariational` in `model.py`, sharing the same frozen
backbone as the dropout model but with a variational head instead. Added
`compare_uncertainty.py` to train both and compare them on the same
images. Full mechanism writeup is in README.md; this section is the
debugging story, in the same spirit as Part 3.

### Obstacle D: the mean barely learned at all

**Symptom:** 50-epoch run, `prior_sigma=0.1`, `kl_weight=1.0`. `val_mae`
crept from 250 to only 230 over the first 41 epochs -- almost no
improvement, and predictions stayed clustered in a narrow band
regardless of the input.

**Why:** `BayesianLinear`'s KL-divergence term pulls every weight's mean
back toward the prior's mean of zero. With `prior_sigma=0.1` that pull is
strong, and it was fighting the head's ability to grow real weight
magnitude away from zero fast enough to express a genuine function of the
input -- a known Bayes-by-Backprop failure mode (the approximate
posterior collapsing onto the prior, sometimes called posterior
collapse), here showing up as very slow convergence rather than a hard
freeze.

**First fix tried:** loosen the prior (`prior_sigma` 0.1 -> 1.0), so
the KL term pulls less aggressively.

### Obstacle E: fixing the mean broke the variance in a new way

**Symptom:** After loosening the prior, the mean recovered nicely --
tracking `true` about as well as the dropout model. But `aleatoric_std`
was now pinned at `~0.01` (the theoretical floor) for almost every
single image, regardless of how wrong the prediction was.

**Why:** Reading the *raw*, pre-soft-clamp `log_var` value directly
(rather than trusting the final `aleatoric_std` number) showed it sitting
at a mean of roughly **-180**, ranging as far as **-390**. `tanh` is
exactly `+-1.0` in float32 past `|x| ~ 20` -- so this was the *exact same
failure* as Obstacle C, just reached differently: not a hard clamp, but
many small training steps walking the raw value straight through the
soft clamp's shrinking gradient region and out into genuine numerical
dead space, where the gradient is zero for real. The likely mechanism:
this branch's variance first inflated to explain away early bad
predictions (the Obstacle-A shortcut again), and the correction back
down overshot through the boundary before the vanishing gradient could
act as a brake.

**Fixes tried, one at a time, isolating what each one actually did:**

1. Lowering `kl_weight` (1.0 -> 0.1), on the theory that a weaker KL pull
   would leave more room for the variance branch to correct itself. No
   effect -- still pinned at the floor for every image.
2. Adding gradient clipping (`torch.nn.utils.clip_grad_norm_` before
   `optimizer.step()` in `train.py`), on the theory that a single
   destabilizing step was throwing the raw value past the boundary.
   This **fixed the mean branch's remaining instability** but did
   **nothing** for the log_var collapse -- confirming that branch's drift
   isn't one big step, it's many small, individually-reasonable-looking
   steps walking steadily in one direction over the whole training run.
3. Widening the soft clamp itself (`tanh(raw / 20)` instead of
   `tanh(raw)`), on the theory that a wider linear region would keep the
   gradient alive over the whole range the raw value was wandering
   into. This made things *worse*: dividing by 20 also flattens the
   gradient by that same factor everywhere else on the curve, including
   right around `raw=0` where this branch normally operates during
   healthy training. That slowdown gave the original variance-inflation
   dynamic far more time to dominate before the branch could
   self-correct, and reintroduced the mean-collapse from Obstacle D at
   the same time (`aleatoric_std` swung to the opposite extreme, 400+).
   Reverted.

**Where it landed:** gradient clipping + `prior_sigma=1.0` +
`kl_weight=0.1` gives a variational model whose **mean and epistemic
estimates both work** -- competitive with the dropout model. Its
**aleatoric estimate does not**: `log_var` still finishes training deeply
saturated (raw mean ~ -202, min ~ -461 in the last full run), reporting
near-total confidence regardless of the image. Three independent levers
were tried and ruled out one at a time, which points at something
structural rather than a tuning problem: the dropout model's head shares
the exact same architecture (one hidden layer feeding both `mean` and
`log_var`) and never shows this collapse, so the actual difference is
`BayesianLinear`'s per-sample weight resampling -- a source of gradient
noise a dropout mask doesn't inject to the same degree, apparently enough
to let this specific branch random-walk into numerically dead territory
over many epochs.

Also added a live per-epoch diagnostic (`model.last_raw_log_var`,
surfaced in `train.py`'s printed line) so this kind of drift can be
watched *during* training from now on, rather than inferred after the
fact from `inference.py`'s output -- which is how the earlier
`kl_weight` and clamp-width changes could be checked against real
evidence instead of another guess.

**Not yet tried:** giving `log_var` its own non-shared first layer
instead of sharing one with `mean`; averaging the loss over several
weight samples per training step to directly reduce the gradient noise
implicated above; or a separate, smaller learning rate for the log_var
output specifically.
