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

---

## Part 5 — Letting a library own the math: BLiTZ

After Part 4 we had a variational model with a working mean, a working
epistemic estimate, and a permanently stuck aleatoric branch, with no
obvious next hyperparameter to try by hand. Rather than keep debugging
`variational.py`'s `BayesianLinear` indefinitely, we swapped it for the
equivalent layer from [BLiTZ](https://github.com/piEsposito/blitz-bayesian-deep-learning)
(`blitz-bayesian-pytorch`), a maintained library implementing the same
Bayes by Backprop idea, as `DistanceNetMobileNetBlitz` in `model.py`.
Learning goal at this point had shifted from "understand the mechanism by
hand" (Part 4's point) to "offload the parts that are now just plumbing."

**What's different from the hand-rolled version:**
- BLiTZ uses a *scale mixture of two Gaussians* prior (`prior_sigma_1`,
  `prior_sigma_2`, `prior_pi`) instead of a single Gaussian — what
  Blundell et al.'s original paper actually proposes.
- Decorating the model with `@variational_estimator` adds `sample_elbo()`,
  which runs `elbo_sample_nbr` (default 3) forward passes per training
  step and averages both the NLL and the KL term over them, instead of
  one pass. `train.py` uses this automatically whenever
  `hasattr(model, "sample_elbo")`. This was the direct test of "average
  over several weight samples," the fix flagged as not-yet-tried at the
  end of Part 4.
- `kl_divergence()` on the new model is a one-line wrapper around BLiTZ's
  own `nn_kl_divergence()`, so `train.py`/`inference.py`'s existing
  `hasattr(model, "kl_divergence")` checks needed no changes at all.

### Obstacle F: BLiTZ's own defaults produced a collapsed epistemic estimate

**Symptom:** a full training run with BLiTZ's own default hyperparameters
produced a mean that tracked `true` well and an `aleatoric_std` that
varied meaningfully across images (66-85 in one test batch) — real
confirmation that the multi-sample-training hypothesis from Part 4 was
correct. But `epistemic_std` came out tiny (0.05-0.10) for every single
image, versus 40-70 for the other models.

**Why:** two of BLiTZ's own defaults. `posterior_rho_init=-7.0` starts
every weight's `sigma` far tighter than the hand-rolled model ever used.
`prior_pi=1` makes the "scale mixture" prior collapse to a single
`N(0, 0.01)`, since a mixing weight of exactly 1 zeroes out the second
component entirely. Both push toward a posterior with little reason to
grow.

**Fix tried, and why it backfired badly:** loosened three parameters at
once — `posterior_rho_init=-3.0`, `prior_pi=0.5`, `prior_sigma_2=1.0`.
This broke the model far more severely than the problem it was meant to
fix: the mean collapsed into a narrow, input-independent band, and
`aleatoric_std` blew up to 250-375. Two mistakes, worth separating:

1. Changed three hyperparameters simultaneously instead of one at a time
   — the exact "isolate one variable" discipline that got Part 4 unstuck
   wasn't applied here.
2. The specific direction chosen fought a real mechanism: BLiTZ's
   `nn_kl_divergence()` is a **Monte Carlo estimate**
   (`log_posterior(w) - log_prior(w)` evaluated at the actual *sampled*
   weight `w`), not the closed-form KL the hand-rolled model computes
   directly from `mu`/`sigma`. Widening `sigma` widens the range of
   weights that get sampled, which widens the variance of that
   log-density estimate itself — so loosening the posterior didn't just
   allow more epistemic spread, it added noise directly to the mechanism
   (`sample_elbo`'s per-sample KL term) this model depends on for the
   stability it was specifically brought in to demonstrate.

Reverted to BLiTZ's defaults.

**Where it landed:** `DistanceNetMobileNetBlitz` ships with BLiTZ's own
defaults — mean and aleatoric both work, epistemic is under-informative
but not corrupting the other two outputs. Not yet revisited; if it is,
change `posterior_rho_init`/`prior_pi`/`prior_sigma_2` one at a time, and
expect the Monte-Carlo-KL mechanism to be more sensitive to loosening
than the closed-form version in `variational.py` was.

---

## Part 6 — A fundamentally different mechanism: Deep Evidential Regression

Every model so far gets its epistemic estimate the same way: perturb the
network (a dropout mask, a resampled weight) and run it many times, using
the spread of the answers as the uncertainty. For a genuinely different
comparison point, and to test whether a library could offload real
complexity (not just re-implement what we already had), we added
`DistanceNetMobileNetDER` using
[TorchUncertainty](https://github.com/ENSTA-U2IS-AI/torch-uncertainty)'s
`NormalInverseGammaLinear` layer and `DERLoss` — **Deep Evidential
Regression** (Amini et al., 2020).

**The idea:** instead of predicting `(mean, log_var)` and needing many
stochastic passes to separate aleatoric from epistemic uncertainty, this
model's head predicts **four numbers** in one deterministic forward pass
— `loc`, `lambda`, `alpha`, `beta`, the parameters of a
Normal-Inverse-Gamma (NIG) distribution over "what the mean and variance
of a Gaussian for this input probably are." Both uncertainty types come
out as closed-form properties of that distribution, no sampling needed:

```
aleatoric_var = beta / (alpha - 1)
epistemic_var = beta / ((alpha - 1) * lambda)
```

Trained with `DERLoss`: the NIG distribution's negative log-likelihood,
plus a regularizer (`reg_weight * |target - loc| * (2*lambda + alpha)`)
whose *intended* job is to shrink `alpha`/`lambda` ("evidence") whenever
the prediction is wrong — that's how this loss represents "I don't know,"
not a bug to be fixed.

**Dependency note:** `torch-uncertainty`'s package unconditionally
imports a full PyTorch Lightning CLI/trainer stack even just to reach
`torch_uncertainty.layers` — importing it pulled in `lightning`, `rich`,
`torchmetrics`, `pandas`, `seaborn`, and more, none of which this project
uses for anything beyond that one import. A noticeably heavier dependency
than BLiTZ's clean, lightweight import.

### Obstacle G: evidence collapsed to its numerical floor, blowing up both variance formulas

**Symptom:** first full run (`reg_weight=1e-2`, the library's own
`min_alpha`/`min_lmbda=1e-6` defaults). Point predictions (`loc`) looked
reasonable, but `epistemic_std` came out in the tens of thousands and
`aleatoric_std` in the thousands, for almost every image.

**Diagnosis:** read the raw evidential parameters directly rather than
trusting the derived stats. `alpha` sat at `1.000001` — exactly its floor
(`1 + min_alpha`) — for 7 of 8 test images, with `lmbda` similarly tiny
(`0.0009-0.003`). Plugging those into the formulas above reproduced the
observed `aleatoric_std`/`epistemic_std` numbers exactly: dividing by a
denominator of `~1e-6` (or `~1e-12` for the epistemic formula, which
divides by `(alpha-1)*lmbda`, both near their floors) turns a small
`beta` into an astronomical variance.

**First fix tried, and why it didn't work:** raised `reg_weight` 100x
(1e-2 -> 1.0), reasoning that a stronger regularizer would push evidence
away from collapsing. A full 50-epoch run showed `alpha`/`lmbda` pinned
at *exactly* their floor for every single epoch, completely unmoved by
the 100x change — this was the tell that `reg_weight` wasn't the
controlling lever at all. Re-reading the regularizer's formula confirmed
why: minimizing `|target - loc| * (2*lambda + alpha)` pushes evidence
*down* whenever the residual is nonzero, and on this task the residual is
essentially never zero (even a well-converged model still has real
error), so that pressure never lets up regardless of how strongly it's
weighted. `reg_weight` controls *how hard* evidence gets pushed down, not
*how low* it's allowed to go.

**Real fix:** raised `min_alpha`/`min_lmbda` from the library's default
of `1e-6` to `0.05`, exposed as constructor parameters on
`DistanceNetMobileNetDER`. This doesn't stop evidence from collapsing —
it bounds how extreme the consequence of that collapse can be: worst-case
`(alpha-1)*lmbda` went from `~1e-12` to `~0.0025`, nine orders of
magnitude better.

### Obstacle H: the floor fix revealed a deeper, still-open problem

**Symptom:** with the raised floors, `epistemic_std`/`aleatoric_std`
stopped being astronomical — but `alpha`'s and `lmbda`'s **mean now
equalled their min**, epoch after epoch. Every image was getting
*identical* evidence, pinned exactly at the new floor. Not exploding
anymore, but also carrying zero information — the model saying the exact
same "I'm not sure" regardless of what it saw.

**Why (best understanding so far):** `DERLoss`'s regularizer creates a
"cop-out" the model has little incentive to leave. Near `alpha=1`, the
NIG's predictive distribution develops very heavy tails, so almost any
residual gets an acceptable log-likelihood without needing an accurate
`loc` or informative `beta`. Since real residual is present on
essentially every training example throughout training, and nothing in
this loss specifically rewards *high* evidence when the model is doing
well on a given input, the model settles into "always claim minimum
confidence" and has no pressure to leave. This matches a documented
critique of the original DER formulation in follow-up literature (not
something specific to this project's setup).

**Ablation tried:** set `reg_weight=0.0` entirely, as a clean test of
whether the regularizer itself (not its magnitude) is the cause. A short
smoke test (5 epochs, 32 samples) showed `alpha` still drifting toward
its floor on its own — so the NLL term alone also favors low evidence for
large residuals, this isn't purely the regularizer's doing — but `lmbda`
showed real per-image spread for the first time (`min=0.156` vs.
`mean=0.211` by epoch 5, instead of min essentially equal to mean every
run before). A full run with `reg_weight=0.0` was kicked off next; last
reported status was "seems to be doing fine," but the exact final
`alpha`/`lmbda` numbers from that run and a fresh `compare_uncertainty.py`
printout haven't been captured here yet — **verify those before trusting
this is actually fixed**, the way every other claimed fix in this project
has been verified with real printed numbers, not just a training curve
that looks OK.

**Not yet tried:** annealing `reg_weight` up from 0 over training (risk:
may just delay Obstacle G's collapse to later epochs, the same
warm-up-band-aid pattern that failed in Obstacle B); a modified
evidential regularizer from follow-up DER literature designed to fix
exactly this "uniform minimum evidence" pathology; checking whether
`torch_uncertainty` ships an alternative loss that already addresses it.

---

## Part 7 — Comparing all four after a real 50-epoch run, and correcting an over-optimistic read

The `reg_weight=0.0` run from Obstacle H finished, and
`compare_uncertainty.py` was run across all four checkpoints on the same
8 test images:

```
true |         dropout          |       variational        |          blitz           |           der
       |      mean ep_std al_std |      mean ep_std al_std |      mean ep_std al_std |      mean ep_std al_std
------------------------------------------------------------------------------------------------------------
 387.2 |     350.7  55.74 109.89 |     346.6  63.83 247.65 |     337.7   0.11  65.57 |     381.3 263.16  58.84
 220.0 |     297.8  45.84  92.28 |     297.4  45.46   0.01 |     285.5   0.12  77.49 |     269.1  90.47  20.23
 429.4 |     326.6  48.33 113.12 |     328.7  60.10   0.01 |     316.8   0.11  99.70 |     371.4 258.44  57.79
 349.0 |     312.0  53.58 106.96 |     329.0  41.63   0.01 |     295.4   0.09  83.98 |     355.2 215.85  48.27
  48.0 |      82.8  15.90  46.25 |      92.3  33.32   0.01 |     144.0   0.08 135.55 |      56.1   8.02   3.89
 487.8 |     312.1  36.16 102.55 |     310.1  50.88   0.01 |     281.7   0.09  76.46 |     342.1 151.17  33.80
 380.8 |     350.3  43.72 111.49 |     336.7  63.21   0.01 |     319.8   0.10  82.07 |     381.5 264.93  59.24
 393.2 |     387.3  58.42 109.97 |     403.3  85.11   0.01 |     357.8   0.13  63.22 |     435.3 276.25  61.77
```

**`dropout`** — unchanged from every previous run. Mean tracks `true`
reasonably (worst at the range's extremes: `48 -> 82.8`, `487.8 -> 312.1`
— the hardest region of the task, per the dataset's own radius clamp).
Both uncertainty types vary sensibly. The stable baseline this whole
project measures everything else against.

**`variational`** — mean quality comparable to dropout. `aleatoric_std`
is `0.01` for 6 of 8 images here (one at exactly `0.01`... wait, 6 of the
7 non-first rows are `0.01`, only the first row shows `247.65`) —
Obstacle E's collapse is still present, unresolved, exactly as documented.
Not a new finding, just a reconfirmation.

**`blitz`** — mean noticeably weaker at the range's extremes than
dropout/DER (`429.4 -> 316.8`, `487.8 -> 281.7`, both large undershoots).
`epistemic_std` is `0.08-0.13` for every image — Obstacle F's collapse,
also unresolved, also just a reconfirmation.

**`der`** — this is the one with something genuinely new to check. Mean
accuracy is competitive with (arguably the best of) the four models
here — `380.8 -> 381.5`, `349.0 -> 355.2` are both excellent. Both
`epistemic_std` and `aleatoric_std` show real per-image spread instead of
one constant value, which is what the `reg_weight=0.0` ablation was
hoping for.

**But read the raw parameters before believing that's actually fixed.**
Checking `alpha`/`lmbda`/`beta` directly for these same 8 images:

```
true= 387.2  loc=  381.3  err=   5.9  lmbda=0.0500  alpha=1.0500  beta=173.1275
true= 220.0  loc=  269.1  err=  49.1  lmbda=0.0500  alpha=1.3837  beta=157.0406
true= 429.4  loc=  371.4  err=  58.0  lmbda=0.0500  alpha=1.0500  beta=166.9803
true= 349.0  loc=  355.2  err=   6.2  lmbda=0.0500  alpha=1.0753  beta=175.4777
true=  48.0  loc=   56.1  err=   8.1  lmbda=0.2353  alpha=5.0105  beta= 60.6642
true= 487.8  loc=  342.1  err= 145.8  lmbda=0.0500  alpha=1.1463  beta=167.1351
true= 380.8  loc=  381.5  err=   0.7  lmbda=0.0500  alpha=1.0500  beta=175.4652
true= 393.2  loc=  435.3  err=  42.0  lmbda=0.0500  alpha=1.0500  beta=190.7812
```

`lmbda` — the parameter that specifically drives *epistemic* uncertainty
(`epistemic_var = beta / ((alpha-1) * lmbda)`) — is pinned at exactly its
floor (`0.0500`) for **7 of the 8 images**, and only escapes for the one
easiest, closest image (`true=48`, the same image that's been the one
outlier throughout this entire DER saga, going all the way back to
Obstacle G). Recomputing `epistemic_var` for those 7 rows using only
`alpha`'s small movements above its own floor (`1.0500` to `1.3837`) and
`beta` reproduces the reported `epistemic_std` values exactly (e.g.
`true=349`: `175.4777 / (0.0753 * 0.05) = 46608`, `sqrt = 215.9`, matching
the printed `215.85`). So the apparent per-image differentiation in
`epistemic_std` is real, but it's coming entirely from `alpha` and
`beta`, not from `lmbda` doing the job it's meant to do. Obstacle H's
core finding — evidence collapsing to a floor value with no real
per-input signal — is **not fixed**, just partially masked: `alpha`
partially escaped the "cop-out" attractor with `reg_weight=0`, `lmbda`
did not, except for the one image easy enough to pull it away.

This corrects the too-optimistic framing at the end of Obstacle H, which
was based on a 5-epoch/32-sample smoke test showing early `lmbda` spread
that did not survive to the end of a full 50-epoch run. Lesson, stated
plainly so it isn't repeated: **a promising trend in a tiny smoke test is
not the same as a verified fix** — this project's own standard, applied
to itself.

One more real observation worth flagging as an open question rather than
a conclusion: `epistemic_std` does not clearly track *accuracy*. The
`true=380.8` row has the smallest error of any image here (`0.7`) and one
of the *highest* reported epistemic uncertainties (`264.93`); the
`true=487.8` row has the largest error (`145.8`) and a comparatively
*lower* epistemic uncertainty (`151.17`) than several much more accurate
predictions. Differentiation across images is not the same thing as
calibration (uncertainty correlating with actual error) — this data has
the former for `alpha`/`beta`, but whether it has the latter is still an
open question, not yet checked systematically.

**Where this actually leaves DER:** point predictions are good — possibly
the best of the four models on this test batch. `alpha`/`beta` produce
real per-image variation; `lmbda` still collapses to its floor for
essentially everything, and whether the resulting uncertainty is
*calibrated* (not just non-constant) hasn't been verified.

---

## Appendix: design rationale not tied to a specific obstacle

Reference material moved here from README.md to keep that document short
and skimmable. Nothing below is chronological — it's the "why" behind
decisions that didn't come from debugging a specific failure.

### Why predict `log_var` instead of `var` directly

Variance must be positive. Predicting `log_var` and exponentiating it
(`var = exp(log_var)`) gets positivity for free from an unconstrained
linear output, no constrained activation needed.

### Why this design actually works, in three points

1. **The mean must keep learning even while the variance branch trains.**
   Beta-NLL decouples the mean's gradient from the variance's current
   value (Obstacle B/Part 3), so the two branches don't fight.
2. **The variance must be free to move wherever the model's actual error
   scale requires**, in either direction — for this 1-500 range, an
   inaccurate mean can mean squared errors in the hundreds of thousands.
   The soft `tanh` bound, calibrated from the task's real worst case
   (Obstacle C), keeps that headroom available.
3. **Each uncertainty type must come from a mechanism that targets what
   it claims to measure.** Aleatoric noise is a property of a single
   input, so it's a direct per-input model output trained with a proper
   scoring rule (NLL). Epistemic uncertainty is about the model's
   confidence in its own weights, so it's estimated by literally
   perturbing the weights (dropout, or real weight resampling) and
   watching how much the answer changes.

### Alternatives considered for the architecture (point estimate)

| Approach | Verdict |
|---|---|
| Plain MLP on flattened pixels | Rejected outright — no notion of spatial locality; would have to relearn "what an edge looks like" separately at every position, and the object's position is random per sample. |
| Small CNN from scratch (`DistanceNet`) | Kept as the simple baseline; needs more data/epochs than a pretrained backbone. |
| **Frozen pretrained backbone + small head (`DistanceNetMobileNet`, chosen)** | ImageNet features already encode general shape/edge/texture detectors, so only a small head has to be learned from a few thousand synthetic images — closer to how real-world "distance from a photo" systems are built. |
| Deeper ResNet / Vision Transformer from scratch | Set aside — more parameters to fit with no pretraining benefit, higher compute, no clear win for a task this constrained. |
| Classical (non-deep) CV, e.g. contour detection + calibrated size formula | Would work unusually well here specifically, since the task's structure is simple enough to hand-engineer. Set aside because the point of the project is learning PyTorch / uncertainty estimation, not because it's a bad idea for this narrow task. |

### Alternatives considered for the uncertainty mechanism

| Approach | Trade-off |
|---|---|
| No uncertainty (bare point regression) | Simplest, but defeats the entire purpose of the project. |
| Heteroscedastic NLL head only (aleatoric) | Cheap, one forward pass, but can't distinguish "genuinely ambiguous input" from "model hasn't learned this region yet." |
| MC Dropout only (epistemic) | Also cheap, but says nothing about irreducible per-input ambiguity — would report low uncertainty on a confidently-wrong-because-genuinely-ambiguous case. |
| **Both combined (Kendall & Gal 2017) — used for the dropout and both variational models** | Best signal per unit of engineering/compute for a project also trying to keep training stable. |
| Deep Ensembles (Lakshminarayanan et al., 2017) | Better-calibrated epistemic estimate than MC Dropout in general, but N× training time/checkpoints/memory — steep for a learning project, and most of that cost would be spent retraining near-identical heads given the frozen backbone. Not implemented. |
| **Full variational / Bayes by Backprop — implemented twice (Part 4, Part 5)** | The "real" version of what MC Dropout approximates. More principled, at the cost of doubled parameter count and a KL term needing its own tuning (see Obstacles D-F). |
| **Deep Evidential Regression — implemented (Part 6)** | Single deterministic pass, no sampling of any kind, cheapest at inference. Trades that for a regularization weight and evidence floor that need their own calibration (see Obstacles G-H). |
| Quantile regression (pinball loss) | More robust to non-Gaussian error distributions, but doesn't naturally decompose into aleatoric vs. epistemic — you get an interval, not a "why." Not implemented. |
| Conformal prediction | Model-agnostic, gives a statistical coverage guarantee the others don't — but it's a calibration wrapper around an existing estimator, not a replacement, and still doesn't decompose the "why." Worth layering on top later. Not implemented. |

The common thread across everything set aside: it either didn't produce
the aleatoric/epistemic decomposition this project specifically wants, or
it multiplied cost (more models, more parameters, less stable training)
for a gain that didn't clearly pay for itself at this project's scale.
