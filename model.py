"""
Models that regress distance from a grayscale image.

Each model outputs (mean, log_var) instead of a single number: mean is the
point estimate, log_var is the model's own estimate of the aleatoric
(data) uncertainty at that input, trained via Gaussian negative
log-likelihood instead of plain MSE. log_var (rather than var) is predicted
because it's unconstrained -- the network can output any real number and we
exponentiate it, so variance is always positive without needing a
constrained activation. It's bounded to a sane range to avoid exp()
overflow/underflow, via a smooth tanh squash rather than torch.clamp --
clamp has exactly zero gradient once a value saturates at its boundary,
which can permanently strand log_var at the limit with no way for training
to ever pull it back (this happened in practice: every prediction reported
identical maximum variance once log_var got clamped at +10).

Both models also keep a Dropout layer active in the head, which is what
lets inference.py run Monte Carlo Dropout: many stochastic forward passes
whose spread approximates epistemic (model) uncertainty, on top of the
per-pass aleatoric estimate.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from blitz.modules import BayesianLinear as BlitzBayesianLinear
from blitz.utils import variational_estimator
from torch_uncertainty.layers.distributions import NormalInverseGammaLinear

from dataset import MIN_DISTANCE, MAX_DISTANCE
from variational import BayesianLinear

LOG_VAR_MIN = -10.0
# Worst-case squared residual is (MAX_DISTANCE - MIN_DISTANCE)**2 -- the
# model predicting one end of the range when the truth is the other. The
# previous fixed ceiling of 10 (var <= ~22,026, std <= ~148) was well
# below that for our 1-500 distance range, so every under-trained
# prediction saturated at the same ceiling regardless of the image -- not
# because the model was "playing it safe," but because it had nowhere
# higher to go. A couple of log-units of headroom above the true worst
# case keeps the ceiling from binding again.
LOG_VAR_MAX = 2.0 * torch.log(torch.tensor(MAX_DISTANCE - MIN_DISTANCE)).item() + 2.0

def _soft_clamp_log_var(raw: torch.Tensor) -> torch.Tensor:
    """Smoothly bounds raw log_var into [LOG_VAR_MIN, LOG_VAR_MAX] via tanh.
    Unlike torch.clamp, the gradient here is never exactly zero, so a
    saturated prediction can still be pulled back during training.

    Tried dividing `raw` by a temperature here to widen the region before
    tanh saturates (to rescue DistanceNetMobileNetVariational's raw log_var,
    which was drifting out to -180/-390 during training and getting stuck
    at the floor) -- reverted. It also flattens the gradient by that same
    factor everywhere else on the curve, including near raw=0 where this
    branch normally operates, which slowed it down enough to reintroduce a
    *different* collapse (mean stuck low, aleatoric_std blown up to 400+)
    instead of fixing the original one. See `last_raw_log_var` below for
    the live per-epoch diagnostic used to actually watch this branch
    during training instead of guessing at the next hyperparameter."""
    return LOG_VAR_MIN + 0.5 * (LOG_VAR_MAX - LOG_VAR_MIN) * (torch.tanh(raw) + 1.0)


class DistanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # -> (batch, 64, 1, 1), any input size works
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),  # (mean, log_var)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        pred_mean, raw_log_var = x.unbind(dim=-1)
        self.last_raw_log_var = raw_log_var.detach()  # for train.py's per-epoch diagnostics
        log_var = _soft_clamp_log_var(raw_log_var)
        return pred_mean, log_var


class DistanceNetMobileNet(nn.Module):
    """
    Distance regressor built on a MobileNetV3-Small backbone pretrained on
    ImageNet. The backbone's conv features are frozen (used only as a fixed
    feature extractor) and a small trainable head maps its output to a
    single distance value.
    """

    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = mobilenet_v3_small(weights=weights)

        self.backbone = backbone.features  # conv stack only, no classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),  # (mean, log_var)
        )

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ImageNet normalization stats, applied inside forward so the
        # caller can keep feeding plain [0, 1] images like DistanceNet does.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            # Keep BatchNorm running stats frozen too, not just the weights.
            self.backbone.eval()
        return self

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # grayscale -> fake RGB
        x = (x - self.mean) / self.std
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)
        pred_mean, raw_log_var = x.unbind(dim=-1)
        self.last_raw_log_var = raw_log_var.detach()  # for train.py's per-epoch diagnostics
        log_var = _soft_clamp_log_var(raw_log_var)
        return pred_mean, log_var


class DistanceNetMobileNetVariational(nn.Module):
    """
    Same frozen MobileNetV3-Small backbone as DistanceNetMobileNet, but the
    head's two Linear layers are replaced with BayesianLinear layers
    (Bayes by Backprop -- see variational.py) instead of relying on
    Dropout for epistemic uncertainty. Where MC Dropout on
    DistanceNetMobileNet approximates "many slightly different models" by
    randomly zeroing head neurons, this head learns an actual distribution
    over every head weight and samples real weights on every forward pass
    -- the "full variational" alternative described in README.md.

    Kept as a separate class (rather than a flag on DistanceNetMobileNet)
    so both can be trained, checkpointed, and compared side by side --
    see compare_uncertainty.py.
    """

    def __init__(self, freeze_backbone: bool = True, prior_sigma: float = 1.0):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = mobilenet_v3_small(weights=weights)

        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            BayesianLinear(576, 64, prior_sigma=prior_sigma),
            nn.ReLU(),
            BayesianLinear(64, 2, prior_sigma=prior_sigma),  # (mean, log_var)
        )

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # grayscale -> fake RGB
        x = (x - self.mean) / self.std
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)
        pred_mean, raw_log_var = x.unbind(dim=-1)
        self.last_raw_log_var = raw_log_var.detach()  # for train.py's per-epoch diagnostics
        log_var = _soft_clamp_log_var(raw_log_var)
        return pred_mean, log_var

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence summed over every BayesianLinear layer in
        the head. train.py adds this to the loss as a regularization term;
        inference.py's predict_with_uncertainty checks for this method to
        tell a variational model apart from a dropout-based one."""
        return sum(m.kl_divergence() for m in self.modules() if isinstance(m, BayesianLinear))


@variational_estimator
class DistanceNetMobileNetBlitz(nn.Module):
    """
    Same idea as DistanceNetMobileNetVariational (Bayes by Backprop head on
    the frozen MobileNetV3-Small backbone), but using blitz-bayesian-pytorch's
    BayesianLinear instead of this project's own hand-rolled one in
    variational.py -- letting a maintained library own the weight-sampling
    and KL-divergence math instead of hand-rolling it.

    Two concrete differences from variational.BayesianLinear, both from the
    library rather than a deliberate design choice here:
    - BLiTZ's prior is a scale mixture of two Gaussians (Blundell et al.'s
      original paper uses this too), not the single Gaussian this project
      wrote by hand -- generally a better-behaved prior for Bayes by
      Backprop.
    - The @variational_estimator decorator adds sample_elbo(), which
      averages the loss over several weight samples per training step.
      train.py uses this for this model instead of a single forward pass
      per step -- directly testing the "average over multiple weight
      samples" hypothesis flagged as the likely fix for
      DistanceNetMobileNetVariational's unresolved aleatoric collapse
      (see SESSION_NOTES.md, Obstacle E).

    Uses BLiTZ's own default prior/posterior-init hyperparameters
    (prior_sigma_1=0.1, prior_sigma_2=0.4, prior_pi=1, posterior_rho_init=
    -7.0) rather than this project's own values -- tried loosening them
    once (posterior_rho_init=-3.0, prior_pi=0.5, prior_sigma_2=1.0) to fix
    a collapsed, near-zero epistemic estimate seen with the defaults, and
    it made things considerably worse: the mean branch collapsed into a
    narrow low band and aleatoric_std blew up to 250-375, both worse than
    with the defaults. Likely cause: BLiTZ's nn_kl_divergence() is a
    Monte Carlo estimate (log_posterior(w) - log_prior(w) at the actual
    *sampled* weight w), not the closed-form KL variational.py computes
    from mu/sigma directly -- widening sigma widened the range of sampled
    weights, which widened the variance of that log-density estimate
    itself, adding noise to the exact mechanism (sample_elbo's per-sample
    KL term) this model relies on for stability. Reverted; the epistemic
    collapse with defaults is a known, smaller-severity open item (see
    README.md) rather than something to fix by changing several priors
    at once again.
    """

    def __init__(
        self,
        freeze_backbone: bool = True,
        prior_sigma_1: float = 0.1,
        prior_sigma_2: float = 0.4,
        prior_pi: float = 1.0,
        posterior_rho_init: float = -7.0,
    ):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = mobilenet_v3_small(weights=weights)

        blitz_kwargs = dict(
            prior_sigma_1=prior_sigma_1,
            prior_sigma_2=prior_sigma_2,
            prior_pi=prior_pi,
            posterior_rho_init=posterior_rho_init,
        )
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            BlitzBayesianLinear(576, 64, **blitz_kwargs),
            nn.ReLU(),
            BlitzBayesianLinear(64, 2, **blitz_kwargs),  # (mean, log_var)
        )

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # grayscale -> fake RGB
        x = (x - self.mean) / self.std
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)
        pred_mean, raw_log_var = x.unbind(dim=-1)
        self.last_raw_log_var = raw_log_var.detach()  # for train.py's per-epoch diagnostics
        log_var = _soft_clamp_log_var(raw_log_var)
        return pred_mean, log_var

    def kl_divergence(self) -> torch.Tensor:
        """Thin wrapper around the nn_kl_divergence() that
        @variational_estimator adds, so train.py/inference.py's existing
        hasattr(model, "kl_divergence") checks work for this model too
        without needing to special-case BLiTZ specifically."""
        return self.nn_kl_divergence()


class DistanceNetMobileNetDER(nn.Module):
    """
    Same frozen MobileNetV3-Small backbone as the other models, but the
    head predicts the four parameters (loc, lmbda, alpha, beta) of a
    Normal-Inverse-Gamma (NIG) distribution instead of a (mean, log_var)
    pair, following Deep Evidential Regression (Amini et al., 2020,
    "Deep Evidential Regression"). Uses torch-uncertainty's
    NormalInverseGammaLinear layer.

    This is a genuinely different mechanism from every other model here:
    no dropout, no weight resampling, no multi-sample inference loop. A
    SINGLE deterministic forward pass gives everything needed for both
    uncertainty types, via closed-form properties of the fitted NIG
    distribution:
        aleatoric_var = beta / (alpha - 1)           -- expected data noise
        epistemic_var = beta / ((alpha - 1) * lmbda)  -- roughly, how much
                                                          "evidence" the
                                                          model has seen
                                                          for this input
    (see torch_uncertainty.utils.distributions.NormalInverseGamma's
    mean_variance/variance_loc properties). inference.py's
    predict_with_uncertainty checks the is_evidential flag set below to
    skip its MC-sampling loop entirely for this model -- there's nothing
    to sample.

    forward() returns the raw parameter dict (each value shaped
    (batch, 1), the NormalInverseGammaLinear layer's native output),
    unlike every other model's (pred_mean, log_var) tuple -- train.py and
    inference.py both branch on is_evidential specifically because this
    output shape and the loss that consumes it (DERLoss, not
    BetaNLLLoss) are genuinely different, not just a different set of
    numbers through the same interface.

    min_alpha/min_lmbda raised well above NormalInverseGammaLinear's own
    default of 1e-6. DERLoss's regularizer (reg_weight * |target - loc| *
    (2*lmbda + alpha)) is *supposed* to shrink alpha/lmbda whenever the
    prediction is wrong -- that's the intended mechanism, not a bug, since
    low evidence is how this loss represents "I don't know." The actual
    problem: a full training run showed alpha and lmbda pinned at exactly
    their floor for all 50 epochs regardless of reg_weight (tried both
    1e-2 and a 100x-larger 1.0, with no change in where they land) --
    because this task's residual is essentially never zero, so that
    pressure never lets up. Both uncertainty formulas divide by these
    values, so a 1e-6 floor turns "the model is unsure" into "the variance
    is astronomically large and useless" (epistemic_std in the tens of
    thousands was observed). Raising the floor doesn't stop evidence from
    shrinking -- it bounds how extreme the consequence of that shrinking
    can be.
    """

    def __init__(
        self,
        freeze_backbone: bool = True,
        min_alpha: float = 0.05,
        min_lmbda: float = 0.05,
    ):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        backbone = mobilenet_v3_small(weights=weights)

        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 64),
            nn.ReLU(),
        )
        # out_features is set internally to 4 * event_dim by this layer --
        # only in_features is passed through to the underlying nn.Linear.
        self.nig_head = NormalInverseGammaLinear(
            nn.Linear, event_dim=1, in_features=64, min_alpha=min_alpha, min_lmbda=min_lmbda
        )

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Also freeze the mode itself here, not just via the train()
            # override below. PyTorch Lightning (train_der.py) snapshots
            # every submodule's train/eval flag individually before its
            # automatic pre-training "sanity check" validation run, and
            # restores that exact snapshot after every later validation
            # too -- silently overriding the train() override's effect for
            # the entire rest of training if the very first snapshot,
            # taken before anyone has called .train() yet, sees a
            # freshly-constructed (default: train-mode) backbone. Starting
            # in eval() here means that first snapshot is already correct.
            self.backbone.eval()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Marks this model for train.py/inference.py's evidential-specific
        # branches -- see the class docstring for why this can't just
        # reuse the (pred_mean, log_var) interface the other models share.
        self.is_evidential = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # grayscale -> fake RGB
        x = (x - self.mean) / self.std
        x = self.backbone(x)
        x = self.pool(x)
        x = self.trunk(x)
        return self.nig_head(x)  # dict: loc, lmbda, alpha, beta, each (batch, 1)


# Lets checkpoints record which class they belong to, so inference.py can
# rebuild the right architecture without the caller having to know it.
MODEL_REGISTRY = {
    "DistanceNet": DistanceNet,
    "DistanceNetMobileNet": DistanceNetMobileNet,
    "DistanceNetMobileNetVariational": DistanceNetMobileNetVariational,
    "DistanceNetMobileNetBlitz": DistanceNetMobileNetBlitz,
    "DistanceNetMobileNetDER": DistanceNetMobileNetDER,
}
