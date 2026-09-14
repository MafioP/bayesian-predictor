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


# Lets checkpoints record which class they belong to, so inference.py can
# rebuild the right architecture without the caller having to know it.
MODEL_REGISTRY = {
    "DistanceNet": DistanceNet,
    "DistanceNetMobileNet": DistanceNetMobileNet,
    "DistanceNetMobileNetVariational": DistanceNetMobileNetVariational,
}
