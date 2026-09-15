"""Training loop for the (mean, log_var) distance regressor."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_uncertainty.losses import DERLoss
from torch_uncertainty.utils.distributions import NormalInverseGamma

from dataset import DistanceDataset
from model import DistanceNet


class BetaNLLLoss(nn.Module):
    """
    Gaussian NLL with a detached var**beta weighting (Seitzer et al., 2022,
    "On the Pitfalls of Heteroscedastic Uncertainty Estimation with
    Probabilistic Neural Networks").

    Plain NLL's gradient into `mean` scales as 1/var: once the model
    inflates variance to cheaply shrink the loss instead of improving the
    mean, that same inflation kills the mean's gradient and freezes it --
    the collapse this project hit twice (once from cold start, once at the
    warmup-to-NLL handoff). Beta-NLL fixes this structurally: with
    beta=1, the mean's gradient becomes var-independent, identical in
    shape to plain MSE, while `log_var` still receives a normal gradient
    pushing it toward the true residual scale.
    """

    def __init__(self, beta: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, mean: torch.Tensor, target: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        var = var.clamp(min=self.eps)
        nll = 0.5 * (torch.log(var) + (target - mean) ** 2 / var)
        weight = var.detach() ** self.beta
        return (weight * nll).mean()


def _blitz_criterion(loss_fn: BetaNLLLoss):
    """Adapts BetaNLLLoss's (mean, target, var) signature to the
    (outputs, labels) signature BLiTZ's sample_elbo expects, where
    `outputs` is whatever the model's forward() returns -- here, our
    (pred_mean, log_var) tuple."""

    def criterion(outputs, target):
        pred_mean, log_var = outputs
        return loss_fn(pred_mean, target, log_var.exp())

    return criterion


def train(
    model_cls: type = DistanceNet,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    train_size: int = 4000,
    val_size: int = 800,
    beta: float = 1.0,
    kl_weight: float = 0.1,
    grad_clip_norm: float = 5.0,
    elbo_sample_nbr: int = 3,
    reg_weight: float = 0.0,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = DistanceDataset(train_size, seed=0)
    val_ds = DistanceDataset(val_size, seed=1)  # different seed -> disjoint samples
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = model_cls().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    loss_fn = BetaNLLLoss(beta=beta)
    # DERLoss's regularizer is reg_weight * |target - loc| * (2*lmbda + alpha)
    # -- it's *meant* to shrink alpha/lmbda ("evidence") whenever the
    # prediction is wrong, which is how this loss represents "I don't
    # know." That's not the bug. Two full runs (reg_weight=1e-2, then a
    # 100x-larger 1.0, both with min_alpha/min_lmbda=0.05) showed alpha and
    # lmbda pinned at exactly their floor for every image, all 50 epochs,
    # in both runs -- because this task's residual is essentially never
    # zero, so that shrinking pressure never lets up, and there's nothing
    # in this loss that specifically rewards high evidence when the model
    # IS doing well on a given input. Set to 0.0 here as a clean ablation:
    # if evidence still collapses uniformly with the regularizer entirely
    # removed, the NLL term itself (not reg_weight's magnitude) is the
    # cause. If it starts differentiating between images instead, that
    # isolates the regularizer as the culprit and 0.0 -- or some small
    # nonzero value -- may be the actual fix, not a diagnostic step.
    der_loss_fn = DERLoss(reg_weight=reg_weight)

    # DistanceNetMobileNetDER (model.py) sets is_evidential and returns a
    # dict of NIG parameters from forward() instead of a (pred_mean,
    # log_var) tuple -- it needs its own loss and its own branch below,
    # not just a different set of numbers through the shared interface.
    is_evidential = getattr(model, "is_evidential", False)
    # BLiTZ models (@variational_estimator, see model.py) expose
    # sample_elbo(), which averages the loss over elbo_sample_nbr weight
    # samples per step instead of one -- use it instead of a single
    # forward pass when available. Our own hand-rolled variational model
    # (variational.py) exposes kl_divergence() but not sample_elbo(), so
    # it still gets the manual single-sample KL-adding path below.
    uses_sample_elbo = not is_evidential and hasattr(model, "sample_elbo")
    is_variational = not is_evidential and not uses_sample_elbo and hasattr(model, "kl_divergence")
    elbo_criterion = _blitz_criterion(loss_fn) if uses_sample_elbo else None
    # Both variational flavors (hand-rolled and BLiTZ) expose last_raw_log_var
    # and kl_divergence(); this gates the raw-log_var diagnostics below for
    # either one, regardless of which training path they take above.
    has_raw_log_var_diagnostic = not is_evidential and hasattr(model, "kl_divergence")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        raw_log_var_min = float("inf")
        raw_log_var_max = float("-inf")
        raw_log_var_sum = 0.0
        raw_log_var_count = 0
        alpha_min = float("inf")
        alpha_sum = 0.0
        lmbda_min = float("inf")
        lmbda_sum = 0.0
        evidential_count = 0
        for images, distances in train_loader:
            images, distances = images.to(device), distances.to(device)

            optimizer.zero_grad()
            if is_evidential:
                out = model(images)
                dist = NormalInverseGamma(out["loc"], out["lmbda"], out["alpha"], out["beta"])
                loss = der_loss_fn(dist, distances.unsqueeze(-1))
                # Watches for evidence collapse (alpha walking down to its
                # floor of 1, which blows up both variance formulas since
                # they divide by (alpha - 1)) instead of waiting to find
                # out post-hoc from inference.py's output, the way the
                # reg_weight=1e-2 collapse first showed up.
                alpha_min = min(alpha_min, out["alpha"].min().item())
                alpha_sum += out["alpha"].sum().item()
                lmbda_min = min(lmbda_min, out["lmbda"].min().item())
                lmbda_sum += out["lmbda"].sum().item()
                evidential_count += out["alpha"].numel()
            elif uses_sample_elbo:
                # sample_elbo runs elbo_sample_nbr forward passes internally
                # and averages both the data-fit loss and the KL term over
                # them -- this is the multi-sample averaging this project's
                # hand-rolled variational model doesn't do, and the leading
                # hypothesis for why that model's aleatoric branch collapses
                # (SESSION_NOTES.md, Obstacle E). model.last_raw_log_var
                # still gets set as a side effect of sample_elbo's internal
                # forward() calls, so the diagnostics below work unchanged.
                loss = model.sample_elbo(
                    inputs=images,
                    labels=distances,
                    criterion=elbo_criterion,
                    sample_nbr=elbo_sample_nbr,
                    complexity_cost_weight=kl_weight / len(train_loader),
                )
            else:
                pred_mean, log_var = model(images)
                loss = loss_fn(pred_mean, distances, log_var.exp())
                if is_variational:
                    # Scaled by the number of minibatches per epoch so that,
                    # summed over one epoch, the total KL contribution matches
                    # a single kl_divergence() call rather than being counted
                    # once per minibatch (Blundell et al., 2015).
                    loss = loss + kl_weight * model.kl_divergence() / len(train_loader)
            loss.backward()
            # Caps how far a single step can move any parameter. Without
            # this, a large gradient (e.g. from a big residual early in
            # training) can push log_var's raw pre-soft-clamp value far
            # enough that tanh saturates to exactly +-1 in float32 --
            # zeroing the gradient for real, the same "stuck at the
            # boundary" failure the soft clamp was meant to prevent, just
            # reached by overshooting through it instead of hitting a hard
            # clamp directly.
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            if has_raw_log_var_diagnostic:
                raw = model.last_raw_log_var
                raw_log_var_min = min(raw_log_var_min, raw.min().item())
                raw_log_var_max = max(raw_log_var_max, raw.max().item())
                raw_log_var_sum += raw.sum().item()
                raw_log_var_count += raw.numel()
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for images, distances in val_loader:
                images, distances = images.to(device), distances.to(device)
                if is_evidential:
                    out = model(images)
                    dist = NormalInverseGamma(out["loc"], out["lmbda"], out["alpha"], out["beta"])
                    val_loss += der_loss_fn(dist, distances.unsqueeze(-1)).item() * images.size(0)
                    val_mae += torch.abs(out["loc"].squeeze(-1) - distances).sum().item()
                else:
                    pred_mean, log_var = model(images)
                    val_loss += loss_fn(pred_mean, distances, log_var.exp()).item() * images.size(0)
                    val_mae += torch.abs(pred_mean - distances).sum().item()
        val_loss /= len(val_ds)
        val_mae /= len(val_ds)

        line = (
            f"epoch {epoch:2d}/{epochs} | "
            f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_mae {val_mae:.4f}"
        )
        if has_raw_log_var_diagnostic:
            # Watches for the raw (pre-soft-clamp) log_var branch drifting
            # toward the +-20-ish range where tanh's gradient underflows to
            # exactly 0 in float32 -- the failure this project hit after
            # gradient clipping fixed the mean branch but left this one
            # stuck. Catching the drift here (live, per epoch) beats
            # inferring it after training from inference.py's output.
            raw_log_var_mean = raw_log_var_sum / raw_log_var_count
            line += (
                f" | raw_log_var[min {raw_log_var_min:.2f}, "
                f"mean {raw_log_var_mean:.2f}, max {raw_log_var_max:.2f}]"
            )
        if is_evidential:
            line += (
                f" | alpha[min {alpha_min:.4f}, mean {alpha_sum / evidential_count:.4f}]"
                f" | lmbda[min {lmbda_min:.4f}, mean {lmbda_sum / evidential_count:.4f}]"
            )
        print(line)

    return model, device


if __name__ == "__main__":
    train()
