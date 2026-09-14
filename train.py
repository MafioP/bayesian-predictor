"""Training loop for the (mean, log_var) distance regressor."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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

    # Variational models (see variational.py) expose kl_divergence(); a
    # dropout-based model like DistanceNetMobileNet doesn't, so this is
    # also how we tell the two apart without an explicit flag.
    is_variational = hasattr(model, "kl_divergence")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        raw_log_var_min = float("inf")
        raw_log_var_max = float("-inf")
        raw_log_var_sum = 0.0
        raw_log_var_count = 0
        for images, distances in train_loader:
            images, distances = images.to(device), distances.to(device)

            optimizer.zero_grad()
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
            if is_variational:
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
                pred_mean, log_var = model(images)
                val_loss += loss_fn(pred_mean, distances, log_var.exp()).item() * images.size(0)
                val_mae += torch.abs(pred_mean - distances).sum().item()
        val_loss /= len(val_ds)
        val_mae /= len(val_ds)

        line = (
            f"epoch {epoch:2d}/{epochs} | "
            f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_mae {val_mae:.4f}"
        )
        if is_variational:
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
        print(line)

    return model, device


if __name__ == "__main__":
    train()
