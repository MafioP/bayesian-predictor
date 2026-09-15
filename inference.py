"""Load a trained checkpoint and run predictions, without retraining.

Usage:
    python inference.py
"""

import torch
import matplotlib.pyplot as plt

from dataset import DistanceDataset
from model import MODEL_REGISTRY

CHECKPOINT_PATH = "distance_net.pt"
MC_SAMPLES = 30


def load_model(checkpoint_path: str = CHECKPOINT_PATH, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_cls = MODEL_REGISTRY[checkpoint["model_cls"]]
    model = model_cls().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, device


def predict_with_uncertainty(
    model, device, image: torch.Tensor, num_samples: int = MC_SAMPLES
) -> dict:
    """
    Runs num_samples stochastic forward passes with dropout kept active
    (Monte Carlo Dropout) and decomposes the resulting spread into:
      - aleatoric_std: average of the model's own per-pass uncertainty
        estimate (log_var head) -- irreducible ambiguity in this image.
      - epistemic_std: spread of the mean predictions across passes --
        the model's own lack of confidence in its weights for this input.
    total_std combines both, per Kendall & Gal (2017).

    image: a single (1, H, W) tensor, values in [0, 1].
    """
    if getattr(model, "is_evidential", False):
        # Deep Evidential Regression (DistanceNetMobileNetDER) needs no
        # sampling loop at all: a single deterministic forward pass
        # already gives every parameter needed for both uncertainty
        # types, via closed-form properties of the fitted
        # Normal-Inverse-Gamma distribution -- see model.py's docstring.
        model.eval()
        with torch.no_grad():
            out = model(image.unsqueeze(0).to(device))
        loc = out["loc"].item()
        lmbda = out["lmbda"].item()
        alpha = out["alpha"].item()
        beta = out["beta"].item()
        aleatoric_var = beta / (alpha - 1)
        epistemic_var = beta / ((alpha - 1) * lmbda)
        return {
            "mean": loc,
            "epistemic_std": epistemic_var ** 0.5,
            "aleatoric_std": aleatoric_var ** 0.5,
            "total_std": (epistemic_var + aleatoric_var) ** 0.5,
        }

    if hasattr(model, "kl_divergence"):
        # Variational (BayesianLinear) models sample fresh weights on
        # every forward call regardless of train/eval mode, so eval() is
        # enough here -- it just keeps BatchNorm/the frozen backbone
        # deterministic, same as it would for any other model.
        model.eval()
    else:
        model.train()  # enables dropout; frozen backbones stay eval via their own train() override

    means = []
    variances = []
    with torch.no_grad():
        for _ in range(num_samples):
            pred_mean, log_var = model(image.unsqueeze(0).to(device))
            means.append(pred_mean.item())
            variances.append(log_var.exp().item())

    means_t = torch.tensor(means)
    variances_t = torch.tensor(variances)

    epistemic_var = means_t.var(unbiased=True).item()
    aleatoric_var = variances_t.mean().item()

    return {
        "mean": means_t.mean().item(),
        "epistemic_std": epistemic_var ** 0.5,
        "aleatoric_std": aleatoric_var ** 0.5,
        "total_std": (epistemic_var + aleatoric_var) ** 0.5,
    }


def show_predictions(model, device, num_samples: int = 8, out_path: str = "predictions.png"):
    ds = DistanceDataset(num_samples, seed=42)

    fig, axes = plt.subplots(2, num_samples // 2, figsize=(2.4 * num_samples // 2, 5.5))
    for i, ax in enumerate(axes.flat):
        image, true_dist = ds[i]
        result = predict_with_uncertainty(model, device, image)

        ax.imshow(image.squeeze(0), cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"true {true_dist:.1f} | pred {result['mean']:.1f} ± {result['total_std']:.1f}\n"
            f"(aleatoric {result['aleatoric_std']:.1f}, epistemic {result['epistemic_std']:.1f})",
            fontsize=8,
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    model, device = load_model()
    print(f"Loaded checkpoint on {device}")
    show_predictions(model, device)
