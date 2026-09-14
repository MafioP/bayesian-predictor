"""
Compare epistemic uncertainty from MC Dropout (DistanceNetMobileNet)
against full variational / Bayes by Backprop (DistanceNetMobileNetVariational)
on the same images, side by side.

Requires both checkpoints to already exist:
    python main.py                      # -> distance_net.pt
    python main.py --model variational  # -> distance_net_variational.pt

Usage:
    python compare_uncertainty.py
"""

from dataset import DistanceDataset
from inference import load_model, predict_with_uncertainty

DROPOUT_CHECKPOINT = "distance_net.pt"
VARIATIONAL_CHECKPOINT = "distance_net_variational.pt"


def main(num_images: int = 8):
    dropout_model, device = load_model(DROPOUT_CHECKPOINT)
    variational_model, _ = load_model(VARIATIONAL_CHECKPOINT, device=device)

    ds = DistanceDataset(num_images, seed=42)  # same seed as inference.py's show_predictions

    header = (
        f"{'true':>6} | "
        f"{'dropout mean':>12} {'ep_std':>7} {'al_std':>7} | "
        f"{'variational mean':>17} {'ep_std':>7} {'al_std':>7}"
    )
    print(header)
    print("-" * len(header))

    for i in range(num_images):
        image, true_dist = ds[i]
        d = predict_with_uncertainty(dropout_model, device, image)
        v = predict_with_uncertainty(variational_model, device, image)
        print(
            f"{true_dist:6.1f} | "
            f"{d['mean']:12.1f} {d['epistemic_std']:7.2f} {d['aleatoric_std']:7.2f} | "
            f"{v['mean']:17.1f} {v['epistemic_std']:7.2f} {v['aleatoric_std']:7.2f}"
        )


if __name__ == "__main__":
    main()
