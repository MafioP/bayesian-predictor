"""
Compare epistemic/aleatoric uncertainty across trained model variants on
the same images.

Requires the checkpoints you want to compare already exist:
    python main.py                      # -> distance_net.pt (MC Dropout)
    python main.py --model variational  # -> distance_net_variational.pt (hand-rolled Bayes by Backprop)
    python main.py --model blitz        # -> distance_net_blitz.pt (BLiTZ's Bayes by Backprop)

Usage:
    python compare_uncertainty.py                       # compares every checkpoint that exists
    python compare_uncertainty.py --models dropout blitz # compares just these two
"""

import argparse
import os

from dataset import DistanceDataset
from inference import load_model, predict_with_uncertainty
from main import DEFAULT_CHECKPOINTS


def main(models, num_images: int = 8):
    loaded = []
    device = None
    for name in models:
        checkpoint = DEFAULT_CHECKPOINTS[name]
        if not os.path.exists(checkpoint):
            print(f"skipping {name!r}: {checkpoint} not found (train it first with `python main.py --model {name}`)")
            continue
        model, device = load_model(checkpoint, device=device)
        loaded.append((name, model))

    if not loaded:
        print("No checkpoints found -- train at least one model first (see usage above).")
        return

    ds = DistanceDataset(num_images, seed=42)  # same seed as inference.py's show_predictions

    header = f"{'true':>6} | " + " | ".join(f"{name:^24}" for name, _ in loaded)
    subheader = f"{'':>6} | " + " | ".join(f"{'mean':>9} {'ep_std':>6} {'al_std':>6}" for _ in loaded)
    print(header)
    print(subheader)
    print("-" * len(subheader))

    for i in range(num_images):
        image, true_dist = ds[i]
        cells = []
        for _, model in loaded:
            r = predict_with_uncertainty(model, device, image)
            cells.append(f"{r['mean']:9.1f} {r['epistemic_std']:6.2f} {r['aleatoric_std']:6.2f}")
        print(f"{true_dist:6.1f} | " + " | ".join(cells))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(DEFAULT_CHECKPOINTS),
        default=list(DEFAULT_CHECKPOINTS),
        help="Which model variants to compare (default: all of them).",
    )
    args = parser.parse_args()
    main(args.models)
