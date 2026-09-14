"""Entry point: train the distance regressor and save a checkpoint.

Run inference.py separately to load the checkpoint and generate predictions
without retraining.

Usage:
    python main.py                    # trains DistanceNetMobileNet (MC Dropout), same as before
    python main.py --model variational  # trains DistanceNetMobileNetVariational (Bayes by Backprop)

Training both under their default checkpoint names lets
compare_uncertainty.py load both and compare their epistemic estimates on
the same images.
"""

import argparse

import torch

from model import DistanceNetMobileNet, DistanceNetMobileNetVariational
from train import train

MODEL_CHOICES = {
    "dropout": DistanceNetMobileNet,
    "variational": DistanceNetMobileNetVariational,
}

# The "dropout" default keeps its original filename so existing checkpoints
# and inference.py's default CHECKPOINT_PATH keep working unchanged.
DEFAULT_CHECKPOINTS = {
    "dropout": "distance_net.pt",
    "variational": "distance_net_variational.pt",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CHOICES, default="dropout")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    model_cls = MODEL_CHOICES[args.model]
    checkpoint_path = args.checkpoint or DEFAULT_CHECKPOINTS[args.model]

    model, device = train(model_cls=model_cls)

    torch.save(
        {"model_cls": model_cls.__name__, "state_dict": model.state_dict()},
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")
