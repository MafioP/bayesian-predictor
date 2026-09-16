"""Entry point: train the distance regressor and save a checkpoint.

Run inference.py separately to load the checkpoint and generate predictions
without retraining.

Usage:
    python main.py                    # trains DistanceNetMobileNet (MC Dropout), same as before
    python main.py --model variational  # trains DistanceNetMobileNetVariational (hand-rolled Bayes by Backprop)
    python main.py --model blitz        # trains DistanceNetMobileNetBlitz (BLiTZ's Bayes by Backprop)
    python main.py --model der          # trains DistanceNetMobileNetDER (Deep Evidential Regression)

Training all four under their default checkpoint names lets
compare_uncertainty.py load them together and compare their uncertainty
estimates on the same images.
"""

import argparse

import torch

from model import (
    DistanceNetMobileNet,
    DistanceNetMobileNetVariational,
    DistanceNetMobileNetBlitz,
    DistanceNetMobileNetDER,
)
from train import train

MODEL_CHOICES = {
    "dropout": DistanceNetMobileNet,
    "variational": DistanceNetMobileNetVariational,
    "blitz": DistanceNetMobileNetBlitz,
    "der": DistanceNetMobileNetDER,
}

# The "dropout" default keeps its original filename so existing checkpoints
# and inference.py's default CHECKPOINT_PATH keep working unchanged.
DEFAULT_CHECKPOINTS = {
    # "dropout": "distance_net.pt",
    # "variational": "distance_net_variational.pt",
    # "blitz": "distance_net_blitz.pt",
    "der": "distance_net_der.pt",
    # Trained by train_der.py (TorchUncertainty's RegressionRoutine/TUTrainer),
    # not by `main.py --model der` -- listed here only so
    # compare_uncertainty.py can load it alongside the other checkpoints.
    "der_tu": "distance_net_der_tu.pt",
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
