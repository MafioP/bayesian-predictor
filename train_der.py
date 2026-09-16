"""
Training for DistanceNetMobileNetDER using TorchUncertainty's own training
routine instead of train.py's hand-rolled loop -- see
https://torch-uncertainty.github.io/auto_tutorials/Regression/tutorial_der_cubic.html

RegressionRoutine (a PyTorch LightningModule) owns the training/validation
step: given dist_family="nig", it takes the raw {loc, lmbda, alpha, beta}
dict our model's forward() already returns, builds a NormalInverseGamma
distribution from it, and calls the loss on (distribution, target) --
exactly what train.py's DER branch does by hand. TUTrainer (a Lightning
Trainer) then runs the actual epoch loop.

Saves a checkpoint in the same {"model_cls", "state_dict"} format
train.py/main.py use, so inference.py and compare_uncertainty.py can load
it like any other model -- as its own checkpoint (distance_net_der_tu.pt
by default), not overwriting distance_net_der.pt from the manual-loop
training path, so the two can be compared directly.

Still intentionally missing: the live per-epoch alpha/lmbda diagnostic
that caught Obstacles G and H in SESSION_NOTES.md. RegressionRoutine logs
its own metrics (RMSE, NLL) during training, which may or may not turn
out to be enough to replace it -- not yet evaluated.
"""

import torch
from torch.utils.data import DataLoader
from torch_uncertainty.losses import DERLoss
from torch_uncertainty.routines import RegressionRoutine
from torch_uncertainty.utils import TUTrainer

from dataset import DistanceDataset
from model import DistanceNetMobileNetDER

CHECKPOINT_PATH = "distance_net_der_tu.pt"


def train_der(
    epochs: int = 70,
    batch_size: int = 32,
    lr: float = 1e-3,
    train_size: int = 6000,
    val_size: int = 1000,
    reg_weight: float = 1e-3,
    min_alpha: float = 0.05,
    min_lmbda: float = 0.05,
    grad_clip_norm: float = 5.0,
    accelerator: str = "auto",
    checkpoint_path: str = CHECKPOINT_PATH,
):
    train_ds = DistanceDataset(train_size, seed=0)
    val_ds = DistanceDataset(val_size, seed=1)  # different seed -> disjoint samples
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DistanceNetMobileNetDER(min_alpha=min_alpha, min_lmbda=min_lmbda)

    def optim_recipe(m: torch.nn.Module) -> torch.optim.Optimizer:
        trainable_params = [p for p in m.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable_params, lr=lr)

    routine = RegressionRoutine(
        model=model,
        output_dim=1,
        loss=DERLoss(reg_weight=reg_weight),
        dist_family="nig",
        optim_recipe=optim_recipe,
    )

    trainer = TUTrainer(
        accelerator=accelerator,
        max_epochs=epochs,
        # Lightning's built-in equivalent of train.py's manual
        # torch.nn.utils.clip_grad_norm_ call -- same reasoning applies
        # here (Obstacle F-adjacent instability from large single steps).
        gradient_clip_val=grad_clip_norm,
    )
    trainer.fit(model=routine, train_dataloaders=train_loader, val_dataloaders=val_loader)

    torch.save(
        {"model_cls": "DistanceNetMobileNetDER", "state_dict": routine.model.state_dict()},
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    return routine, trainer


if __name__ == "__main__":
    train_der()
