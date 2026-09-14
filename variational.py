"""
Bayes by Backprop (Blundell et al., 2015, "Weight Uncertainty in Neural
Networks") -- the "full variational" alternative to MC Dropout for
epistemic uncertainty, flagged as a future upgrade in README.md.

MC Dropout approximates "many slightly different models" cheaply, by
randomly zeroing neurons in an otherwise ordinary, deterministic layer.
Bayes by Backprop does the real thing instead: every weight gets its own
learned Gaussian distribution (mean + spread) rather than a single value,
and a fresh weight is *sampled* from that distribution on every forward
pass. Running the same input through many times and looking at the spread
of outputs is still how you get an epistemic estimate, but the randomness
now comes from actual weight uncertainty rather than a dropout mask.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear where each weight and bias is a
    Gaussian distribution, not a single number.

    - `mu` is the distribution's mean -- initialized like an ordinary
      Linear layer's weights.
    - `sigma` is its spread, stored as an unconstrained `rho` and passed
      through `softplus` to keep it positive with a smooth, never-zero
      gradient -- the same reasoning as this project's soft-clamped
      log_var (see model.py): a hard positivity constraint (e.g. clamping
      a raw value at some floor) can zero out the gradient and strand a
      parameter where training can never move it again.
    - Every forward pass draws w = mu + sigma * eps, eps ~ N(0, 1) (the
      "reparameterization trick"): sampling happens on a fixed random
      draw multiplied and added to differentiable tensors, so gradients
      can flow into mu and sigma even though the layer is stochastic.

    `rho` is initialized so that `sigma` starts out equal to `prior_sigma`
    -- i.e. the learned distribution starts (almost) matching the prior,
    since `mu` starts small too. Starting instead with a much smaller
    `sigma` (closer to deterministic) sounds safer but backfires: the KL
    term below is largely a `log(prior_sigma / sigma)` ratio, so an
    initial `sigma` far below the prior creates a huge, entirely
    artificial KL penalty before a single gradient step has happened,
    which then dominates and fights the data-fit term from step one.
    Starting at the prior keeps that ratio near zero, so training starts
    from a KL-neutral point and lets the data decide which weights need
    to move away from the prior, and by how much.

    Without a countervailing pressure, sigma would otherwise shrink toward
    zero over training (an ordinary deterministic layer has strictly lower
    loss on the training data than a noisy one) -- see `kl_divergence()`
    for the term that keeps it from collapsing.
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.prior_sigma = prior_sigma

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))

        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.zeros_(self.bias_mu)
        # Inverse of softplus: the rho that makes softplus(rho) == prior_sigma.
        rho_init = math.log(math.expm1(prior_sigma))
        nn.init.constant_(self.weight_rho, rho_init)
        nn.init.constant_(self.bias_rho, rho_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)

        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """
        KL(q(w) || p(w)), summed over every weight and bias in this layer:
        q is the learned per-weight Gaussian, p is a fixed N(0,
        prior_sigma^2) prior shared by all weights. This is the
        regularization term that keeps sigma from collapsing to zero --
        without it, the loss below has nothing pushing back against
        "just become a deterministic layer," since noise can only hurt
        the data-fit term. Closed-form because both q and p are Gaussian.
        """
        return self._kl_term(self.weight_mu, F.softplus(self.weight_rho)) + self._kl_term(
            self.bias_mu, F.softplus(self.bias_rho)
        )

    def _kl_term(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        prior_sigma = self.prior_sigma
        return (
            math.log(prior_sigma)
            - torch.log(sigma)
            + (sigma**2 + mu**2) / (2 * prior_sigma**2)
            - 0.5
        ).sum()
