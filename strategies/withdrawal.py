"""Withdrawal rule: "divide what you have by the years you have left".

Every year, withdraw K times (current wealth / years left, this year included). After good
years you spend more, after bad years less, so the pot adjusts instead of running dry, and
little is left unused at the end.

    rate = min(K * W / n, 99.5% of W)        n = 30, 29, ..., 1

- K = 0.9 keeps a 10% buffer. On the train scenarios K = 1.0 looks better (they almost never
  run out), but on lower-return scenarios it roughly doubles the ruin probability.
- The 99.5% cap matters in the last year (n = 1): withdrawing all wealth leaves a margin of 0,
  which `simulate` counts as ruin.

This rule only decides HOW MUCH to withdraw; it works with any portfolio weights. It is
differentiable, so it can sit inside a weights-training loop that runs through `simulate`.
"""

import torch

K = 0.9
MAX_SHARE_OF_WEALTH = 0.995


def spend_down_rate(obs, k=K):
    """Withdrawal rate (as a share of the initial capital) for every scenario, shape (N,)."""
    years_left = obs.n_years - obs.year
    return torch.minimum(k * obs.wealth / years_left, MAX_SHARE_OF_WEALTH * obs.wealth)
