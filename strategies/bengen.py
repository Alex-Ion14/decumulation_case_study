"""The Bengen "4% rule": the baseline that every strategy is compared with.

As in Bengen's original 1994 study: every year, rebalance to 50% US stocks and 50%
intermediate-term (5-year) US Treasuries, and withdraw 4% of the initial capital
(adjusted for inflation).
"""

import torch

ALLOCATION = {
    "US equity": 0.50,
    "US Treasury 5Y": 0.50,
}  # assets that are not listed get weight 0
WITHDRAWAL_RATE = 0.04


class Strategy:
    def fit(self, train, cfg):
        pass  # nothing to learn: the rule is fixed

    def act(self, obs):
        weights = torch.tensor([ALLOCATION.get(name, 0.0) for name in obs.asset_names], dtype=torch.float64)
        return weights, WITHDRAWAL_RATE
