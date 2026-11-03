"""Optimised Bengen strategy: the best static allocation and fixed withdrawal rate.

Like the 4% rule, this strategy uses the same allocation and withdrawal rate in every year
and every scenario, but it *learns* them from the train scenarios with gradient ascent:

    weights = softmax(allocation_logits)                     (non-negative, sum to 1)
    rate    = wd_min + (wd_max - wd_min) * sigmoid(rate_logit)  (stays within the bounds)

We maximise the score J = alpha * norm_wd - (1 - alpha) * ruin_prob. Ruin is a yes/no
event and has no useful gradient, so during training it is replaced by a smooth version:
sigmoid(-lowest_margin / tau). The temperature tau decreases during training, so the
smooth version gets closer and closer to the real ruin indicator.

This file also produces the frontier in main.py. If you want to build on it, copy it to
strategies/run.py first, so the frontier stays the same.
"""

import torch

from utils import DTYPE, evaluate, simulate

STEPS = 300  # gradient steps, each on all train scenarios
LEARNING_RATE = 0.05
TAU_START = 0.1  # smoothness of the ruin approximation at the start of training ...
TAU_END = 0.005  # ... and at the end


class Strategy:
    def fit(self, train, cfg):
        self.cfg = cfg
        self.allocation_logits = torch.zeros(train.n_assets, dtype=DTYPE, requires_grad=True)  # equal weights
        self.rate_logit = torch.tensor(-2.0, dtype=DTYPE, requires_grad=True)  # rate of about 4%
        optimizer = torch.optim.Adam([self.allocation_logits, self.rate_logit], lr=LEARNING_RATE)

        for step in range(STEPS):
            tau = TAU_START * (TAU_END / TAU_START) ** (step / (STEPS - 1))
            out = simulate(self, train, cfg)
            lowest_margin = out["margins"].min(dim=0).values  # <= 0 means ruined
            smooth_ruin_prob = torch.sigmoid(-lowest_margin / tau).mean()
            smooth_j = cfg.alpha * out["norm_wd"].mean() - (1 - cfg.alpha) * smooth_ruin_prob

            optimizer.zero_grad()
            (-smooth_j).backward()  # minimising -J maximises J
            optimizer.step()

            if (step + 1) % 100 == 0:
                m = evaluate(self, train, cfg)
                print(f"  step {step + 1}: train J {m['J']:.4f}, ruin {m['ruin_prob']:.4f}, norm_wd {m['norm_wd']:.4f}")

        weights, rate = self.act(None)
        print(f"  learned withdrawal rate: {rate.item():.2%}")
        for name, weight in zip(train.asset_names, weights.tolist()):
            print(f"    {name:<34s} {weight:7.2%}")

    def act(self, obs):
        weights = torch.softmax(self.allocation_logits, dim=0)
        rate = self.cfg.wd_min + (self.cfg.wd_max - self.cfg.wd_min) * torch.sigmoid(self.rate_logit)
        return weights, rate
