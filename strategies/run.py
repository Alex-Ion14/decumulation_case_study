"""Submission: a portfolio-weights model combined with the spend-down withdrawal rule.

The two parts are separate:
  - HOW MUCH to withdraw: `strategies/withdrawal.py`, "withdraw 90% of wealth / years left".
  - HOW to invest: a weights model. `StaticWeights` below is a stand-in (7 learned weights,
    the same every year) to be replaced by the weights team's model. Any model works as long
    as it has `parameters()` (the tensors to train) and `weights(obs)` (shape (A,) or (N, A)).

Why the weights are trained WITH the withdrawal rule in the loop: the best weights depend on
how you withdraw. Spending that adapts to wealth absorbs bad years, so it can carry different
risk than a fixed 4% withdrawal.

Why on train + a stressed copy of train: with the spend-down rule the train scenarios almost
never run out of money, so the ruin term in the loss gives no signal and the optimiser
simply buys more risk. That looked better on train and worse on validation. Adding a copy of
the train scenarios with lower returns (shares -3.5%/yr, bonds -1%, cash -1.5%) brings the
ruin signal back.

No other files were changed.
"""

import torch

from strategies.withdrawal import spend_down_rate
from utils import DTYPE, Scenarios, evaluate, simulate

STEPS = 300
LEARNING_RATE = 0.05
TAU_START, TAU_END = 0.1, 0.005  # smoothness of the ruin approximation (as in bengen_optimised)
STRESS = {"growth": 0.035, "bonds": 0.01, "cash": 0.015}  # lower yearly returns in the stressed copy


class StaticWeights:
    """Stand-in for the weights team's model: one learned weight per asset, every year."""

    def __init__(self, n_assets):
        self.logits = torch.zeros(n_assets, dtype=DTYPE, requires_grad=True)

    def parameters(self):
        return [self.logits]

    def weights(self, obs):
        return torch.softmax(self.logits, dim=0)


def with_stressed_copy(sc):
    """The scenarios plus a copy with lower returns (growth: 3 equity/REIT assets, bonds: 3, cash: 1)."""
    shift = torch.tensor([STRESS["growth"]] * 3 + [STRESS["bonds"]] * 3 + [STRESS["cash"]], dtype=DTYPE)
    return Scenarios(torch.cat([sc.returns, sc.returns - shift], dim=1),
                     torch.cat([sc.inflation, sc.inflation], dim=1),
                     torch.cat([sc.market, sc.market], dim=1), sc.asset_names, sc.market_names)


class Strategy:
    def __init__(self, weights_model=None):
        self.weights_model = weights_model

    def fit(self, train, cfg):
        if self.weights_model is None:
            self.weights_model = StaticWeights(train.n_assets)
        scenarios = with_stressed_copy(train)
        optimizer = torch.optim.Adam(self.weights_model.parameters(), lr=LEARNING_RATE)

        for step in range(STEPS):
            tau = TAU_START * (TAU_END / TAU_START) ** (step / (STEPS - 1))
            out = simulate(self, scenarios, cfg)  # the withdrawal rule is inside self.act
            smooth_ruin = torch.sigmoid(-out["margins"].min(dim=0).values / tau).mean()
            smooth_j = cfg.alpha * out["norm_wd"].mean() - (1 - cfg.alpha) * smooth_ruin
            optimizer.zero_grad()
            (-smooth_j).backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                m = evaluate(self, train, cfg)
                print(f"  step {step + 1}: train J {m['J']:.4f}, ruin {m['ruin_prob']:.4f}, norm_wd {m['norm_wd']:.4f}")

        if isinstance(self.weights_model, StaticWeights):
            for name, w in zip(train.asset_names, self.weights_model.weights(None).tolist()):
                print(f"    {name:<20s} {w:7.2%}")

    def act(self, obs):
        return self.weights_model.weights(obs), spend_down_rate(obs)
