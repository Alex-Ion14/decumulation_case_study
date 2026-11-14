"""Submission: a portfolio-weights model combined with the spend-down withdrawal rule.

The two parts are separate:
  - HOW MUCH to withdraw: `strategies/withdrawal.py`, "withdraw 90% of wealth / years left".
  - HOW to invest: a weights model, chosen with WEIGHTS below. The default ("yield_trained")
    is the weights team's yield model (yield_mlp/), TRAINED IN `fit`: 50 small networks, each
    on its own block bootstrap of the train scenarios, with a fixed 4% withdrawal (so our
    withdrawal rule plays no part in choosing the weights). The portfolio is the average of
    the 50 networks' portfolios for last year's 5Y and 30Y Treasury yields. See
    `strategies/yield_weights.py`. Takes about 2 minutes.
    Any model works as long as it has `parameters()` (the tensors to train, or none) and
    `weights(obs)` (shape (A,) or (N, A)).

Challenge rules: with "yield_trained", `fit` only uses the train scenarios it receives, so the
scores on data/validation.parquet are a fair check. The other tables ("yearly", "yield") were
learned beforehand from train AND validation scenarios (see yield_mlp/README.md); with those,
`fit` does nothing and only --test data/test_set.parquet is a fair check.

The notes below are about WEIGHTS = "learned" (7 weights trained with our withdrawal rule).

Why the weights are trained WITH the withdrawal rule in the loop: the best weights depend on
how you withdraw. Spending that adapts to wealth absorbs bad years, so it can carry different
risk than a fixed 4% withdrawal.

Why on train + a stressed copy of train: with the spend-down rule the train scenarios almost
never run out of money, so the ruin term in the loss gives no signal and the optimiser
simply buys more risk. That looked better on train and worse on validation. Adding a copy of
the train scenarios with lower returns (shares -3.5%/yr, bonds -1%, cash -1.5%) brings the
ruin signal back.

Why bootstrapped batches: every training step draws BATCH new scenarios from the train
scenarios with the data team's block bootstrap (copied into `strategies/block_bootstrap.py`, blocks of BLOCK_YEARS years),
plus their stressed copy. The model then never sees the same 100 scenarios twice and cannot
memorise them. This only uses the train scenarios that `fit` receives, so it keeps to the
challenge rules. Set BATCH = None to train on the train scenarios themselves.

No files outside strategies/ and experiments/ were changed by the withdrawal team.
"""

import os
import sys

import torch

# The imports below live in the repository root. main.py puts it on the path already; this
# line also makes `python strategies/run.py` (or an editor's Run button) work.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from strategies.block_bootstrap import BlockBootstrap  # noqa: E402
from strategies.withdrawal import K, spend_down_rate  # noqa: E402
from strategies.yield_weights import AveragedYieldWeights, train_yield_networks  # noqa: E402
from utils import DTYPE, Scenarios, evaluate, simulate  # noqa: E402

STEPS = 300
BATCH = 256  # bootstrapped scenarios per step (None: use the train scenarios as they are)
BLOCK_YEARS = 5  # length of a bootstrap block; the data team recommends 5 to 10
LEARNING_RATE = 0.05
TAU_START, TAU_END = 0.1, 0.005  # smoothness of the ruin approximation (as in bengen_optimised)
STRESS = {"growth": 0.035, "bonds": 0.01, "cash": 0.015}  # lower yearly returns in the stressed copy

# Which weights to use when no weights model is passed in:
#   "yield_trained" : the weights team's yield model, trained in fit (strategies/yield_weights.py);
#   "yearly"  : the weights team's yearly table (yield_mlp/yearly_table.py): one fixed
#               portfolio per year, the average of what their yield model holds that year;
#   "yield"   : the weights team's yield table (yield_mlp/hardcoded_strategy.py): the
#               portfolio depends on last year's 5Y and 30Y Treasury yields;
#   "bengen"  : the fixed 4% rule mix (50% US equity, 50% US Treasury 5Y);
#   "learned" : StaticWeights, 7 weights trained together with the withdrawal rule.
# "yield_trained" and "learned" are trained in fit; the others are fixed.
WEIGHTS = "yield_trained"
BENGEN_ALLOCATION = {"US equity": 0.5, "US Treasury 5Y": 0.5}


def reorder(table_assets, asset_names):
    """Index that puts weights listed in `table_assets` order into `asset_names` order."""
    if sorted(table_assets) != sorted(asset_names):
        raise ValueError(f"the weights table has assets {table_assets}, the simulator {asset_names}")
    return torch.tensor([table_assets.index(name) for name in asset_names])


class YearlyTableWeights:
    """The weights team's yearly table: portfolio number t in year t, the same in every scenario."""

    def __init__(self, asset_names):
        from yield_mlp import yearly_table

        table = torch.tensor(yearly_table.YEARLY_PORTFOLIOS, dtype=DTYPE)  # (30, 7)
        self.table = table[:, reorder(yearly_table.ASSETS, asset_names)]

    def parameters(self):
        return []

    def weights(self, obs):
        return self.table[obs.year]


class YieldTableWeights:
    """The weights team's yield table: the portfolio follows last year's 5Y and 30Y yields."""

    def __init__(self, asset_names):
        from yield_mlp import hardcoded_strategy

        self.table = hardcoded_strategy
        self.order = reorder(hardcoded_strategy.ASSETS, asset_names)

    def parameters(self):
        return []

    def weights(self, obs):
        if obs.year == 0:  # no yields known yet
            return torch.tensor(self.table.FIRST_YEAR_PORTFOLIO, dtype=DTYPE)[self.order]
        last = obs.market_history[-1]
        five_year = last[:, obs.market_names.index("US_ZC_5Y")]
        thirty_year = last[:, obs.market_names.index("US_ZC_30Y")]
        return self.table.look_up_portfolio(five_year, thirty_year)[:, self.order]


class FixedWeights:
    """The same given allocation in every year and every scenario; nothing to train."""

    def __init__(self, allocation, asset_names):
        self.w = torch.tensor([allocation.get(name, 0.0) for name in asset_names], dtype=DTYPE)

    def parameters(self):
        return []

    def weights(self, obs):
        return self.w


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
    def __init__(self, weights_model=None, k=K, seed=0, weights=WEIGHTS):
        self.weights_model = weights_model
        self.k = k  # withdrawal factor: withdraw k * wealth / years left
        self.seed = seed  # for the bootstrap draws
        self.weights = weights  # used only when no weights_model is given (see WEIGHTS)

    def fit(self, train, cfg):
        if self.weights_model is None:
            models = {
                "yield_trained": lambda: AveragedYieldWeights(train_yield_networks(train, seed=self.seed)),
                "yearly": lambda: YearlyTableWeights(train.asset_names),
                "yield": lambda: YieldTableWeights(train.asset_names),
                "bengen": lambda: FixedWeights(BENGEN_ALLOCATION, train.asset_names),
                "learned": lambda: StaticWeights(train.n_assets),
            }
            if self.weights not in models:
                raise ValueError(f"unknown weights {self.weights!r}: use one of {list(models)}")
            self.weights_model = models[self.weights]()
        if not list(self.weights_model.parameters()):
            return  # fixed weights: nothing to train
        sampler = BlockBootstrap(train, block_years=BLOCK_YEARS, seed=self.seed) if BATCH else None
        optimizer = torch.optim.Adam(self.weights_model.parameters(), lr=LEARNING_RATE)

        for step in range(STEPS):
            tau = TAU_START * (TAU_END / TAU_START) ** (step / (STEPS - 1))
            batch = sampler.sample(BATCH) if sampler else train
            out = simulate(self, with_stressed_copy(batch), cfg)  # the withdrawal rule is inside self.act
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
        return self.weights_model.weights(obs), spend_down_rate(obs, self.k)


if __name__ == "__main__":
    # `python strategies/run.py [options]` is the same as `python main.py strategies/run.py [options]`
    import main

    sys.argv = [os.path.join(ROOT, "main.py"), os.path.abspath(__file__), *sys.argv[1:]]
    main.main()
