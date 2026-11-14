"""Check the block bootstrap: do the new scenarios behave like the original ones?

    python -B bootstrap/demo.py

It scores a few fixed strategies (a fixed allocation and a fixed withdrawal rate) on the
original train scenarios and on bootstrapped scenarios with different block lengths. A good
bootstrap gives about the same ruin probabilities as the original scenarios. With only 100
original scenarios, their own ruin probabilities are uncertain by a few percentage points
(the "+/-" row).
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # so that this also runs from inside the folder

from bootstrap import BlockBootstrap  # noqa: E402
from utils import ASSET_NAMES, Config, evaluate, load_scenarios  # noqa: E402

N_BOOTSTRAP = 20_000
BLOCK_YEARS = [1, 3, 5, 10]
PORTFOLIOS = {
    "50/50 equity/5Y": {"US equity": 0.5, "US Treasury 5Y": 0.5},
    "100% equity": {"US equity": 1.0},
    "100% cash": {"Cash": 1.0},
}
RATES = [0.04, 0.05]


class FixedStrategy:
    def __init__(self, allocation, rate):
        self.weights = torch.tensor([allocation.get(name, 0.0) for name in ASSET_NAMES], dtype=torch.float64)
        self.rate = rate

    def act(self, obs):
        return self.weights, self.rate


def ruin_probabilities(scenarios, cfg):
    return [evaluate(FixedStrategy(allocation, rate), scenarios, cfg)["ruin_prob"]
            for allocation in PORTFOLIOS.values() for rate in RATES]


def main():
    cfg = Config()
    train = load_scenarios(os.path.join(HERE, "..", "data", "train.parquet"))
    columns = [f"{name} @{rate:.0%}" for name in PORTFOLIOS for rate in RATES]

    print(f"Ruin probability on {train.n_scenarios} original and {N_BOOTSTRAP} bootstrapped scenarios")
    print(f"  {'':<22s}" + "".join(f"{column:>22s}" for column in columns))

    original = ruin_probabilities(train, cfg)
    print(f"  {'original':<22s}" + "".join(f"{p:>22.1%}" for p in original))
    uncertainty = [(p * (1 - p) / train.n_scenarios) ** 0.5 for p in original]
    print(f"  {'+/-':<22s}" + "".join(f"{u:>22.1%}" for u in uncertainty))

    for block_years in BLOCK_YEARS:
        new = BlockBootstrap(train, block_years=block_years, seed=0).sample(N_BOOTSTRAP)
        row = ruin_probabilities(new, cfg)
        print(f"  {f'blocks of {block_years} years':<22s}" + "".join(f"{p:>22.1%}" for p in row))


if __name__ == "__main__":
    main()
