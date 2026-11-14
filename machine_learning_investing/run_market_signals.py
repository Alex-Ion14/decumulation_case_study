"""Experiment 3: which information does the network use, and what does it do with it?

    python -B machine_learning_investing/run_market_signals.py

Trains the market-aware MLP on the first N_FOLDS sets and asks two questions on the held-out
scenarios:

  Importance  How much does the score drop when one input (or one group of inputs) is
              shuffled between the scenarios, so that it no longer says anything? An input
              the network does not rely on can be shuffled for free.
  Direction   How does the portfolio change when one input goes from low (1 standard
              deviation below its average) to high (1 above), with all other inputs as
              they were?

Writes signals_importance.csv and signals_direction.csv to machine_learning_investing/results/.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # so that this also runs as a script

from machine_learning_investing.data import load_fold  # noqa: E402
from machine_learning_investing.run_comparison import MODELS, N_PROCESSES, RESULTS  # noqa: E402
from machine_learning_investing.training import metrics, train  # noqa: E402
from utils import ASSET_NAMES, Config  # noqa: E402

N_FOLDS = 8
N_SHUFFLES = 5
MODEL = "MLP, with market"
GROUPS = {
    "wealth and time": ["year", "log wealth", "log funded ratio"],
    "interest rates and spreads": ["5Y yield", "30Y yield", "T-bill yield", "break-even inflation",
                                   "10Y real yield", "5Y minus T-bill", "30Y minus 5Y"],
    "risk indicators": ["log VIX", "credit spread", "unemployment"],
    "last year's returns and inflation": ["last return US equity", "last return EM equity", "last return US REIT",
                                          "last return 5Y", "last return 30Y", "last return TIPS",
                                          "last return cash", "last inflation"],
    "US equity trend (3 and 5 years)": ["US equity 3y average", "US equity 5y average"],
}


class Probe(nn.Module):
    """Sits between the features and the network: can shuffle columns and remembers what it saw."""

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.shuffle, self.generator, self.seen = [], torch.Generator(), []

    def forward(self, obs):
        x = self.features(obs)
        if self.shuffle:
            x = x.clone()
            order = torch.randperm(len(x), generator=self.generator)  # the same order for a whole group
            x[:, self.shuffle] = x[order][:, self.shuffle]
        self.seen.append(x)
        return x


def run_fold(k):
    torch.set_num_threads(1)
    cfg, fold = Config(), load_fold(k)
    make, settings = MODELS[MODEL]
    policy = train(make(), fold.train, fold.validation, cfg, seed=k, **settings)[0]
    names = policy.features.names
    probe = policy.features = Probe(policy.features)
    base = metrics(policy, fold.held_out, cfg)["score"]
    seen = torch.cat(probe.seen[1:])  # the inputs of all years except year 0 (no market history yet)

    importance = []
    for label, columns in [*[(name, [name]) for name in names], *[(f"group: {g}", c) for g, c in GROUPS.items()]]:
        probe.shuffle = [names.index(c) for c in columns]
        probe.generator.manual_seed(k)
        shuffled = sum(metrics(policy, fold.held_out, cfg)["score"] for _ in range(N_SHUFFLES)) / N_SHUFFLES
        importance.append({"fold": k, "input": label, "score_drop": base - shuffled})
    probe.shuffle = []

    direction = []
    with torch.no_grad():
        for j, name in enumerate(names[:-1]):  # not the "has history" flag
            low, high = seen.clone(), seen.clone()
            low[:, j], high[:, j] = seen[:, j] - 1, seen[:, j] + 1
            if j < 3:  # the state features are not standardised: use their own spread
                spread = seen[:, j].std()
                low[:, j], high[:, j] = seen[:, j] - spread, seen[:, j] + spread
            change = (torch.softmax(policy.net(high), 1) - torch.softmax(policy.net(low), 1)).mean(dim=0)
            direction.append({"fold": k, "input": name, **dict(zip(ASSET_NAMES, change.tolist()))})
    return importance, direction


def main():
    os.makedirs(RESULTS, exist_ok=True)
    with ProcessPoolExecutor(N_PROCESSES) as pool:
        results = list(pool.map(run_fold, range(N_FOLDS)))
    importance = pd.DataFrame([row for rows, _ in results for row in rows])
    direction = pd.DataFrame([row for _, rows in results for row in rows])
    importance.to_csv(os.path.join(RESULTS, "signals_importance.csv"), index=False)
    direction.to_csv(os.path.join(RESULTS, "signals_direction.csv"), index=False)

    pd.set_option("display.width", 250)
    summary = importance.groupby("input", sort=False).score_drop.agg(["mean", "std", lambda s: (s > 0).mean()])
    summary.columns = ["score_drop", "std_over_folds", "folds_with_a_drop"]
    print(f"Importance: drop in held-out score when an input is shuffled ({N_FOLDS} folds)")
    print(summary.sort_values("score_drop", ascending=False).round(3).to_string())
    print("\nDirection: change in portfolio weight (percentage points) when an input goes from low to high")
    print((100 * direction.groupby("input", sort=False)[ASSET_NAMES].mean()).round(1).to_string())


if __name__ == "__main__":
    main()
