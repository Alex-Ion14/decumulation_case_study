"""Experiment 4: does the network need all its inputs?

    python -B machine_learning_investing/run_input_selection.py

Experiment 3 showed that the network leans on the interest rates and hardly on anything
else. Here the market-aware MLP is trained again with fewer and fewer inputs, on the same
folds and scored on the same held-out scenarios as in experiment 1. If a smaller set of
inputs scores as well, it is the better model: less to overfit and easier to explain.

Writes input_selection_folds.csv to machine_learning_investing/results/ and prints a summary.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # so that this also runs as a script

from machine_learning_investing.data import load_fold  # noqa: E402
from machine_learning_investing.features import MARKET_NAMES, STATE_NAMES  # noqa: E402
from machine_learning_investing.policies import MLPPolicy, StaticPolicy  # noqa: E402
from machine_learning_investing.run_comparison import N_FOLDS, N_PROCESSES, NETWORK, RESULTS  # noqa: E402
from machine_learning_investing.training import metrics, train  # noqa: E402
from utils import Config  # noqa: E402

RATES = ["5Y yield", "30Y yield", "T-bill yield", "break-even inflation", "10Y real yield", "5Y minus T-bill",
         "30Y minus 5Y"]
FLAG = ["has history"]  # tells the network that year 0 has no market information yet
# name: the inputs the network gets (None: all 24)
INPUT_SETS = {
    "All inputs": None,
    "Rates and spreads + wealth and time": STATE_NAMES + RATES + FLAG,
    "Rates and spreads only": RATES + FLAG,
    "Three yields (5Y, 30Y, T-bill)": ["5Y yield", "30Y yield", "T-bill yield"] + FLAG,
    "Two yields (5Y, 30Y)": ["5Y yield", "30Y yield"] + FLAG,
    "Everything except rates and spreads": [name for name in STATE_NAMES + MARKET_NAMES if name not in RATES] + FLAG,
}
BASELINE = "Static (no inputs)"
FULL = "All inputs"


def run_fold(k):
    torch.set_num_threads(1)
    cfg, fold = Config(), load_fold(k)
    models = {BASELINE: (StaticPolicy, {"learning_rate": 0.05}),
              **{name: (lambda inputs=inputs: MLPPolicy(inputs=inputs), NETWORK) for name, inputs in INPUT_SETS.items()}}
    rows = []
    for name, (make, settings) in models.items():
        torch.manual_seed(k)  # the same starting weights wherever the model is in the list
        policy = train(make(), fold.train, fold.validation, cfg, seed=k, **settings)[0]
        n_inputs = len(policy.features.names) if hasattr(policy, "features") else 0
        for part in ("train", "held_out"):
            rows.append({"fold": k, "inputs": name, "n_inputs": n_inputs, "part": part,
                         **metrics(policy, getattr(fold, part), cfg)})
    return rows


def main():
    os.makedirs(RESULTS, exist_ok=True)
    with ProcessPoolExecutor(N_PROCESSES) as pool:
        folds = pd.DataFrame([row for rows in pool.map(run_fold, range(N_FOLDS)) for row in rows])
    folds.to_csv(os.path.join(RESULTS, "input_selection_folds.csv"), index=False)

    held = folds[folds.part == "held_out"]
    table = held.groupby("inputs", sort=False)[
        ["n_inputs", "score", "ruin_prob", "median_final_wealth", "p10_final_wealth", "turnover"]].mean()
    for metric in ("score", "median_final_wealth"):
        wide = held.pivot(index="fold", columns="inputs", values=metric)
        difference = wide.sub(wide[FULL], axis=0)
        table[f"{metric}_minus_all"] = difference.mean()
        table[f"{metric}_folds_at_least_as_good"] = (difference >= 0).mean()
    train_part = folds[folds.part == "train"].groupby("inputs", sort=False)
    table["train_score"] = train_part.score.mean()
    table["train_median_final_wealth"] = train_part.median_final_wealth.mean()
    pd.set_option("display.width", 300)
    print(f"Held-out results, averaged over {N_FOLDS} folds ('minus_all': difference with the network with all inputs)")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
