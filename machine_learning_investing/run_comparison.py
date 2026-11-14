"""Experiment 1: which kind of investing model works best?

    python -B machine_learning_investing/run_comparison.py

For each of the first N_FOLDS bootstrapped sets, every model is trained on the set's train
scenarios and scored on three parts (see data.py): the train scenarios, the bootstrapped
validation scenarios and the original held-out scenarios. The held-out scores are the ones
that count: the model has never seen those scenarios in any form.

Writes three tables to machine_learning_investing/results/:
  comparison_folds.csv        one row per fold, model and part, with all metrics
  comparison_scenarios.csv    one row per fold, model and held-out scenario (final wealth, ruined)
  comparison_allocations.csv  one row per fold, model and year: the average held-out portfolio
and prints a summary. plots.py turns the tables into figures.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # so that this also runs as a script

from machine_learning_investing.data import load_fold  # noqa: E402
from machine_learning_investing.policies import (BayesianMLPPolicy, FixedPolicy, MLPPolicy, RNNPolicy,  # noqa: E402
                                                 StaticPolicy)
from machine_learning_investing.training import metrics, outcomes, train  # noqa: E402
from utils import ASSET_NAMES, Config  # noqa: E402

RESULTS = os.path.join(HERE, "results")
N_FOLDS = 20
N_PROCESSES = 7

# name: (function that makes a new model, training settings)
NETWORK = {"learning_rate": 0.003, "batch_size": 128}
MODELS = {
    "Bengen 50/50": (lambda: FixedPolicy({"US equity": 0.5, "US Treasury 5Y": 0.5}), {}),
    "Glide path 70->30": (lambda: FixedPolicy({"US equity": 0.7, "US Treasury 5Y": 0.3},
                                              {"US equity": 0.3, "US Treasury 5Y": 0.7}), {}),
    "Static (learned mix)": (StaticPolicy, {"learning_rate": 0.05}),
    "MLP, wealth and time": (lambda: MLPPolicy(use_market=False), NETWORK),
    "MLP, with market": (MLPPolicy, NETWORK),
    "GRU, with market": (lambda: RNNPolicy("gru"), NETWORK),
    "LSTM, with market": (lambda: RNNPolicy("lstm"), NETWORK),
    "Bayesian MLP, with market": (BayesianMLPPolicy, {**NETWORK, "steps": 600, "min_steps": 100}),
}


def run_fold(k):
    torch.set_num_threads(1)  # the folds run next to each other, one processor each
    cfg, fold = Config(), load_fold(k)
    fold_rows, scenario_rows, allocation_rows = [], [], []
    for name, (make, settings) in MODELS.items():
        start = time.time()
        policy, _ = train(make(), fold.train, fold.validation, cfg, seed=k, **settings)
        for part in ("train", "validation", "held_out"):
            fold_rows.append({"fold": k, "model": name, "part": part,
                              **metrics(policy, getattr(fold, part), cfg), "seconds": time.time() - start})
        out = outcomes(policy, fold.held_out, cfg)
        for i, history_id in enumerate(fold.held_out_ids):
            scenario_rows.append({"fold": k, "model": name, "history_id": history_id,
                                  "origin": fold.held_out_origin[i],
                                  "final_wealth": out["final_wealth"][i].item(),
                                  "ruined": bool(out["ruined"][i])})
        has_money = out["wealth"][:-1] > 0  # (T, N): the portfolio only matters while there is money
        for t in range(fold.held_out.n_years):
            average = out["weights"][t][has_money[t]].mean(dim=0)
            allocation_rows.append({"fold": k, "model": name, "year": t + 1,
                                    "scenarios": int(has_money[t].sum()), **dict(zip(ASSET_NAMES, average.tolist()))})
    return fold_rows, scenario_rows, allocation_rows


def summarise(folds):
    """Average held-out results per model, and how often a model beats the learned static mix."""
    held = folds[folds.part == "held_out"]
    table = held.groupby("model", sort=False)[
        ["score", "ruin_prob", "median_final_wealth", "p10_final_wealth", "growth_share", "turnover"]].mean()
    score = held.pivot(index="fold", columns="model", values="score")
    difference = score.sub(score["Static (learned mix)"], axis=0)
    table.insert(1, "score_std_over_folds", score.std())
    table["folds_better_than_static"] = (difference > 0).mean()
    table["train_score"] = folds[folds.part == "train"].groupby("model", sort=False)["score"].mean()
    return table


def main():
    os.makedirs(RESULTS, exist_ok=True)
    start = time.time()
    with ProcessPoolExecutor(N_PROCESSES) as pool:
        results = list(pool.map(run_fold, range(N_FOLDS)))
    folds, scenarios, allocations = (pd.DataFrame([row for result in results for row in result[i]]) for i in range(3))
    folds.to_csv(os.path.join(RESULTS, "comparison_folds.csv"), index=False)
    scenarios.to_csv(os.path.join(RESULTS, "comparison_scenarios.csv"), index=False)
    allocations.to_csv(os.path.join(RESULTS, "comparison_allocations.csv"), index=False)

    pd.set_option("display.width", 250)
    print(f"{N_FOLDS} folds in {time.time() - start:.0f} seconds. Held-out results, averaged over the folds:")
    print(summarise(folds).round(3).to_string())
    print("\nHeld-out scenarios by original file (ruin probability and median final wealth):")
    by_origin = scenarios.groupby(["model", "origin"], sort=False).agg(
        ruin_prob=("ruined", "mean"), median_final_wealth=("final_wealth", "median"))
    print(by_origin.unstack().round(3).to_string())


if __name__ == "__main__":
    main()
