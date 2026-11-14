"""Experiment 2: how sure are the models about the portfolio they propose?

    python -B machine_learning_investing/run_confidence.py

Two ways to get a confidence, both on the first N_FOLDS bootstrapped sets:

  Deep ensemble   N_MEMBERS networks of the same kind, each trained on a different random
                  half of the train scenarios and from a different random start. If the
                  proposal depends on which scenarios a network happened to see, the members
                  disagree.
  Bayesian MLP    one network that learns a range for each of its weights. N_DRAWS networks
                  are drawn from those ranges.

In both cases the portfolio is the average of the proposals, and the confidence is the
*agreement*: the share of the portfolio that a member has in common with the average
portfolio (100% = all members propose the same portfolio).

Writes to machine_learning_investing/results/:
  confidence_folds.csv      held-out metrics of the averaged portfolios and of single networks
  confidence_decisions.csv  one row per method, held-out scenario and year: the agreement,
                            the average portfolio and the wealth at that moment
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
from machine_learning_investing.policies import BayesianMLPPolicy, Ensemble  # noqa: E402
from machine_learning_investing.run_comparison import MODELS, N_PROCESSES, RESULTS  # noqa: E402
from machine_learning_investing.training import metrics, outcomes, train  # noqa: E402
from utils import ASSET_NAMES, Config  # noqa: E402

N_FOLDS = 8
N_MEMBERS = 10
N_DRAWS = 30
MEMBER_MODEL = "MLP, with market"  # the kind of network in the deep ensemble


def run_fold(k):
    torch.set_num_threads(1)
    cfg, fold = Config(), load_fold(k)
    make, settings = MODELS[MEMBER_MODEL]
    half = fold.train.n_scenarios // 2

    members = []
    for m in range(N_MEMBERS):
        generator = torch.Generator().manual_seed(1000 * k + m)
        part = fold.train.subset(torch.randperm(fold.train.n_scenarios, generator=generator)[:half])
        members.append(train(make(), part, fold.validation, cfg, seed=1000 * k + m, **settings)[0])

    make_bayesian, bayesian_settings = MODELS["Bayesian MLP, with market"]
    bayesian = train(make_bayesian(), fold.train, fold.validation, cfg, seed=k, **bayesian_settings)[0]

    policies = {
        "Deep ensemble": Ensemble(members),
        "Single network": members[0],
        "Bayesian MLP, draws": Ensemble(bayesian.draw_members(N_DRAWS, seed=k)),
        "Bayesian MLP, average network": bayesian,
    }
    fold_rows = [{"fold": k, "method": name, **metrics(policy, fold.held_out, cfg)}
                 for name, policy in policies.items()]

    decision_rows = []
    for name, policy in policies.items():
        if not isinstance(policy, Ensemble):
            continue
        out = outcomes(policy, fold.held_out, cfg)
        for t in range(fold.held_out.n_years):
            for i, history_id in enumerate(fold.held_out_ids):
                decision_rows.append({
                    "fold": k, "method": name, "history_id": history_id, "origin": fold.held_out_origin[i],
                    "year": t + 1, "agreement": policy.agreement[t][i].item(),
                    "wealth": out["wealth"][t, i].item(), "ruined": bool(out["ruined"][i]),
                    "final_wealth": out["final_wealth"][i].item(),
                    **{asset: policy.average_weights[t][i, a].item() for a, asset in enumerate(ASSET_NAMES)},
                })
    return fold_rows, decision_rows


def main():
    os.makedirs(RESULTS, exist_ok=True)
    start = time.time()
    with ProcessPoolExecutor(N_PROCESSES) as pool:
        results = list(pool.map(run_fold, range(N_FOLDS)))
    folds = pd.DataFrame([row for fold_rows, _ in results for row in fold_rows])
    decisions = pd.DataFrame([row for _, decision_rows in results for row in decision_rows])
    folds.to_csv(os.path.join(RESULTS, "confidence_folds.csv"), index=False)
    decisions.to_csv(os.path.join(RESULTS, "confidence_decisions.csv"), index=False)

    pd.set_option("display.width", 250)
    print(f"{N_FOLDS} folds in {time.time() - start:.0f} seconds. Held-out results, averaged over the folds:")
    print(folds.groupby("method", sort=False)[
        ["score", "ruin_prob", "median_final_wealth", "p10_final_wealth", "growth_share", "turnover"]
    ].mean().round(3).to_string())

    alive = decisions[decisions.wealth > 0]
    print("\nAgreement (only years in which there is still money):")
    print(alive.groupby("method", sort=False).agreement.describe(percentiles=[0.1, 0.5, 0.9]).round(3).to_string())
    print("\nAverage agreement by period:")
    period = pd.cut(alive.year, [0, 1, 5, 10, 20, 30], labels=["year 1", "2-5", "6-10", "11-20", "21-30"])
    print(alive.groupby(["method", period], sort=False, observed=True).agreement.mean().unstack().round(3).to_string())
    print("\nAverage agreement per scenario: ruined or not, and by original file:")
    per_scenario = alive.groupby(["method", "fold", "history_id", "origin", "ruined"], sort=False).agreement.mean()
    print(per_scenario.groupby(["method", "ruined"], sort=False).agg(["mean", "count"]).round(3).to_string())
    print(per_scenario.groupby(["method", "origin"], sort=False).agg(["mean", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()
