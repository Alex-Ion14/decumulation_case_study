"""Refit and compare investment policies on the branch's bootstrap datasets."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import torch

from bootstrap.datasets import (
    BLOCK_YEARS, N_SETS, N_TRAIN_SCENARIOS, POOL, TEST_SET, load_set, set_path,
)
from bootstrap.sampler import BlockBootstrap
from machine_learning_investing.policies import GROWTH, SEED
from machine_learning_investing.study import POLICIES, calibration, fit_policies, path_results, summarize
from strategies.bengen import WITHDRAWAL_RATE
from utils import Config, load_scenarios

BAYESIAN_COMPARISON = ("bengen", "static_fit", "gru", "bayesian_gru")


def portfolio_metrics(policy, scenarios, cfg):
    metrics = summarize(path_results(policy, scenarios, cfg))
    ruin_rate = metrics["ruins"] / metrics["paths"]
    metrics["ruin_rate"] = ruin_rate
    metrics["case_score"] = cfg.alpha * metrics["normalized_withdrawals"] - (1 - cfg.alpha) * ruin_rate
    return metrics


def evaluate_bayesian_gru(policy, scenarios, cfg):
    widths = []
    growth = [scenarios.asset_names.index(asset) for asset in GROWTH]

    class ObserveDraws:
        def act(self, obs):
            if obs.year == 0:
                return policy.act(obs)
            draws = policy.allocation_draws(obs)
            shares = torch.cat((draws, draws[..., growth].sum(dim=-1, keepdim=True)), dim=-1)
            bounds = torch.quantile(shares, shares.new_tensor([0.05, 0.95]), dim=0)
            widths.append(bounds[1] - bounds[0])
            return draws.mean(dim=0), WITHDRAWAL_RATE

    metrics = portfolio_metrics(ObserveDraws(), scenarios, cfg)
    names = [*scenarios.asset_names, "growth_total"]
    intervals = dict(zip(names, torch.cat(widths).mean(dim=0).tolist()))
    return metrics, intervals


def evaluate_set(index, names=POLICIES):
    train, validation = load_set(index)
    cfg = Config()
    policies = fit_policies(train, cfg, SEED, names)
    result = {
        "set": index,
        "training_paths": train.n_scenarios,
        "validation_paths": validation.n_scenarios,
        "policies": {name: portfolio_metrics(policy, validation, cfg)
                     for name, policy in policies.items() if name != "bayesian_gru"},
    }
    if "bayes_mean" in policies:
        result["bayesian_forecast"] = calibration(policies["bayes_mean"].forecast, validation)
    if "bayesian_gru" in policies:
        metrics, intervals = evaluate_bayesian_gru(policies["bayesian_gru"], validation, cfg)
        result["policies"]["bayesian_gru"] = metrics
        result["bayesian_gru_weight_widths"] = intervals
    return result


def mean_metrics(rows):
    return {name: float(np.mean([row[name] for row in rows]))
            for name in ("median_terminal", "mean_terminal", "p10_terminal", "ruin_rate",
                         "normalized_withdrawals", "utility", "case_score")}


def aggregate(sets):
    results = {}
    for name in sets[0]["policies"]:
        metrics = [row["policies"][name] for row in sets]
        medians = [m["median_terminal"] for m in metrics]
        result = mean_metrics(metrics)
        result["median_terminal_set_p10_p90"] = np.quantile(medians, (0.1, 0.9)).tolist()
        result["ruins"] = sum(m["ruins"] for m in metrics)
        result["paths"] = sum(m["paths"] for m in metrics)
        result["utility_wins"] = sum(
            name == max(row["policies"], key=lambda n: row["policies"][n]["utility"])
            for row in sets
        )
        for reference in ("bengen", "static_fit", "gru"):
            checks = [row["policies"][reference] for row in sets]
            result[f"sets_with_higher_utility_than_{reference}"] = sum(
                m["utility"] > baseline["utility"] for m, baseline in zip(metrics, checks)
            )
            result[f"sets_with_higher_median_than_{reference}"] = sum(
                m["median_terminal"] > baseline["median_terminal"] for m, baseline in zip(metrics, checks)
            )
        results[name] = result
    return results


def summarize_calibration(sets):
    rows = [row["bayesian_forecast"] for row in sets]
    return {
        name: float(np.mean([row[name] for row in rows]))
        for name in ("rmse", "coverage_90", "mean_model_sd", "mean_predictive_sd",
                     "brier", "base_rate_brier")
    }


def evaluate_holdout(selected, compare_bayesian):
    # Refit with the entire allowed pool after selection. Then check intact paths.
    cfg = Config()
    pool = load_scenarios(POOL)
    train = BlockBootstrap(pool, BLOCK_YEARS, seed=SEED).sample(N_TRAIN_SCENARIOS)
    holdout = load_scenarios(TEST_SET)
    names = tuple(dict.fromkeys(("bengen", "static_fit", "gru", selected)))
    if compare_bayesian and "bayesian_gru" not in names:
        names += ("bayesian_gru",)
    policies = fit_policies(train, cfg, SEED, names)
    return {name: portfolio_metrics(policy, holdout, cfg) for name, policy in policies.items()}


def one_thread():
    torch.set_num_threads(1)


def print_table(title, results):
    print(f"\n{title}")
    print(f"{'policy':<18} {'median':>8} {'mean':>8} {'ruin':>8} {'spend':>8} {'utility':>8} {'J':>8}")
    for name, m in sorted(results.items(), key=lambda row: -row[1]["utility"]):
        print(f"{name:<18} {m['median_terminal']:>8.3f} {m['mean_terminal']:>8.3f} "
              f"{m['ruin_rate']:>8.1%} {m['normalized_withdrawals']:>8.1%} "
              f"{m['utility']:>8.3f} {m['case_score']:>8.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", type=int, nargs="+", default=range(N_SETS),
                        choices=range(N_SETS), metavar="INDEX")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--holdout", action="store_true", help="also score the reserved intact paths")
    parser.add_argument("--bayesian-gru", action="store_true",
                        help="compare the variational GRU with the original GRU and static benchmarks")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    indices = sorted(set(args.sets))
    missing = [index for index in indices if not Path(set_path(index)).is_file()]
    if missing:
        parser.error(f"missing bootstrap sets {missing}; run python -B bootstrap/datasets.py first")
    if args.holdout and not all(Path(file).is_file() for file in (POOL, TEST_SET)):
        parser.error("missing pool or reserved test file; run python -B bootstrap/datasets.py first")

    one_thread()
    names = BAYESIAN_COMPARISON if args.bayesian_gru else POLICIES
    sets = []
    if args.jobs == 1:
        for index in indices:
            sets.append(evaluate_set(index, names))
            print(f"Completed set {index:02d} ({len(sets)}/{len(indices)})", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(indices)), initializer=one_thread) as pool:
            tasks = [pool.submit(evaluate_set, index, names) for index in indices]
            for task in as_completed(tasks):
                sets.append(task.result())
                print(f"Completed set {sets[-1]['set']:02d} ({len(sets)}/{len(indices)})", flush=True)
        sets.sort(key=lambda row: row["set"])

    summary = aggregate(sets)
    selected = max(summary, key=lambda name: summary[name]["utility"])
    output = {
        "set_ids": indices,
        "seed": SEED,
        "block_years": BLOCK_YEARS,
        "selected_by_mean_validation_utility": selected,
        "validation_means_across_sets": summary,
        "sets": sets,
    }
    if args.bayesian_gru:
        widths = [row["bayesian_gru_weight_widths"] for row in sets]
        output["bayesian_gru_mean_90pct_weight_interval_width"] = {
            asset: float(np.mean([row[asset] for row in widths])) for asset in widths[0]
        }
    else:
        output["bayesian_forecast_means_across_sets"] = summarize_calibration(sets)
    print_table("Bootstrap validation (mean across sets)", summary)
    print(f"Selected on bootstrap validation: {selected}", flush=True)
    if args.holdout:
        output["reserved_intact_test"] = evaluate_holdout(selected, args.bayesian_gru)
        print_table("Reserved intact test", output["reserved_intact_test"])
    default_name = "bayesian_gru_results.json" if args.bayesian_gru else "bootstrap_results.json"
    out = args.out or Path(__file__).with_name(default_name)
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
