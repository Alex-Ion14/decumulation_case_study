"""Compare annual investment policies on complete retirement paths.

Run from the repository root with python -m machine_learning_investing.study.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import torch

from machine_learning_investing.policies import (
    HURDLE, SEED, FixedMix, NeuralAllocation, TipsForecast, YieldAllocation, mix,
    mixture_cdf, portfolio_utility, tips_spread,
)
from strategies.bengen import Strategy as Bengen
from utils import Config, load_scenarios, simulate

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FOLDS = 4
BOOTSTRAPS = 4000
ENSEMBLE_OFFSETS = (0, 101, 102, 103)
POLICIES = ("bengen", "growth_60", "growth_100", "us_equity_100", "static_fit",
            "ridge", "mlp", "gru", "bayes_mean", "bayes_confidence")


def fit_policies(train, cfg, seed, names=POLICIES):
    forecast = None
    policies = {}
    for name in names:
        if name == "bengen":
            policy = Bengen()
        elif name in ("growth_60", "growth_100"):
            policy = FixedMix(mix(train.asset_names, 0.60 if name == "growth_60" else 1.0))
        elif name == "us_equity_100":
            policy = FixedMix(torch.tensor([float(asset == "US equity")
                                            for asset in train.asset_names]))
        elif name in ("static_fit", "mlp", "gru", "bayesian_gru"):
            policy = NeuralAllocation(name, train, cfg, seed)
        elif name == "ridge" or train.n_years < 2:
            policy = YieldAllocation(name, train) if train.n_years > 1 else FixedMix(
                mix(train.asset_names, 0.60)
            )
        else:
            if forecast is None:
                forecast = TipsForecast(train, seed)
            policy = YieldAllocation(name, train, forecast)
        policies[name] = policy
    return policies


@torch.no_grad()
def path_results(policy, scenarios, cfg):
    out = simulate(policy, scenarios, cfg)
    terminal = out["wealth"][-1]
    norm_wd = out["norm_wd"]
    ruin = (out["margins"] <= 0).any(dim=0)
    return np.column_stack((terminal, norm_wd, ruin, portfolio_utility(terminal, norm_wd)))


def cross_validate(train, cfg):
    fold_count = min(FOLDS, train.n_scenarios)
    if fold_count < 2:
        raise ValueError("Model comparison needs at least two training scenarios")
    paths = np.arange(train.n_scenarios)
    folds = np.array_split(np.random.default_rng(SEED).permutation(paths), fold_count)
    results = {name: np.empty((len(paths), 4)) for name in POLICIES}
    for i, held_out in enumerate(folds, start=1):
        policies = fit_policies(train.subset(np.setdiff1d(paths, held_out)), cfg, SEED + i)
        check = train.subset(held_out)
        for name, policy in policies.items():
            results[name][held_out] = path_results(policy, check, cfg)
        print(f"Training cross-validation fold {i}/{fold_count} complete", flush=True)
    return results


def choose(results):
    return max(results, key=lambda name: results[name][:, 3].mean())


def summarize(rows):
    terminal, normalized, ruined, utility = rows.T
    return {
        "median_terminal": float(np.median(terminal)),
        "mean_terminal": float(terminal.mean()),
        "p10_terminal": float(np.quantile(terminal, 0.10)),
        "ruins": int(ruined.sum()),
        "paths": len(rows),
        "normalized_withdrawals": float(normalized.mean()),
        "utility": float(utility.mean()),
    }


def paired_intervals(rows, reference, indices):
    sample = rows[indices]
    base = reference[indices]
    differences = {
        "median_terminal": np.median(sample[..., 0], axis=1) - np.median(base[..., 0], axis=1),
        "ruin_rate": (sample[..., 2] - base[..., 2]).mean(axis=1),
        "utility": (sample[..., 3] - base[..., 3]).mean(axis=1),
    }
    return {name: np.quantile(values, (0.025, 0.975)).tolist()
            for name, values in differences.items()}


class WeightAverage:
    def __init__(self, policies):
        self.policies = policies

    def act(self, obs):
        decisions = [policy.act(obs) for policy in self.policies]
        weights, rates = zip(*decisions)
        return torch.stack(weights).mean(dim=0), rates[0]


def seed_sensitivity(base, train, test, cfg):
    policies = {SEED: base}
    for offset in ENSEMBLE_OFFSETS[1:]:
        seed = SEED + offset
        policies[seed] = NeuralAllocation("gru", train, cfg, seed)
    return {
        "individual": {seed: summarize(path_results(policy, test, cfg))
                       for seed, policy in policies.items()},
        "ensemble": summarize(path_results(WeightAverage(list(policies.values())), test, cfg)),
    }


@torch.no_grad()
def calibration(forecast, scenarios):
    observed = tips_spread(scenarios)
    draws = [forecast.predict(scenarios.market[:year], scenarios.market_names)
             for year in range(1, scenarios.n_years)]
    means, sigmas = (torch.stack(values, dim=1) for values in zip(*draws))
    pit = mixture_cdf(observed, means, sigmas)
    events = observed > HURDLE
    prob = 1 - mixture_cdf(HURDLE, means, sigmas)
    bins = []
    edges = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    for lower, upper in zip(edges, edges[1:]):
        in_bin = (prob >= lower) & ((prob < upper) if upper < 1 else (prob <= 1))
        if in_bin.any():
            bins.append({"bounds": [lower, upper], "n": int(in_bin.sum()),
                         "forecast": prob[in_bin].mean().item(),
                         "observed": events[in_bin].float().mean().item()})
    return {
        "observations": observed.numel(),
        "rmse": (means.mean(dim=0) - observed).square().mean().sqrt().item(),
        "coverage_90": ((pit >= 0.05) & (pit <= 0.95)).float().mean().item(),
        "mean_model_sd": means.std(dim=0, unbiased=False).mean().item(),
        "mean_predictive_sd": (means.var(dim=0, unbiased=False)
                               + sigmas.square().mean(dim=0)).sqrt().mean().item(),
        "brier": (prob - events.float()).square().mean().item(),
        "base_rate_brier": (forecast.base_probability - events.float()).square().mean().item(),
        "reliability": bins,
    }


def print_table(label, rows):
    print(f"\n{label}")
    print(f"{'policy':<18} {'median':>8} {'mean':>8} {'P10':>8} {'ruins':>8} {'spend':>8} {'utility':>8}")
    for name, metrics in rows.items():
        ruins = f"{metrics['ruins']}/{metrics['paths']}"
        print(f"{name:<18} {metrics['median_terminal']:>8.3f} {metrics['mean_terminal']:>8.3f} "
              f"{metrics['p10_terminal']:>8.3f} {ruins:>8} "
              f"{metrics['normalized_withdrawals']:>8.3f} {metrics['utility']:>8.3f}")


def allocation_profile(policy, scenarios, cfg):
    first = []
    later = []
    turnover = []

    class ObservedPolicy:
        def act(self, obs):
            weights, rate = policy.act(obs)
            weights = weights.double().expand(scenarios.n_scenarios, -1)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            if obs.year == 0:
                first.append(weights.mean(dim=0))
            else:
                later.append(weights)
                turnover.append((weights - obs.holdings).abs().sum(dim=-1))
            return weights, rate

    with torch.no_grad():
        simulate(ObservedPolicy(), scenarios, cfg)
    return {
        "first_year": dict(zip(scenarios.asset_names, first[0].tolist())),
        "later_years": (dict(zip(scenarios.asset_names, torch.cat(later).mean(dim=0).tolist()))
                        if later else {}),
        "mean_l1_turnover": torch.cat(turnover).mean().item() if turnover else 0.0,
    }


def plot_comparison(metrics, out):
    data = sorted(metrics.items(), key=lambda item: item[1]["median_terminal"])
    labels = {
        "bengen": "Bengen 50/50", "growth_60": "60% growth", "growth_100": "100% growth",
        "us_equity_100": "100% US equity", "static_fit": "Fitted static mix",
        "ridge": "Ridge TIPS rule", "mlp": "Feed-forward net", "gru": "GRU",
        "bayes_mean": "Bayesian mean", "bayes_confidence": "Bayesian confidence filter",
    }
    names = [labels[name] for name, _ in data]
    colors = ["#126948" if name == "gru" else "#497089" for name, _ in data]
    fig, (wealth, ruin) = plt.subplots(1, 2, figsize=(10, 5.1), sharey=True,
                                      constrained_layout=True)
    wealth.barh(names, [m["median_terminal"] for _, m in data], color=colors)
    ruin.barh(names, [100 * m["ruins"] / m["paths"] for _, m in data], color=colors)
    wealth.set_xlabel("Median real terminal wealth (× starting capital)")
    ruin.set_xlabel("Retirements that ran out of money (%)")
    for axes in (wealth, ruin):
        axes.spines[["top", "right", "left"]].set_visible(False)
        axes.grid(axis="x", color="#e0e7ec")
        axes.set_axisbelow(True)
        axes.tick_params(axis="y", length=0)
    fig.suptitle("Portfolio outcomes under fixed real withdrawals")
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/train.parquet")
    parser.add_argument("--test", type=Path, default=ROOT / "data/validation.parquet")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--plot", type=Path, help="save a comparison chart")
    args = parser.parse_args()

    torch.set_num_threads(1)
    cfg = Config()
    train = load_scenarios(args.train)
    cv = cross_validate(train, cfg)
    selected = choose(cv)
    print(f"Selected using training folds: {selected}", flush=True)
    policies = fit_policies(train, cfg, SEED)
    test = load_scenarios(args.test)
    validation = {name: path_results(policy, test, cfg) for name, policy in policies.items()}
    indices = np.random.default_rng(SEED).integers(
        len(validation["bengen"]), size=(BOOTSTRAPS, test.n_scenarios)
    )
    stress_years = min(5, test.n_years)
    us_returns = test.returns[:stress_years, :, test.asset_names.index("US equity")]
    stress = ((1 + us_returns) / (1 + test.inflation[:stress_years])).prod(dim=0).numpy() < 1
    cost_cfg = replace(cfg, transaction_cost=2 * cfg.transaction_cost)
    report = {
        "selected_on_training": selected,
        "training_scenarios": train.n_scenarios,
        "test_scenarios": test.n_scenarios,
        "test_years": test.n_years,
        "test_allocation_profiles": {name: allocation_profile(policies[name], test, cfg)
                                     for name in dict.fromkeys(("bengen", "static_fit", selected))},
        "training_folds": {name: summarize(rows) for name, rows in cv.items()},
        "test": {name: summarize(rows) for name, rows in validation.items()},
        "test_double_cost": {name: summarize(path_results(policy, test, cost_cfg))
                             for name, policy in policies.items()},
        "test_negative_early_equity": {name: summarize(rows[stress])
                                        for name, rows in validation.items()} if stress.any() else {},
        "test_vs_bengen_95_pct_intervals": {
            name: paired_intervals(rows, validation["bengen"], indices)
            for name, rows in validation.items() if name != "bengen"
        },
        "test_vs_fitted_static_95_pct_intervals": {
            name: paired_intervals(rows, validation["static_fit"], indices)
            for name, rows in validation.items() if name != "static_fit"
        },
        "test_gru_seed_sensitivity": seed_sensitivity(policies["gru"], train, test, cfg),
    }
    forecast = getattr(policies["bayes_mean"], "forecast", None)
    if forecast is not None and test.n_years > 1:
        report["test_bayesian_forecast"] = calibration(forecast, test)
        print(json.dumps(report["test_bayesian_forecast"], indent=2))
    print_table("Training cross-validation", report["training_folds"])
    print_table("Test", report["test"])
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out}")
    if args.plot:
        plot_comparison(report["test"], args.plot)


if __name__ == "__main__":
    main()
