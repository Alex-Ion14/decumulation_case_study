"""Compare fixed and adaptive policies with the original retirement simulator.

Run from the repository root: python -B market_research/retirement_study.py
Use --test to score the same frozen selection process on a separate holdout.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from retirement_strategy import GLIDES, Policy, Signals, Strategy
from ml_forecasts import BOOTSTRAPS, ROOT, SEED, load_scenarios, path_interval, splits
from forecast_study import real_returns
from utils import Config, simulate


SPENDING_LEVELS = (0.90, 1.00, 1.10)
PRIMARY_GLIDES = ("static", "declining", "rising")
ALPHAS = (0.10, 0.25, 0.50, 0.75)
BASELINE = Policy(spend="fixed", multiplier=0.04, defensive="five")
OVERLAYS = {
    "cash spread only": {},
    "always five-year": {"defensive": "five"},
    "cash spread plus TIPS": {"defensive": "tips"},
    "late funded de-risking": {"lock_late": True},
    "TIPS plus late funded de-risking": {"defensive": "tips", "lock_late": True},
    "crash alert cuts equity to cash": {"crash_to": "cash"},
    "crash alert cuts equity to defensive sleeve": {"crash_to": "defensive"},
}


def paths(policy, scenarios, config, signals):
    with torch.no_grad():
        out = simulate(Strategy(policy).attach(config, signals), scenarios, config)
    margins = out["margins"]
    active = margins > 0
    annual_return = out["wealth"][1:] / margins.clamp_min(1e-12) - 1
    return {
        "last_decade_loss_years": ((annual_return[-10:] <= -0.10) & active[-10:]).sum(dim=0).numpy(),
        "last_decade_active_years": active[-10:].sum(dim=0).numpy(),
        "ruin": (out["margins"] <= 0).any(dim=0).double().numpy(),
        "norm_wd": out["norm_wd"].numpy(),
        "final": out["wealth"][-1].numpy(),
        "total_wd": out["withdrawals"].sum(dim=0).numpy(),
        "first_wd": out["withdrawals"][0].numpy(),
        "early_wd": out["withdrawals"][:10].mean(dim=0).numpy(),
        "late_wd": out["withdrawals"][-10:].mean(dim=0).numpy(),
        "below_4pct_share": (out["withdrawals"] < 0.04 - 1e-9).double().mean(dim=0).numpy(),
    }


def summaries(out):
    ruin = out["ruin"].mean()
    normalized = out["norm_wd"].mean()
    return {
        "ruins": int(out["ruin"].sum()),
        "ruin_prob": ruin,
        "norm_wd": normalized,
        "mean_total_withdrawal": out["total_wd"].mean(),
        "first_withdrawal": out["first_wd"].mean(),
        "early_mean_withdrawal": out["early_wd"].mean(),
        "late_mean_withdrawal": out["late_wd"].mean(),
        "below_4pct_share": out["below_4pct_share"].mean(),
        "median_final_wealth": torch.as_tensor(out["final"]).median().item(),
        "final_wealth_p10": np.quantile(out["final"], 0.10),
        "last_decade_return_loss_rate": (
            out["last_decade_loss_years"].sum() / max(out["last_decade_active_years"].sum(), 1)
        ),
        "scores": {str(alpha): alpha * normalized - (1 - alpha) * ruin for alpha in ALPHAS},
    }


def nested_results(candidates, train, cfg):
    results = {name: {} for name in candidates}
    for included, check in splits(train.n_scenarios, 5, SEED):
        train_fold, test_fold = train.subset(included), train.subset(check)
        fitted = Signals(train_fold)
        for name, candidate in candidates.items():
            fold = paths(candidate, test_fold, cfg, fitted)
            for metric, values in fold.items():
                results[name].setdefault(metric, np.empty(train.n_scenarios))[check] = values
    return results


def selection(cv, cfg):
    # Rank by the official configured score on grouped training predictions.
    # Dict insertion order breaks exact ties in favor of the simpler base rule.
    return max(cv, key=lambda name: np.mean(
        cfg.alpha * cv[name]["norm_wd"] - (1 - cfg.alpha) * cv[name]["ruin"]
    ))


def paired(a, b, cfg, bootstrap):
    wd = a["norm_wd"] - b["norm_wd"]
    ruin = a["ruin"] - b["ruin"]
    utility = cfg.alpha * wd - (1 - cfg.alpha) * ruin
    return {
        "norm_wd_difference": wd.mean(),
        "ruin_difference": ruin.mean(),
        "J_difference": utility.mean(),
        "J_ci95_unadjusted": path_interval(utility, bootstrap),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "validation.parquet")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cfg = Config()
    train_file = ROOT / "data" / "train.parquet"
    if args.test.resolve() == train_file.resolve():
        parser.error("--test must be separate from the training file")
    train = load_scenarios(train_file)

    fixed = {f"fixed 4% {name}": replace(BASELINE, glide=name) for name in PRIMARY_GLIDES}
    core = {f"{glide} x {level:.2f}": Policy(glide=glide, multiplier=level)
            for glide in PRIMARY_GLIDES for level in SPENDING_LEVELS}
    core_cv = nested_results({**fixed, **core}, train, cfg)
    core_winner = selection({name: core_cv[name] for name in core}, cfg)
    overlays = {name: replace(core[core_winner], **changes) for name, changes in OVERLAYS.items()}
    overlay_cv = nested_results(overlays, train, cfg)
    non_crash = {name: out for name, out in overlay_cv.items()
                 if overlays[name].crash_to == "off"}
    overlay_winner = selection(non_crash, cfg)

    # No choices above depend on the supplied public or private test file.
    test = load_scenarios(args.test)
    if train.asset_names != test.asset_names or train.n_years != test.n_years:
        raise ValueError("The test file must use the same assets and horizon")
    fitted = Signals(train)
    all_policies = {**fixed, **core, **overlays}
    test_paths = {name: paths(policy, test, cfg, fitted) for name, policy in all_policies.items()}
    # Follow-up sensitivity uses the public validation again. The primary
    # training-only selection above stays fixed.
    cautious = {
        f"{glide} x {level:.2f} with TIPS": Policy(glide=glide, multiplier=level,
                                                 defensive="tips")
        for glide in GLIDES for level in (0.80, 0.85, 0.90)
    }
    cautious_cv = nested_results(cautious, train, cfg)
    cautious_paths = {name: paths(policy, test, cfg, fitted) for name, policy in cautious.items()}
    us = test.asset_names.index("US equity")
    real = real_returns(test.returns.numpy(), test.inflation.numpy())
    bad_first_five = np.prod(1 + real[:5, :, us], axis=0) < 1
    stress_paths = {name: {key: value[bad_first_five] for key, value in out.items()}
                    for name, out in {**test_paths, **cautious_paths}.items()}

    bootstrap = np.random.default_rng(SEED + 2).integers(
        test.n_scenarios, size=(BOOTSTRAPS, test.n_scenarios)
    )
    report = {
        "test_file": args.test.name,
        "training_paths": train.n_scenarios,
        "test_paths": test.n_scenarios,
        "alpha_for_selection": cfg.alpha,
        "best_core_on_training_cv": core_winner,
        "selected_non_crash_policy_on_training_cv": overlay_winner,
        "selected_parameters": asdict(all_policies[overlay_winner]),
        "training_group_cv": {name: summaries(out) for name, out in {**core_cv, **overlay_cv}.items()},
        "test": {name: summaries(out) for name, out in test_paths.items()},
        "exploratory_cautious_spending": {
            "training_group_cv": {name: summaries(out) for name, out in cautious_cv.items()},
            "test": {name: summaries(out) for name, out in cautious_paths.items()},
        },
        "test_negative_first_five_equity_years": {
            "paths": int(bad_first_five.sum()),
            "strategies": {name: summaries(out) for name, out in stress_paths.items()}
            if bad_first_five.any() else {},
        },
        "test_vs_bengen": paired(test_paths[overlay_winner], test_paths["fixed 4% static"],
                                 cfg, bootstrap),
        "test_overlays_vs_cash_spread_only": {
            name: paired(test_paths[name], test_paths["cash spread only"], cfg, bootstrap)
            for name in overlays if name != "cash spread only"
        },
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
