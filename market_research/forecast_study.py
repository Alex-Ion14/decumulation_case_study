"""Evaluate one-year market signals on the challenge's synthetic scenarios.

Run from the repository root: python -B market_research/forecast_study.py
The script reads the supplied Parquet files and writes JSON to stdout.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import Config, load_scenarios


US, EM, REIT, FIVE, THIRTY, TIPS, CASH = range(7)
SWITCH_HURDLE = 0.02  # Approximate round-trip cost for an all-in switch.
FOLDS = 5
SEED = 2026
BOOTSTRAPS = 10_000


@dataclass(frozen=True)
class Test:
    name: str
    feature: str
    target: str
    pair: tuple = ()
    naive: str = ""


# The candidate list, fitting method and evaluation rules were fixed before
# looking at validation returns. Each candidate uses just one lagged feature.
TESTS = (
    Test("Inflation persistence", "inflation", "inflation", naive="inflation"),
    Test("5Y yield mean reversion", "yield_5y", "change_5y", naive="zero"),
    Test("30Y yield mean reversion", "yield_30y", "change_30y", naive="zero"),
    Test("Real yield mean reversion", "real_yield_10y", "change_real_10y", naive="zero"),
    Test("Nominal cash carry", "bill_yield", "nominal_cash", naive="bill_yield"),
    Test("Real cash carry", "real_bill_proxy", "real_cash", naive="real_bill_proxy"),
    Test("US equity momentum", "us_real_return", "us_cash", (US, CASH)),
    Test("US equity and volatility", "vix", "us_cash", (US, CASH)),
    Test("EM equity momentum", "em_real_return", "em_cash", (EM, CASH)),
    Test("REIT momentum", "reit_real_return", "reit_cash", (REIT, CASH)),
    Test("5Y Treasury vs cash", "five_bill_spread", "five_cash", (FIVE, CASH)),
    Test("30Y vs 5Y Treasury", "yield_30y", "thirty_five", (THIRTY, FIVE)),
    Test("TIPS vs 5Y Treasury", "inflation_gap", "tips_five", (TIPS, FIVE)),
)


def real_returns(nominal, inflation):
    return (1 + nominal) / (1 + inflation[..., None]) - 1


def forecast_features(nominal, inflation, market, market_names):
    """Features from completed years. Use the final row at an Observation decision."""
    column = {name: market[..., i] for i, name in enumerate(market_names)}
    five = column["US_ZC_5Y"] / 100
    thirty = column["US_ZC_30Y"] / 100
    ten = column["US_ZC_10Y"] / 100
    breakeven = column["US_BEI_10Y"] / 100
    bills = column["US_TBILL_3M"] / 100
    real = real_returns(nominal, inflation)
    return {
        "inflation": inflation,
        "yield_5y": five,
        "yield_30y": thirty,
        "real_yield_10y": (1 + ten) / (1 + breakeven) - 1,
        "bill_yield": bills,
        "real_bill_proxy": (1 + bills) / (1 + inflation) - 1,
        "vix": column["US_VIX"],
        "us_real_return": real[..., US],
        "em_real_return": real[..., EM],
        "reit_real_return": real[..., REIT],
        "five_bill_spread": five - bills,
        "inflation_gap": inflation - breakeven,
        "zero": np.zeros_like(inflation),
    }


def study_arrays(scenarios):
    nominal = scenarios.returns.numpy()
    inflation = scenarios.inflation.numpy()
    real = real_returns(nominal, inflation)
    features = forecast_features(nominal, inflation, scenarios.market.numpy(),
                                 scenarios.market_names)
    targets = {
        "inflation": inflation[1:],
        "change_5y": np.diff(features["yield_5y"], axis=0),
        "change_30y": np.diff(features["yield_30y"], axis=0),
        "change_real_10y": np.diff(features["real_yield_10y"], axis=0),
        "nominal_cash": nominal[1:, :, CASH],
        "real_cash": real[1:, :, CASH],
    }
    for test in TESTS:
        if test.pair:
            first, second = test.pair
            targets[test.target] = real[1:, :, first] - real[1:, :, second]
    return {name: value[:-1] for name, value in features.items()}, targets, real


@dataclass
class Line:
    feature_mean: float
    target_mean: float
    slope: float
    positive_rate: float
    residuals: np.ndarray

    def predict(self, x):
        return self.target_mean + self.slope * (x - self.feature_mean)

    def probability_positive(self, prediction):
        # The empirical distribution of training residuals gives a probability.
        return 1 - np.searchsorted(self.residuals, -prediction, side="right") / len(self.residuals)


def fit_line(x, y):
    feature_mean, target_mean = x.mean(), y.mean()
    centered = x - feature_mean
    variance = np.mean(centered ** 2)
    slope = np.mean(centered * (y - target_mean)) / variance if variance else 0.0
    return Line(feature_mean, target_mean, slope, (y > 0).mean(),
                np.sort((y - target_mean - slope * centered).ravel()))


def error_metrics(y, predicted, baseline, probability, base_probability):
    actual_up = y > 0
    mse = np.mean((y - predicted) ** 2)
    base_mse = np.mean((y - baseline) ** 2)
    return {
        "rmse": np.sqrt(mse),
        "baseline_rmse": np.sqrt(base_mse),
        "mse_skill": 1 - mse / base_mse,
        "accuracy": np.mean((probability > 0.5) == actual_up),
        "baseline_accuracy": np.mean((base_probability > 0.5) == actual_up),
        "brier": np.mean((probability - actual_up) ** 2),
        "baseline_brier": np.mean((base_probability - actual_up) ** 2),
    }


def cross_validate(x, y, folds):
    prediction = np.empty_like(y)
    probability = np.empty_like(y)
    baseline = np.empty_like(y)
    base_probability = np.empty_like(y)
    slopes = []
    for held_out in folds:
        included = np.ones(y.shape[1], dtype=bool)
        included[held_out] = False
        line = fit_line(x[:, included], y[:, included])
        prediction[:, held_out] = line.predict(x[:, held_out])
        probability[:, held_out] = line.probability_positive(prediction[:, held_out])
        baseline[:, held_out] = line.target_mean
        base_probability[:, held_out] = line.positive_rate
        slopes.append(line.slope)
    return {**error_metrics(y, prediction, baseline, probability, base_probability),
            "fold_slopes": slopes}


def path_interval(values, bootstrap, family_size=1):
    # Each bootstrap draw samples complete validation paths, preserving the
    # serial dependence of annual results. Correct within each test family.
    means = values[bootstrap].mean(axis=1)
    tail = 0.025 / family_size
    return np.quantile(means, [tail, 1 - tail]).tolist()


def policy_check(test, static, predicted, real, asset_names, bootstrap):
    first, second = test.pair
    alternate = second if static == first else first
    advantage = -predicted if static == first else predicted
    chosen = np.where(advantage > SWITCH_HURDLE, alternate, static)
    positions = np.concatenate([np.full((1, chosen.shape[1]), static), chosen])
    switches = positions[1:] != positions[:-1]
    gross = np.take_along_axis(real[1:], chosen[..., None], axis=-1)[..., 0]
    net = (1 + gross) * (1 - 2 * Config().transaction_cost * switches) - 1
    base = real[1:, :, static]
    log_edge = (np.log1p(net) - np.log1p(base)).mean(axis=0)
    family_size = sum(bool(candidate.pair) for candidate in TESTS)
    alternate_selected = chosen == alternate
    realized_advantage = real[1:, :, alternate] - base
    alternate_ci = path_interval(
        (np.log1p(net) - np.log1p(real[1:, :, alternate])).mean(axis=0), bootstrap
    )
    return {
        "static_asset": asset_names[static],
        "alternate_asset": asset_names[alternate],
        "alternate_share": alternate_selected.mean(),
        "switch_share": switches.mean(),
        "net_return": net.mean(),
        "static_return": base.mean(),
        "net_edge": (net - base).mean(),
        "net_edge_early": (net[:9] - base[:9]).mean(),
        "net_edge_late": (net[9:] - base[9:]).mean(),
        "annual_log_edge": log_edge.mean(),
        "log_edge_ci95_family": path_interval(log_edge, bootstrap, family_size),
        "net_p05": np.quantile(net, 0.05),
        "static_p05": np.quantile(base, 0.05),
        # Descriptive follow-up on the same validation data. The decision rule
        # above is identical to the one used in the primary test.
        "supplementary": {
            "net_annual_geometric_return": np.expm1(np.log1p(net).mean()),
            "static_annual_geometric_return": np.expm1(np.log1p(base).mean()),
            "alternate_return": real[1:, :, alternate].mean(),
            "alternate_p05": np.quantile(real[1:, :, alternate], 0.05),
            "alternate_annual_geometric_return": np.expm1(np.log1p(real[1:, :, alternate]).mean()),
            "log_edge_vs_alternate_ci95_unadjusted": alternate_ci,
            "realized_advantage_when_alternate_selected": (
                realized_advantage[alternate_selected].mean() if alternate_selected.any() else None
            ),
        },
    }


def evaluate_test(test, train, validation, folds, bootstrap, asset_names):
    train_x = train[0][test.feature]
    train_y = train[1][test.target]
    val_x = validation[0][test.feature]
    val_y = validation[1][test.target]
    if not all(np.isfinite(x).all() for x in (train_x, train_y, val_x, val_y)):
        raise ValueError(f"Nonfinite feature or target in {test.name}")
    line = fit_line(train_x, train_y)
    prediction = line.predict(val_x)
    probability = line.probability_positive(prediction)
    base = line.target_mean
    squared_gain = ((val_y - base) ** 2 - (val_y - prediction) ** 2).mean(axis=0)
    cuts = np.quantile(train_x, [1 / 3, 2 / 3])
    bins = np.digitize(val_x, cuts)
    report = {
        "feature": test.feature,
        "target": test.target,
        "fit": {
            "feature_mean": line.feature_mean,
            "target_mean": line.target_mean,
            "slope": line.slope,
            "positive_rate": line.positive_rate,
        },
        "train_group_cv": cross_validate(train_x, train_y, folds),
        "validation": error_metrics(val_y, prediction, base, probability, line.positive_rate),
        "validation_squared_gain_ci95_family": path_interval(squared_gain, bootstrap, len(TESTS)),
        "validation_early": error_metrics(val_y[:9], prediction[:9], base,
                                          probability[:9], line.positive_rate),
        "validation_late": error_metrics(val_y[9:], prediction[9:], base,
                                         probability[9:], line.positive_rate),
        "validation_feature_terciles": [
            {"count": int(mask.sum()), "actual_mean": val_y[mask].mean() if mask.any() else None,
             "predicted_mean": prediction[mask].mean() if mask.any() else None,
             "positive_rate": np.mean(val_y[mask] > 0) if mask.any() else None}
            for mask in (bins == 0, bins == 1, bins == 2)
        ],
        "train_feature_tercile_cutoffs": cuts.tolist(),
    }
    if test.naive:
        naive = validation[0][test.naive]
        report["naive_rule"] = test.naive
        report["naive_rmse"] = np.sqrt(np.mean((val_y - naive) ** 2))
    if test.pair:
        first, second = test.pair
        static = first if line.target_mean > 0 else second
        report["instrument_check"] = policy_check(test, static, prediction, validation[2],
                                                  asset_names, bootstrap)
        if line.slope:
            hurdle = -SWITCH_HURDLE if static == first else SWITCH_HURDLE
            report["instrument_check"]["feature_switch_cutoff"] = (
                line.feature_mean + (hurdle - line.target_mean) / line.slope
            )
        if test.feature == "five_bill_spread":
            # Follow-up on the same validation data with the original hurdle.
            raw_check = policy_check(test, static, val_x, validation[2], asset_names, bootstrap)
            raw_check.pop("log_edge_ci95_family")  # Outside the primary family.
            report["raw_spread_sensitivity"] = {
                "forecast_rmse": np.sqrt(np.mean((val_y - val_x) ** 2)),
                "forecast_sign_accuracy": np.mean((val_x > 0) == (val_y > 0)),
                "instrument_check": raw_check,
            }
    return report


def asset_summary(scenarios, real):
    inflation = scenarios.inflation.numpy()
    inflation_stress = inflation >= 0.04
    equity_stress = real[..., US] <= -0.10
    result = {}
    for index, name in enumerate(scenarios.asset_names):
        yearly = real[..., index]
        result[name] = {
            "mean": yearly.mean(),
            "sd": yearly.std(),
            "p05": np.quantile(yearly, 0.05),
            "loss_rate": (yearly < 0).mean(),
            "median_path_cagr": np.expm1(np.median(np.log1p(yearly).mean(axis=0))),
            "mean_when_inflation_at_least_4pct": yearly[inflation_stress].mean(),
            "mean_when_us_equity_down_at_least_10pct": yearly[equity_stress].mean(),
        }
    return {
        "paths": scenarios.n_scenarios,
        "years": scenarios.n_years,
        "stress_year_counts": {"inflation": int(inflation_stress.sum()),
                               "equity": int(equity_stress.sum())},
        "assets": result,
        "real_return_correlations": np.corrcoef(real.reshape(-1, real.shape[-1]).T).tolist(),
    }


def main():
    torch.set_num_threads(1)
    train_path, val_path = (ROOT / "data" / name for name in ("train.parquet", "validation.parquet"))
    train_sc, val_sc = load_scenarios(train_path), load_scenarios(val_path)
    train_ids, val_ids = (set(pd.read_parquet(p, columns=["history_id"])["history_id"])
                          for p in (train_path, val_path))
    if train_ids & val_ids:
        raise ValueError("Train and validation scenario IDs overlap")
    train, val = study_arrays(train_sc), study_arrays(val_sc)
    if train_sc.asset_names != val_sc.asset_names or train_sc.n_years != val_sc.n_years:
        raise ValueError("Train and validation must share assets and horizon")
    rng = np.random.default_rng(SEED)
    folds = np.array_split(rng.permutation(train_sc.n_scenarios), FOLDS)
    bootstrap = rng.integers(val_sc.n_scenarios, size=(BOOTSTRAPS, val_sc.n_scenarios))
    report = {
        "units": "Decimal returns and yields except VIX. MSE and Brier use squared units.",
        "scored_decision_years": "2 through 30, split into 2–10 and 11–30",
        "training": asset_summary(train_sc, train[2]),
        "validation": asset_summary(val_sc, val[2]),
        "year_2_starting_state_mean": {
            name: {"train": train[0][name][0].mean(), "validation": val[0][name][0].mean()}
            for name in ("yield_5y", "yield_30y", "bill_yield", "real_yield_10y", "inflation", "vix")
        },
        "tests": {test.name: evaluate_test(test, train, val, folds, bootstrap, train_sc.asset_names)
                  for test in TESTS},
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
