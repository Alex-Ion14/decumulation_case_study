"""Test ridge forecasts using the case's market, macro, and global data.

Run from the repository root: python -B market_research/ml_forecasts.py
The only output is JSON on stdout. No extra packages are required.
"""
from dataclasses import dataclass
import json

import numpy as np
import torch

from forecast_study import (
    BOOTSTRAPS, CASH, EM, FIVE, REIT, ROOT, SEED, SWITCH_HURDLE, THIRTY, TIPS, US,
    Config, fit_line, forecast_features, load_scenarios, path_interval, real_returns,
)


PENALTIES = (0.01, 0.1, 1.0, 10.0)
FEATURE_SETS = ("market", "market_macro", "market_macro_global")
EARLY_YEARS = 8  # Years 3–10, followed by 11–30.


@dataclass(frozen=True)
class Target:
    name: str
    baseline: str = "mean"
    feature: str = ""
    pair: tuple = ()
    static_asset: int = -1
    binary: bool = False


TARGETS = (
    Target("5Y Treasury vs cash", "raw", "five_bill_spread", (FIVE, CASH), FIVE),
    Target("30Y vs 5Y Treasury", "line", "yield_30y", (THIRTY, FIVE), FIVE),
    Target("TIPS vs 5Y Treasury", "line", "inflation_gap", (TIPS, FIVE), FIVE),
    Target("US equity vs cash", "line", "us_real_return", (US, CASH), US),
    Target("EM equity vs US equity", pair=(EM, US), static_asset=US),
    Target("US REIT vs US equity", pair=(REIT, US), static_asset=US),
    Target("Real cash return", "line", "real_bill_proxy"),
    Target("Inflation"),
    Target("US equity real loss of at least 10%", binary=True),
    Target("Inflation at least 4%", binary=True),
)


def panel(scenarios):
    nominal = scenarios.returns.numpy()
    inflation = scenarios.inflation.numpy()
    real = real_returns(nominal, inflation)
    market = scenarios.market.numpy()
    names = scenarios.market_names
    columns = {name: market[..., index] for index, name in enumerate(names)}
    features = forecast_features(nominal, inflation, market, names)

    def level(name):
        return columns[name][1:-1]

    def change(name):
        return level(name) - columns[name][:-2]

    def growth(name):
        previous = columns[name][:-2]
        return change(name) / np.abs(previous)

    # The most recent data in each feature row is from the completed year.
    # All variants use the same decision years 3–30 to allow annual changes.
    market_inputs = {
        name: features[name][1:-1] for name in (
            "five_bill_spread", "yield_5y", "yield_30y", "real_yield_10y",
            "bill_yield", "vix", "us_real_return", "em_real_return", "reit_real_return",
        )
    }
    market_inputs.update({
        "30Y minus 5Y yield": level("US_ZC_30Y") - level("US_ZC_5Y"),
        "5Y yield annual change": change("US_ZC_5Y"),
    })
    macro_inputs = {
        "lagged inflation": features["inflation"][1:-1],
        "breakeven inflation": level("US_BEI_10Y"),
        "US credit spread": level("US_GZ_SPREAD"),
        "US unemployment": level("US_UNRATE_NSA"),
        "US unemployment annual change": change("US_UNRATE_NSA"),
        "US payroll growth": growth("US_PAYROLLS_NSA"),
        "US industrial production growth": growth("US_IP_NSA"),
        "US real GDP growth": growth("US_GDP_REAL_NSA"),
    }
    global_inputs = {
        "German 10Y minus US 10Y yield": level("DE_ZC_10Y") - level("US_ZC_10Y"),
        "Japan 10Y minus US 10Y yield": level("JP_PAR_10Y") - level("US_ZC_10Y"),
        "Euro area financial stress": level("EA_CISS"),
        "EURUSD annual change": growth("EURUSD"),
        "USDJPY annual change": growth("USDJPY"),
        "Brent annual change": growth("BRENT_SPOT"),
        "Copper annual change": growth("COPPER"),
        "Gold annual change": growth("GOLD"),
    }
    feature_groups = dict(zip(FEATURE_SETS, (
        market_inputs, {**market_inputs, **macro_inputs},
        {**market_inputs, **macro_inputs, **global_inputs},
    )))
    matrices = {name: np.stack(tuple(group.values()), axis=-1)
                for name, group in feature_groups.items()}
    if not all(np.isfinite(x).all() for x in matrices.values()):
        raise ValueError("Nonfinite predictors after transforming scenario data")
    targets = {
        "Real cash return": real[2:, :, CASH],
        "Inflation": inflation[2:],
        "US equity real loss of at least 10%": (real[2:, :, US] <= -0.10).astype(float),
        "Inflation at least 4%": (inflation[2:] >= 0.04).astype(float),
    }
    for target in TARGETS:
        if target.pair:
            first, second = target.pair
            targets[target.name] = real[2:, :, first] - real[2:, :, second]
    return {
        "matrices": matrices, "feature_names": {k: list(v) for k, v in feature_groups.items()},
        "simple_inputs": {name: value[1:-1] for name, value in features.items()},
        "targets": targets, "real": real[2:],
    }


@dataclass
class Ridge:
    low: np.ndarray
    high: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    intercept: float
    weights: np.ndarray

    def predict(self, x):
        scaled = (np.clip(x, self.low, self.high) - self.center) / self.scale
        return self.intercept + scaled @ self.weights


def fit_ridge(x, y, penalty):
    x = x.reshape(-1, x.shape[-1])
    y = y.ravel()
    low, high = np.quantile(x, [0.01, 0.99], axis=0)
    clipped = np.clip(x, low, high)
    center = clipped.mean(axis=0)
    sd = clipped.std(axis=0)
    scale = np.where(sd > 0, sd, 1)
    scaled = (clipped - center) / scale
    y_mean = y.mean()
    # Minimize average squared error plus penalty times squared coefficient norm.
    covariance = scaled.T @ scaled / len(y)
    covariance += penalty * np.eye(x.shape[1])
    weights = np.linalg.solve(covariance, scaled.T @ (y - y_mean) / len(y))
    return Ridge(low, high, center, scale, y_mean, weights)


def predict(model, x, binary):
    prediction = model.predict(x)
    return np.clip(prediction, 0, 1) if binary else prediction


def splits(paths, count, seed):
    for held_out in np.array_split(np.random.default_rng(seed).permutation(paths), count):
        yield np.setdiff1d(np.arange(paths), held_out), held_out


def choose_penalty(x, y, binary):
    # Prefer the strongest shrinkage within one path-level standard error of
    # the best inner-fold loss. Never use the outer fold for this choice.
    losses = np.empty((len(PENALTIES), y.shape[1]))
    for train, check in splits(y.shape[1], 4, SEED + 1):
        for index, penalty in enumerate(PENALTIES):
            model = fit_ridge(x[:, train], y[:, train], penalty)
            predictions = predict(model, x[:, check], binary)
            losses[index, check] = np.mean((predictions - y[:, check]) ** 2, axis=0)
    mean_loss = losses.mean(axis=1)
    best = mean_loss.argmin()
    tolerance = losses[best].std(ddof=1) / np.sqrt(y.shape[1])
    return PENALTIES[np.flatnonzero(mean_loss <= mean_loss[best] + tolerance)[-1]]


def fit_simple(target, x, y):
    if target.baseline == "line":
        return fit_line(x, y).predict
    if target.baseline == "raw":
        return lambda x: x
    return lambda x: np.full(x.shape, y.mean())


def score(y, prediction, binary):
    squared_error = np.mean((y - prediction) ** 2)
    if binary:
        return {"brier": squared_error, "mean_probability": prediction.mean()}
    return {
        "rmse": np.sqrt(squared_error),
        "direction_accuracy": np.mean((y > 0) == (prediction > 0)),
    }


def error_skill(y, prediction, baseline):
    return 1 - np.mean((y - prediction) ** 2) / np.mean((y - baseline) ** 2)


def switching(target, prediction, real, full_switch_cost=2 * Config().transaction_cost):
    """Instrument comparison with identical decision thresholds and costs."""
    first, second = target.pair
    static = target.static_asset
    other = second if static == first else first
    advantage = -prediction if static == first else prediction
    chosen = np.where(advantage > SWITCH_HURDLE, other, static)
    previous = np.vstack((np.full(chosen.shape[1], static), chosen[:-1]))
    switched = chosen != previous
    gross = np.take_along_axis(real, chosen[..., None], axis=-1)[..., 0]
    net = (1 + gross) * (1 - full_switch_cost * switched) - 1
    return net, {
        "net_real_return": net.mean(), "net_p05": np.quantile(net, 0.05),
        "alternate_share": np.mean(chosen == other), "switch_share": switched.mean(),
    }


def train_target(target, data):
    y = data["targets"][target.name]
    base_x = data["simple_inputs"].get(target.feature, np.zeros_like(y))
    predictions = {k: np.empty_like(y) for k in ("simple", *FEATURE_SETS)}
    chosen = {k: [] for k in FEATURE_SETS}
    for train, check in splits(y.shape[1], 5, SEED):
        simple = fit_simple(target, base_x[:, train], y[:, train])
        predictions["simple"][:, check] = simple(base_x[:, check])
        for name, x in data["matrices"].items():
            penalty = choose_penalty(x[:, train], y[:, train], target.binary)
            chosen[name].append(penalty)
            model = fit_ridge(x[:, train], y[:, train], penalty)
            predictions[name][:, check] = predict(model, x[:, check], target.binary)

    simple = fit_simple(target, base_x, y)
    stats = {"simple": score(y, predictions["simple"], target.binary)}
    models = {}
    for name, x in data["matrices"].items():
        penalty = choose_penalty(x, y, target.binary)
        models[name] = fit_ridge(x, y, penalty)
        stats[name] = {
            **score(y, predictions[name], target.binary),
            "skill_vs_simple": error_skill(y, predictions[name], predictions["simple"]),
            "outer_fold_penalties": chosen[name],
            "final_penalty": penalty,
            "largest_standardized_weights": sorted(
                zip(data["feature_names"][name], models[name].weights.tolist()),
                key=lambda item: abs(item[1]), reverse=True
            )[:5],
        }
    return simple, models, stats


def validate_target(target, fitted, data, bootstrap, asset_names):
    simple, models, training = fitted
    y = data["targets"][target.name]
    predictions = {"simple": simple(data["simple_inputs"].get(target.feature, np.zeros_like(y)))}
    predictions.update({name: predict(model, data["matrices"][name], target.binary)
                        for name, model in models.items()})
    baseline = predictions["simple"]
    results = {"simple": score(y, baseline, target.binary)}
    compare_to = dict(zip(FEATURE_SETS, ("simple", *FEATURE_SETS[:-1])))
    forecast_family_size = len(TARGETS) * (2 * len(FEATURE_SETS) - 1)
    for name, pred in predictions.items():
        if name == "simple":
            continue
        previous = predictions[compare_to[name]]
        results[name] = {
            **score(y, pred, target.binary),
            "skill_vs_simple": error_skill(y, pred, baseline),
            "early_skill_vs_simple": error_skill(y[:EARLY_YEARS], pred[:EARLY_YEARS],
                                                baseline[:EARLY_YEARS]),
            "late_skill_vs_simple": error_skill(y[EARLY_YEARS:], pred[EARLY_YEARS:],
                                               baseline[EARLY_YEARS:]),
            "squared_error_gain_vs_simple_ci95_family": path_interval(
                ((y - baseline) ** 2 - (y - pred) ** 2).mean(axis=0),
                bootstrap, forecast_family_size,
            ),
            "incremental_skill": error_skill(y, pred, previous),
            "incremental_gain_ci95_family": path_interval(
                ((y - previous) ** 2 - (y - pred) ** 2).mean(axis=0),
                bootstrap, forecast_family_size,
            ),
        }
    if target.pair:
        base_net, base_stats = switching(target, baseline, data["real"])
        static = data["real"][..., target.static_asset]
        results["simple"]["instrument_check"] = base_stats
        results["static"] = {"asset": asset_names[target.static_asset], "mean_real_return": static.mean(),
                             "p05": np.quantile(static, 0.05)}
        switch_family_size = sum(bool(t.pair) for t in TARGETS) * len(FEATURE_SETS)
        for name in FEATURE_SETS:
            net, stats = switching(target, predictions[name], data["real"])
            log_gain = (np.log1p(net) - np.log1p(base_net)).mean(axis=0)
            results[name]["instrument_check"] = {
                **stats,
                "net_edge_vs_simple": (net - base_net).mean(),
                "annual_log_edge_ci95_family": path_interval(log_gain, bootstrap, switch_family_size),
            }
    return {"nested_training_cv": training, "validation": results,
            "validation_event_rate": y.mean() if target.binary else None}


def tips_sensitivity(train, validation, macro_model, asset_names):
    """Exploratory follow-up with the same paths and decision hurdle."""
    target = next(target for target in TARGETS if target.name == "TIPS vs 5Y Treasury")
    group = "market_macro"
    names = ["real_yield_10y", "breakeven inflation", "yield_5y"]
    columns = [train["feature_names"][group].index(name) for name in names]
    x_train = train["matrices"][group][..., columns]
    x_val = validation["matrices"][group][..., columns]
    train_y = train["targets"][target.name]
    val_y = validation["targets"][target.name]
    penalty = choose_penalty(x_train, train_y, binary=False)
    reduced = fit_ridge(x_train, train_y, penalty)
    predictions = {
        "three_yield_inputs": reduced.predict(x_val),
        "market_macro": macro_model.predict(validation["matrices"][group]),
    }
    return {
        "three_yield_inputs": names,
        "reduced_model_penalty": penalty,
        "default_instrument": asset_names[target.static_asset],
        "static_real_return": validation["real"][..., target.static_asset].mean(),
        "models": {
            name: {
                **score(val_y, pred, binary=False),
                "cost_sensitivity": {
                    str(cost): switching(target, pred, validation["real"], cost)[1]
                    for cost in (0.01, 0.02, 0.04)
                },
            }
            for name, pred in predictions.items()
        },
    }


def main():
    torch.set_num_threads(1)
    train = load_scenarios(ROOT / "data" / "train.parquet")
    train_data = panel(train)
    fitted = {target.name: train_target(target, train_data) for target in TARGETS}

    validation = load_scenarios(ROOT / "data" / "validation.parquet")
    val_data = panel(validation)
    if train.asset_names != validation.asset_names or train.n_years != validation.n_years:
        raise ValueError("Scenario files differ in asset order or horizon")
    bootstrap = np.random.default_rng(SEED + 2).integers(
        validation.n_scenarios, size=(BOOTSTRAPS, validation.n_scenarios)
    )
    report = {
        "years": "3–30 (early 3–10; late 11–30)",
        "paths": {"train": train.n_scenarios, "validation": validation.n_scenarios},
        "feature_sets": train_data["feature_names"],
        "tips_exploratory_followup": tips_sensitivity(
            train_data, val_data, fitted["TIPS vs 5Y Treasury"][1]["market_macro"],
            validation.asset_names,
        ),
        "targets": {target.name: validate_target(
            target, fitted[target.name], val_data, bootstrap, validation.asset_names
        ) for target in TARGETS},
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
