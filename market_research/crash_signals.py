"""Evaluate next-year crash warnings using annual and optional monthly data.

Run from the repository root: python -B market_research/crash_signals.py
"""
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
import torch

from ml_forecasts import BOOTSTRAPS, CASH, FIVE, ROOT, SEED, US, load_scenarios
from forecast_study import EM, real_returns
from ml_forecasts import path_interval, splits


ALERT_SHARE = 0.20
CRASH_CUTOFF = -0.20
PENALTY = 0.10
ANNUAL_FEATURES = (
    "vix", "US credit spread", "US unemployment annual change", "five_bill_spread",
    "us_real_return", "em_real_return", "5Y yield annual change", "lagged inflation",
)


def annual_feature_rows(nominal, inflation, market, market_names):
    """Features for each completed year after the first, with no forward data."""
    col = {name: market[..., i] for i, name in enumerate(market_names)}
    real = real_returns(nominal, inflation)
    five = col["US_ZC_5Y"]
    return np.stack((
        col["US_VIX"][1:], col["US_GZ_SPREAD"][1:], np.diff(col["US_UNRATE_NSA"], axis=0),
        (five[1:] - col["US_TBILL_3M"][1:]) / 100, real[1:, :, US], real[1:, :, EM],
        np.diff(five, axis=0), inflation[1:],
    ), axis=-1)


def annual_inputs(scenarios):
    """Aligned to decisions in years 3–30 and limited to annual Observation data."""
    nominal, inflation = scenarios.returns.numpy(), scenarios.inflation.numpy()
    features = annual_feature_rows(nominal, inflation, scenarios.market.numpy(),
                                   scenarios.market_names)
    return features[:-1], real_returns(nominal, inflation)[2:]


def monthly_us_equity(file, paths, years):
    columns = ["history_id", "period", "variable", "value"]
    equity = pd.read_parquet(file, columns=columns, filters=[("variable", "==", "US_EQ_MKT")])
    equity = equity.sort_values(["history_id", "period"])
    counts = equity.groupby("history_id").size()
    if len(counts) != paths or not (counts == 12 * years).all():
        raise ValueError("Unexpected US equity monthly panel")
    if equity.duplicated(["history_id", "period"]).any():
        raise ValueError("Duplicate US equity month")
    monthly = equity["value"].to_numpy().reshape(paths, years, 12).transpose(1, 0, 2) / 100
    return monthly, set(counts.index)


def monthly_metrics(monthly):
    log_gross = np.log1p(monthly)
    log_index = np.concatenate([np.zeros_like(log_gross[..., :1]), log_gross.cumsum(-1)], axis=-1)
    log_drawdown = log_index - np.maximum.accumulate(log_index, axis=-1)
    drawdown = np.expm1(log_drawdown.min(axis=-1))
    features = np.stack([
        np.expm1(log_gross[..., -3:].sum(axis=-1)),
        np.expm1(log_gross[..., -6:].sum(axis=-1)),
        log_gross.std(axis=-1) * np.sqrt(12),
        np.expm1(log_drawdown[..., -1]),
        drawdown,
    ], axis=-1)
    return features[1:-1], drawdown[2:]


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class CrashModel:
    low: np.ndarray
    high: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def predict(self, x):
        scaled = (np.clip(x, self.low, self.high) - self.center) / self.scale
        return sigmoid(self.weights[0] + scaled @ self.weights[1:])


def fit_crash(x, y):
    """Penalized logistic regression, with fixed regularization and train-only scaling."""
    x = x.reshape(-1, x.shape[-1])
    y = y.ravel().astype(float)
    low, high = np.quantile(x, [0.01, 0.99], axis=0)
    clipped = np.clip(x, low, high)
    center = clipped.mean(axis=0)
    sd = clipped.std(axis=0)
    scale = np.where(sd > 0, sd, 1)
    design = np.column_stack([np.ones(len(y)), (clipped - center) / scale])
    weights = np.zeros(design.shape[-1])
    rate = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    weights[0] = np.log(rate / (1 - rate))
    penalty = np.diag([0, *([PENALTY] * x.shape[-1])])
    for _ in range(30):
        probability = sigmoid(design @ weights)
        variance = probability * (1 - probability)
        hessian = (design.T * variance) @ design / len(y) + penalty
        gradient = design.T @ (probability - y) / len(y) + penalty @ weights
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.linalg.norm(step) < 1e-9:
            break
    return CrashModel(low, high, center, scale, weights)


def crash_forecast(x, y, check):
    model = fit_crash(x, y)
    threshold = np.quantile(model.predict(x), 1 - ALERT_SHARE)
    probability = model.predict(check)
    return probability, probability > threshold, threshold


def ranking_scores(y, p):
    y, p = y.ravel(), p.ravel()
    order = np.argsort(-p, kind="stable")
    labels, ordered_p = y[order], p[order]
    boundaries = np.flatnonzero(np.r_[ordered_p[1:] != ordered_p[:-1], True])
    tp = np.r_[0, labels.cumsum()[boundaries]]
    fp = np.r_[0, boundaries + 1] - tp
    recall, fpr = tp / tp[-1], fp / fp[-1]
    auc = np.sum(np.diff(fpr) * (recall[1:] + recall[:-1]) / 2)
    ap = np.sum(np.diff(recall) * tp[1:] / (tp[1:] + fp[1:]))
    return {"auc": auc, "average_precision": ap}


def classify(y, p, alert, baseline):
    true_alert = y & alert
    return {
        **ranking_scores(y, p),
        "event_rate": y.mean(),
        "events": int(y.sum()),
        "brier": np.mean((p - y) ** 2),
        "constant_brier": np.mean((baseline - y) ** 2),
        "alert_rate": alert.mean(),
        "precision": true_alert.sum() / alert.sum() if alert.any() else None,
        "recall": true_alert.sum() / y.sum(),
    }


def mean_subset(x, mask):
    return x[mask].mean() if mask.any() else None


def alert_returns(real, event, alert, bootstrap):
    equity, cash, five = (real[..., index] for index in (US, CASH, FIVE))
    previous = np.vstack([np.zeros(alert.shape[1], dtype=bool), alert[:-1]])
    costs = 0.01 * (alert != previous)
    result = {}
    for name, defensive in (("cash", cash), ("five_year", five)):
        moved = (1 + np.where(alert, defensive, equity)) * (1 - costs) - 1
        edge = np.log1p(moved) - np.log1p(equity)
        result[name] = {
            "defensive_return_in_crash_year": mean_subset(defensive, event),
            "defensive_minus_equity_in_alert_year": mean_subset(defensive - equity, alert),
            "defensive_minus_equity_on_false_alarm": mean_subset(defensive - equity, alert & ~event),
            "net_mean_return_edge": (moved - equity).mean(),
            "annual_log_edge_ci95_family": path_interval(edge.mean(axis=0), bootstrap, 16),
        }
    return result


def assess(name, x, y, x_val, y_val, real_val, bootstrap):
    cv = np.empty_like(y, dtype=float)
    cv_alerts = np.empty_like(y)
    cv_base = np.empty_like(y, dtype=float)
    for train, check in splits(y.shape[1], 5, SEED):
        cv[:, check], cv_alerts[:, check], _ = crash_forecast(x[:, train], y[:, train], x[:, check])
        cv_base[:, check] = y[:, train].mean()
    p, alerts, threshold = crash_forecast(x, y, x_val)
    event_gap = (y_val - y.mean()) ** 2 - (y_val - p) ** 2
    result = {
        "training_path_cv": classify(y, cv, cv_alerts, cv_base),
        "validation": classify(y_val, p, alerts, y.mean()),
        "alert_threshold": threshold,
        "validation_brier_gain_ci95_family": path_interval(event_gap.mean(axis=0), bootstrap, 8),
    }
    if name == "VIX":
        # Rank-only baseline, with a high-VIX alert fixed from training.
        vix_alert = x_val[..., 0] > np.quantile(x[..., 0], 1 - ALERT_SHARE)
        result["high_vix_rule"] = classify(y_val, x_val[..., 0], vix_alert, y.mean())
        result["high_vix_rule"].pop("brier")  # Raw VIX is a rank score, not a probability.
    result["validation_alert_returns"] = alert_returns(real_val, y_val, alerts, bootstrap)
    return result


def main():
    torch.set_num_threads(1)
    train_file = ROOT / "data" / "train.parquet"
    validation_file = ROOT / "data" / "validation.parquet"
    train = load_scenarios(train_file)
    validation = load_scenarios(validation_file)
    train_x, train_real = annual_inputs(train)
    val_x, val_real = annual_inputs(validation)
    train_month, train_ids = monthly_us_equity(train_file, train.n_scenarios, train.n_years)
    val_month, val_ids = monthly_us_equity(validation_file, validation.n_scenarios, validation.n_years)
    if train_ids & val_ids:
        raise ValueError("Training and validation scenario IDs overlap")
    np.testing.assert_allclose(np.prod(1 + train_month, axis=-1) - 1, train.returns[..., US])
    np.testing.assert_allclose(np.prod(1 + val_month, axis=-1) - 1, validation.returns[..., US])
    train_month_x, train_dd = monthly_metrics(train_month)
    val_month_x, val_dd = monthly_metrics(val_month)
    train_events = {"real annual equity loss at least 20%": train_real[..., US] <= CRASH_CUTOFF,
                    "intrayear nominal drawdown at least 20%": train_dd <= CRASH_CUTOFF}
    val_events = {"real annual equity loss at least 20%": val_real[..., US] <= CRASH_CUTOFF,
                  "intrayear nominal drawdown at least 20%": val_dd <= CRASH_CUTOFF}
    bootstrap = np.random.default_rng(SEED + 2).integers(
        validation.n_scenarios, size=(BOOTSTRAPS, validation.n_scenarios)
    )
    specs = {
        "VIX": (train_x[..., :1], val_x[..., :1]),
        "lagged equity return": (train_x[..., 4:5], val_x[..., 4:5]),
        "annual market and macro": (train_x, val_x),
        "annual plus previous-year monthly data": (
            np.concatenate([train_x, train_month_x], axis=-1),
            np.concatenate([val_x, val_month_x], axis=-1),
        ),
    }
    result = {
        "decision_years": "3–30",
        "annual_features": ANNUAL_FEATURES,
        "monthly_features": ["last 3-month return", "last 6-month return", "annualized volatility",
                             "end-of-year drawdown", "worst drawdown in last completed year"],
        "events": {event: {
            name: assess(name, x, train_events[event], x_val, val_events[event], val_real, bootstrap)
            for name, (x, x_val) in specs.items()
        } for event in train_events},
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
