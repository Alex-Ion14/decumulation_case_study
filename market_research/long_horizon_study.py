"""Forecast all seven assets and test modest allocation changes.

Run python -B market_research/long_horizon_study.py from the repository root.
The script uses the existing requirements and emits JSON to stdout.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from forecast_study import BOOTSTRAPS, CASH, FIVE, ROOT, SEED, TIPS, US, path_interval, real_returns
from ml_forecasts import fit_ridge, load_scenarios, splits
from retirement_study import paired, paths, summaries
from retirement_strategy import DEFAULT_POLICY, Policy, Signals, Strategy
from utils import Config, simulate


HORIZONS = (1, 3, 5, 10)
MATURITIES = np.array([2, 5, 10, 30])
TENOR_NAMES = tuple(f"US_ZC_{year}Y" for year in MATURITIES)
RIDGE_PENALTY = 0.10
VAR_PENALTY = 0.10
NEIGHBORS = 30
TAIL_HORIZON = 3
TAIL_CUTOFF = -0.20
MAX_EQUITY_TILT = 0.10
FORECAST_METHODS = ("direct_ridge", "curve_var", "blend")
OVERLAYS = ("equity_forecast", "defensive_forecast", "downside_analogs", "equity_trend")
ATTRIBUTION = ("equity_leg_only", "treasury_leg_only", "carry_only", "always_less_equity")
MEASURES = ("US equity", "EM equity", "US REIT", "US Treasury 5Y", "US Treasury 30Y",
            "US TIPS 10Y", "Cash", "Inflation", "5Y yield change", "30Y yield change")


def curve_loadings(maturities):
    # Fixed Nelson–Siegel decay. This gives a curvature peak near 3 years.
    time = 0.60 * np.asarray(maturities)
    slope = -np.expm1(-time) / time
    return np.stack([np.ones_like(time), slope, slope - np.exp(-time)], axis=-1)


LOADINGS = curve_loadings(MATURITIES)
CURVE_FIT = np.linalg.pinv(LOADINGS).T


def state_rows(nominal, inflation, market, names):
    """End-of-year state. Returns supplied here are from completed years."""
    def col(name):
        return market[..., names.index(name)]
    curve = np.stack([col(name) for name in TENOR_NAMES], axis=-1) / 100
    factors = np.log1p(curve) @ CURVE_FIT
    bills = np.log1p(col("US_TBILL_3M") / 100)
    real_tips = np.log1p(curve[..., 2]) - np.log1p(col("US_BEI_10Y") / 100)
    real = real_returns(nominal, inflation)
    other = np.stack([bills, real_tips, np.log1p(inflation),
                      *np.moveaxis(np.log1p(real[..., :3]), -1, 0),
                      np.log(col("US_VIX")), col("US_GZ_SPREAD")], axis=-1)
    return np.concatenate([factors, other], axis=-1), np.log1p(curve)


def data_panel(sc):
    nominal, inflation = sc.returns.numpy(), sc.inflation.numpy()
    state, observed_curve = state_rows(nominal, inflation, sc.market.numpy(), sc.market_names)
    return state, observed_curve, real_returns(nominal, inflation)


def future_mean(x, horizon):
    """For state row i, average the realized values in rows i+1 through i+h."""
    cumulative = x.cumsum(axis=0)
    return (cumulative[horizon:] - cumulative[:-horizon]) / horizon


def outcomes(panel, horizon):
    state, curve, real = panel
    asset_log = future_mean(np.log1p(real), horizon)
    inflation_log = future_mean(state[..., 5], horizon)
    yields = np.expm1(curve[..., [1, 3]])
    change = yields[horizon:] - yields[:-horizon]
    return np.concatenate([asset_log, inflation_log[..., None], change], axis=-1)


@dataclass
class CurveVAR:
    center: np.ndarray
    scale: np.ndarray
    intercept: np.ndarray
    transition: np.ndarray

    def predict(self, current, curve, horizon):
        x = current
        aggregate = np.zeros((*x.shape[:-1], 8))
        # Start the first step at observed yields, keeping the fit residual out
        # of today's bond prices. Later steps use the projected factor curve.
        raw_curve = curve[..., [1, 3]]
        for _ in range(horizon):
            next_x = self.center + (
                self.intercept + ((x - self.center) / self.scale) @ self.transition
            ) * self.scale
            next_curve = next_x[..., :3] @ LOADINGS[[1, 3]].T
            nominal_bonds = rolling_bond_log(raw_curve, next_curve, np.array([5, 30]))
            real_tips = rolling_bond_log(x[..., 4], next_x[..., 4], 10)
            inflation = next_x[..., 5]
            real_cash = 0.5 * (x[..., 3] + next_x[..., 3]) - inflation
            aggregate += np.concatenate([
                next_x[..., 6:9], nominal_bonds - inflation[..., None],
                real_tips[..., None], real_cash[..., None], inflation[..., None],
            ], axis=-1)
            x, raw_curve = next_x, next_curve
        delta = np.expm1(raw_curve) - np.expm1(curve[..., [1, 3]])
        return np.concatenate([aggregate / horizon, delta], axis=-1)


def rolling_bond_log(start_yield_log, end_yield_log, maturity):
    # Constant-maturity price move plus estimated within-year carry.
    return maturity * (start_yield_log - end_yield_log) + 0.5 * (
        start_yield_log + end_yield_log
    )


def fit_curve_var(state):
    flat = state.reshape(-1, state.shape[-1])
    center, scale = flat.mean(axis=0), flat.std(axis=0)
    scaled = (state - center) / np.where(scale > 0, scale, 1)
    x = scaled[:-1].reshape(-1, state.shape[-1])
    y = scaled[1:].reshape(x.shape)
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    x, y = x - x_mean, y - y_mean
    covariance = x.T @ x / len(x) + VAR_PENALTY * np.eye(x.shape[-1])
    transition = np.linalg.solve(covariance, x.T @ y / len(x))
    radius = np.max(np.abs(np.linalg.eigvals(transition)))
    transition *= min(1, 0.98 / max(radius, 1e-12))
    return CurveVAR(center, np.where(scale > 0, scale, 1), y_mean - x_mean @ transition,
                    transition)


@dataclass
class Forecasts:
    curve_var: CurveVAR
    direct: list
    mean: np.ndarray

    def predict(self, state, curve, horizon):
        direct = np.stack([line.predict(state) for line in self.direct], axis=-1)
        projected = self.curve_var.predict(state, curve, horizon)
        return {"constant": np.broadcast_to(self.mean, direct.shape),
                "direct_ridge": direct, "curve_var": projected,
                "blend": (direct + projected) / 2}


def fit_forecasts(panel, horizon, var=None):
    state = panel[0]
    y = outcomes(panel, horizon)
    # Use training means for log returns/inflation and unchanged yields for
    # the yield forecast. Each output has the same fixed ridge penalty.
    mean = y.mean(axis=(0, 1))
    mean[-2:] = 0
    direct = [fit_ridge(state[:-horizon], y[..., i], RIDGE_PENALTY)
              for i in range(y.shape[-1])]
    return Forecasts(var or fit_curve_var(state), direct, mean)


@dataclass
class DownsideAnalogs:
    x: np.ndarray
    events: np.ndarray
    center: np.ndarray
    scale: np.ndarray

    def predict(self, state):
        # One closest observation per training path, then the closest paths.
        # Distances use state available at the decision. The attached outcomes
        # come entirely from the training paths.
        shape = state.shape[:-1]
        standardized = (state.reshape(-1, state.shape[-1]) - self.center) / self.scale
        flat = self.x.reshape(-1, self.x.shape[-1])
        norms = np.square(flat).sum(axis=-1)
        probabilities = []
        for start in range(0, len(standardized), 128):
            query = standardized[start:start + 128]
            distance = norms - 2 * query @ flat.T
            distance = distance.reshape(len(query), *self.x.shape[:-1])
            best_year = distance.argmin(axis=1)
            columns = np.arange(self.x.shape[1])[None, :]
            best_distance = distance[np.arange(len(query))[:, None], best_year, columns]
            closest_paths = np.argsort(best_distance, axis=1, kind="stable")[:, :NEIGHBORS]
            labels = self.events[best_year[np.arange(len(query))[:, None], closest_paths],
                                 closest_paths]
            probabilities.append((labels.sum(axis=1) + 1) / (labels.shape[1] + 2))
        return np.concatenate(probabilities).reshape(shape)


def downside_event(panel):
    """Any cumulative real 50/50 loss of 20% over the following 3 years."""
    mix = 0.5 * (panel[2][..., US] + panel[2][..., FIVE])
    future = np.stack([mix[year:len(mix) - TAIL_HORIZON + year]
                       for year in range(1, TAIL_HORIZON + 1)], axis=-1)
    return ((1 + future).cumprod(axis=-1) <= 1 + TAIL_CUTOFF).any(axis=-1)


def fit_analogs(panel):
    x = panel[0][:-TAIL_HORIZON]
    center = x.mean(axis=(0, 1))
    sd = x.std(axis=(0, 1))
    scale = np.where(sd > 0, sd, 1)
    return DownsideAnalogs((x - center) / scale, downside_event(panel), center, scale)


def subset(panel, indices):
    return tuple(values[:, indices] for values in panel)


def forecast_errors(y, p, baseline, bootstrap=None, benchmark="constant"):
    loss = (y - p) ** 2
    base = (y - baseline) ** 2
    result = {
        "rmse": np.sqrt(loss.mean(axis=(0, 1))).tolist(),
        f"mse_skill_vs_{benchmark}": (1 - loss.mean(axis=(0, 1)) / base.mean(axis=(0, 1))).tolist(),
    }
    if bootstrap is not None:
        family = len(HORIZONS) * len(MEASURES) * len(FORECAST_METHODS)
        result["mse_gain_ci95_family"] = [
            path_interval(column, bootstrap, family) for column in (base - loss).mean(axis=0).T
        ]
    return dict(zip(MEASURES, (dict(zip(result, values)) for values in zip(*result.values()))))


def downside_scores(y, p, base, bootstrap=None):
    brier = (y - p) ** 2
    baseline = (y - base) ** 2
    result = {"event_rate": y.mean(), "predicted_rate": p.mean(),
              "brier": brier.mean(), "constant_brier": baseline.mean()}
    if bootstrap is not None:
        result["brier_gain_ci95"] = path_interval((baseline - brier).mean(axis=0), bootstrap)
    return result


def cross_validation(train):
    n = train[0].shape[1]
    actual = {h: outcomes(train, h) for h in HORIZONS}
    output = {h: {name: np.empty_like(actual[h]) for name in ("constant", *FORECAST_METHODS)}
              for h in HORIZONS}
    event = downside_event(train)
    risk, base_risk = np.empty_like(event, dtype=float), np.empty_like(event, dtype=float)
    for fit, check in splits(n, 5, SEED):
        fit_panel, test_panel = subset(train, fit), subset(train, check)
        var = fit_curve_var(fit_panel[0])
        for h in HORIZONS:
            model = fit_forecasts(fit_panel, h, var)
            for name, pred in model.predict(test_panel[0][:-h], test_panel[1][:-h], h).items():
                output[h][name][:, check] = pred
        analogs = fit_analogs(fit_panel)
        risk[:, check] = analogs.predict(test_panel[0][:-TAIL_HORIZON])
        base_risk[:, check] = analogs.events.mean()
    return ({h: {name: forecast_errors(actual[h], pred, values["constant"])
                 for name, pred in values.items()} for h, values in output.items()},
            downside_scores(event, risk, base_risk))


class LongHorizonStrategy(Strategy):
    """Apply one bounded overlay to an existing retirement policy."""
    def __init__(self, policy, overlay, model, analogs):
        super().__init__(policy)
        self.overlay = overlay
        self.model = model
        self.analogs = analogs
        self.equity_history = []

    def act(self, obs):
        weights, rate = self.allocation(obs)
        self.equity_history.append(weights[:, US].clone())
        return weights, rate

    def allocation(self, obs):
        weights, rate = super().act(obs)
        if obs.year == 0:
            return weights, rate
        if self.overlay == "always_less_equity":
            return shift_equity(weights, np.full(len(weights), -MAX_EQUITY_TILT)), rate
        if self.overlay in ("equity_forecast", "defensive_forecast", "downside_analogs",
                            "equity_leg_only", "treasury_leg_only", "carry_only"):
            if obs.n_years - obs.year < TAIL_HORIZON:
                return weights, rate
            state, curve = state_rows(obs.return_history[-1].numpy(),
                                      obs.inflation_history[-1].numpy(),
                                      obs.market_history[-1].numpy(), obs.market_names)
            if self.overlay == "downside_analogs":
                risk = self.analogs.predict(state)
                cut = np.where(risk > self.analogs.events.mean() + 0.10, -MAX_EQUITY_TILT, 0)
                return shift_equity(weights, cut), rate
            mean = self.model.mean
            if self.overlay == "carry_only":
                expected = carry_forecast(state, curve, mean)
            else:
                expected = self.model.predict(state, curve, TAIL_HORIZON)["blend"]
            if self.overlay != "defensive_forecast":
                equity_edge = expected[:, US] - mean[US]
                treasury_edge = expected[:, FIVE] - mean[FIVE]
                edge = equity_edge - treasury_edge
                if self.overlay == "equity_leg_only":
                    edge = equity_edge
                if self.overlay == "treasury_leg_only":
                    edge = -treasury_edge
                tilt = np.where(np.abs(edge) > 0.02, np.sign(edge) * MAX_EQUITY_TILT, 0)
                return shift_equity(weights, tilt), rate

            alternatives = np.array([FIVE, TIPS, CASH])
            current = weights[:, alternatives].argmax(dim=1).numpy()
            forecast = expected[:, alternatives]
            best = forecast.argmax(axis=1)
            gap = forecast[np.arange(len(best)), best] - forecast[np.arange(len(best)), current]
            # Move half the existing defensive sleeve only if the annualized
            # advantage exceeds 2 points. Keep equity and withdrawals unchanged.
            transfer = torch.where(torch.from_numpy(gap > 0.02),
                                   0.5 * (1 - weights[:, US]), 0.0)
            rows = torch.arange(len(best))
            weights[rows, alternatives[current]] -= transfer
            weights[rows, alternatives[best]] += transfer
            return weights, rate

        lagged = obs.return_history[-1]
        tilt = np.where(lagged[:, US] < lagged[:, CASH], -MAX_EQUITY_TILT, 0)
        return shift_equity(weights, tilt), rate


def shift_equity(weights, tilt):
    equity = weights[:, US].clone()
    new = (equity + torch.from_numpy(tilt)).clamp(0, 1)
    weights *= ((1 - new) / (1 - equity))[:, None]
    weights[:, US] = new
    return weights


def verify_asset_order(sc):
    if list(sc.asset_names) != list(MEASURES[:7]):
        raise ValueError("Unexpected simulator asset order")


def evaluate_overlays(test, cfg, signals, models, analogs, bootstrap):
    # Keep both previously studied policies fixed for this new experiment.
    policies = {"declining": DEFAULT_POLICY,
                "middle peak": Policy(glide="middle peak", multiplier=0.80, defensive="tips")}
    out = {}
    for name, policy in policies.items():
        baseline = paths(policy, test, cfg, signals)
        report = {"baseline": summaries(baseline)}
        for overlay in (*OVERLAYS, *ATTRIBUTION):
            strategy = LongHorizonStrategy(policy, overlay, models[3], analogs).attach(cfg, signals)
            with torch.no_grad():
                raw = simulate(strategy, test, cfg)
            withdrawals = raw["withdrawals"]
            ruined = (raw["margins"] <= 0).any(dim=0).double().numpy()
            values = {"ruin": ruined, "norm_wd": raw["norm_wd"].numpy()}
            equity = torch.stack(strategy.equity_history).numpy()
            report[overlay] = {
                "mean_equity": equity.mean(),
                "mean_equity_first_decade": equity[:10].mean(),
                "mean_equity_last_decade": equity[-10:].mean(),
                "ruins": int(ruined.sum()),
                "norm_wd": values["norm_wd"].mean(),
                "J": cfg.alpha * values["norm_wd"].mean() - (1 - cfg.alpha) * ruined.mean(),
                "mean_total_withdrawal": withdrawals.sum(dim=0).mean().item(),
                **paired(values, baseline, cfg, bootstrap),
            }
        out[name] = report
    return out


def carry_forecast(state, curve, mean):
    """Hold current yields constant, with mean inflation and growth-asset returns."""
    result = np.broadcast_to(mean, (*state.shape[:-1], len(mean))).copy()
    result[..., [FIVE, FIVE + 1]] = curve[..., [1, 3]] - mean[7]
    result[..., TIPS] = state[..., 4]
    result[..., CASH] = state[..., 3] - mean[7]
    return result


def overlay_cv(train_sc, train, cfg, bootstrap):
    """Cross-fit the original equity-spread overlay on whole training paths."""
    result = {}
    policies = {"declining": DEFAULT_POLICY,
                "middle peak": Policy(glide="middle peak", multiplier=0.80, defensive="tips")}
    for name, policy in policies.items():
        metrics = ("ruin", "norm_wd")
        pairs = [{metric: np.empty(train_sc.n_scenarios) for metric in metrics} for _ in range(2)]
        for fit, check in splits(train_sc.n_scenarios, 5, SEED):
            sc = train_sc.subset(check)
            signals = Signals(train_sc.subset(fit), tips=True, crash=False)
            forecast = fit_forecasts(subset(train, fit), 3)
            baseline = paths(policy, sc, cfg, signals)
            strategy = LongHorizonStrategy(policy, "equity_forecast", forecast, None)
            with torch.no_grad():
                raw = simulate(strategy.attach(cfg, signals), sc, cfg)
            pairs[0]["ruin"][check] = (raw["margins"] <= 0).any(dim=0).double().numpy()
            pairs[0]["norm_wd"][check] = raw["norm_wd"].numpy()
            for metric in metrics:
                pairs[1][metric][check] = baseline[metric]
        result[name] = paired(*pairs, cfg, bootstrap)
    return result


def verify_disjoint_scenarios(train_file, test_file):
    import pandas as pd

    def ids(file):
        return set(pd.read_parquet(file, columns=["history_id"])["history_id"])
    if ids(train_file) & ids(test_file):
        raise ValueError("Training and test scenario IDs overlap")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "validation.parquet")
    args = parser.parse_args()
    torch.set_num_threads(1)
    train_file = ROOT / "data" / "train.parquet"
    if args.test.resolve() == train_file.resolve():
        parser.error("--test must be separate from the training file")
    train_sc = load_scenarios(train_file)
    verify_asset_order(train_sc)
    train = data_panel(train_sc)
    cv, cv_risk = cross_validation(train)
    var = fit_curve_var(train[0])
    models = {h: fit_forecasts(train, h, var) for h in HORIZONS}
    analogs = fit_analogs(train)
    signals = Signals(train_sc, tips=True, crash=False)
    verify_disjoint_scenarios(train_file, args.test)
    test_sc = load_scenarios(args.test)
    verify_asset_order(test_sc)
    if test_sc.n_years != train_sc.n_years:
        raise ValueError("The test file must use the same horizon as training")
    test = data_panel(test_sc)
    boot = np.random.default_rng(SEED + 2).integers(
        test_sc.n_scenarios, size=(BOOTSTRAPS, test_sc.n_scenarios)
    )
    results = {}
    carry_sensitivity = {}
    for h in HORIZONS:
        y = outcomes(test, h)
        predictions = models[h].predict(test[0][:-h], test[1][:-h], h)
        carry = carry_forecast(test[0][:-h], test[1][:-h], models[h].mean)
        carry_sensitivity[h] = {
            name: forecast_errors(y, pred, carry, benchmark="carry")
            for name, pred in {"constant": predictions["constant"],
                               "carry": carry, **{name: predictions[name]
                                                   for name in FORECAST_METHODS}}.items()
        }
        results[h] = {name: forecast_errors(y, pred, predictions["constant"],
                                            boot if name != "constant" else None)
                      for name, pred in predictions.items()}
    downside = downside_scores(downside_event(test), analogs.predict(test[0][:-TAIL_HORIZON]),
                               analogs.events.mean(), boot)
    report = {
        "test_file": args.test.name, "paths": [train_sc.n_scenarios, test_sc.n_scenarios],
        "training_group_cv": cv, "test_forecasts": results,
        "three_year_portfolio_downside": {"train_group_cv": cv_risk, "test": downside},
        "strategy_overlays": evaluate_overlays(test_sc, Config(), signals, models, analogs, boot),
        "carry_benchmark_exploratory": carry_sensitivity,
        "equity_overlay_training_cv_exploratory": overlay_cv(
            train_sc, train, Config(), np.random.default_rng(SEED + 2).integers(
                train_sc.n_scenarios, size=(BOOTSTRAPS, train_sc.n_scenarios)
            ),
        ),
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
