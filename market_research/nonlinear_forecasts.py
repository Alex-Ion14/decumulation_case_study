"""Compare exact Gaussian processes, small neural nets, and ridge on case data.

Run from the repository root: python -B market_research/nonlinear_forecasts.py
The script uses the existing NumPy and PyTorch dependencies and prints JSON.
"""
from dataclasses import dataclass
import json

import numpy as np
import torch
from torch import nn

from ml_forecasts import (
    BOOTSTRAPS, EARLY_YEARS, ROOT, SEED, TARGETS, choose_penalty, error_skill,
    fit_ridge, fit_simple, load_scenarios, panel, path_interval, score, splits,
    switching,
)


# Groups from the previous ridge study. This is a public-validation follow-up.
EXPERIMENTS = {
    "5Y Treasury vs cash": "market",
    "TIPS vs 5Y Treasury": "market_macro",
    "Real cash return": "market",
    "US equity vs cash": "market_macro",
}
METHODS = ("ridge", "ridge_3year", "gp", "mlp", "gru")
GP_LENGTHS = (0.65, 1.6)
GP_NOISES = (0.35, 0.9)  # Observation standard deviation in scaled target units.
HIDDEN = 8
MAX_EPOCHS = 80
PATIENCE = 12
PATH_BATCH_SIZE = 24


@dataclass
class Scaling:
    low: np.ndarray
    high: np.ndarray
    center: np.ndarray
    scale: np.ndarray

    def transform(self, x):
        return (np.clip(x, self.low, self.high) - self.center) / self.scale


def fit_scaling(x):
    flat = x.reshape(-1, x.shape[-1])
    low, high = np.quantile(flat, [0.01, 0.99], axis=0)
    clipped = np.clip(flat, low, high)
    sd = clipped.std(axis=0)
    return Scaling(low, high, clipped.mean(axis=0), np.where(sd > 0, sd, 1))


def lagged_rows(x):
    """Concatenate current and previous two feature rows, padding at the start."""
    row = np.arange(len(x))
    return np.concatenate([x[np.maximum(row - lag, 0)] for lag in range(3)], axis=-1)


def gp_kernel(x, z, relative_length):
    dimensions = x.shape[-1]
    distance = torch.cdist(x, z).square() / dimensions
    nonlinear = torch.exp(-distance / (2 * relative_length ** 2))
    return x @ z.T / dimensions + 0.25 * nonlinear


@dataclass
class GaussianProcess:
    scaling: Scaling
    train_x: torch.Tensor
    target_mean: float
    target_sd: float
    relative_length: float
    noise: float
    cholesky: torch.Tensor
    coefficients: torch.Tensor

    def predict(self, x):
        flat = self.scaling.transform(x).reshape(-1, x.shape[-1])
        check = torch.from_numpy(flat)
        cross_covariance = gp_kernel(check, self.train_x, self.relative_length)
        mean = (cross_covariance @ self.coefficients).squeeze(-1)
        solved = torch.linalg.solve_triangular(self.cholesky, cross_covariance.T, upper=False)
        prior_variance = check.square().mean(dim=-1) + 0.25
        variance = (prior_variance - solved.square().sum(dim=0)).clamp_min(0)
        sd = (variance + self.noise ** 2).sqrt() * self.target_sd
        return (mean.numpy().reshape(x.shape[:-1]) * self.target_sd + self.target_mean,
                sd.numpy().reshape(x.shape[:-1]))


def fit_gp(x, y, relative_length, noise):
    scaling = fit_scaling(x)
    train_x = torch.from_numpy(scaling.transform(x).reshape(-1, x.shape[-1]))
    mean, sd = y.mean(), y.std()
    centered = torch.from_numpy((y.ravel() - mean) / sd).unsqueeze(-1)
    covariance = gp_kernel(train_x, train_x, relative_length)
    covariance.diagonal().add_(noise ** 2 + 1e-8)
    cholesky = torch.linalg.cholesky(covariance)
    coefficients = torch.cholesky_solve(centered, cholesky)
    return GaussianProcess(scaling, train_x, mean, sd, relative_length, noise,
                           cholesky, coefficients)


class FeedForward(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.linear = nn.Linear(inputs, 1)
        self.nonlinear = nn.Sequential(nn.Linear(inputs, HIDDEN), nn.Tanh(),
                                       nn.Dropout(0.1), nn.Linear(HIDDEN, 1))

    def forward(self, x):
        return (self.linear(x) + self.nonlinear(x)).squeeze(-1)


class Recurrent(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.linear = nn.Linear(inputs, 1)
        self.gru = nn.GRU(inputs, HIDDEN, batch_first=True)
        self.output = nn.Sequential(nn.Dropout(0.1), nn.Linear(HIDDEN, 1))

    def forward(self, x):
        hidden, _ = self.gru(x)
        return (self.linear(x) + self.output(hidden)).squeeze(-1)


@dataclass
class NeuralForecast:
    network: nn.Module
    scaling: Scaling
    target_mean: float
    target_sd: float

    def inputs(self, x):
        # Network layout is paths by years by features.
        return torch.as_tensor(self.scaling.transform(x).swapaxes(0, 1), dtype=torch.float32)

    @torch.no_grad()
    def predict(self, x):
        self.network.eval()
        prediction = self.network(self.inputs(x)).numpy().T.astype(np.float64)
        return prediction * self.target_sd + self.target_mean


def fit_network(network_type, x, y, seed, epochs, check=None):
    torch.manual_seed(seed)
    shuffle = np.random.default_rng(seed)
    model = NeuralForecast(network_type(x.shape[-1]), fit_scaling(x), y.mean(), y.std())
    inputs = model.inputs(x)
    targets = torch.as_tensor(((y - model.target_mean) / model.target_sd).T,
                              dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.network.parameters(), lr=0.005, weight_decay=0.02)
    if check is not None:
        check_inputs = model.inputs(check[0])
        check_y = torch.as_tensor(((check[1] - model.target_mean) / model.target_sd).T,
                                  dtype=torch.float32)
    best_epoch = epochs
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.network.train()
        order = shuffle.permutation(len(inputs))
        for start in range(0, len(inputs), PATH_BATCH_SIZE):
            batch = order[start:start + PATH_BATCH_SIZE]
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model.network(inputs[batch]), targets[batch])
            loss.backward()
            optimizer.step()

        if check is not None:
            model.network.eval()
            with torch.no_grad():
                check_loss = nn.functional.mse_loss(model.network(check_inputs), check_y).item()
            if check_loss < best_loss:
                best_epoch, best_loss = epoch, check_loss
            if epoch - best_epoch >= PATIENCE:
                break
    return model, best_epoch


def fit_candidates(x, y, seed, test):
    """Select hyperparameters inside these training paths and refit on all of them."""
    inner, check = next(splits(y.shape[1], 5, seed))
    inner_x, inner_y = x[:, inner], y[:, inner]
    check_x, check_y = x[:, check], y[:, check]
    predictions = {}
    choices = {}

    for method, inputs, check_inputs in (("ridge", x, test),
                                         ("ridge_3year", lagged_rows(x), lagged_rows(test))):
        penalty = choose_penalty(inputs, y, binary=False)
        predictions[method] = fit_ridge(inputs, y, penalty).predict(check_inputs)
        choices[method] = {"penalty": penalty}

    errors = {}
    for length in GP_LENGTHS:
        for noise in GP_NOISES:
            gp = fit_gp(inner_x, inner_y, length, noise)
            mean, _ = gp.predict(check_x)
            errors[length, noise] = np.mean((check_y - mean) ** 2)
    length, noise = min(errors, key=errors.get)
    gp = fit_gp(x, y, length, noise)
    predictions["gp"], gp_sd = gp.predict(test)
    choices["gp"] = {"relative_length": length, "noise_sd": noise}

    for method, kind in (("mlp", FeedForward), ("gru", Recurrent)):
        _, best_epoch = fit_network(kind, inner_x, inner_y, seed, MAX_EPOCHS,
                                    check=(check_x, check_y))
        network, _ = fit_network(kind, x, y, seed, best_epoch)
        predictions[method] = network.predict(test)
        choices[method] = {"epochs": best_epoch,
                           "parameters": sum(p.numel() for p in network.network.parameters())}
        if method == "gru":
            # A decision must be identical when later observations are removed.
            prefix = min(6, len(test))
            np.testing.assert_allclose(network.predict(test[:prefix]),
                                       predictions[method][:prefix], atol=1e-7, rtol=1e-6)
    return predictions, gp_sd, choices


def cross_validate(x, y, simple_x, target):
    predictions = {name: np.empty_like(y) for name in ("simple", *METHODS)}
    gp_sd = np.empty_like(y)
    choices = []
    for fold, (train, check) in enumerate(splits(y.shape[1], 5, SEED)):
        fitted, uncertainty, parameters = fit_candidates(x[:, train], y[:, train],
                                                          SEED + fold, x[:, check])
        simple = fit_simple(target, simple_x[:, train], y[:, train])
        predictions["simple"][:, check] = simple(simple_x[:, check])
        for name, prediction in fitted.items():
            predictions[name][:, check] = prediction
        gp_sd[:, check] = uncertainty
        choices.append(parameters)
    return predictions, gp_sd, choices


def report_scores(y, predictions):
    ridge = predictions["ridge"]
    return {
        name: {
            **score(y, pred, binary=False),
            "mse_gain_vs_ridge": error_skill(y, pred, ridge),
        }
        for name, pred in predictions.items()
    }


def run_target(target, group, train, validation, bootstrap):
    x, y = train["matrices"][group], train["targets"][target.name]
    x_val, y_val = validation["matrices"][group], validation["targets"][target.name]
    simple_x = train["simple_inputs"].get(target.feature, np.zeros_like(y))
    simple_val = validation["simple_inputs"].get(target.feature, np.zeros_like(y_val))
    cv, cv_gp_sd, fold_choices = cross_validate(x, y, simple_x, target)
    predictions, gp_sd, choices = fit_candidates(x, y, SEED + 5, x_val)
    predictions["simple"] = fit_simple(target, simple_x, y)(simple_val)
    if target.name == "TIPS vs 5Y Treasury":
        # Carry forward the earlier three-input alternative on the same test years.
        names = ("real_yield_10y", "breakeven inflation", "yield_5y")
        columns = [train["feature_names"][group].index(name) for name in names]
        small_x, small_val = x[..., columns], x_val[..., columns]
        penalty = choose_penalty(small_x, y, binary=False)
        predictions["ridge_3yield"] = fit_ridge(small_x, y, penalty).predict(small_val)

    ridge = predictions["ridge"]
    validation_scores = report_scores(y_val, predictions)
    for name, pred in predictions.items():
        error_delta = (y_val - ridge) ** 2 - (y_val - pred) ** 2
        if name in METHODS[1:]:
            validation_scores[name]["mse_gain_ci95_family"] = path_interval(
                error_delta.mean(axis=0), bootstrap, len(EXPERIMENTS) * (len(METHODS) - 1)
            )
        validation_scores[name].update({
            "early_mse_gain_vs_ridge": error_skill(y_val[:EARLY_YEARS], pred[:EARLY_YEARS],
                                                  ridge[:EARLY_YEARS]),
            "late_mse_gain_vs_ridge": error_skill(y_val[EARLY_YEARS:], pred[EARLY_YEARS:],
                                                 ridge[EARLY_YEARS:]),
        })

    if target.pair:
        base_net, _ = switching(target, ridge, validation["real"])
        family_size = sum(bool(t.pair) and t.name in EXPERIMENTS for t in TARGETS)
        family_size *= len(METHODS) - 1
        for name, pred in predictions.items():
            net, stats = switching(target, pred, validation["real"])
            log_gain = (np.log1p(net) - np.log1p(base_net)).mean(axis=0)
            stats["net_edge_vs_ridge"] = (net - base_net).mean()
            if name in METHODS[1:]:
                stats["log_edge_ci95_family"] = path_interval(log_gain, bootstrap, family_size)
            validation_scores[name]["instrument_check"] = stats

    # A Gaussian predictive interval and a version scaled using training OOF
    # residuals. Both are checked on the public validation paths.
    cv_standardized_error = np.abs(y - cv["gp"]) / cv_gp_sd
    calibrated_multiplier = np.quantile(cv_standardized_error, 0.90)
    gp_error = np.abs(y_val - predictions["gp"])
    ridge_half_width = np.quantile(np.abs(y - cv["ridge"]), 0.90)
    ridge_error = np.abs(y_val - ridge)
    result = {
        "feature_group": group,
        "training_nested_cv": report_scores(y, cv),
        "fold_hyperparameters": fold_choices,
        "final_hyperparameters": choices,
        "validation": validation_scores,
        "gp_nominal_90pct_intervals": {
            "gaussian_coverage": np.mean(gp_error <= 1.644854 * gp_sd),
            "gaussian_mean_width": np.mean(2 * 1.644854 * gp_sd),
            "train_oof_scaled_coverage": np.mean(gp_error <= calibrated_multiplier * gp_sd),
            "train_oof_scaled_mean_width": np.mean(2 * calibrated_multiplier * gp_sd),
            "train_oof_multiplier": calibrated_multiplier,
            "ridge_constant_width_coverage": np.mean(ridge_error <= ridge_half_width),
            "ridge_constant_width": 2 * ridge_half_width,
        },
    }
    if "ridge_3yield" in predictions:
        # Same public paths as the primary comparison, so treat this as exploratory.
        net_gp, _ = switching(target, predictions["gp"], validation["real"])
        net_small, _ = switching(target, predictions["ridge_3yield"], validation["real"])
        edge = net_gp - net_small
        log_gain = (np.log1p(net_gp) - np.log1p(net_small)).mean(axis=0)
        result["tips_gp_vs_3yield_followup"] = {
            "net_edge_early": edge[:EARLY_YEARS].mean(),
            "net_edge_late": edge[EARLY_YEARS:].mean(),
            "net_edge": edge.mean(),
            "log_edge_ci95_unadjusted": path_interval(log_gain, bootstrap),
            "net_returns_at_double_cost": {
                name: switching(target, predictions[name], validation["real"], 0.02)[1]
                for name in ("gp", "ridge_3yield")
            },
        }
    return result


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train = panel(load_scenarios(ROOT / "data" / "train.parquet"))
    validation = panel(load_scenarios(ROOT / "data" / "validation.parquet"))
    val_paths = validation["real"].shape[1]
    bootstrap = np.random.default_rng(SEED + 2).integers(val_paths, size=(BOOTSTRAPS, val_paths))
    results = {
        target.name: run_target(target, EXPERIMENTS[target.name], train, validation, bootstrap)
        for target in TARGETS if target.name in EXPERIMENTS
    }
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
