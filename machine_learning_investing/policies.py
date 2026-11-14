"""Fixed-spending allocation policies and their training objectives."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from machine_learning_investing.bayesian_gru import KL_WEIGHT, VariationalGRU
from market_research.retirement_strategy import HURDLE, Signals, yield_inputs
from strategies.bengen import WITHDRAWAL_RATE
from utils import DTYPE, simulate

LOOKBACK = 4
SEED = 42
WIDTH = 16
DROPOUT = 0.20
MC_DRAWS = 128
CONFIDENCE = 0.60
SHORTFALL_PENALTY = 3.0
YIELD_NAMES = ("US_TBILL_3M", "US_ZC_2Y", "US_ZC_5Y", "US_ZC_10Y",
               "US_ZC_30Y", "US_BEI_10Y", "US_GZ_SPREAD", "US_VIX")
GROWTH = {"US equity": 0.50, "EM equity": 0.30, "US REIT": 0.20}


def portfolio_utility(terminal, norm_wd):
    """Reward real growth with diminishing returns and penalize missed spending."""
    return torch.log1p(terminal) - SHORTFALL_PENALTY * (1 - norm_wd)


def mix(asset_names, growth_share):
    weights = {name: growth_share * share for name, share in GROWTH.items()}
    weights["US Treasury 5Y"] = 1 - growth_share
    return torch.tensor([weights.get(name, 0.0) for name in asset_names], dtype=DTYPE)


class FixedMix:
    def __init__(self, weights):
        self.weights = weights

    def act(self, obs):
        return self.weights, WITHDRAWAL_RATE


def policy_features(returns, inflation, market, market_names):
    rates = market[..., [market_names.index(name) for name in YIELD_NAMES]] / 100
    real = (1 + returns) / (1 + inflation.unsqueeze(-1)) - 1
    return torch.cat((real, inflation.unsqueeze(-1), rates), dim=-1).float()


def tips_features(market, market_names):
    _, yields = yield_inputs(market.numpy(), market_names)
    return torch.from_numpy(yields).float()


def tips_spread(scenarios):
    tips = scenarios.asset_names.index("US TIPS 10Y")
    five = scenarios.asset_names.index("US Treasury 5Y")
    return ((scenarios.returns[1:, :, tips] - scenarios.returns[1:, :, five])
            / (1 + scenarios.inflation[1:]))


class FeatureScale:
    def __init__(self, features):
        flat = features.flatten(0, 1)
        self.center = flat.mean(dim=0)
        self.scale = flat.std(dim=0, unbiased=False).clamp_min(1e-6)

    def __call__(self, features):
        return ((features - self.center) / self.scale).clamp(-4, 4)


def windows(features):
    """Each row ends at the last completed year, with a flag on observed rows."""
    observed = F.pad(features, (0, 1), value=1)
    padded = F.pad(observed, (0, 0, 0, 0, LOOKBACK - 1, 0))
    return padded.unfold(0, LOOKBACK, 1).movedim(-1, -2)


class Encoder(nn.Module):
    def __init__(self, kind, n_features, seed=SEED):
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.network = nn.Sequential(nn.Linear(LOOKBACK * n_features, WIDTH), nn.Tanh())
        else:
            self.network = nn.GRU(n_features, WIDTH, batch_first=True)
            if kind == "bayesian_gru":
                self.network = VariationalGRU(self.network, seed)

    def forward(self, inputs):
        if self.kind == "mlp":
            return self.network(inputs.flatten(start_dim=-2))
        flat = inputs.reshape(-1, *inputs.shape[-2:])
        if self.kind == "bayesian_gru":
            return self.network(flat).reshape(*inputs.shape[:-2], WIDTH)
        _, hidden = self.network(flat)
        return hidden[0].reshape(*inputs.shape[:-2], WIDTH)


class PolicyNetwork(nn.Module):
    def __init__(self, kind, assets, n_features, seed=SEED):
        super().__init__()
        initial = 0.9 * mix(assets, 0.65) + 0.1 / len(assets)
        self.logits = nn.Parameter(initial.float().log())
        self.encoder = None
        if kind != "static_fit":
            self.encoder = Encoder(kind, n_features, seed)
            self.tilt = nn.Linear(WIDTH, len(assets))
            nn.init.zeros_(self.tilt.weight)
            nn.init.zeros_(self.tilt.bias)

    def forward(self, inputs=None):
        if inputs is None:
            return self.logits.softmax(dim=-1)
        return self.weights_from_state(self.encoder(inputs))

    def weights_from_state(self, state):
        return (self.logits + 1.5 * torch.tanh(self.tilt(state))).softmax(dim=-1)


class AllocationTable:
    """Use in training only. Each decision was computed from the preceding rows."""
    def __init__(self, weights):
        self.weights = weights

    def act(self, obs):
        return self.weights[obs.year], WITHDRAWAL_RATE


class NeuralAllocation:
    def __init__(self, kind, train, cfg, seed):
        torch.manual_seed(seed)
        features = None
        if train.n_years == 1:
            kind = "static_fit"
        if kind != "static_fit":
            panel = policy_features(train.returns[:-1], train.inflation[:-1],
                                    train.market[:-1], train.market_names)
            self.scale = FeatureScale(panel)
            features = windows(self.scale(panel))
        n_features = 0 if features is None else features.shape[-1]
        self.model = PolicyNetwork(kind, train.asset_names, n_features, seed + 1)
        self.posterior = self.model.encoder.network if kind == "bayesian_gru" else None
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.025, weight_decay=0.02)
        for _ in range(65):
            optimizer.zero_grad()
            if features is None:
                policy = FixedMix(self.model())
            else:
                later = self.model(features)
                first = self.model().expand(1, train.n_scenarios, -1)
                policy = AllocationTable(torch.cat((first, later)))
            out = simulate(policy, train, cfg)
            loss = -portfolio_utility(out["wealth"][-1], out["norm_wd"]).mean()
            if self.posterior is not None:
                loss = loss + KL_WEIGHT * self.posterior.kl_per_weight()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
        self.model.eval()

    def current_inputs(self, obs):
        features = policy_features(obs.return_history[-LOOKBACK:], obs.inflation_history[-LOOKBACK:],
                                   obs.market_history[-LOOKBACK:], obs.market_names)
        return windows(self.scale(features))[-1]

    @torch.no_grad()
    def allocation_draws(self, obs):
        states = self.posterior.posterior_states(self.current_inputs(obs))
        return self.model.weights_from_state(states)

    @torch.no_grad()
    def act(self, obs):
        if self.model.encoder is None or obs.year == 0:
            return self.model(), WITHDRAWAL_RATE
        if self.posterior is not None:
            return self.allocation_draws(obs).mean(dim=0), WITHDRAWAL_RATE
        return self.model(self.current_inputs(obs)), WITHDRAWAL_RATE


class TipsNetwork(nn.Module):
    """Gaussian return-spread forecast with Monte Carlo dropout on the GRU state."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder("gru", 4)
        self.head = nn.Linear(WIDTH, 2)

    def distribution(self, state):
        location, raw_scale = self.head(state).unbind(dim=-1)
        return location, F.softplus(raw_scale) + 0.05

    def forward(self, inputs):
        return self.distribution(F.dropout(self.encoder(inputs), DROPOUT, self.training))


class TipsForecast:
    def __init__(self, train, seed):
        torch.manual_seed(seed)
        panel = tips_features(train.market[:-1], train.market_names)
        self.scale = FeatureScale(panel)
        inputs = windows(self.scale(panel)).flatten(0, 1)
        spread = tips_spread(train).float().flatten()
        self.center = spread.mean()
        self.sigma = spread.std(unbiased=False).clamp_min(1e-6)
        self.base_probability = (spread > HURDLE).float().mean().item()
        target = (spread - self.center) / self.sigma
        self.model = TipsNetwork()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.008, weight_decay=0.02)
        for _ in range(45):
            for idx in torch.randperm(len(target)).split(256):
                optimizer.zero_grad()
                mean, std = self.model(inputs[idx])
                loss = (std.log() + 0.5 * ((target[idx] - mean) / std).square()).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
        self.model.eval()
        # Use the same sampled readouts for every scenario and decision.
        generator = torch.Generator().manual_seed(seed + 1)
        self.masks = (torch.rand(MC_DRAWS, 1, WIDTH, generator=generator) > DROPOUT) / (1 - DROPOUT)

    @torch.no_grad()
    def predict(self, market, names):
        inputs = windows(self.scale(tips_features(market[-LOOKBACK:], names)))[-1]
        state = self.model.encoder(inputs)
        means, sigmas = self.model.distribution(state.unsqueeze(0) * self.masks)
        means = means * self.sigma + self.center
        sigmas = sigmas * self.sigma
        return means, sigmas


def mixture_cdf(value, means, sigmas):
    return torch.special.ndtr((value - means) / sigmas).mean(dim=0)


class YieldAllocation:
    """Keep the growth mix fixed and choose 5Y, TIPS, or cash for the rest."""
    def __init__(self, kind, train, forecast=None):
        self.kind = kind
        self.base = mix(train.asset_names, 0.60)
        self.ridge = Signals(train, tips=True, crash=False).tips if kind == "ridge" else None
        self.forecast = forecast

    def act(self, obs):
        if obs.year == 0 or (self.ridge is None and self.forecast is None):
            return self.base, WITHDRAWAL_RATE
        spread, yields = yield_inputs(obs.market_history[-1].numpy(), obs.market_names)
        cash_advantage = torch.from_numpy(-spread)
        hurdle = cash_advantage.clamp_min(HURDLE)
        if self.ridge is not None:
            tips_advantage = torch.from_numpy(self.ridge.predict(yields))
            take_tips = tips_advantage > hurdle
        else:
            means, sigmas = self.forecast.predict(obs.market_history, obs.market_names)
            take_tips = means.mean(dim=0) > hurdle
            if self.kind == "bayes_confidence":
                take_tips &= (1 - mixture_cdf(hurdle, means, sigmas)) >= CONFIDENCE
        take_cash = (cash_advantage > HURDLE) & ~take_tips
        weights = self.base.repeat(len(take_tips), 1)
        weights[:, obs.asset_names.index("US Treasury 5Y")] = 0.40 * ~(take_cash | take_tips)
        weights[:, obs.asset_names.index("US TIPS 10Y")] = 0.40 * take_tips
        weights[:, obs.asset_names.index("Cash")] = 0.40 * take_cash
        return weights, WITHDRAWAL_RATE
