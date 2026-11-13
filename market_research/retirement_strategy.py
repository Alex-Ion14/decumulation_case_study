"""Candidate retirement policy compatible with the repository simulator.

Run the comparisons with: python -B market_research/retirement_study.py
"""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch

# main.py can load a strategy by path, without adding this directory to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crash_signals import annual_feature_rows, annual_inputs, fit_crash
from crash_signals import ALERT_SHARE, CRASH_CUTOFF
from ml_forecasts import fit_ridge
from utils import DTYPE


GLIDES = {
    "static": (0.50, 0.50),
    "declining": (0.70, 0.30),
    "rising": (0.30, 0.70),
    "middle peak": (0.40, 0.30),
}
ANNUITY_RATE = 0.01
TIPS_PENALTY = 0.10
HURDLE = 0.02
MAX_SPENDING = 0.08


@dataclass(frozen=True)
class Policy:
    glide: str = "static"
    spend: str = "guarded"
    multiplier: float = 1.0
    defensive: str = "spread"
    lock_late: bool = False
    crash_to: str = "off"


def yield_inputs(market, names):
    def value(name):
        return market[..., names.index(name)] / 100
    five, bill = value("US_ZC_5Y"), value("US_TBILL_3M")
    ten, bei = value("US_ZC_10Y"), value("US_BEI_10Y")
    real_ten = (1 + ten) / (1 + bei) - 1
    return five - bill, np.stack((real_ten, bei, five), axis=-1)


class Signals:
    def __init__(self, train, tips=True, crash=True):
        self.tips = None
        self.crash = None
        if tips:
            _, yields = yield_inputs(train.market.numpy(), train.market_names)
            tips_index = train.asset_names.index("US TIPS 10Y")
            five_index = train.asset_names.index("US Treasury 5Y")
            return_spread = (train.returns[1:, :, tips_index] - train.returns[1:, :, five_index])
            target = return_spread.numpy() / (1 + train.inflation[1:].numpy())
            self.tips = fit_ridge(yields[:-1], target, TIPS_PENALTY)
        if crash:
            inputs, returns = annual_inputs(train)
            us = train.asset_names.index("US equity")
            self.crash = fit_crash(inputs, returns[..., us] <= CRASH_CUTOFF)
            self.crash_threshold = np.quantile(self.crash.predict(inputs), 1 - ALERT_SHARE)


DEFAULT_POLICY = Policy(glide="declining", multiplier=0.90, defensive="tips")


class Strategy:
    """Annual allocation plus real spending based on wealth and years remaining."""

    def __init__(self, policy=DEFAULT_POLICY):
        self.policy = policy

    def fit(self, train, cfg):
        self.attach(cfg, Signals(train, tips=self.policy.defensive == "tips",
                                 crash=self.policy.crash_to != "off"))

    def attach(self, cfg, signals):
        self.cfg = cfg
        self.signals = signals
        return self

    def act(self, obs):
        policy = self.policy
        years_left = obs.n_years - obs.year
        fraction = obs.year / max(obs.n_years - 1, 1)
        first, last = GLIDES[policy.glide]
        if policy.glide == "middle peak":
            peak = 0.65
            weight = np.interp(fraction, (0.0, 1 / 3, 1.0), (first, peak, last))
        else:
            weight = first + fraction * (last - first)
        equity = torch.full_like(obs.wealth, float(weight))
        # A 1% real discount rate defines an annuity due. The first withdrawal
        # comes immediately and the spending rate is recomputed once per year.
        discount = 1 / (1 + ANNUITY_RATE)
        annuity = (1 - discount ** years_left) / (1 - discount)
        funded_spend = obs.wealth / annuity
        if policy.lock_late and years_left <= 10:
            # Limit equity to the surplus above the remaining 4% real spending
            # goal. Equity is capped only once wealth covers that goal.
            goal = 0.04 * years_left
            surplus_share = (1 - goal / obs.wealth.clamp_min(1e-9)).clamp_min(0)
            equity = torch.where(obs.wealth >= goal, torch.minimum(equity, surplus_share), equity)

        cuts_to_cash = torch.zeros_like(equity)
        if policy.crash_to != "off" and obs.year >= 2:
            features = annual_feature_rows(
                obs.return_history[-2:].numpy(), obs.inflation_history[-2:].numpy(),
                obs.market_history[-2:].numpy(), obs.market_names,
            )[0]
            alert = torch.from_numpy(self.signals.crash.predict(features)
                                     > self.signals.crash_threshold)
            cut = torch.where(alert, equity.clamp(max=0.20), 0.0)
            equity -= cut
            if policy.crash_to == "cash":
                cuts_to_cash = cut

        cash_choice = torch.zeros_like(equity, dtype=torch.bool)
        tips_choice = torch.zeros_like(cash_choice)
        if policy.defensive in ("spread", "tips") and obs.year:
            spread, yields = yield_inputs(obs.market_history[-1].numpy(), obs.market_names)
            cash_advantage = torch.from_numpy(-spread)
            cash_choice = cash_advantage > HURDLE
            if policy.defensive == "tips":
                tips_advantage = torch.from_numpy(self.signals.tips.predict(yields))
                tips_choice = (tips_advantage > HURDLE) & (~cash_choice | (tips_advantage > cash_advantage))
                cash_choice &= ~tips_choice

        remaining = 1 - equity - cuts_to_cash
        weights = torch.zeros((len(equity), len(obs.asset_names)), dtype=DTYPE)
        weights[:, obs.asset_names.index("US equity")] = equity
        weights[:, obs.asset_names.index("US Treasury 5Y")] = remaining * (~(cash_choice | tips_choice))
        weights[:, obs.asset_names.index("Cash")] = cuts_to_cash + remaining * cash_choice
        weights[:, obs.asset_names.index("US TIPS 10Y")] = remaining * tips_choice

        if policy.spend == "fixed":
            return weights, policy.multiplier
        rate = (funded_spend * policy.multiplier).clamp(self.cfg.wd_min, MAX_SPENDING)
        return weights, rate
