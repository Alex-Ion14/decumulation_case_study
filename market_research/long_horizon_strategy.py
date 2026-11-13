"""Holdout candidate using the bounded stock-versus-Treasury return tilt.

Implements the project's Strategy.fit / Strategy.act interface. The underlying
middle-peak policy is the 0.80 spending variant from the retirement study.
"""
from pathlib import Path
import sys

# The repository's strategy loader imports this file directly by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from long_horizon_study import LongHorizonStrategy, data_panel, fit_forecasts, verify_asset_order
from retirement_strategy import Policy, Signals


class Strategy:
    def fit(self, train, cfg):
        verify_asset_order(train)
        policy = Policy(glide="middle peak", multiplier=0.80, defensive="tips")
        forecasts = fit_forecasts(data_panel(train), 3)
        self.policy = LongHorizonStrategy(policy, "equity_forecast", forecasts, None)
        self.policy.attach(cfg, Signals(train, tips=True, crash=False))

    def act(self, obs):
        return self.policy.allocation(obs)
