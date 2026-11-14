"""What an investing model gets to see at the start of a year.

Everything is computed from ``obs`` (see utils.Observation), so only from the past:

  state features   how far we are (year), how much money is left (wealth) and how that
                   compares with the withdrawals that still have to be paid (funded ratio)
  market features  interest rates and spreads at the end of last year, last year's real
                   returns of the 7 assets, last year's inflation, and the average real US
                   equity return over the last 3 and 5 years

The market features follow market_research/: yields say a lot about next year's bond and cash
returns, and very little is known in advance about equity returns. In year 0 there is no
market history yet, so the market features are 0 and the flag ``has_history`` is 0.
"""

import torch
from torch import nn

from utils import DTYPE

STATE_NAMES = ["year", "log wealth", "log funded ratio"]
MARKET_NAMES = [
    "5Y yield", "30Y yield", "T-bill yield", "break-even inflation", "10Y real yield",
    "5Y minus T-bill", "30Y minus 5Y", "log VIX", "credit spread", "unemployment",
    "last return US equity", "last return EM equity", "last return US REIT", "last return 5Y",
    "last return 30Y", "last return TIPS", "last return cash", "last inflation",
    "US equity 3y average", "US equity 5y average",
]


def _market_block(market, returns, inflation, names):
    """Raw market features at the start of a year.

    market (N, V) is the last row of the market history; returns (t, N, A) and inflation (t, N)
    are the full histories so far (t >= 1).
    """
    v = {name: market[:, names.index(name)] for name in
         ["US_ZC_5Y", "US_ZC_30Y", "US_ZC_10Y", "US_TBILL_3M", "US_BEI_10Y", "US_VIX", "US_GZ_SPREAD",
          "US_UNRATE_NSA"]}
    real = (1 + returns) / (1 + inflation[:, :, None]) - 1  # (t, N, A) real returns so far
    equity_log = torch.log1p(real[:, :, 0])
    columns = [
        v["US_ZC_5Y"], v["US_ZC_30Y"], v["US_TBILL_3M"], v["US_BEI_10Y"], v["US_ZC_10Y"] - v["US_BEI_10Y"],
        v["US_ZC_5Y"] - v["US_TBILL_3M"], v["US_ZC_30Y"] - v["US_ZC_5Y"],
        torch.log(v["US_VIX"].clamp(min=1.0)), v["US_GZ_SPREAD"], v["US_UNRATE_NSA"],
        *real[-1].unbind(dim=1), inflation[-1],
        equity_log[-3:].mean(dim=0), equity_log[-5:].mean(dim=0),
    ]
    return torch.stack(columns, dim=1)


class Features(nn.Module):
    """Turns an observation into a table of numbers (one row per scenario) for a model."""

    def __init__(self, withdrawal_rate, use_market=True):
        super().__init__()
        self.withdrawal_rate = withdrawal_rate
        self.use_market = use_market
        self.names = STATE_NAMES + (MARKET_NAMES + ["has history"] if use_market else [])
        self.register_buffer("mean", torch.zeros(len(MARKET_NAMES), dtype=DTYPE))
        self.register_buffer("std", torch.ones(len(MARKET_NAMES), dtype=DTYPE))

    @property
    def size(self):
        return len(self.names)

    def fit(self, train):
        """Learn the average and spread of every market feature, to bring them to the same scale."""
        if not self.use_market:
            return self
        blocks = [_market_block(train.market[t - 1], train.returns[:t], train.inflation[:t], train.market_names)
                  for t in range(1, train.n_years)]
        blocks = torch.cat(blocks)
        self.mean.copy_(blocks.mean(dim=0))
        self.std.copy_(blocks.std(dim=0).clamp(min=1e-6))
        return self

    def forward(self, obs):
        years_left = obs.n_years - obs.year
        funded_ratio = obs.wealth / (self.withdrawal_rate * years_left)  # 1 = exactly enough without any return
        columns = [
            torch.full_like(obs.wealth, 2 * obs.year / obs.n_years - 1),
            torch.log(obs.wealth + 0.05),
            torch.log(funded_ratio + 0.05),
        ]
        state = torch.stack(columns, dim=1)
        if not self.use_market:
            return state
        if obs.year == 0:
            market = torch.zeros(len(obs.wealth), len(MARKET_NAMES) + 1, dtype=DTYPE)
        else:
            block = _market_block(obs.market_history[-1], obs.return_history, obs.inflation_history,
                                  obs.market_names)
            block = ((block - self.mean) / self.std).clamp(-5, 5)
            market = torch.cat([block, torch.ones(len(obs.wealth), 1, dtype=DTYPE)], dim=1)
        return torch.cat([state, market], dim=1)
