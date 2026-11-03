"""Shared code: settings, scenario data, the market simulator and the evaluation.

The data contains nominal returns and interest rates. The simulator corrects them for
inflation once, so all amounts in the simulation are *real* and measured in units of the
initial capital: a wealth of 1.0 equals the initial capital, and a withdrawal rate of 0.04
means withdrawing 4% of the initial capital (in today's money) that year.

This code is an example and you may change it, but we recommend keeping the market
simulation (``simulate``) and the evaluation (``evaluate``) the same, so that results can be
compared with the baseline and with other groups.
"""

from dataclasses import dataclass

import pandas as pd
import torch

DTYPE = torch.float64


# ── Settings ─────────────────────────────────────────────────────────────────

@dataclass
class Config:
    transaction_cost: float = 0.005  # cost per unit of traded wealth when rebalancing (0.5%)
    wd_min: float = 0.03  # every year at least 3% of the initial capital is withdrawn ...
    wd_max: float = 0.12  # ... and at most 12%
    discount_factor: float = 0.99  # yearly discount on withdrawals in the score
    alpha: float = 0.25  # score J = alpha * norm_wd - (1 - alpha) * ruin_prob


# ── Scenario data ────────────────────────────────────────────────────────────

@dataclass
class Scenarios:
    returns: torch.Tensor  # (T, N, A) nominal yearly return of every asset, per year and scenario
    inflation: torch.Tensor  # (T, N)  yearly US inflation
    market: torch.Tensor  # (T, N, V) optional extra information: every variable in the data at the end of every year
    asset_names: list  # A names
    market_names: list  # V names

    @property
    def n_years(self):
        return self.returns.shape[0]

    @property
    def n_scenarios(self):
        return self.returns.shape[1]

    @property
    def n_assets(self):
        return self.returns.shape[2]

    def subset(self, idx):
        """Select some scenarios, e.g. a minibatch or your own validation split."""
        return Scenarios(self.returns[:, idx], self.inflation[:, idx], self.market[:, idx],
                         self.asset_names, self.market_names)


# The 7 assets you can invest in (see load_scenarios for how their returns are computed).
ASSET_NAMES = ["US equity", "EM equity", "US REIT", "US Treasury 5Y", "US Treasury 30Y", "US TIPS 10Y", "Cash"]


def load_scenarios(path):
    """Read a parquet file with monthly scenarios and aggregate it into yearly scenarios.

    The file has the columns history_id (the scenario), period ("YYYY-MM"), variable and
    value. First the monthly returns of the 7 assets (all in US dollars) are computed:
      - US equity, EM equity, US REIT: the monthly returns in the data (in %);
      - US Treasury 5Y / 30Y: a zero-coupon Treasury bond that always has 5 / 30 years left
        until it matures, priced with the US zero-coupon yields;
      - US TIPS 10Y: the same for an inflation-linked bond with 10 years left. It is priced with
        the real yield (10-year nominal yield minus break-even inflation) and also earns inflation;
      - Cash: one month of the 3-month T-bill yield.
    Then the months are aggregated into years: returns and inflation (the change in the US
    consumer price index) are compounded over the 12 months of every year, and of all other
    variables the value at the end of every year is kept. The returns stay nominal here;
    ``simulate`` corrects them for inflation.

    The data has no month before the first month, so in the first month the bonds and cash
    simply earn their yield and inflation is 0.
    """
    df = pd.read_parquet(path)
    wide = df.pivot(index=["history_id", "period"], columns="variable", values="value").sort_index()
    wide = wide.groupby(level="history_id").ffill()  # quarterly variables keep their last value
    n_scenarios = wide.index.get_level_values("history_id").nunique()
    n_months = len(wide) // n_scenarios
    if n_scenarios * n_months != len(wide) or n_months % 12 != 0:
        raise ValueError(f"{path}: every scenario must contain the same whole number of years")

    names = list(wide.columns)
    data = torch.tensor(wide.to_numpy(), dtype=DTYPE).reshape(n_scenarios, n_months, len(names))
    data = data.transpose(0, 1)  # (months, N, V)
    v = {name: data[:, :, i] for i, name in enumerate(names)}  # monthly values of every variable

    def last_month(x):
        return torch.cat([x[:1], x[:-1]])  # the first month uses its own value

    def zero_coupon_return(yield_pct, maturity):
        """Monthly return of a bond bought at last month's yield and valued at this month's yield."""
        y = yield_pct / 100
        return (1 + last_month(y)) ** maturity / (1 + y) ** (maturity - 1 / 12) - 1

    inflation = v["US_CPI_NSA"] / last_month(v["US_CPI_NSA"]) - 1
    real_yield_pct = 100 * ((1 + v["US_ZC_10Y"] / 100) / (1 + v["US_BEI_10Y"] / 100) - 1)
    monthly_returns = torch.stack([
        v["US_EQ_MKT"] / 100,  # US equity
        v["EM_EQ_MKT"] / 100,  # EM equity
        v["US_REIT"] / 100,  # US REIT
        zero_coupon_return(v["US_ZC_5Y"], 5),  # US Treasury 5Y
        zero_coupon_return(v["US_ZC_30Y"], 30),  # US Treasury 30Y
        (1 + zero_coupon_return(real_yield_pct, 10)) * (1 + inflation) - 1,  # US TIPS 10Y
        (1 + last_month(v["US_TBILL_3M"]) / 100) ** (1 / 12) - 1,  # Cash
    ], dim=-1)  # (months, N, 7)

    n_years = n_months // 12

    def compound_per_year(monthly):
        return (1 + monthly).reshape(n_years, 12, *monthly.shape[1:]).prod(dim=1) - 1

    returns = compound_per_year(monthly_returns)
    if torch.isnan(returns).any() or torch.isnan(inflation).any():
        raise ValueError(f"{path}: missing values in the asset or inflation variables")
    return Scenarios(returns=returns, inflation=compound_per_year(inflation), market=data[11::12],
                     asset_names=ASSET_NAMES, market_names=names)


# ── Simulation ───────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """What a strategy knows at the start of year t, for all N scenarios at once.

    Only the past is included, so a strategy cannot look into the future.
    """
    year: int  # t = 0, 1, ..., T-1
    n_years: int  # T
    asset_names: list  # A names
    wealth: torch.Tensor  # (N,)      current wealth
    holdings: torch.Tensor  # (N, A)    current weights after last year's price moves (None in year 0)
    wealth_history: torch.Tensor  # (t+1, N)  wealth at the start of years 0..t
    withdrawal_history: torch.Tensor  # (t, N)    withdrawals paid in years 0..t-1
    return_history: torch.Tensor  # (t, N, A) asset returns in years 0..t-1
    inflation_history: torch.Tensor  # (t, N)    inflation in years 0..t-1
    market_names: list  # optional extra information (can be ignored): V names of all variables in the data
    market_history: torch.Tensor  # (t, N, V) optional: value of every variable at the end of years 0..t-1


def simulate(strategy, sc, cfg):
    """Run a strategy on all scenarios. Differentiable, so it can be used for training.

    Every year, in every scenario:
      1. the strategy chooses portfolio weights and a withdrawal rate (``strategy.act``);
      2. the portfolio is rebalanced to these weights, which costs
         transaction_cost * (amount traded). The first investment is free;
      3. the withdrawal rate (clipped to [wd_min, wd_max]) is withdrawn. If there is not
         enough wealth, the scenario is *ruined* and whatever is left is paid out;
      4. the rest earns the portfolio return, corrected for inflation.

    Returns a dict with:
      wealth      (T+1, N) wealth at the start of every year; the last row is the final wealth
      withdrawals (T, N)   withdrawals paid
      margins     (T, N)   wealth left after the requested withdrawal (<= 0 means ruined)
      norm_wd     (N,)     discounted withdrawals, divided by those of a 4% strategy that is
                           never ruined (so a value of 1 means "as good as the 4% rule")
    """
    n_years, n, n_assets = sc.n_years, sc.n_scenarios, sc.n_assets
    wealth = torch.ones(n, dtype=DTYPE)
    holdings = None  # filled after the first year
    wealth_list, withdrawal_list, margin_list = [wealth], [], []

    for t in range(n_years):
        obs = Observation(
            year=t, n_years=n_years, asset_names=sc.asset_names, wealth=wealth, holdings=holdings,
            wealth_history=torch.stack(wealth_list),
            withdrawal_history=torch.stack(withdrawal_list) if t > 0 else torch.zeros(0, n, dtype=DTYPE),
            return_history=sc.returns[:t],
            inflation_history=sc.inflation[:t],
            market_names=sc.market_names,
            market_history=sc.market[:t],
        )

        # 1. decide
        weights, rate = strategy.act(obs)
        weights = torch.as_tensor(weights, dtype=DTYPE).expand(n, n_assets)  # also accepts shape (A,)
        if not (weights >= 0).all():
            raise ValueError("weights must be non-negative (no short selling) and not NaN")
        weights = weights / weights.sum(dim=1, keepdim=True)
        rate = torch.as_tensor(rate, dtype=DTYPE).expand(n).clamp(cfg.wd_min, cfg.wd_max)

        # 2. rebalance
        if holdings is not None:
            traded = (weights - holdings).abs().sum(dim=1)
            wealth = wealth * (1.0 - cfg.transaction_cost * traded)

        # 3. withdraw
        margin = wealth - rate
        withdrawal_list.append(torch.minimum(rate, wealth))
        margin_list.append(margin)

        # 4. grow
        returns = sc.returns[t]
        wealth = margin.clamp(min=0.0) * (1.0 + (weights * returns).sum(dim=1)) / (1.0 + sc.inflation[t])
        holdings = weights * (1.0 + returns)
        holdings = holdings / holdings.sum(dim=1, keepdim=True)
        wealth_list.append(wealth)

    withdrawals = torch.stack(withdrawal_list)
    discounts = cfg.discount_factor ** torch.arange(1, n_years + 1, dtype=DTYPE)
    discounted_withdrawals = (discounts[:, None] * withdrawals).sum(dim=0)
    return {
        "wealth": torch.stack(wealth_list),
        "withdrawals": withdrawals,
        "margins": torch.stack(margin_list),
        "norm_wd": discounted_withdrawals / (0.04 * discounts.sum()),
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(strategy, sc, cfg):
    """Metrics of a strategy on a set of scenarios:

    norm_wd   : average normalised withdrawals (1 = the 4% rule without ruin)
    ruin_prob : fraction of scenarios that run out of money in at least one year
    J         : alpha * norm_wd - (1 - alpha) * ruin_prob for alpha = cfg.alpha (higher is better)
    The other metrics are for information only.
    """
    with torch.no_grad():
        out = simulate(strategy, sc, cfg)
    ruin_prob = (out["margins"] <= 0).any(dim=0).double().mean().item()
    norm_wd = out["norm_wd"].mean().item()
    return {
        "J": cfg.alpha * norm_wd - (1 - cfg.alpha) * ruin_prob,
        "ruin_prob": ruin_prob,
        "norm_wd": norm_wd,
        "mean_withdrawal": out["withdrawals"].mean().item(),
        "median_final_wealth": out["wealth"][-1].median().item(),
    }
