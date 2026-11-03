# Decumulation strategy challenge

A retiree has a pot of savings and needs to live off it for **30 years**. Each year they
choose **how to invest** the remaining wealth and **how much to withdraw**. Withdrawing more
is better (because who doesn't like money), but running out of money (*ruin*) is very bad.

A classic approach to this problem is the **Bengen "4% rule"**. Its allocation is fixed (as in
Bengen's original study: 50% US stocks and 50% intermediate-term US Treasuries) and rebalanced
every year. Every year it withdraws 4% of the initial capital, adjusted for inflation.

**Your task:** design and optimise a strategy that performs better than the baseline, measured
on a hidden holdout set of scenarios (see [How strategies are evaluated](#how-strategies-are-evaluated)).
You may go beyond a fixed allocation
and withdrawal rate. Your strategy can react to wealth, time, past returns, inflation and
anything else that is known at the moment of the decision. However, keep in mind that
**interpretability is key**, so do not overcomplicate your strategy. The goal is to find the
sweet spot between simplicity and performance. So when creating your strategy, try to consider how difficult it would be to explain in words to someone with little financial knowledge.

## Quickstart

```bash
pip install -r requirements.txt
python main.py                          # the 4% rule vs. the optimised example
python main.py strategies/run.py   # your strategy vs. the 4% rule
python main.py strategies/run.py --no-frontier   # faster: skip the frontier plot
```

`main.py` trains on `data/train.parquet`, prints the scores on the train and the validation
scenarios (`data/validation.parquet`) and saves `frontier.png` (see [The frontier](#the-frontier)).

## Files

The code in this repository is a **starting point**: you may change all of it. We do recommend
keeping the market simulation and the evaluation the same (see [Rules](#rules)).

| File | What it is | Should you change it? |
|------|-----------|--------------------|
| `main.py` | Trains and scores strategies, and plots the frontier. We evaluate with it. | You may, as long as the command in [Submission](#submission-and-evaluation) still works |
| `utils.py` | Settings, loading the scenarios, the market simulator (`simulate`) and the score (`evaluate`) | You may, but preferably keep `simulate` and `evaluate` the same |
| `strategies/bengen.py` | The 4% rule: the baseline | Better not: everyone is compared with it |
| `strategies/bengen_optimised.py` | Example: learns the best static allocation and withdrawal rate. It also produces the frontier. | To build on it, copy it to `strategies/run.py`, so the frontier stays the same |
| `strategies/run.py` | **Your submission** | Yes |
| `data/train.parquet`, `data/validation.parquet` | The scenarios (see [Scenario data](#scenario-data)) | No |

## The market

All amounts are **real** (corrected for inflation) and in units of the initial capital, so a
wealth of 1.0 equals the initial capital. You can invest in 7 assets, all in US dollars:

| Asset | Role | Computed from (see [Scenario data](#scenario-data)) |
|-------|------|-----------------|
| `US equity` | growth | the monthly return `US_EQ_MKT` |
| `EM equity` | growth, international diversification | the monthly return `EM_EQ_MKT` |
| `US REIT` | real estate | the monthly return `US_REIT` |
| `US Treasury 5Y` | intermediate-term government bonds | a zero-coupon bond with always 5 years left, priced with the yield `US_ZC_5Y` |
| `US Treasury 30Y` | long-term government bonds | a zero-coupon bond with always 30 years left, priced with `US_ZC_30Y` |
| `US TIPS 10Y` | inflation-linked government bonds | a zero-coupon bond with always 10 years left, priced with the real yield (`US_ZC_10Y` minus the break-even inflation `US_BEI_10Y`), that also earns inflation |
| `Cash` | safe and liquid | the 3-month T-bill yield `US_TBILL_3M` |

The returns and interest rates in the data are **nominal** (not corrected for inflation).
Inflation is the change in the US consumer price index `US_CPI_NSA`, and the simulator corrects
for it once, in step 4 below. Every year, in every scenario:

1. Your strategy chooses **portfolio weights** and a **withdrawal rate**.
2. The portfolio is **rebalanced** to these weights. This costs 0.5% of the amount traded;
   the first investment is free.
3. The withdrawal rate is **clipped to [3%, 12%]** of the initial capital and withdrawn. If the
   wealth is not enough, the scenario is **ruined** and whatever is left is paid out.
4. The rest earns the portfolio return, corrected for inflation.

The settings are in `Config` in `utils.py`.

## How strategies are evaluated

A good strategy withdraws a lot and rarely runs out of money. We measure both on the holdout
scenarios:

- `norm_wd`: the discounted withdrawals `Σ_t 0.99^(t+1) · withdrawal_t`, averaged over the
  scenarios and divided by the same number for the 4% rule without ruin. So 1 means
  "as much as the 4% rule".
- `ruin_prob`: the fraction of scenarios that run out of money in at least one year.

How to weigh one against the other is a matter of taste, so **we do not rank on a single
number**. We look at the trade-off between the two: where your strategy lies compared with the
baseline and [the frontier](#the-frontier), and how it scores on

```
J = alpha · norm_wd − (1 − alpha) · ruin_prob        (higher is better)
```

for a range of values of `alpha` (from caring only about ruin, alpha = 0, to caring only
about withdrawals, alpha = 1). `main.py` prints `J` for several alphas. `Config.alpha` (0.25)
is only the alpha that the example strategy trains on and that the score tables use; you may
train on another alpha, or on a different objective altogether. We also look at how
interpretable your strategy is.

`main.py` also shows the mean withdrawal and the median final wealth, for information only.

## The frontier

Which strategy is "best" depends on alpha, i.e. on how much you care about withdrawals
compared to ruin. To show this, `main.py` trains the static example `bengen_optimised.py`
once for every alpha in `FRONTIER_ALPHAS`. It then plots the test ruin probability and
`norm_wd` of each trained version as a curve in `frontier.png`, and adds every strategy you
evaluated as a star.

The curve shows the best that a static allocation with a fixed withdrawal rate, trained on the
train scenarios, can do on the test scenarios. A star **above or to the left of the curve**
gets more withdrawals for the same risk of ruin, so it beats every static Bengen strategy.

## Writing a strategy

Write your strategy in a file called **`strategies/run.py`** (always this name, so there is no
need to use your group name). It needs a class named `Strategy` with two methods:

```python
import torch

class Strategy:
    def fit(self, train, cfg):
        """Called once with the train scenarios. Learn your parameters here."""

    def act(self, obs):
        """Called at the start of every year, for all scenarios at once."""
        # one weight per asset, in the order of obs.asset_names:
        # US equity, EM equity, US REIT, US Treasury 5Y, US Treasury 30Y, US TIPS 10Y, Cash
        weights = torch.tensor([0.5, 0, 0, 0.5, 0, 0, 0], dtype=torch.float64)
        rate = torch.where(obs.wealth < 0.5, 0.03, 0.04)  # withdraw less when wealth is low
        return weights, rate
```

What `act` returns:
- `weights`: the same weights for every scenario (shape `(A,)`) or one row per scenario
  (shape `(N, A)`). They must be non-negative and are scaled to sum to 1.
- `rate`: one number, or one number per scenario (shape `(N,)`).

`obs` tells you everything that is known at that moment (only the past, never the future):

| Field | Shape | Meaning |
|-------|-------|---------|
| `year`, `n_years` | number | the current year t (starting at 0) and the total number of years |
| `asset_names` | list | the names of the A assets |
| `wealth` | (N,) | current wealth in every scenario |
| `holdings` | (N, A) | current weights after last year's price moves (`None` in year 0) |
| `wealth_history` | (t+1, N) | wealth at the start of every year so far |
| `withdrawal_history` | (t, N) | withdrawals paid so far |
| `return_history` | (t, N, A) | asset returns so far |
| `inflation_history` | (t, N) | inflation so far |
| `market_names` | list | optional: the names of the V extra variables in the data, e.g. `"US_VIX"`, `"US_ZC_10Y"` |
| `market_history` | (t, N, V) | optional: the value of every extra variable at the end of every year so far |

**The extra variables are optional.** The simulation only uses the asset returns and
inflation; `market_names` and `market_history` are extra information that you **can safely
ignore**. A good strategy can be built from wealth, time, past returns and inflation alone.
Only use the extra variables if you have a clear idea why they would help (and keep the risk
of overfitting in mind).

**Your own state.** You decide which information from `obs` your strategy uses, for example
wealth, time left, or last year's returns. If you do want to use an extra variable: the
10-year yield at the end of last year, for example, is
`obs.market_history[-1, :, obs.market_names.index("US_ZC_10Y")]` (only when `obs.year > 0`).
Your strategy may also remember things between calls to `act`, by storing them on `self`. If
you need information that is not in `obs`, you may add a field to `Observation` in
`utils.py`; please mention this in your submission.

**Training.** You are free to choose how you train: gradient ascent through `simulate` (as in
`bengen_optimised.py`), a grid search over a few parameters, or anything else. Useful tools:
- `utils.evaluate(strategy, scenarios, cfg)` returns the score of a strategy.
- `utils.simulate(strategy, scenarios, cfg)` returns full paths, and is differentiable.
- `train.subset(indices)` selects scenarios, e.g. for minibatches or your own validation split.

## Rules

- The code is an example: you may change everything. We **recommend keeping the market
  simulation (`simulate`), the evaluation (`evaluate`) and the settings in `Config` the same**,
  so that your results can be compared with the baseline and with the other groups. If you
  change them, explain why in your submission.
- `fit` may only use the train scenarios it receives.
- Decisions in `act` may only use `obs` and what was learned in `fit` (no looking into the
  future).
- Extra helper files in `strategies/` are fine (import them with
  `from strategies.my_helpers import ...`), and so are extra packages (add them to
  `requirements.txt`).
- Keep `fit` reasonably fast: at most about 15 minutes on a laptop.

## Submission and evaluation

Hand in your copy of the repository, with your strategy in `strategies/run.py`. At the top of
that file, briefly describe your approach and any changes you made to other files. We run:

```bash
python main.py strategies/run.py --test holdout.parquet --out results.json
```

We also read your code, so clarity counts.

## Scenario data

The `data` folder contains the scenarios as parquet files. All scenarios are **fully
synthetic**: they are simulated, not historical or real market data.

- `train.parquet`: 100 scenarios. `fit` is trained on these.
- `validation.parquet`: 80 other scenarios, to check how well your strategy works on
  scenarios it was not trained on. `main.py` uses them as test scenarios.
- We evaluate on a hidden **holdout** file with yet other scenarios.

Every scenario covers 30 years, month by month (periods `2101-01` to `2130-12`). The files have
one row per scenario, month and variable:

| history_id | period | variable | value |
|------------|--------|----------|-------|
| h015bf1f4 | 2101-01 | US_EQ_MKT | -1.763641 |
| h015bf1f4 | 2101-01 | US_ZC_10Y | 4.767836 |
| ... | | | |

There are 50 variables, such as equity returns in % per month (`US_EQ_MKT`), interest rates
in % per year (`US_ZC_10Y`, `US_TBILL_3M`), price indices (`US_CPI_NSA`), unemployment
(`US_UNRATE_NSA`) and volatility (`US_VIX`). Only a few of them are needed for the 7 assets and
inflation; **you can ignore all the others**.

`load_scenarios` in `utils.py` reads a file and **aggregates it into years**:
- It computes the monthly returns of the 7 assets and compounds them over the 12 months of
  every year.
- Inflation is compounded in the same way.
- It also keeps the end-of-year values of all 50 variables, as optional extra information in
  `obs.market_history`.

In the first month there is no previous month, so the bonds and cash simply earn their yield
and inflation is 0.

> **Watch out for overfitting.** With only 100 train scenarios, a strategy can easily become
> too tailored to them. The validation scenarios start in a different economic situation (for
> example, lower interest rates), and the holdout scenarios may differ again. Always check the
> validation scores, not just the train scores.
