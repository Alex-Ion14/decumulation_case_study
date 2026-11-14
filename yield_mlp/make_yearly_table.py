"""Write the average portfolio of the yield model for each of the 30 years to yearly_table.py.

    python -B yield_mlp/make_yearly_table.py

The yield model (hardcoded_strategy.py) chooses its portfolio from two bond yields, so in
the same year it holds a different portfolio in every scenario. This file runs it on the 153
pool scenarios, notes the portfolio it chose in every year of every scenario, and averages
those per year. The result is one portfolio per year: a table with 30 rows.

IMPORTANT: a table per year is NOT the yield model. It can no longer react to the yields,
which is where the yield model gets its extra return from. On the pool scenarios the year
table ends with a median final wealth of about 2.2 times the starting capital, the yield
table with about 3.5. Use yearly_table.py only if you need a portfolio that depends on
nothing but the year.
"""

import os
import sys

import torch

# Tell Python where the project folder is, so it can find utils.py (see yield_mlp.py).
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(THIS_FOLDER)
sys.path.insert(0, PROJECT_FOLDER)

import hardcoded_strategy  # noqa: E402  the yield model as a table (in this folder)
from utils import ASSET_NAMES, Config, load_scenarios, simulate  # noqa: E402

OUTPUT_FILE = os.path.join(THIS_FOLDER, "yearly_table.py")


class Notebook:
    """Wraps a strategy and writes down the portfolio it chooses every year."""

    def __init__(self, strategy):
        self.strategy = strategy
        self.portfolios = []  # one entry per year: the portfolio in every scenario

    def act(self, obs):
        weights, rate = self.strategy.act(obs)
        number_of_scenarios = len(obs.wealth)
        # In year 0 the strategy returns one portfolio for all scenarios; repeat it per scenario.
        self.portfolios.append(weights.expand(number_of_scenarios, len(ASSET_NAMES)))
        return weights, rate


# The text of yearly_table.py. @...@ is replaced by the real numbers below.
TEMPLATE = r"""'''The average portfolio of the yield model for each of the 30 years.

THIS FILE WAS WRITTEN BY yield_mlp/make_yearly_table.py. Do not change the numbers by hand.

IMPORTANT: this is NOT the yield model. It is the average of what the yield model holds in
each year, over the 153 pool scenarios. It depends on nothing but the year, so it cannot
react to the bond yields. On the pool scenarios it ends with a median final wealth of about
@YEAR_MEDIAN@ times the starting capital, against about @YIELD_MEDIAN@ for the yield table in
hardcoded_strategy.py.

Use it as a strategy of the case:

    python main.py yield_mlp/yearly_table.py --no-frontier
'''

import torch

WITHDRAWAL_RATE = 0.04

# The order of the 7 numbers in every portfolio below.
ASSETS = @ASSETS@

# YEARLY_PORTFOLIOS[0] is the portfolio for the first year, YEARLY_PORTFOLIOS[29] for the last.
# Because of rounding the weights can add up to 0.999 or 1.001; the simulator scales them to 1.
YEARLY_PORTFOLIOS = [
@ROWS@
]


class Strategy:
    '''The strategy in the form that main.py of the case expects.'''

    def fit(self, train, cfg):
        pass  # nothing to learn: the table is fixed

    def act(self, obs):
        # obs.year counts from 0 (first year) to 29 (last year).
        return torch.tensor(YEARLY_PORTFOLIOS[obs.year], dtype=torch.float64), WITHDRAWAL_RATE
"""


def median_final_wealth(strategy, scenarios):
    with torch.no_grad():
        return simulate(strategy, scenarios, Config())["wealth"][-1].median().item()


if __name__ == "__main__":
    # The 153 pool scenarios: everything except the test set.
    pool = load_scenarios(os.path.join(PROJECT_FOLDER, "data", "pool.parquet"))

    # Run the yield model and write down its portfolios.
    notebook = Notebook(hardcoded_strategy.Strategy())
    with torch.no_grad():
        outcome = simulate(notebook, pool, Config())

    # Average per year, only over the scenarios that still have money at the start of that
    # year (a portfolio of nothing means nothing).
    rows = []
    for year in range(pool.n_years):
        has_money = outcome["wealth"][year] > 0
        average = notebook.portfolios[year][has_money].mean(dim=0)
        numbers = ", ".join(f"{weight:.3f}" for weight in average.tolist())
        rows.append(f"    [{numbers}],  # year {year + 1}")

    # Write the file once, read it back to measure it, then write it again with the
    # measured results in its explanation.
    def write(year_median, yield_median):
        text = TEMPLATE.replace("@ASSETS@", str(ASSET_NAMES)).replace("@ROWS@", "\n".join(rows))
        text = text.replace("@YEAR_MEDIAN@", year_median).replace("@YIELD_MEDIAN@", yield_median)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
            file.write(text)

    write("?", "?")
    import yearly_table  # noqa: E402  the file we just wrote
    year_median = median_final_wealth(yearly_table.Strategy(), pool)
    yield_median = median_final_wealth(hardcoded_strategy.Strategy(), pool)
    write(f"{year_median:.1f}", f"{yield_median:.1f}")

    print(f"Wrote {os.path.relpath(OUTPUT_FILE, PROJECT_FOLDER)}")
    print(f"Median final wealth on the {pool.n_scenarios} pool scenarios: "
          f"year table {year_median:.2f}, yield table {yield_median:.2f}")
    print("\nyear  " + "  ".join(f"{name:>15}" for name in ASSET_NAMES))
    for year, row in enumerate(yearly_table.YEARLY_PORTFOLIOS):
        print(f"{year + 1:>4}  " + "  ".join(f"{100 * weight:>14.1f}%" for weight in row))
