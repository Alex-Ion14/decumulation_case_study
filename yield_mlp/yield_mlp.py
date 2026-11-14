"""A small neural network that chooses the portfolio from two bond yields.

    python -B yield_mlp/yield_mlp.py

THE IDEA
--------
Every year the model looks at two numbers that are known at that moment:
    - the yield of a 5-year US Treasury bond at the end of last year
    - the yield of a 30-year US Treasury bond at the end of last year
From these two numbers it decides how to spread the money over the 7 assets.
Nothing else is used: no wealth, no year, no other market data.

The withdrawal is not decided here. Every year 4% of the starting capital is taken out
(corrected for inflation), as in the Bengen rule. Another team works on the withdrawals.

In the very first year no yield "at the end of last year" exists yet. For that year the
model assumes that both yields are at their normal (average) level.

ONE MODEL OR MANY?
------------------
A single network depends on which data set it happened to be trained on. So we train one
network on each of the 50 data sets and invest in the AVERAGE of the 50 portfolios they propose.
Where the networks agree the average is clear-cut; where they disagree it is a compromise.

THE RESULT AS A TABLE
---------------------
Because the model only looks at two numbers, everything it does fits in a table: "for
this 5-year yield and this 30-year yield, hold this portfolio". At the end this file
writes that table to hardcoded_strategy.py. That file needs no neural network and no
training: it only looks up the portfolio in the table.

WHAT IS A NEURAL NETWORK HERE?
------------------------------
A formula with about 1,400 adjustable numbers. It turns the two yields into 7 portfolio
weights. At the start the adjustable numbers are random, so the portfolio is arbitrary.
"Training" means: try the formula on many scenarios, see how well the retiree ends up, and
nudge every adjustable number a tiny bit in the direction that gives a better result.
Repeat a few hundred times.

HOW THIS FILE IS ORGANISED
--------------------------
    1. Settings          the numbers you may want to change
    2. The model         the formula: two yields in, seven weights out
    3. The score         how we measure whether a result is good
    4. Training          the loop that improves the model step by step
    5. Averaging         many trained models that act as one
    6. The table         writing the averaged portfolios to hardcoded_strategy.py
    7. Running it        train on every data set, average, print and write the table
"""

import copy
import os
import sys

import torch

# Python only finds the other files of this project (utils.py, strategies/, ...) if it knows
# where the project folder is. These two lines tell it: "one folder above this file".
PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_FOLDER)

from machine_learning_investing.data import load_fold  # noqa: E402  loads the scenarios (see section 5)
from strategies.bengen import Strategy as BengenRule  # noqa: E402  the 50/50 baseline of the case
from utils import ASSET_NAMES, Config, simulate  # noqa: E402  the market simulator of the case


# ══════════════════════════════════════════════════════════════════════════════
# 1. SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# The two numbers the model looks at. These are names of variables in the scenario data.
# Both are in percent per year (4.2 means 4.2%).
YIELDS = ["US_ZC_5Y", "US_ZC_30Y"]

WITHDRAWAL_RATE = 0.04  # 4% of the starting capital per year, fixed

# How many of the 50 bootstrapped data sets to use (see bootstrap/README.md). One network is
# trained on each, and their portfolios are averaged. Each set takes about 15 seconds, so
# all 50 take about 12 minutes. Use a smaller number (for example 5) for a quick try-out.
NUMBER_OF_DATA_SETS = 50

# The yields (in percent) for which the table in hardcoded_strategy.py gets an entry. The
# same steps are used for the 5-year and the 30-year yield. In the data both yields lie
# between 0% and 13%, and 99.9% of the time below 12%.
TABLE_YIELDS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

# Size of the network: how many in-between numbers it computes on the way from the two
# yields to the seven weights. More = a more flexible formula, but easier to overfit.
HIDDEN_SIZE = 32

# Training settings. These values were tried out in machine_learning_investing/.
TRAINING_STEPS = 300  # how many times the model is nudged
SCENARIOS_PER_STEP = 128  # each step looks at this many randomly picked scenarios
LEARNING_RATE = 0.003  # how big every nudge is
CHECK_EVERY = 20  # every 20 steps we check the model on scenarios it is not trained on

# In the score, one ruined scenario costs as much as this many points (see section 3).
RUIN_PENALTY = 1.0

NUMBER_OF_ASSETS = len(ASSET_NAMES)  # 7
DECIMAL = torch.float64  # the simulator calculates with high-precision decimal numbers


# ══════════════════════════════════════════════════════════════════════════════
# 2. THE MODEL
# ══════════════════════════════════════════════════════════════════════════════

class YieldModel:
    """Two yields in, seven portfolio weights out."""

    def __init__(self):
        # The network: three steps, each one a set of adjustable numbers.
        #   step 1: combine the 2 yields into 32 in-between numbers
        #   step 2: combine those 32 numbers into 32 new ones
        #   step 3: combine those into 7 numbers, one "preference" per asset
        # "Linear" means: multiply every input by an adjustable number and add them up.
        # "Tanh" squeezes every in-between number into the range -1 to 1. Without it the
        # three steps would collapse into one simple multiplication, and the model could
        # only learn straight-line relations between yields and weights.
        self.network = torch.nn.Sequential(
            torch.nn.Linear(len(YIELDS), HIDDEN_SIZE), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), torch.nn.Tanh(),
            torch.nn.Linear(HIDDEN_SIZE, NUMBER_OF_ASSETS),
        ).to(DECIMAL)

        # Start close to "the same amount in every asset, whatever the yields are": make the
        # adjustable numbers of the last step small. Training then starts from a sensible
        # portfolio instead of a random one.
        with torch.no_grad():
            self.network[-1].weight.mul_(0.1)
            self.network[-1].bias.zero_()

        # The typical level of each yield and how much it varies. Filled in by
        # learn_typical_yields() below.
        self.typical_yield = torch.zeros(len(YIELDS), dtype=DECIMAL)
        self.typical_variation = torch.ones(len(YIELDS), dtype=DECIMAL)

    def learn_typical_yields(self, scenarios):
        """Look up the average of each yield and how much it varies in the train scenarios.

        A network learns best when its inputs are small numbers around 0. So we never feed
        it "5.1%" but "0.8 above normal", where normal and the size of one unit come from
        the train scenarios.
        """
        # scenarios.market holds every variable at the end of every year:
        # one value per year, per scenario and per variable.
        columns = [scenarios.market[:, :, scenarios.market_names.index(name)] for name in YIELDS]
        yields = torch.stack(columns, dim=-1)  # year x scenario x 2
        self.typical_yield = yields.mean(dim=(0, 1))
        self.typical_variation = yields.std(dim=(0, 1))

    def portfolio_for(self, yields):
        """The portfolio for given yields. One row per scenario: [5-year yield, 30-year yield] in percent."""
        compared_to_normal = (yields - self.typical_yield) / self.typical_variation
        preferences = self.network(compared_to_normal)  # 7 numbers per scenario, can be anything
        # "softmax" turns the 7 preferences into 7 weights that are all positive and add up
        # to 1 (= 100% of the money). A higher preference gives a higher weight.
        return torch.softmax(preferences, dim=1)

    def first_year_portfolio(self):
        """The portfolio for the first year: the one that belongs to normal yields."""
        return self.portfolio_for(self.typical_yield[None, :])[0]

    def act(self, obs):
        """Called by the simulator at the start of every year, for all scenarios at once.

        obs holds everything that is known at that moment (see README.md of the case).
        We must return the portfolio weights and the withdrawal rate.
        """
        if obs.year == 0:
            # First year: no yields known yet. One fixed portfolio for all scenarios.
            return self.first_year_portfolio(), WITHDRAWAL_RATE

        # Later years. obs.market_history[-1] is the last row of the history: the values
        # of all variables at the end of last year, for every scenario.
        end_of_last_year = obs.market_history[-1]
        columns = [end_of_last_year[:, obs.market_names.index(name)] for name in YIELDS]
        yields = torch.stack(columns, dim=1)  # scenario x 2
        return self.portfolio_for(yields), WITHDRAWAL_RATE

    def adjustable_numbers(self):
        """Everything that training is allowed to change."""
        return list(self.network.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# 3. THE SCORE
# ══════════════════════════════════════════════════════════════════════════════
#
# After 30 years every scenario ends with a final wealth (1.0 = the starting capital,
# 0 = ran out of money along the way, which we call "ruined"). One number sums it up:
#
#     score = average of log(1 + final wealth)  -  RUIN_PENALTY x share of ruined scenarios
#
# Why the logarithm? It makes going from 1 to 2 count as much as going from 2 to 4. Without
# it, a few scenarios with an enormous final wealth would dominate the average, and the
# model would gamble. With it, the model cares about the typical scenario.

def results(strategy, scenarios):
    """Run a strategy on scenarios and summarise how it went. Nothing is learned here."""
    with torch.no_grad():  # "just calculate, do not prepare for nudging"
        outcome = simulate(strategy, scenarios, Config())
    final_wealth = outcome["wealth"][-1]  # one number per scenario
    # outcome["margins"] is the wealth left right after each year's withdrawal.
    # Zero or less in any year means the money ran out.
    ruined = (outcome["margins"] <= 0).any(dim=0)
    share_ruined = ruined.double().mean().item()
    return {
        "score": torch.log1p(final_wealth).mean().item() - RUIN_PENALTY * share_ruined,
        "ruined": share_ruined,
        "median final wealth": final_wealth.median().item(),
        "worst 10% final wealth": final_wealth.quantile(0.10).item(),
    }


def score_for_training(outcome, softness):
    """The same score, in a form the training can work with.

    Training works by asking: "if I change this adjustable number a tiny bit, does the score
    go up or down?" Ruin is a yes/no event: a tiny change almost never flips it, so the
    answer would always be "no difference" and the model would never learn to avoid ruin.
    We therefore replace yes/no by "how close to ruin": a number between 0 and 1 that rises
    smoothly as the lowest wealth of a scenario approaches zero. `softness` sets how gradual
    this is: large = very gradual, small = almost the real yes/no.
    """
    margins = outcome["margins"]  # year x scenario

    # After a ruin the wealth stays 0 whatever the model does. So we only look at the years
    # up to and including the first ruin, to point the training at the year it went wrong.
    ruined_this_year = (margins <= 0).to(DECIMAL)
    ruined_in_an_earlier_year = torch.cat([torch.zeros_like(margins[:1]), ruined_this_year[:-1]]).cumsum(dim=0) > 0
    lowest_margin = margins.masked_fill(ruined_in_an_earlier_year, float("inf")).min(dim=0).values

    closeness_to_ruin = torch.sigmoid(-lowest_margin / softness)  # between 0 and 1, per scenario
    final_wealth = outcome["wealth"][-1]
    return torch.log1p(final_wealth).mean() - RUIN_PENALTY * closeness_to_ruin.mean()


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_model(train_scenarios, validation_scenarios, seed=0, show_progress=False):
    """Make a new model and improve it step by step. Returns the best version.

    seed: any whole number. It fixes the random starting numbers of the network, so the
          same seed always gives the same model. Different seeds give different starts.

    train_scenarios:      the model learns from these.
    validation_scenarios: the model never learns from these. We only use them to check,
                          every 20 steps, how the model does on scenarios it does not know,
                          and we keep the version that did best there. This guards against
                          "overfitting": learning the train scenarios by heart instead of
                          learning something that works in general.
    """
    torch.manual_seed(seed)  # makes the random starting numbers the same on every run
    model = YieldModel()
    model.learn_typical_yields(train_scenarios)

    # "Adam" is the standard recipe for nudging the adjustable numbers. weight_decay pulls
    # every number slightly towards 0, which keeps the formula from becoming extreme.
    optimizer = torch.optim.Adam(model.adjustable_numbers(), lr=LEARNING_RATE, weight_decay=0.0001)

    best_model = None
    best_validation_score = -float("inf")

    for step in range(1, TRAINING_STEPS + 1):
        # Start with a very gradual version of ruin (0.1) and end close to the real
        # yes/no (0.01). Early on the model needs clear directions; later, precision.
        progress = (step - 1) / (TRAINING_STEPS - 1)  # runs from 0 to 1
        softness = 0.1 * (0.01 / 0.1) ** progress

        # Pick a random handful of train scenarios. A different handful every step keeps
        # the model from fitting one particular group too closely.
        random_order = torch.randperm(train_scenarios.n_scenarios)
        batch = train_scenarios.subset(random_order[:SCENARIOS_PER_STEP])

        # Run 30 years of retirement on these scenarios with the current model ...
        outcome = simulate(model, batch, Config())
        score = score_for_training(outcome, softness)

        # ... and nudge. The three lines below are the heart of the learning:
        optimizer.zero_grad()  # forget the directions of the previous step
        (-score).backward()  # work out, for every adjustable number, which way improves the score
        #                      (the minus is there because the recipe lowers things, and we want a higher score)
        torch.nn.utils.clip_grad_norm_(model.adjustable_numbers(), 1.0)  # never nudge too hard at once
        optimizer.step()  # do the nudge

        # Every 20 steps: how does the model do on scenarios it is not trained on?
        if step % CHECK_EVERY == 0:
            validation_score = results(model, validation_scenarios)["score"]
            if validation_score > best_validation_score:
                best_validation_score = validation_score
                best_model = copy.deepcopy(model)  # keep a copy of this version
            if show_progress:
                print(f"  step {step:3d}: score on validation scenarios {validation_score:.3f}")

    return best_model


# ══════════════════════════════════════════════════════════════════════════════
# 5. AVERAGING
# ══════════════════════════════════════════════════════════════════════════════

class AverageOfModels:
    """Several trained models that act as one: it invests in the average of their portfolios."""

    def __init__(self, models):
        self.models = models

    def portfolio_for(self, yields):
        """The average portfolio for given yields. One row per scenario: [5-year, 30-year] in percent."""
        portfolios = [model.portfolio_for(yields) for model in self.models]
        return torch.stack(portfolios).mean(dim=0)  # the average over the models

    def first_year_portfolio(self):
        portfolios = [model.first_year_portfolio() for model in self.models]
        return torch.stack(portfolios).mean(dim=0)

    def act(self, obs):
        """Called by the simulator: ask every model for its portfolio and take the average."""
        portfolios = [model.act(obs)[0] for model in self.models]
        return torch.stack(portfolios).mean(dim=0), WITHDRAWAL_RATE


# ══════════════════════════════════════════════════════════════════════════════
# 6. THE TABLE
# ══════════════════════════════════════════════════════════════════════════════
#
# Note on "averaging the weights": we average the PORTFOLIO weights that the networks
# propose, not the adjustable numbers inside the networks. Two networks can reach the same
# portfolio with completely different inside numbers, so the average of their inside numbers
# means nothing. The average of their portfolios is simply a compromise portfolio.

HARDCODED_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hardcoded_strategy.py")

# The text of hardcoded_strategy.py. The words in capitals between @ signs are replaced by
# the real numbers in write_hardcoded_strategy() below.
HARDCODED_TEMPLATE = r"""'''The yield model as a table: no neural network and no training needed.

THIS FILE WAS WRITTEN BY yield_mlp/yield_mlp.py. Do not change the numbers by hand; run
that file again to make a new table.

What it does: every year it looks at the 5-year and the 30-year US Treasury yield at the end
of last year and looks up the portfolio that belongs to them. The portfolios in the table
are the average of the portfolios proposed by @NUMBER_OF_MODELS@ small neural networks.
The withdrawal is fixed at 4% of the starting capital.

Use it as a strategy of the case:

    python main.py yield_mlp/hardcoded_strategy.py --no-frontier

or copy the tables and the function look_up_portfolio into your own strategies/run.py.
'''

import torch

WITHDRAWAL_RATE = 0.04

# The order of the 7 numbers in every portfolio below.
ASSETS = @ASSETS@

# The portfolio for the first year, when no yield of "last year" is known yet.
FIRST_YEAR_PORTFOLIO = @FIRST_YEAR@

# The yields (in percent) for which the table has an entry. The same steps are used for the
# 5-year yield and for the 30-year yield.
TABLE_YIELDS = @TABLE_YIELDS@

# PORTFOLIOS[i][j] is the portfolio when the 5-year yield is TABLE_YIELDS[i] and the 30-year
# yield is TABLE_YIELDS[j]. Every portfolio is 7 weights in the order of ASSETS. Because of
# rounding they can add up to 0.999 or 1.001; the simulator scales them to exactly 1.
PORTFOLIOS = [
@PORTFOLIOS@
]


def look_up_portfolio(five_year_yield, thirty_year_yield):
    '''The portfolio for the given yields (in percent), for all scenarios at once.

    Yields that fall between two entries of the table get a mix of the portfolios around
    them: a yield of 3.25% gets 75% of the entry for 3% and 25% of the entry for 4%. This is
    done for both yields at the same time, so four entries are mixed. Yields below the
    lowest or above the highest entry get the portfolio of the nearest entry.
    '''
    table = torch.tensor(PORTFOLIOS, dtype=torch.float64)
    lowest, highest = TABLE_YIELDS[0], TABLE_YIELDS[-1]
    step = TABLE_YIELDS[1] - TABLE_YIELDS[0]

    # Where are we in the table? Position 3.25 means: a quarter of the way from entry 3 to entry 4.
    row_position = (five_year_yield.clamp(lowest, highest) - lowest) / step
    column_position = (thirty_year_yield.clamp(lowest, highest) - lowest) / step

    # The entry just below (the last entry has no next one, so stop one before the end) ...
    row = row_position.floor().long().clamp(max=len(TABLE_YIELDS) - 2)
    column = column_position.floor().long().clamp(max=len(TABLE_YIELDS) - 2)
    # ... and how far we are towards the next entry (0 = not at all, 1 = all the way).
    to_next_row = (row_position - row)[:, None]
    to_next_column = (column_position - column)[:, None]

    return ((1 - to_next_row) * (1 - to_next_column) * table[row, column]
            + to_next_row * (1 - to_next_column) * table[row + 1, column]
            + (1 - to_next_row) * to_next_column * table[row, column + 1]
            + to_next_row * to_next_column * table[row + 1, column + 1])


class Strategy:
    '''The strategy in the form that main.py of the case expects.'''

    def fit(self, train, cfg):
        pass  # nothing to learn: the table is fixed

    def act(self, obs):
        if obs.year == 0:
            return torch.tensor(FIRST_YEAR_PORTFOLIO, dtype=torch.float64), WITHDRAWAL_RATE
        end_of_last_year = obs.market_history[-1]
        five_year_yield = end_of_last_year[:, obs.market_names.index("@FIVE_YEAR_NAME@")]
        thirty_year_yield = end_of_last_year[:, obs.market_names.index("@THIRTY_YEAR_NAME@")]
        return look_up_portfolio(five_year_yield, thirty_year_yield), WITHDRAWAL_RATE
"""


def write_hardcoded_strategy(model, number_of_models):
    """Ask the model for its portfolio at every combination of table yields and write the file."""
    def as_text(portfolio):  # 7 weights -> "[0.123, 0.456, ...]"
        return "[" + ", ".join(f"{weight:.3f}" for weight in portfolio.tolist()) + "]"

    blocks = []
    with torch.no_grad():
        for five_year in TABLE_YIELDS:
            lines = [f"    [  # 5-year yield {five_year:g}%"]
            for thirty_year in TABLE_YIELDS:
                portfolio = model.portfolio_for(torch.tensor([[five_year, thirty_year]], dtype=DECIMAL))[0]
                lines.append(f"        {as_text(portfolio)},  # 30-year yield {thirty_year:g}%")
            lines.append("    ],")
            blocks.append("\n".join(lines))
        first_year = as_text(model.first_year_portfolio())

    text = HARDCODED_TEMPLATE
    text = text.replace("@NUMBER_OF_MODELS@", str(number_of_models))
    text = text.replace("@ASSETS@", str(ASSET_NAMES))
    text = text.replace("@FIRST_YEAR@", first_year)
    text = text.replace("@TABLE_YIELDS@", str(TABLE_YIELDS))
    text = text.replace("@PORTFOLIOS@", "\n".join(blocks))
    text = text.replace("@FIVE_YEAR_NAME@", YIELDS[0]).replace("@THIRTY_YEAR_NAME@", YIELDS[1])
    with open(HARDCODED_FILE, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"\nWrote the table to {os.path.relpath(HARDCODED_FILE, PROJECT_FOLDER)}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. RUNNING IT
# ══════════════════════════════════════════════════════════════════════════════

def print_results(title, strategies, scenarios):
    print(f"\n{title} ({scenarios.n_scenarios} scenarios)")
    print(f"  {'':<22}{'score':>8}{'ruined':>9}{'median final wealth':>22}{'worst 10% final wealth':>25}")
    for name, strategy in strategies.items():
        r = results(strategy, scenarios)
        print(f"  {name:<22}{r['score']:>8.2f}{r['ruined']:>9.1%}{r['median final wealth']:>22.2f}"
              f"{r['worst 10% final wealth']:>25.2f}")


def print_what_the_model_does(model):
    """Show the portfolio for a few combinations of the two yields.

    Because the model only looks at two numbers, its whole behaviour fits in a table.
    The full table is in hardcoded_strategy.py; this is a short version to read.
    """
    with torch.no_grad():
        starting = model.first_year_portfolio()
    print("\nWhat the averaged model does")
    print("  first year (no yields known yet):")
    print("    " + ", ".join(f"{name} {weight:.0%}" for name, weight in zip(ASSET_NAMES, starting.tolist())))
    print("\n  later years, by the yields at the end of the year before (weights in %):")
    print(f"    {'5Y yield':>8} {'30Y yield':>9} |" + "".join(f"{name:>17}" for name in ASSET_NAMES))
    for five_year in [1.0, 3.0, 5.0, 7.0]:
        for thirty_year in [2.0, 4.0, 6.0]:
            with torch.no_grad():
                weights = model.portfolio_for(torch.tensor([[five_year, thirty_year]], dtype=DECIMAL))[0]
            print(f"    {five_year:>7.0f}% {thirty_year:>8.0f}% |" + "".join(f"{100 * w:>17.0f}" for w in weights.tolist()))


# The line below means: "only do this when the file is run directly", so that other files
# can borrow YieldModel and train_model from this file without starting a training run.
if __name__ == "__main__":
    models = []
    never_seen = {"model": [], "Bengen": []}  # the results of every set, to average at the end

    print(f"Training one network on each of {NUMBER_OF_DATA_SETS} data sets.")
    print("Results on the scenarios that the network of that set has never seen:")
    print(f"  {'set':>4}{'median final wealth':>22}{'ruined':>9}   |{'Bengen: median':>16}{'ruined':>9}")
    for set_number in range(NUMBER_OF_DATA_SETS):
        # load_fold gives three groups of scenarios that belong to one bootstrapped set:
        #   data.train       500 bootstrapped scenarios: the network learns from these
        #   data.validation  125 bootstrapped scenarios: to pick the best version during training
        #   data.held_out    31 original scenarios this network has never seen in any form
        data = load_fold(set_number)
        model = train_model(data.train, data.validation, seed=set_number)
        models.append(model)

        ours, bengen = results(model, data.held_out), results(BengenRule(), data.held_out)
        never_seen["model"].append(ours)
        never_seen["Bengen"].append(bengen)
        print(f"  {set_number:>4}{ours['median final wealth']:>22.2f}{ours['ruined']:>9.1%}   |"
              f"{bengen['median final wealth']:>16.2f}{bengen['ruined']:>9.1%}")

    def average(name, key):
        return sum(r[key] for r in never_seen[name]) / len(never_seen[name])

    print(f"  {'avg':>4}{average('model', 'median final wealth'):>22.2f}{average('model', 'ruined'):>9.1%}   |"
          f"{average('Bengen', 'median final wealth'):>16.2f}{average('Bengen', 'ruined'):>9.1%}")
    print(f"  average score: single networks {average('model', 'score'):.2f}, Bengen {average('Bengen', 'score'):.2f}")

    # All networks together act as one model.
    averaged_model = AverageOfModels(models)
    print_what_the_model_does(averaged_model)
    write_hardcoded_strategy(averaged_model, len(models))

    # Check that the table does the same as the networks it was made from. We use the
    # scenarios of the last data set. NOTE: the averaged model has seen every pool scenario
    # in some form during training, so these numbers say nothing about how good it is on
    # new scenarios. They only show whether the table copies the networks faithfully.
    import hardcoded_strategy  # the file we just wrote (it is in the same folder as this file)
    print_results("Check: is the table the same as the averaged networks?",
                  {"averaged networks": averaged_model, "table (hardcoded)": hardcoded_strategy.Strategy()},
                  data.train)
