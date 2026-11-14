# The yield model: small neural networks that invest on two bond yields

Two files:

| File | What it is |
|---|---|
| [yield_mlp.py](yield_mlp.py) | Trains the model. Written to be read from top to bottom; every step is explained in comments. |
| [hardcoded_strategy.py](hardcoded_strategy.py) | The trained model as a table of numbers. Written by `yield_mlp.py`. Needs no neural network and no training. |
| [yearly_table.py](yearly_table.py) | The model's **average** portfolio for each of the 30 years. Written by [make_yearly_table.py](make_yearly_table.py). This is not the yield model: it depends only on the year, cannot react to the yields, and ends with a median final wealth of about 2.2 on the pool scenarios against 3.5 for the yield table. |

## What the model does

Every year it looks at two numbers, both known at that moment:

- the yield of a 5-year US Treasury bond at the end of last year
- the yield of a 30-year US Treasury bond at the end of last year

From these it decides how to spread the money over the 7 assets. It uses nothing else. The
withdrawal is fixed at 4% of the starting capital per year, as in the Bengen rule.

One network is trained on each of 20 bootstrapped data sets, and the model invests in the
**average of the 20 portfolios** they propose. A single network depends on which data set it
happened to get; the average does not.

## Run it

From the project folder, after `python -B bootstrap/datasets.py` has created the data sets:

```bash
python -B yield_mlp/yield_mlp.py
```

This takes about 5 minutes. It trains the 20 networks, prints how each does on scenarios it
has never seen, prints a short table of what the averaged model does and writes
`hardcoded_strategy.py`. It does not read the test set.

## The hardcoded table

Because the model only looks at two numbers, everything it does fits in a table: for this
5-year yield and this 30-year yield, hold this portfolio. `hardcoded_strategy.py` contains:

- `FIRST_YEAR_PORTFOLIO`: the portfolio for the first year, when no yield is known yet
- `PORTFOLIOS`: one portfolio for every combination of a 5-year yield and a 30-year yield
  of 0%, 1%, ..., 12% (169 portfolios of 7 weights)
- `look_up_portfolio(...)`: finds the portfolio for any yields; between two entries it mixes
  the neighbouring portfolios
- `Strategy`: the model in the form the case expects, so it runs with `main.py`:

```bash
python main.py yield_mlp/hardcoded_strategy.py --no-frontier
```

To use it in `strategies/run.py`, copy the three tables and `look_up_portfolio`, and call
the function from your own `act` as `Strategy.act` in the file does.

The numbers are the averaged **portfolio** weights. The numbers inside the 20 networks
cannot be averaged: two networks can reach the same portfolio with completely different
inside numbers, so their average means nothing.

## What the averaged model does

A short version of the table (weights in % of the portfolio):

| 5-year yield | 30-year yield | US equity | EM equity | US REIT | Treasury 5Y | Treasury 30Y | TIPS | Cash |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| first year | | 21 | 37 | 18 | 0 | 23 | 0 | 0 |
| 1% | 2% | 34 | 37 | 26 | 0 | 2 | 0 | 0 |
| 3% | 4% | 25 | 39 | 20 | 0 | 15 | 0 | 0 |
| 5% | 4% | 15 | 31 | 14 | 1 | 38 | 1 | 0 |
| 5% | 6% | 11 | 30 | 9 | 1 | 48 | 1 | 0 |
| 7% | 6% | 7 | 23 | 6 | 2 | 59 | 1 | 0 |
| 7% | 2% | 9 | 14 | 8 | 34 | 27 | 6 | 3 |

In words: when yields are low, hold almost only equity and real estate. The higher the
yields, the more goes into 30-year Treasuries, up to more than half. Only when the 5-year
yield is far above the 30-year yield (the last row, which is rare) does it also hold
5-year Treasuries.

## How well it works

Fixed 4% withdrawal. Final wealth is in units of the starting capital.

**The single networks, each on the 31 original scenarios it has never seen** (average over
the 20 data sets; this is a fair check):

| | Score | Ruined | Median final wealth |
|---|---:|---:|---:|
| Bengen 50/50 | 0.44 | 12.7% | 0.6 |
| One yield network | 1.36 | 9.8% | 3.4 |

The network ends with a higher median wealth than Bengen in all 20 data sets (2.7 to 4.6
against 0.4 to 0.9).

**The averaged model and its table, on the 153 pool scenarios** (not a fair check: every
one of these scenarios was used to train some of the networks):

| | Ruined | Median final wealth |
|---|---:|---:|
| Bengen 50/50 | 13.7% | 0.7 |
| Hardcoded table | 11.1% | 3.5 |

The table copies the averaged networks closely: on 500 scenarios the two give a median
final wealth of 2.77 and 2.78 and 6.8% and 6.4% ruined.

## What to keep in mind

- **The averaged model has no fair check yet.** The 20 networks together have seen every
  pool scenario. Earlier, averaging 10 larger networks scored slightly better than one of
  them (see `machine_learning_investing/REPORT.md`), but for this model only the test set
  could tell, and that has not been done.
- **It depends on how this simulator moves interest rates.** The model earns its extra
  return by holding long bonds when yields are high. If the hidden holdout behaves
  differently, that can hurt.
- **Challenge rules.** The case README says a strategy may only learn from the train
  scenarios it gets in `fit`. This table was learned beforehand from pool scenarios, which
  include validation scenarios, and `fit` does nothing. Mention that in the submission.
- **The score is a choice.** The networks are trained to end with a high wealth in the
  typical scenario without running out of money (section 3 of `yield_mlp.py`), not on the
  median final wealth alone.
