# Machine learning for the investment decision

**A small neural network that looks at interest rates beats every fixed portfolio we tried.**
With a fixed withdrawal of 4% (the Bengen assumption), it ends with a median wealth of about
3.4 times the initial capital and runs out of money in 9.5% of the scenarios it has never seen.
The best fixed mix reaches 2.3 times and 11.9%; the Bengen 50/50 portfolio 0.6 times and 12.7%.
Memory (GRU, LSTM) and Bayesian weights do not improve on the plain network.

These numbers come from the 153 pool scenarios only. **The test set has not been used.**

## What was tested

Every model chooses the weights of the 7 assets at the start of each year. The withdrawal is
fixed at 4% of the initial capital, in real terms. One number summarises a result:

    score = average of log(1 + final wealth)  -  ruin probability

It rewards a large final wealth in the typical scenario and punishes ruin twice (a ruined
scenario ends with nothing and also adds to the ruin probability). This score is a choice;
the tables also show ruin and final wealth separately.

Each model was trained on the 500 bootstrapped train scenarios of a set, stopped at its best
score on the set's 125 bootstrapped validation scenarios, and then scored on the **31
original scenarios behind that validation part**, which the model never saw in any form.
This was repeated for the first 20 sets.

| Model | What it can react to |
|---|---|
| Bengen 50/50 | nothing: 50% US equity, 50% 5-year Treasuries |
| Glide path 70->30 | the year: US equity from 70% down to 30%, the rest 5-year Treasuries |
| Static (learned mix) | nothing: one mix of the 7 assets, learned from the data |
| MLP, wealth and time | the year, the wealth and how it compares with the remaining withdrawals |
| MLP, with market | the same, plus interest rates, spreads, VIX, unemployment, last year's returns and inflation |
| GRU / LSTM, with market | the same inputs, plus a memory of all earlier years |
| Bayesian MLP, with market | the same inputs; every network weight has an uncertainty |

## Results on scenarios the models never saw

Averages over 20 folds. Final wealth is in units of the initial capital.

| Model | Score | Ruin probability | Median final wealth | 10% worst final wealth | Sold per year | Better than static in |
|---|---:|---:|---:|---:|---:|---:|
| Bengen 50/50 | 0.44 | 12.7% | 0.6 | 0.02 | 4% | 0 of 20 folds |
| Glide path 70->30 | 0.48 | 12.9% | 0.7 | 0.02 | 4% | 0 of 20 |
| Static (learned mix) | 1.05 | 11.9% | 2.3 | 0.06 | 6% | — |
| MLP, wealth and time | 1.06 | 12.3% | 2.2 | 0.05 | 5% | 11 of 20 |
| **MLP, with market** | **1.36** | **9.5%** | **3.4** | **0.19** | 20% | 20 of 20 |
| GRU, with market | 1.36 | 9.2% | 3.3 | 0.18 | 17% | 20 of 20 |
| LSTM, with market | 1.34 | 9.2% | 3.2 | 0.20 | 15% | 20 of 20 |
| Bayesian MLP, with market | 1.36 | 9.5% | 3.5 | 0.18 | 19% | 20 of 20 |

"Sold per year" is the share of the portfolio that is traded away each year; its cost (0.5%
of the amount traded) is included in all results. See `results/comparison_models.png`.

What this says:

- **More growth assets explain the first jump.** The learned static mix holds about 75% in
  equity and real estate instead of 50%. That multiplies the
  median final wealth by more than three, but hardly changes ruin.
- **Market information explains the second jump.** The market-aware network beats the
  static mix in all 20 folds, by 0.31 score points on average (0.02 in the worst fold, 0.53
  in the best).
- **Wealth and time alone add nothing** over a fixed mix (+0.01).
- **The four market-aware networks are equal.** Their average scores differ by 0.00 to 0.02,
  while the difference in a single fold swings by about 0.08 to 0.10. Memory does not help,
  so the plain MLP is the one to keep: it is the simplest.
- **Ruin stays close to 1 in 10.** With a fixed 4% withdrawal the investment side can only do
  so much; lowering ruin further is mainly a job for the withdrawal rule.

### Not every scenario benefits

The pool mixes scenarios from the original train file (good markets) and the original
validation file (worse markets). The market-aware network helps a lot in the bad ones and
costs a little in the good ones:

| Held-out scenarios from | Ruin, static mix | Ruin, MLP with market | Median final wealth, static | Median final wealth, MLP |
|---|---:|---:|---:|---:|
| original train file | 1.8% | 4.7% | 2.8 | 4.3 |
| original validation file | 24.3% | 15.4% | 1.1 | 1.8 |

## What the network does with the market (market analysis)

`run_market_signals.py` shuffles one input at a time between scenarios, so it no longer means
anything, and measures how much the held-out score drops (8 folds).

| Input shuffled | Drop in score |
|---|---:|
| all interest rates and spreads together | 0.55 |
| 30-year yield | 0.14 |
| 5-year yield | 0.11 |
| 5-year yield minus T-bill yield | 0.08 |
| VIX, credit spread and unemployment together | 0.06 |
| last year's returns and inflation together | 0.04 |
| wealth and time together | 0.01 |
| US equity trend over 3 and 5 years | 0.00 |

- **The network is an interest-rate strategy.** Without the rates it scores 0.82 on these 8
  folds, below the static mix (1.09): it then makes its bond bets blindly.
- **Main rule it found: buy long bonds when yields are high.** When the 30-year yield goes
  from low to high (one standard deviation either side of its average), the network moves 18
  percentage points into 30-year Treasuries, taken from the three growth assets. In
  this simulator yields drift back to their average, so a high yield is followed by falling
  yields and rising bond prices.
- **A wide credit spread works the same way:** 12 points out of emerging-market equity, into
  30-year and 5-year Treasuries.
- **It does not try to time equities from past equity returns.** This matches
  `market_research/`: bonds and cash are predictable from yields, equities are not.

On average the network holds about 41% emerging-market equity, 22% 30-year Treasuries, 16% US
real estate, 14% US equity and little else (`results/allocation_by_year.png`). The average
hardly changes over the 30 years; the changes are between scenarios, driven by the rates.
`results/allocation_all_models.png` shows the same picture for every model in the table above.

## How confident is the model?

Two ways to get a confidence were tried (`run_confidence.py`, 8 folds):

- **Deep ensemble:** 10 networks, each trained on a different random half of the train
  scenarios and from a different random start.
- **Bayesian MLP:** one network that learns a range for each of its weights; 30 networks are
  drawn from those ranges.

In both cases the portfolio is the average of the proposals. The confidence is the
**agreement**: the share of the portfolio that a network has in common with the average
portfolio. 100% means all networks propose the same portfolio.

| | Deep ensemble | Bayesian MLP |
|---|---:|---:|
| Average agreement | 85% | 82% |
| 1 in 10 decisions is below | 76% | 74% |
| Agreement in year 1 (no market information yet) | 89% | 86% |
| Agreement in scenarios that end in ruin / that survive | 83% / 85% | 79% / 82% |
| Share of decisions in ruined scenarios, least-sure fifth of decisions | 12.6% | 14.3% |
| Same, other four fifths | 5% to 7% | 3% to 7% |

- **The networks agree on most of the portfolio, and disagree on the bond bets.** Agreement
  is lowest when the average portfolio holds more Treasuries, TIPS and cash, so when the model
  moves away from its usual equity-heavy mix.
- **Low confidence is a warning sign, a modest one.** Decisions in the least-sure fifth are
  about twice as often part of a scenario that ends in ruin. The difference between ruined
  and surviving scenarios is 2 to 3 percentage points of agreement, with only about 20
  ruined scenarios in this experiment.
- **The two methods partly agree with each other** (correlation 0.53 per decision).
- **Averaging helps a little.** The deep ensemble scores 1.40 against 1.37 for one of its
  own members on the same folds (better in 5 of 8), with ruin 8.5% against 10.1%.
- **The level of the Bayesian confidence is partly a choice.** With the textbook strength of
  the prior, the Bayesian network learns nothing: every weight keeps the prior's uncertainty
  and the drawn networks propose random portfolios. A much weaker prior was chosen on the
  validation scenarios. Where the Bayesian network is more or less sure is informative; the
  82% itself is not. The deep ensemble has no such knob and is the one to trust more.

See `results/confidence_by_year.png`.

## What to be careful about

- **Not checked on the test set.** `data/test_set.parquet` was not read. All numbers are from
  held-out scenarios of the pool, and training stopped at the best score on bootstrapped
  scenarios made from those same held-out scenarios, so they are slightly optimistic.
- **The folds overlap.** The 20 folds share scenarios: there are 152 different held-out
  scenarios in total, each used in several folds. "20 of 20 folds" is strong, but it is not
  20 independent experiments.
- **The gain depends on how this simulator moves interest rates.** The network profits from
  yields that return to their average. If the hidden holdout behaves differently, the bond
  bets can hurt: without rate information the network is worse than a fixed mix.
- **The portfolio leans on emerging-market equity.** It has the highest return in the
  original train file, but `market_research/REPORT.md` found US equity ahead in the
  validation file. This ranking is not stable.
- **A neural network is hard to explain,** and the challenge asks for an interpretable
  strategy. The main behaviour can be written as a rule (a growth-heavy mix, shifted into
  30-year Treasuries when long yields are high); that rule has not been tested here.
- **Challenge rules.** The case README says `fit` may only use the train scenarios it
  receives. These models were trained on sets that include validation scenarios.
- **Fixed 4% withdrawal.** Everything here assumes it. A different withdrawal rule changes
  the wealth path and probably the best portfolio; the models should be retrained together
  with the withdrawal team's rule.

## Files and how to run

Run from the repository root, after `python -B bootstrap/datasets.py` has created the sets.

| File | What it does | Time on 8 processors |
|---|---|---|
| `data.py` | loads a set with its train, validation and original held-out scenarios | |
| `features.py` | the inputs of the networks | |
| `policies.py` | the models | |
| `training.py` | the score, the training loop and the metrics | |
| `run_comparison.py` | experiment 1: all models on 20 folds | 5 minutes |
| `run_confidence.py` | experiment 2: deep ensemble and Bayesian confidence on 8 folds | 5 minutes |
| `run_market_signals.py` | experiment 3: which inputs matter and what they change | a few minutes (not timed) |
| `plots.py` | the four figures in `results/` | seconds |

The tables behind this report are the CSV files in `results/`. The figures are not in git
(the repository ignores PNG files); `plots.py` recreates them from the CSV files.
