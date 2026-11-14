# Block bootstrap of the train scenarios

Makes new 30-year scenarios out of the 100 train scenarios, so a strategy that is trained
with many gradient steps (for example a neural network) sees different scenarios in every
step and cannot simply memorise the original 100.

## How it works

Think of every scenario as a diary of 30 pages, one page per year. A page holds everything
that happened that year: the returns of the 7 assets, inflation and the 50 market variables.
A new scenario is made by copying a *block* of consecutive pages from a randomly chosen diary,
then a block from another one, and so on until there are 30 pages.

- A page is always copied whole, so the assets keep moving together as they do in the data.
- Pages are copied several years in a row, so patterns between years (interest rates that stay
  low for years, stocks that recover after a bad period) survive inside a block.
- The first block of every new scenario is cut short at random, so the places where two blocks
  meet do not fall in the same years in every scenario.

## Use

```python
from bootstrap import BlockBootstrap, split_scenarios

fit_part, check_part = split_scenarios(train, holdout_fraction=0.2, seed=0)
sampler = BlockBootstrap(fit_part, block_years=5, seed=0)

for step in range(STEPS):
    batch = sampler.sample(256)            # new scenarios in every step
    out = simulate(self, batch, cfg)       # ... the usual training step
evaluate(self, check_part, cfg)            # scenarios that were never cut up
```

`sample` returns the same `Scenarios` object as `load_scenarios`, so it works with `simulate`,
`evaluate` and `subset`. Run `python -B bootstrap/demo.py` from the repository root to compare
bootstrapped scenarios with the original ones.

## Ready-made datasets for the neural network

`python -B bootstrap/datasets.py` merges the 100 train and 80 validation scenarios and writes:

| File | What it is |
|------|-----------|
| `data/test_set.parquet` | **The test set: do not use it for training or for choosing a model.** 15 random train scenarios and 12 random validation scenarios, in the original monthly format. |
| `data/pool.parquet` | The other 153 scenarios (85 from train, 68 from validation), in the original monthly format. |
| `data/split_manifest.csv` | For each of the 180 scenarios: which file it came from and whether it is in the test set or the pool. |
| `data/bootstrap_sets/set_00.pt` ... `set_49.pt` | 50 sets, each with 500 bootstrapped train scenarios and 125 bootstrapped validation scenarios (blocks of 5 years). Not in git (208 MB); the script recreates exactly the same files. |

```python
from bootstrap.datasets import load_set

train, validation = load_set(0)   # set 0 to 49; two Scenarios objects, as from load_scenarios
```

For every set the 153 pool scenarios are first split into 122 for training and 31 for
validation (a different split for every set, with scenarios of both original files on each
side), and only then bootstrapped. So within a set, no original scenario is behind both the
train and the validation scenarios. Across sets this does not hold: a scenario that is used
for validation in one set is used for training in most others.

The test scenarios are chosen once. Running the script again reuses `split_manifest.csv` and
does not read or rewrite the test set; the sizes and the block length at the top of
`datasets.py` can be changed freely.

Two warnings:

- **The bootstrap softens the bad scenarios.** Some of the validation scenarios have poor
  equity returns for the whole 30 years. Gluing blocks of different scenarios together mixes
  these with better periods. The 4% rule is ruined in 13.7% of the 153 pool scenarios and in
  11.2% of the bootstrapped train scenarios; the worst 5% of 30-year real equity returns is
  -1.4% per year in the pool and -0.2% in the bootstrapped scenarios. Longer blocks
  (`BLOCK_YEARS`) soften less.
- **Challenge rules.** The case README says that `fit` may only use the train scenarios it
  receives. A network trained on these sets has also seen validation scenarios, so the scores
  that `main.py` prints on `data/validation.parquet` are no longer a fair check: use
  `data/test_set.parquet` for that, and mention the approach in the submission.

## Choosing `block_years`

Use 5 to 10. The table shows the ruin probability of three fixed strategies on the original
train scenarios and on 20,000 bootstrapped ones (output of `demo.py`):

| | 50/50 equity/5Y, 4% | 50/50 equity/5Y, 5% | 100% equity, 4% | 100% equity, 5% | 100% cash, 4% | 100% cash, 5% |
|---|---|---|---|---|---|---|
| original (100 scenarios) | 6.0% | 33.0% | 9.0% | 19.0% | 66.0% | 87.0% |
| uncertainty of the original | ±2.4% | ±4.7% | ±2.9% | ±3.9% | ±4.7% | ±3.4% |
| blocks of 1 year | 8.3% | 32.6% | 11.8% | 23.2% | 85.3% | 100.0% |
| blocks of 3 years | 6.6% | 29.9% | 10.1% | 20.2% | 75.8% | 99.6% |
| blocks of 5 years | 6.0% | 29.6% | 8.9% | 19.7% | 71.8% | 98.4% |
| blocks of 10 years | 6.0% | 29.2% | 8.9% | 18.6% | 67.5% | 95.6% |

Single years (`block_years=1`) make cash look far riskier than it is and are best avoided.
From 5 years on, the mixed portfolio and equity are within the uncertainty of the original 100
scenarios. Cash stays too risky even with blocks of 10 years (at a 5% withdrawal: 95.6%
against 87.0%), because its return moves very slowly; keep that in mind for cash-heavy
strategies.

## Limits

- **Nothing new is invented.** Every year is a copy of a train year. The validation scenarios
  have clearly lower equity and cash returns than the train scenarios; a bootstrap of the
  train scenarios does not prepare a strategy for that.
- **Seams.** Where two blocks meet, the market variables jump: the yield at the end of one
  block says nothing about the bond return at the start of the next. If a strategy uses
  `market_history`, ask for the seam positions with `sample(n, return_seams=True)`.
- **Split first, bootstrap second.** Use `split_scenarios` before creating the sampler,
  otherwise years of the same scenario are in both the training and the check data.
- **Long-term patterns are only partly kept.** Over 30 years, equity in the train scenarios
  is less risky than the bootstrap suggests (bad decades tend to be followed by good ones, and
  blocks of 5 to 10 years are too short to capture that). The bootstrapped scenarios are
  therefore somewhat pessimistic about equity.
