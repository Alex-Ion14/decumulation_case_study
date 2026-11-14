# Machine learning for retirement investment

## Bayesian GRU for portfolio allocation

The variational GRU learns a Gaussian mean and standard deviation for each entry of the input-to-hidden and hidden-to-hidden weight matrices. Biases, allocation logits, and the allocation head are point estimates. Each training step samples a pair of matrices and reuses them across the entire sequence. The loss is negative portfolio utility plus 0.02 times the average KL divergence per weight from a zero-mean Gaussian prior with standard deviation 0.25. The initial posterior standard deviation is 0.03. This is generalized Bayesian inference with portfolio utility as the learning objective. [Bayesian recurrent networks](https://arxiv.org/abs/1704.02798) · [Generalized Bayesian updating](https://doi.org/10.1111/rssb.12158)

The policy averages 32 posterior allocation draws to choose weights. The reported 90% intervals show how the allocation varies across sampled recurrent weights for the same market history. They quantify uncertainty under the fitted model about the allocation and exclude future market-return risk. The first-year allocation uses the learned unconditional weights.

```powershell
python -B -m machine_learning_investing.bootstrap_study --bayesian-gru --jobs 4 --holdout
```

This compares the Bayesian GRU with the deterministic GRU, fitted static mix, and Bengen using identical splits, simulator settings, and withdrawal schedules. It writes `bayesian_gru_results.json`.

| Policy | Average split median wealth | Mean wealth | Ruin rate | Wealth and spending score |
| --- | ---: | ---: | ---: | ---: |
| Bayesian GRU | 2.455× | 4.220× | 10.48% | 1.184 |
| Deterministic GRU | 2.377× | 4.195× | 11.12% | 1.168 |
| Fitted static mix | 1.987× | 3.241× | 12.66% | 1.022 |
| Bengen 50/50 | 0.699× | 0.893× | 11.30% | 0.513 |

Across the 50 splits, the Bayesian GRU beat the deterministic GRU on the wealth and spending score in 34 splits and on median terminal wealth in 28. Each year after the first, the mean width of the central 90% allocation interval was **6.58 percentage points** for total growth exposure and **7.26 points** for emerging-market equity. The portfolio gains versus the deterministic GRU were modest across bootstrap splits. The comparison combines the effects of weight sampling, the KL penalty, and posterior averaging.

On the 27 reserved intact paths, the Bayesian GRU ended with **4.12× median and 5.40× mean** wealth, with **2 ruins**. The deterministic GRU ended with **2.67× median and 4.93× mean**, with **4 ruins**. Bengen had **3 ruins**. This is one fixed model specification and training seed per split. The reserved sample is small, and the challenge's hidden holdout remains untested.

## Block bootstrap results

We refitted all ten policies on each of the branch's 50 bootstrap splits using `bootstrap.datasets.load_set`. Each split has 500 training and 125 validation paths. Every policy requests the same 4% real withdrawal. The table averages each metric across splits, including each split's median terminal wealth. Wealth is in multiples of starting capital.

| Policy | Average median wealth | Mean wealth | Ruin rate | Wealth and spending score | Case score at α = 0.25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU | 2.377× | 4.20× | 11.12% | 1.168 | 0.160 |
| Feed-forward neural net | 2.379× | 4.10× | 10.90% | 1.166 | 0.162 |
| 100% diversified growth | 2.068× | 3.84× | 15.39% | 1.038 | 0.125 |
| Fitted static mix | 1.987× | 3.24× | 12.66% | 1.022 | 0.148 |
| 100% US equity | 1.664× | 3.27× | 17.73% | 0.915 | 0.106 |
| Ridge TIPS rule | 1.308× | 1.68× | 7.38% | 0.808 | 0.192 |
| Bayesian TIPS mean | 1.311× | 1.68× | 7.39% | 0.808 | 0.192 |
| Bayesian TIPS confidence filter | 1.258× | 1.62× | 7.76% | 0.788 | 0.189 |
| 60% diversified growth | 1.068× | 1.38× | 9.70% | 0.696 | 0.174 |
| Bengen 50/50 | 0.699× | 0.89× | 11.30% | 0.513 | 0.162 |

The GRU had the highest mean terminal wealth and led the wealth and spending score in 27 of 50 splits. The feed-forward model led that score in 21 splits, and the 100% diversified growth policy led in two. Both neural models beat Bengen's median terminal wealth in 50 of 50 splits. Against the fitted static mix, they did so in 42 of 50 and 45 of 50 splits, respectively.

The ridge TIPS rule had the lowest average ruin rate and the highest case score by a small margin over the Bayesian mean rule. The Bayesian forecast's 90% predictive intervals covered 90.2% of realized spreads on average. The Brier score was 0.203 versus 0.239 for the training event rate. Adding the confidence filter lowered terminal wealth and raised ruin relative to the mean rule.

### Reserved intact test

After selection by average validation wealth and spending score, we drew 500 bootstrap paths from the full training pool and refitted the GRU and static benchmark. We compared them with Bengen on the branch's 27 reserved paths.

| Policy | Median wealth | Mean wealth | Ruins / 27 | Normalized withdrawals |
| --- | ---: | ---: | ---: | ---: |
| GRU | 2.67× | 4.93× | 4 | 95.8% |
| Fitted static mix | 3.22× | 4.32× | 4 | 96.4% |
| Bengen 50/50 | 0.83× | 0.93× | 3 | 97.4% |

The bootstrap source audit verified that each split has 122 training and 31 validation source scenarios with no overlap within the split. The reserved test IDs are outside the training pool. Source scenarios recur across splits, so split win counts describe consistency on this dataset. Five-year blocks preserve short sequences and can break links between the previous year's signals and the next year's returns at block boundaries. The reserved test paths come from the original files used in earlier research. The challenge's hidden holdout remains untested. [Bootstrap design](../bootstrap/README.md)

### Reproduce the bootstrap study

```powershell
python -B bootstrap/datasets.py
python -B -m machine_learning_investing.bootstrap_study --jobs 4 --holdout
```

Use `--sets 0` for a single split or `--out` to keep a separate results file. The default run compares all 50 splits and writes `bootstrap_results.json` with each split's results. `--holdout` also scores the selected policy, the GRU, and benchmarks on the reserved intact paths after selection.

## Original train and validation comparison

The **GRU** finished with the highest median and mean real wealth on the 80 public validation paths. It ended at **1.34×** starting capital at the median versus **0.61×** for Bengen. It also ran out of money in **23 of 80** paths versus **18 of 80** for Bengen. With the same ruin count as Bengen, the simpler ridge TIPS rule finished at **1.00×** at the median.

## Portfolio results

Every policy requests 4% of starting capital at the beginning of each year in real terms. In nominal terms, the first withdrawal is 4% of initial wealth, and subsequent withdrawals increase or decrease with cumulative inflation. The shared simulator handles inflation, trading costs, and shortfalls. Terminal wealth includes depleted portfolios at zero. Normalized withdrawals are discounted spending as a fraction of the full 30-year schedule.

| Allocation policy | Median terminal wealth | Mean terminal wealth | Ruins / 80 | Normalized withdrawals |
| --- | ---: | ---: | ---: | ---: |
| Bengen 50/50 | 0.61× | 0.91× | 18 | 96.3% |
| 60% diversified growth and 40% 5Y Treasuries | 0.63× | 1.31× | 20 | 95.9% |
| 100% diversified growth | 0.91× | 4.29× | 28 | 91.6% |
| 100% US equity | 0.71× | 4.02× | 25 | 90.9% |
| Fitted static allocation | 0.90× | 3.53× | 27 | 92.8% |
| Ridge TIPS rule | 1.00× | 1.67× | 18 | 96.2% |
| Feed-forward neural allocation | 1.24× | 4.19× | 24 | 92.7% |
| **GRU neural allocation** | **1.34×** | **4.92×** | **23** | **93.4%** |
| Bayesian-style TIPS forecast, mean rule | 1.00× | 1.68× | 19 | 96.2% |
| Bayesian-style TIPS forecast, confidence rule | 0.95× | 1.65× | 20 | 96.0% |

The GRU won the four-fold comparison on the 100 training paths. Folds held out complete paths. The ranking maximizes the mean of `log(1 + real terminal wealth) − 3 × (1 − normalized withdrawals)`. The log limits the influence of outsized winners, and the factor of three assigns a cost to missed spending. This factor is a modeling choice. The official evaluator at alpha 0.25 scored the GRU at 0.018 and Bengen at 0.072 because its objective favors withdrawals and survival. This study uses the conventional median, which averages the middle two paths for an even-sized sample. The shared evaluator uses the lower middle value.

### How the allocation reacts

The two neural policies read up to four completed years of real asset returns, inflation, US yields, credit spread and VIX. They produce long-only weights for all seven assets and learn from the same net-of-cost simulator used to score them. At the first decision they use a trained unconditional allocation. Growth in the benchmarks means 50% US equity, 30% emerging-market equity and 20% US REITs within the growth allocation. The three TIPS rules keep 60% in that mix and apply cash and TIPS signals to the 40% defensive sleeve, which defaults to 5Y Treasuries. [Earlier market research](../market_research/ML_REPORT.md)

The GRU averaged **85% in growth assets** after the first year on validation, including **47% EM equity** and **30% US REITs**. The 10th-percentile terminal wealth was zero for every candidate. Among the 26 validation paths where US stocks lost purchasing power over the first five years, the GRU had 11 ruins versus 14 for Bengen. At double the default trading cost, the GRU still led on median terminal wealth at 1.27×, with 25 ruins.

The paired 95% path-bootstrap interval for the GRU's median wealth difference versus Bengen was **−0.003× to +2.26×**. Against the fitted static strategy it was **−0.23× to +1.26×**. When rerun with three alternate seeds, the GRU median ranged from **0.68× to 1.11×**, with **26–27** ruins. Averaging target weights across those three seeds and the original seed gave **1.20×** median wealth and **22** ruins. The seed and ensemble checks are exploratory and leave the original model as the evaluator entry point.

## Bayesian-style forecast

The probabilistic GRU predicts next year's real TIPS return minus the real 5Y Treasury return. It uses prior-year sequences of 10Y real yield, 10Y breakeven inflation and 5Y nominal yield. Monte Carlo dropout on the GRU's readout uses 128 masks to approximate model uncertainty. Each draw also predicts a conditional spread volatility. A mixture of those forecasts gives a predictive distribution for the realized spread. [Gal and Ghahramani on Monte Carlo dropout](https://proceedings.mlr.press/v48/gal16.html)

- The nominal 90% predictive interval covered **88.4%** of 2,320 annual decisions. Average standard deviation across model means was **0.55 percentage points**, compared with **4.30 points** for the full predictive distribution.
- For TIPS beating the 5Y Treasury by more than two percentage points, the Brier score was **0.183** versus **0.234** from the training event rate. In the 60–80% forecast bin, the event happened **64.9%** of the time against a **69.5%** mean forecast.
- Requiring at least 60% predicted probability of clearing the switching hurdle reduced median wealth from **1.00× to 0.95×** and increased ruins from **19 to 20**. These probabilities describe the next year's bond-return spread.

The case data is synthetic, and prior market research used the same validation file. The hidden holdout remains untested.

## Reproduce

Run from the repository root with the existing dependencies.

```powershell
python -m machine_learning_investing.study --plot machine_learning_investing/comparison.png
python main.py machine_learning_investing/strategy.py --no-frontier
```

The study generates [results.json](results.json) with portfolio summary statistics, cost and seed checks, bootstrap intervals and forecast diagnostics. `--test path/to/holdout.parquet` evaluates the same comparison on a separate file. The study accepts `--out` to keep separate result files. The shared evaluator writes its usual `frontier.png` in the working directory.
