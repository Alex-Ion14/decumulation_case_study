# Crash warnings and risk-aware retirement strategies

**Keep the crash alert out of the default strategy.** Use a scheduled equity allocation, adjust spending to the remaining portfolio, and use the yield signals to choose among five-year Treasuries, TIPS, and cash. The training-selected policy starts at 70% US equity and finishes at 30%. A lower-spending policy that takes its largest equity risk in the middle of retirement is the strongest candidate for a separate risk-first check.

## Can we see crashes coming?

The tests forecast two distinct next-year outcomes. An annual crash is a US equity real return of −20% or worse. An intrayear crash is a nominal drawdown of 20% or more measured from the highest month-end level reached since the start of that year, including the opening level. The second measure can capture a crash followed by a recovery. Monthly closes cannot capture the full intramonth drawdown.

| Outcome and warning model | Event frequency | AUC | Crash frequency on alerts | Share of crashes caught |
| --- | ---: | ---: | ---: | ---: |
| Annual crash, high VIX rule | 7.6% | 0.472 | 7.8% | 25.1% |
| Annual crash, annual market and macro | 7.6% | 0.470 | 7.6% | 14.0% |
| Annual crash, annual and prior-year monthly inputs | 7.6% | 0.504 | 9.7% | 21.6% |
| Intrayear crash, high VIX rule | 14.9% | 0.520 | 18.0% | 29.6% |
| Intrayear crash, annual market and macro | 14.9% | 0.525 | 16.2% | 35.3% |
| Intrayear crash, annual and prior-year monthly inputs | 14.9% | 0.583 | 19.7% | 35.3% |

AUC of 0.50 corresponds to chance ranking. The annual inputs included VIX, credit spread, change in unemployment, the yield curve, equity returns, yield changes, and inflation. The monthly extension added recent equity returns, volatility, and drawdowns calculated entirely from the previous year. The current strategy interface exposes annual observations. The monthly extension is a research check and would need a data-interface change to run in the challenge.

For the strongest intrayear warning, four out of five alerts were false alarms and nearly two thirds of crashes went unflagged. Its Brier improvement had a family-adjusted interval that included zero. A full exit from equity to cash on those alerts reduced annual mean real return by 2.26 percentage points after costs. Five-year Treasuries reduced it by 1.42 points. On false alarms, cash trailed equity by 13.0 points on average before trading costs. The annual-data crash model also failed to improve the retirement policy when cutting only 20 percentage points of equity. Both cash and the defensive sleeve versions had 9 ruins in 80 paths, the same as their matched base policy, and both paid out less. Among actual annual crash years, five-year Treasuries averaged +4.05% real and cash averaged −1.06%. This is a description of the crash years, with no use of the realized crash in any decision.

## Strategy and validation results

The training-selected policy takes these steps each year.

1. Reduce US equity linearly from 70% to 30% over 30 years. Invest the remainder in the defensive sleeve.
2. Default the sleeve to five-year Treasuries. Estimate the cash advantage from the three-month bill yield minus the five-year Treasury yield. Estimate the TIPS advantage with a three-input ridge model using the 10-year real yield, 10-year breakeven inflation, and five-year Treasury yield. Choose the largest predicted advantage over 2 percentage points. The first decision uses the five-year Treasury because the observation has no yield history.
3. Estimate the real spending that current wealth could support over the remaining years at a 1% real discount rate. Withdraw 90% of that estimate, clipped to 3–8% of initial capital. Recompute annually. This starts at 3.45% of initial capital.

| Policy | Ruins / 80 | Normalized withdrawals | Case score at alpha 0.25 | Median terminal wealth | Last-decade losses of at least 10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bengen, 50% equity and fixed 4% | 18 | 0.963 | 0.072 | 60.0% | 6.9% |
| Fixed 4%, equity declines 70% → 30% | 18 | 0.952 | 0.069 | 59.1% | 3.4% |
| Training-selected, 70% → 30%, spending factor 0.90 | 8 | 1.339 | 0.260 | 12.7% | 3.5% |
| Middle peak, 40% → 65% → 30%, factor 0.80 | 3 | 1.293 | 0.295 | 27.2% | 4.1% |
| Middle peak, 40% → 65% → 30%, factor 0.85 | 4 | 1.312 | 0.291 | 20.2% | 4.1% |
| Rising, 30% → 70%, factor 0.85 | 3 | 1.296 | 0.296 | 15.8% | 11.5% |

The loss column is the share of last-decade years with a post-withdrawal portfolio return of −10% or worse, among years with a positive post-withdrawal balance. Terminal wealth is a percentage of initial capital. Normalized withdrawals divide discounted payouts by those of a fixed 4% rule that survives all 30 years. The adaptive policy pays out more during the simulation and ends with less median wealth. It pays below 4% in 33.1% of all years. If maintaining a stable withdrawal or a bequest matters, track those outcomes alongside the case score.

The **40% → 65% → 30% middle-peak policy at the 0.80 spending factor** is the candidate I would test if reducing ruin and late risk takes priority. It reaches 65% equity around year 11 and starts spending at 3.07%. The 0.85 version starts at 3.26% and increases average payouts with one additional ruin on validation. In the 26 public paths where US equity lost purchasing power over the first five years, Bengen ran out in 14 paths, the training-selected declining policy in 7, and the 0.80 middle-peak policy in 3. This is a stress breakdown, with the realized first-five-year return used only to summarize the results. It illustrates why taking the most risk at the start of retirement can be costly even if the final allocation is conservative. [Pfau and Kitces on sequence risk and glide paths](https://www.financialplanningassociation.org/article/journal/JAN14-reducing-retirement-risk-rising-equity-glide-path)

## What has been checked

- The crash models use fixed regularization and a training threshold that targets the top 20% of scores. Five training folds keep whole scenarios together. The public validation contains 80 paths, 2,240 eligible annual decisions, 171 annual crashes, and 334 intrayear crashes. Model comparisons bootstrap entire paths.
- Primary retirement selection uses the case's default score with alpha 0.25. It ranks three glide paths and three spending factors by grouped training scores, then selects a defensive overlay on those same training folds. The training scores used to select the policy can be optimistic. The 0.80 and 0.85 spending factors and middle-peak glide were added after reviewing the primary public comparison. Treat their results as exploratory. The test set does not refit models or select policies.
- Replaying the training-selected strategy in public validation improved the official score by 0.188 versus Bengen. The paired 95% bootstrap interval was +0.143 to +0.240. The 80 validation scenarios are synthetic and previously used for research. Hidden holdout performance is unknown.
- All investment and withdrawal policies use the original simulator, including annual rebalancing costs and the 3% minimum withdrawal. The retirement script compares simple policies over a limited grid. The result is a case-study candidate and measures the specified trade-off between withdrawals and ruin.

## Reproduce

Run these from the repository root with the existing dependencies.

```text
python -B market_research/crash_signals.py
python -B market_research/retirement_study.py
python -B market_research/retirement_study.py --test path/to/holdout.parquet
```

Both scripts print JSON. The local results are [crash_results.json](crash_results.json) and [retirement_results.json](retirement_results.json). The repository ignores JSON, so teammates can regenerate them. The default `Strategy` in [retirement_strategy.py](retirement_strategy.py) implements the training-selected declining policy.
