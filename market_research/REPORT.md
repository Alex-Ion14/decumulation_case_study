# One-year market forecasts for the decumulation case

## Instrument decision

**Use the five-year Treasury as the fixed-income default. Consider cash when the latest five-year Treasury yield is more than 2 percentage points below the three-month bill yield.** This is a candidate signal for the strategy owner to test in the hidden holdout. It applies to the choice between those two instruments. At the first decision, both histories are empty, so use the five-year default.

Use the last completed year's market observations. The data already expresses yields in percentage units, so subtract the two values and compare with `-2.0`.

```python
market = obs.market_history[-1]  # only when obs.year > 0
five = market[:, obs.market_names.index("US_ZC_5Y")]
bills = market[:, obs.market_names.index("US_TBILL_3M")]
prefer_cash = (five - bills) < -2.0
```

The forecast is that next year's five-year Treasury return minus cash return will tend to track today's yield spread. In validation, when the five-year yield was at least 1.08 points below bills, the Treasury lagged cash by 2.87 points on average and won just 15% of the time. When it was at least 1.26 points above bills, it led by 4.44 points on average and won 84% of the time. These bins use the training set's tercile cutoffs. The stricter 2-point decision hurdle gives room for a potential switch to cash and back.

| Five-year Treasury vs cash | Training group CV MSE improvement | Validation RMSE | Validation direction accuracy | Net annual real return | Validation annual 5th percentile |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant training forecast / always hold 5Y for return comparison | — | 5.60 pp | 37.2% | 1.41% | −6.80% |
| Fitted spread rule | 39.7% | 3.87 pp | 77.8% | 1.71% | −6.19% |
| Raw spread rule | — | 3.88 pp | 79.4% | 1.71% | −6.19% |
| Always hold cash | — | — | — | −0.74% | −3.89% |

Direction accuracy is for which instrument beats the other before costs. The constant forecast uses the training mean for the amount and training base rate for the direction. A direction chosen with hindsight to always say “five-year Treasury wins” would score 62.8% on validation. The fitted forecast reduced validation mean squared error by 52.1%. Its fitted spread slope ranged from 1.09 to 1.11 across the five training folds. The fitted switch threshold is −1.93 pp.

The raw rule chose cash in 9.8% of validation decisions. In those years cash beat the five-year Treasury by 3.47 pp on average before switching costs. The overall improvement after costs was +0.30 pp per year over always holding the five-year Treasury, with gains in years 2–10 (+0.23 pp) and 11–30 (+0.33 pp). For the primary fitted rule, the family-adjusted 95% scenario-bootstrap interval for the annual log-return improvement was +0.15 to +0.51 pp. The raw rule was a simplicity check on the same validation paths after the primary tests and still needs its own confirmation on the hidden holdout.

## Other instruments and signals

- **US equity** is the clearest growth candidate in validation. Its median path CAGR was 3.72%, versus 3.24% for EM equity and 2.79% for US REIT. Its one-year 5th percentile was −23.0%, versus −30.9% and −29.4% respectively. EM and REIT returns had correlations of 0.85 and 0.62 with US equity. In training, EM had the higher median CAGR, so treat the growth ranking as regime-sensitive. The lagged-return and VIX tests provide little support for annual equity or REIT timing. The lagged-return slopes were mildly negative, with 0.5–1.0% validation MSE gains and virtually no direction improvement.
- **US Treasury 30Y** is a secondary research candidate. The starting 30-year yield predicted its spread to the five-year Treasury with 56.0% direction accuracy and 4.4% MSE improvement. Switching to the five-year Treasury when the fitted expected advantage exceeded 2 pp produced a net geometric annual return of 2.08%, versus 1.05% for always holding 30Y and 1.26% for always holding 5Y. The 5th-percentile year was −27.1% for the switch rule versus −6.8% for 5Y. That downside makes the five-year Treasury the more defensible fixed-income default.
- **US TIPS 10Y** is worth considering for inflation exposure. During the 342 validation years with inflation at least 4%, TIPS averaged −1.18% real versus −3.09% for 5Y, −11.03% for 30Y and −3.22% for cash. Its unconditional 5th-percentile real return was −10.04%, versus −6.79% for 5Y. The prior inflation-minus-breakeven signal slightly worsened the TIPS-versus-5Y forecast in validation. The inflation-stress comparison describes what happened in the same year and gives no decision-time trigger.
- **Cash** had the narrowest real return distribution. The starting bill yield forecasts nominal cash returns closely (0.29 pp validation RMSE). Bill yield minus prior inflation is useful for estimating purchasing power, with 77.7% accuracy for a positive versus nonpositive real cash return in the fitted test. Validation cash lost purchasing power in 67.4% of all years.
- **Yield direction** showed modest mean reversion. Starting levels reduced MSE for the next annual change by 2.8–3.0% for five-year, 30-year and 10-year real yields. Direction accuracy was 52–55%. Lagged inflation alone worsened the validation inflation forecast versus a constant training mean.

### Validation asset profiles

All figures are real returns. Stress-year statistics are contemporaneous descriptions.

| Asset | Annual mean | 5th percentile | Median 30-year CAGR | Mean when US equity fell at least 10% |
| --- | ---: | ---: | ---: | ---: |
| US equity | 5.60% | −23.01% | 3.72% | −19.59% |
| EM equity | 6.54% | −30.94% | 3.24% | −21.27% |
| US REIT | 5.57% | −29.38% | 2.79% | −14.20% |
| US Treasury 5Y | 1.44% | −6.79% | 1.11% | 3.02% |
| US Treasury 30Y | 3.65% | −30.15% | 0.80% | 8.69% |
| US TIPS 10Y | 1.55% | −10.04% | 1.32% | 2.42% |
| Cash | −0.73% | −3.87% | −1.53% | −1.17% |

## How the tests limit overfitting

- The case supplies synthetic data for 100 training paths and 80 separate validation paths, each 30 years long. The files have disjoint path IDs. The results describe this generator, and hidden holdout performance is unknown. The observed start-of-year-2 bill yield averaged 2.96% in training and 1.60% in validation.
- The study uses the repository's annual loader. It computes real return as `(1 + nominal_return) / (1 + inflation) - 1`. Features at the end of one year forecast the following year. The first forecast is for year 2. Asset profiles include all 30 years.
- Thirteen one-feature tests, five whole-path training folds, and a 2 pp switching hurdle were set before validation scoring. Fits use one slope and an intercept. The primary baselines are the training mean for return forecasts and the training event frequency for direction probabilities. Training-fold fits and validation fits use only their respective training paths. The reported intervals bootstrap whole validation paths with Bonferroni adjustments within the forecast-error and switching-test families. This preserves within-path dependence. Validation is public and any follow-up comparisons on these same paths are exploratory.
- The instrument check starts in a fixed instrument selected by the sign of the training mean spread. Each full switch costs 1% under the simulator's 0.5% charge on each side. The 2 pp hurdle covers the approximate round-trip cost. The results measure an isolated instrument choice with no withdrawals. The larger strategy can have different turnover, exposure, and ruin outcomes.

## Reproduce

From the repository root, run `python -B market_research/forecast_study.py` with the repository requirements installed. The script prints JSON with forecast and realized returns by signal bin, training-fold slopes, and scenario-bootstrap intervals. The local copy is in [results.json](results.json). The repository ignores JSON files, so teammates can regenerate it from the script.
