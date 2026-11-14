# Long-horizon market forecasting for the retirement case

**Keep a simple yield-based forecast as the benchmark. Test a bounded three-year stock-versus-Treasury allocation tilt on the hidden holdout.** The case data contains useful information about future bond and cash returns. It gave little reliable information about the long-term direction of equities or REITs. An allocation can still improve if it responds to changes in what the defensive assets are likely to earn.

## What the literature suggests

| Method | How it fits the case |
| --- | --- |
| Yield-curve factors | Diebold and Li forecast the yield curve through its level, slope, and curvature. The case supplies four US zero-coupon maturities, enough for a simplified three-factor model. I fitted the factors with a fixed decay and estimated their dynamics jointly with cash yields, real yields, inflation, growth-asset returns, VIX, and credit spreads. [Diebold and Li](https://www.nber.org/papers/w10048) |
| Direct return forecasts and a constant benchmark | Welch and Goyal found weak out-of-sample equity forecasts relative to a historical mean. Campbell and Thompson found modest gains under economically motivated restrictions. I compared fixed-penalty ridge models with a training-mean forecast and with current yield carry for fixed income. [Welch and Goyal](https://www.nber.org/papers/w10483) · [Campbell and Thompson](https://www.nber.org/papers/w11468) |
| Multi-asset forecasts | Campbell and Viceira model how the risks and returns of stocks, bonds, and cash change with the holding horizon. I compared direct forecasts with iterated forecasts from a regularized vector autoregression and an equal-weight blend. The model is an adaptation of the framework. [Campbell and Viceira](https://www.nber.org/papers/w11119) |
| Forward-rate bond signals | Cochrane and Piazzesi's approach uses a particular set of short-maturity forwards. The case lacks their full set of maturities. The curve model uses observed tenors, and the carry benchmark tests how much they already tell us. [Cochrane and Piazzesi](https://web.stanford.edu/~piazzesi/cp.pdf) |
| Equity valuation and trend | CAPE requires prices and a history of earnings, which the case does not provide. The published time-series momentum strategy uses many futures markets and typically rebalances monthly. I tested a small annual, long-only adaptation that cuts equity after it underperforms bills. [Shiller's data](https://www.econ.yale.edu/~shiller/data.htm) · [Moskowitz, Ooi, and Pedersen](https://fairmodel.econ.yale.edu/ec439/mosk.pdf) |

## Forecast results

Forecasts use the state at the end of the latest completed year. The target is the average annual real log return of each investable asset over the next 1, 3, 5, or 10 years. Inflation uses average annual log inflation. The yield targets are changes to the end of each horizon, compared with an unchanged-yield forecast. Bond forecasts use the case's rolling zero-coupon mechanics with an approximation to within-year carry. Five training folds leave out whole 30-year paths. The final models use 100 training paths and are scored on 80 separate public paths. Bootstrap comparisons keep complete paths together because adjacent multiyear targets overlap. [Boudoukh, Richardson, and Whitelaw on overlapping horizons](https://www.nber.org/papers/w11841)

Public validation MSE improvement versus the training-mean forecast, for the equal-weight blend. A negative number means higher error.

| Asset | 1 year | 3 years | 5 years | 10 years |
| --- | ---: | ---: | ---: | ---: |
| US equity | +1.1% | −0.4% | −1.2% | −3.5% |
| EM equity | +0.5% | −1.6% | −2.6% | −5.2% |
| US REIT | −1.5% | −6.4% | −9.2% | −9.9% |
| US Treasury 5Y | +15.6% | +62.7% | +76.0% | +71.5% |
| US Treasury 30Y | +1.6% | +16.7% | +26.4% | +40.9% |
| US TIPS 10Y | +22.1% | +43.2% | +61.6% | +82.2% |
| Cash | +68.6% | +81.3% | +81.5% | +73.7% |
| Inflation | +2.1% | −2.7% | −1.8% | −4.0% |

**Current yield is the stronger benchmark.** It assumes the latest yields stay flat, uses the training-mean inflation forecast for nominal instruments, and uses the latest real yield for TIPS. Against that benchmark, the blend improved three-year MSE by 0.9% for 5Y Treasuries, 5.8% for TIPS, and 6.7% for 30Y Treasuries, while worsening cash MSE by 17.5%. At five and ten years, the blend continued to improve TIPS and 30Y forecasts by about 10–13%, but carry was better for 5Y Treasuries and cash. These carry comparisons were added after reviewing the first results and are descriptive. The iterated curve forecast on its own performed poorly for the change in five-year yields. Direct ridge modestly improved five- and 30-year yield-change forecasts at horizons up to five years.

The historical-analog test used 30 distinct training paths nearest to the current market state to estimate the probability of a real 20% loss in a 50/50 stock and 5Y portfolio over the following three years. The event occurred in 3.15% of validation decisions. The analog probability averaged 5.13%, and its Brier score was 0.03137 versus 0.03060 for the training base rate. The analogs also failed to improve retirement outcomes when used for a small equity cut.

## Portfolio impact

The tilt uses the forecasted three-year US equity return minus the forecasted three-year five-year Treasury return. It compares that spread with its training average. If the difference exceeds two percentage points of annualized log return in either direction, it changes the scheduled equity weight by ten percentage points in the corresponding direction. The existing policy sets the composition of the rest of the portfolio and the annual withdrawal. The tilt needs a completed year of history and at least three years remaining. It is a decision rule for the synthetic case.

| Policy | 70% → 30% base, ruins / 80 | Normalized withdrawals | 40% → 65% → 30% base, ruins / 80 | Normalized withdrawals |
| --- | ---: | ---: | ---: | ---: |
| Base policy | 8 | 1.339 | 3 | 1.293 |
| Three-year stock-versus-Treasury tilt | 5 | 1.366 | 2 | 1.319 |
| Treasury forecast only, equity at its training mean | 6 | 1.361 | 3 | 1.314 |
| Treasury carry only, equity at its training mean | 6 | 1.360 | 3 | 1.313 |
| Equity forecast only, Treasury at its training mean | 8 | 1.342 | 3 | 1.295 |
| Always cut equity by 10 points after the first year | 6 | 1.338 | 2 | 1.289 |
| Reallocate within the defensive sleeve using 3Y forecasts | 9 | 1.341 | 3 | 1.294 |
| Cut equity on three-year downside analogs | 8 | 1.339 | 3 | 1.292 |
| Cut equity after US stocks trail cash in the prior year | 9 | 1.333 | 6 | 1.286 |

The combined tilt had positive score changes on held-out training folds for both policies, around +0.003 at alpha 0.25, without changing ruin counts there. Public validation gains were larger and include only three fewer ruins for the declining strategy and one fewer for the middle-peak strategy. The leg-by-leg checks reuse public validation and show that Treasury opportunities explain most of the withdrawal improvement. The case's fixed 4% Bengen policy had 18 ruins and 0.963 normalized withdrawals on the same paths.

## What to carry forward

- **Test on the hidden holdout** with the existing policy, the three-year tilt, and the carry-only version fixed before scoring. Compare normalized withdrawals, ruin, and terminal wealth. The runnable strategy implements the middle-peak version with the combined tilt, which scored 0.311 at alpha 0.25 on the public paths. Those public paths informed earlier research and provide an exploratory comparison.
- **Avoid long-horizon stock, EM, and REIT timing with these inputs.** The multiyear forecasts fell short of a constant mean, and the equity-only tilt barely changed results. The annual momentum adaptation reduced withdrawals and increased ruin on both policies. The result is specific to the annual case implementation.
- **Avoid cash exits based on analog crash alerts and avoid replacing the short-horizon defensive rule with three-year rankings.** Both choices failed to improve the withdrawal–ruin trade-off consistently. Also avoid inserting real-world CAPE values into synthetic scenario paths. An external valuation test needs a separate historical dataset with the inputs aligned as of the decision date.
- **Keep long-horizon forecasts in perspective.** Several horizons share most of their realized returns, and a 10-year forecast is an average over many outcomes. Its accuracy says little about the timing of a crash or how much a retiree can safely withdraw next year.

## Reproduce

Run `python -B market_research/long_horizon_study.py` from the repository root. To score a separate holdout, add `--test path/to/holdout.parquet`. The script prints the forecast, downside, and portfolio results. The local [long_horizon_results.json](long_horizon_results.json) is ignored by the repository and can be regenerated. The [long_horizon_strategy.py](long_horizon_strategy.py) entry point uses the case's normal strategy interface.
