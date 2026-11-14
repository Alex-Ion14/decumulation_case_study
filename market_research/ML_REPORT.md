# Machine learning for annual market decisions

The clearest new candidate is **switching between the 10-year TIPS and the five-year Treasury**. A regularized model also improved forecasts of cash purchasing power. The tests give little reason to add ML to the existing five-year Treasury versus cash rule or to time equities and REITs.

## Forecast results

All errors are on the 80 public validation scenarios, using decision years 3–30. Each row compares a model with the listed simple forecast on the same years. RMSE is in percentage points. The Brier score measures squared probability error, with lower values better.

| Forecast | Simple comparison | Ridge inputs shown | Simple error | Ridge error | Validation squared error reduction |
| --- | --- | --- | ---: | ---: | ---: |
| TIPS return minus 5Y Treasury return | Prior inflation minus breakeven | US market + macro | 5.79 pp | 4.61 pp | 36.7% |
| Real cash return | Bill yield and prior inflation combined | US market | 1.86 pp | 1.45 pp | 39.4% |
| 5Y Treasury return minus cash return | Raw 5Y-minus-bill yield spread | US market | 3.89 pp | 3.85 pp | 2.0% |
| 30Y Treasury return minus 5Y Treasury return | Starting 30Y yield | US market + macro + global | 20.00 pp | 20.28 pp | −2.9% |
| US equity return minus cash return | Prior US equity return | US market + macro + global | 18.99 pp | 18.93 pp | 0.6% |
| EM equity return minus US equity return | Training mean | US market + macro + global | 13.07 pp | 13.05 pp | 0.2% |
| US REIT return minus US equity return | Training mean | US market + macro + global | 18.76 pp | 18.75 pp | 0.1% |
| Inflation | Training mean | US market | 1.52 pp | 1.49 pp | 4.3% |
| Inflation reaches 4% | Training event frequency | US market + macro + global | 0.1247 Brier | 0.1221 Brier | 2.1% |
| US equity loses at least 10% in real terms | Training event frequency | US market + macro | 0.1652 Brier | 0.1648 Brier | 0.2% |

The fitted 5Y-versus-cash forecast was marginally more accurate than the raw spread, but the cost-adjusted improvement was only +0.034 pp per year. Its family-adjusted interval for log-return improvement includes zero. Keep the raw spread rule as the benchmark. The real cash model correctly classified gains and losses in purchasing power 85.0% of the time versus 77.5% for the simpler forecast. This target can inform cash use but does not by itself rank cash against a bond.

## TIPS versus five-year Treasury

The US market and macro model correctly predicted which instrument would win **70.0%** of the time versus **50.8%** for the single-feature comparison. Its squared error fell **36.7%** in nested training CV and **36.7%** in validation. Adding the US macro group to the market-only model lowered validation MSE a further **15.0%**. Adding the global group lowered it another **1.0%**. The standardized coefficients point to **10-year breakeven inflation, 10-year real yield, and the five-year Treasury yield** as the main inputs. The coefficients are associations within the simulator. Starting real and nominal yields affect the carry and price sensitivity of the two instruments, and breakeven inflation helps describe their relative pricing.

The isolated switching check starts in the five-year Treasury and chooses TIPS when the predicted next-year real return advantage exceeds 2 pp. Under the simulator's 1% charge for a full switch, it earned **2.28%** per year on average versus **1.40%** for always holding the five-year Treasury. It switched in **30%** of decisions and its annual 5th percentile was **−7.13%**, versus **−6.81%** for the static Treasury. The family-adjusted 95% interval for annual log-return improvement against the simple switching forecast was **+0.58 to +1.06 pp**. This test evaluates the isolated instrument choice before integration with withdrawals or other holdings.

A follow-up fit using only those three yield inputs had 4.56 pp RMSE and 70.3% direction accuracy. The fixed decision rule remained ahead of static 5Y when the simulated full-switch charge doubled to 2%, but the advantage disappeared at 4%.

| Charge per full switch | Market + macro mean real return | Three-yield model mean real return | Always hold 5Y |
| --- | ---: | ---: | ---: |
| 1% | 2.28% | 2.31% | 1.40% |
| 2% | 1.97% | 1.98% | 1.40% |
| 4% | 1.35% | 1.31% | 1.40% |

The three-input model is the simpler candidate for the hidden holdout. I chose those inputs after inspecting the full model, and the follow-up reuses the public validation paths. The three-input result is exploratory. Test it alongside the original 5Y/cash rule before deciding how both forecasts interact.

## What the additional data contributed

- **US macro data** included breakeven and lagged inflation, credit spreads, unemployment, payrolls, industrial production, and GDP. It helped most on TIPS relative returns and modestly on high-inflation probabilities. The TIPS follow-up suggests that most of its useful information comes from yields and breakeven inflation.
- **Global data** included German and Japanese yields, euro-area financial stress, EUR/USD and USD/JPY, and Brent, copper, and gold returns. In this test it produced small incremental gains for TIPS and high-inflation forecasts. It did not produce a reliable gain over the simple forecasts for the three equity and REIT comparisons or a meaningful early warning for equity drawdowns.
- **Real historical data** could support a separate external test. Start with observed Treasury and TIPS yields and investable total-return series, and align macro observations to their original release dates. The St. Louis Fed's ALFRED archive provides historical vintages for US macro data. A real-market test would need to account for differences between observed instruments and the simulator's constant-maturity zero-coupon returns. [ALFRED vintage data](https://alfred.stlouisfed.org/help/downloaddata)

## How the experiment limits overfitting

The model is ridge regression, a linear forecast with a penalty that shrinks coefficients. The three prespecified groups have 11, 19 and 27 features. Each predictor uses data through the last completed year. Year-over-year changes require two completed years. The tests therefore start in year 3, and the first-year choice continues to use a static default.

The 100 training paths are split into five outer folds. Four inner folds choose among four penalty strengths, favoring stronger shrinkage when errors are close. Feature clipping and scaling are estimated within each fit. Whole paths stay together throughout, and the public validation paths are scored after the training fits are frozen. The scenario bootstrap resamples full paths and adjusts intervals within the forecast and trading comparison families. The public validation set was used in the earlier study, so the hidden holdout remains the next independent check for these candidates. [Nested cross-validation reference](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)

## Reproduce

From the repository root, run `python -B market_research/ml_forecasts.py`. The script uses the existing requirements and prints the detailed JSON. The local output is [ml_results.json](ml_results.json). The repository ignores JSON files, so teammates can regenerate it from the script.
