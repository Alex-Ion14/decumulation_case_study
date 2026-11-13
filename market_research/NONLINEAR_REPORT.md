# Gaussian processes and neural networks for annual forecasts

The three-yield ridge model remains the leading candidate for TIPS versus five-year Treasuries. The Gaussian process earned slightly more in the isolated TIPS switching test, but had worse forecast RMSE and its return improvement versus the three-yield model was small and uncertain. The feed-forward and recurrent models showed no consistent advantage on the public validation scenarios.

## Forecast accuracy

Validation RMSE in percentage points, using decision years 3–30. Each target uses the same features across models. The three-year ridge receives the current feature row and two lagged rows, repeating the first available row at the start. The GRU uses the available sequence through the decision date.

| One-year target | Ridge | Three-year ridge | Gaussian process | Feed-forward net | GRU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5Y Treasury minus cash | **3.853** | 3.935 | 3.854 | 3.915 | 3.877 |
| 10Y TIPS minus 5Y Treasury | **4.610** | 4.636 | 4.642 | 4.643 | 4.764 |
| US equity minus cash | **18.954** | 19.001 | 19.214 | 19.068 | 19.107 |
| Real cash return | 1.451 | 1.464 | **1.443** | 1.470 | 1.474 |

The cash Gaussian process reduced mean squared error by 1.2% versus ridge, and the family-adjusted interval for the improvement crossed zero. Its direction accuracy was 85.6%, versus 85.0% for ridge. All three nonlinear equity models reduced MSE within the training folds and increased it on the public validation set. The GRU showed no validation MSE improvement on any of the four targets. These tests use annual observations, so they do not answer whether finer-grained histories would help with a shorter-horizon decision.

## TIPS switching

Each model starts with the five-year Treasury and selects TIPS when its forecasted next-year real return advantage exceeds 2 percentage points. The simulator's charge is 1% per full switch in this isolated test.

| TIPS forecast | Validation RMSE | Direction accuracy | Net annual real return | 5th-percentile year | Switch frequency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always hold 5Y | — | — | 1.40% | −6.81% | 0% |
| Three-yield ridge | 4.56 pp | 70.3% | 2.31% | −6.99% | 32.1% |
| Full ridge | 4.61 pp | 70.0% | 2.28% | −7.13% | 30.2% |
| Gaussian process | 4.64 pp | 69.9% | 2.34% | −6.93% | 29.4% |
| Feed-forward net | 4.64 pp | 70.4% | 2.31% | −7.03% | 29.2% |
| GRU | 4.76 pp | 69.2% | 2.29% | −7.03% | 27.4% |

The GP added 0.056 pp per year versus full ridge. The family-adjusted 95% interval for annual log-return improvement was +0.003 to +0.110 pp. Against the three-yield ridge it added 0.032 pp, with an exploratory, unadjusted log-return interval of −0.014 to +0.077 pp. At a 2% charge per full switch, the GP earned 2.03% versus 1.98% for the three-yield ridge. These are comparisons on the same public validation paths and cover an isolated two-instrument decision without withdrawals.

The GP's nominal 90% TIPS prediction interval covered 92.5% of outcomes and averaged 16.34 pp wide. Scaling it with out-of-fold training residuals narrowed it to 13.96 pp, with 87.4% coverage. The simple ridge residual interval achieved 88.2% coverage at 14.22 pp wide. The GP can express uncertainty for each decision, but coverage varied across the targets and was close to the simple residual interval for TIPS after training-based calibration. These intervals estimate future realized return spreads and were evaluated independently of the switching hurdle.

## Implementation and validation

- The exact Gaussian process uses a linear plus radial-basis kernel and Gaussian observation noise. It fits all the rows in each training split. Two kernel lengths and two noise scales compete on a held-out subset of training paths. The mean forecast is used for the instrument choice. [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/)
- The feed-forward network has a linear branch and a single eight-unit nonlinear layer. The recurrent network uses a single unidirectional eight-unit GRU with the same kind of linear branch. Epoch counts come from early stopping on training-only paths. Both networks refit on the full training portion for the selected epoch count. The script checks that GRU predictions for the same prefix agree when later observations are removed. [PyTorch GRU documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- The experiment uses five whole-path folds inside the 100 training scenarios and scores the final fits on 80 separate public scenarios. Clipping and scaling are fitted inside each training split. Validation intervals resample complete paths and adjust for the primary model comparisons. The public validation paths informed the earlier ridge study and the choice of targets for this follow-up. The hidden holdout is still needed for a fresh comparison.
- The simple 5Y/cash spread rule remains the practical benchmark. For TIPS, the three-yield ridge and the GP are the most useful pair to compare on the hidden holdout. The GP's potential value is a forecast distribution and slightly lower turnover in this experiment.

Run `python -B market_research/nonlinear_forecasts.py` from the repository root. It uses the existing NumPy and PyTorch dependencies, prints JSON, and takes about a minute and a half on the CPU used here. The local output is [nonlinear_results.json](nonlinear_results.json). The repository ignores JSON files, so teammates can regenerate it.
