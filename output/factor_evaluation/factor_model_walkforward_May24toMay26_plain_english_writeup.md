# May24toMay26 Crypto Factor Walk-Forward: Plain-English Companion

This note explains the notebook in non-technical language. It reflects the broader CoinGecko Pro-backed research dataset, the formal `$10M` liquidity universe, purged rolling folds, and the stricter model-selection gate.

## The Short Version

The practical question is:

> Do any of the tested factor models look strong enough to recommend for a liquid crypto long/short strategy?

The current answer is:

> No. The stricter process did not select a production-worthy model.

The best reporting-only L/S basket is:

| Item | Result |
|---|---:|
| Recommendation status | No model selected |
| Reporting-only model | Learned global |
| Portfolio variant | Sticky L/S |
| Liquidity universe | `$10M` trailing 7D average daily volume |
| Primary horizon | 14D |
| Fundamentals weight | 55% |
| Momentum weight | 45% |
| Flows weight | 0% |
| Factor improvement weight | 0% |

The key distinction is important: the notebook still shows a basket so we can inspect what the model would have owned, but the strict gate says not to treat it as a strategy recommendation.

## What Changed

The latest run made four changes:

- `$10M` liquidity is now a formal candidate, not just an appendix sensitivity.
- The selector is L/S-first; long-only, short-only, and regime-switch candidates no longer win on weak evidence.
- The selector now requires positive validation return, positive validation Sharpe, positive average rolling-fold return, and at least 3 positive rolling folds.
- A sticky L/S variant was added to reduce short-basket churn by keeping prior shorts if they remain near the bottom ranks.

## Fixed Split Versus Rolling Folds

The notebook still keeps the final untouched test period separate:

- Training: May 1, 2024 to April 30, 2025
- Validation: May 1, 2025 to October 31, 2025
- Test: November 1, 2025 onward, label-complete by horizon

It also now includes purged rolling folds before the untouched test period. Each fold trains on earlier data, leaves a `horizon + 1` day gap, validates on the next isolated window, and never uses the final test period.

## What Validation Said

The best reporting-only L/S row was:

| Model / variant / universe | Validation return | Sharpe | Max drawdown | Positive rolling folds |
|---|---:|---:|---:|---:|
| Learned global sticky L/S, `$10M` universe | -4.8% | -0.04 | -26.6% | 1 of 4 |

That fails the stricter gate. In plain English: it was the least-bad L/S candidate in the preferred liquidity universe, but it was not actually good enough.

## Untouched Test Result

The reporting-only sticky L/S basket did better in the untouched test period, but that result was not allowed to rescue the model. The test period must stay untouched for selection decisions.

| Holding period | Rebalances | Total return | Sharpe | Max drawdown | Hit rate | Excess vs BTC |
|---|---:|---:|---:|---:|---:|---:|
| 7D | 26 | +2.2% | 0.32 | -6.7% | 57.7% | +31.1% |
| 14D | 13 | +8.3% | 1.43 | -4.2% | 61.5% | +37.2% |
| 30D | 6 | +10.2% | 2.46 | -1.0% | 66.7% | +39.5% |

This is encouraging as a diagnostic, but it is not a valid reason to select the model because the validation and rolling gates failed before test.

## Monthly L/S Basket Read

The monthly 14D basket table shows the reporting-only L/S constituents and realized forward returns. It is useful for understanding concentration and stability:

- Longs were fairly consistent: BTC, BNB, WBT, TRX, ADA, and CC appeared repeatedly.
- Shorts became stickier by design: HYPE, ENA, AAVE, UNI, and related bottom-ranked names persisted across months.
- The monthly 14D net returns were mixed but positive overall in test: +0.6%, +4.0%, +0.3%, -0.9%, -1.7%, +0.1%.

## What "Factor Improvement: 0%" Means

The learned-global weights assign no weight to factor improvement:

```text
55% fundamentals
45% momentum
0% flows
0% factor improvement
```

That means the training process did not find enough unique value in separately rewarding tokens whose factor profile recently improved. It does not prove factor improvement is bad; it only says this training setup preferred current fundamentals and momentum.

## Bottom Line

The latest run is more conservative and more honest.

The conclusion is:

> Keep the `$10M` universe, keep the rolling-fold framework, and keep investigating sticky L/S baskets, but do not call the current model selected. The validation and rolling evidence is still too weak.
