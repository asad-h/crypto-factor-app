# Next Steps: May24toMay26 Research Track

## Fundamentals Redefinitions

Stable across Train, Validation, Test, and the mixed-regime fold_2:

| signal | Train IC | Validation IC | Test IC | fold_2 IC |
| --- | ---: | ---: | ---: | ---: |
| `tvl_chg_8w` | 0.1921 | 0.0334 | 0.0652 | 0.1357 |
| `trading_fees_chg_4w` | 0.0231 | 0.0295 | 0.0258 | 0.0368 |
| `revenue_chg_4w` | 0.0329 | 0.0168 | 0.0222 | 0.0448 |

`dex_volume_chg_4w` is positive on Train, Validation, and fold_2, but not Test, so it is useful as a validation-only research candidate but not a stable all-window result.

## Weight Robustness

No weight scheme clears the no-leakage strict-gate proxy. All schemes have negative validation return and Sharpe, and none get beyond 1 of 4 positive fold-retrained rolling folds.

Closest diagnostics:

- Best validation deficit: `baseline_learned_global` at validation return -4.84%, Sharpe -0.043, 1/4 positive folds.
- Best rolling-average deficit: `shrunk_to_momentum_anchor` at rolling average -2.88%, but validation return is -8.83% and Sharpe is -0.198.

## Regime-Stratified Gate

The informational regime-stratified sub-gate rejects the diagnostic selected basket (`learned_global` / `sticky_l_s` / `eligible_sensitivity`). Worst sufficient bullish sub-window return is -96.1%; worst sufficient bearish sub-window return is -48.8%. This confirms the basket is not robust to the mixed-regime validation structure.

## Beta Decomposition

For the selected primary-horizon test row, total return is +22.19%, beta-attributed return is +6.16%, and dispersion alpha is +16.03%. The beta share is about 27.7%, below 50%, but this does not reverse the audit because the candidate fails validation, rolling, and regime-stratified gates.

## Recommendation

No candidate clears. The next research-track promotion attempt should start from the stable fundamentals deltas above, especially `tvl_chg_8w`, `trading_fees_chg_4w`, and `revenue_chg_4w`, then rerun selection on validation-only rules. Do not promote the current diagnostic basket.
