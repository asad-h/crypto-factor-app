# Claude Handover: May24toMay26 Crypto Factor Research

## Research Objectives

The goal of this research layer is to evaluate whether a liquid crypto long/short factor strategy has enough historical evidence to justify further development.

The specific objectives are:

- Build an auditable historical dataset from May 1, 2024 through the latest label-complete May 2026 data.
- Avoid obvious leakage: lag fundamentals/flows, use only as-of signal values, enter at `t+1` close, exit at `t+h+1` close, and keep test labels out of model selection.
- Compare factor families: fundamentals, momentum, flows, and factor improvement.
- Compare model candidates: current composite, equal-family, heuristic weights, learned global weights, and regime-specific weights.
- Test whether BTC 20-week moving-average regimes improve signal weighting or exposure rules.
- Evaluate long/short, long-only, short-only, regime-switching, and sticky L/S variants.
- Promote the `$10M` trailing 7D volume universe to a formal candidate, not just an appendix sensitivity.
- Produce enough artifacts for a non-technical reviewer to understand the conclusion without reading code.

This is research-only. It is not Streamlit/dashboard production work.

## Where To Work

Relevant files:

- `crypto_factor_model/research/`
- `test_research_walkforward.py`
- `notebooks/factor_model_walkforward_May24toMay26.ipynb`
- `cache/research/factor_research_May24toMay26.parquet`
- `output/factor_evaluation/`

Do not modify dashboard files unless the user explicitly asks for production integration.

The CoinGecko key used for the refresh was handled through runtime configuration and was not persisted in repo files.

## Current Dataset And Outputs

Main dataset:

- `cache/research/factor_research_May24toMay26.parquet`
- Shape from latest verification: `150675` rows x `130` columns
- Coverage: May 1, 2024 through May 5, 2026
- Universe: 205 tokens

Executed notebook:

- `notebooks/factor_model_walkforward_May24toMay26.ipynb`
- Includes coverage, leakage checks, BTC regime chart, signal ICs, validation selection, rolling folds, untouched test results, BTC-relative results, constituents, and monthly L/S basket table.

Plain-English summary:

- `output/factor_evaluation/factor_model_walkforward_May24toMay26_plain_english_writeup.md`

Monthly L/S basket table:

- `output/factor_evaluation/ls_basket_monthly_May24toMay26.csv`

## Most Recent Rule Change: Never Short HYPE

User explicitly requested that HYPE never appear in the short basket.

Implementation:

- `NEVER_SHORT_TOKENS = ("HYPE",)` in `crypto_factor_model/research/constants.py`
- `run_portfolio_backtest()` applies the exclusion to the shortable row before constructing positions.
- This affects all short-producing variants, including always L/S, sticky L/S, short-only, bearish regime-switch, and regime-aware L/S.
- HYPE can still be evaluated as an asset and can still be long if ranked high enough; it is only banned from short positions.

Verification after regeneration:

- `output/factor_evaluation/constituents_May24toMay26.csv`: 0 rows where `token == HYPE` and `side == Short`
- `output/factor_evaluation/ls_basket_monthly_May24toMay26.csv`: 0 rows with HYPE in `short_tokens`
- Synthetic test added: HYPE is bottom-ranked but does not enter the short basket.

## Current Model Result

The stricter selector still says:

- `selection_status`: `NO_STRICT_PASS`
- `strict_gate_passed`: `false`
- `final_recommendation`: `No model selected; reporting basket is diagnostic only`

The reporting-only L/S candidate is:

- model: `learned_global`
- variant: `sticky_l_s`
- universe: `eligible_sensitivity` (`$10M` trailing 7D volume)
- primary horizon: `14D`
- weights: 55% fundamentals, 45% momentum, 0% flows, 0% factor improvement

Why it failed selection:

- Validation return: `-4.8%`
- Validation Sharpe: `-0.04`
- Rolling positive folds: `1 / 4`
- Rolling average return: `-10.7%`
- Rolling average Sharpe: `-0.68`

Do not describe this as selected for production. It is a diagnostic basket only.

## Current Issues With The Model

The main issues are:

- Validation is negative. The best reporting L/S candidate loses money in the validation window.
- Rolling support is weak. Only 1 of 4 purged rolling folds is positive.
- Test-period improvement is not selection evidence. The HYPE-excluded diagnostic basket performs well in test, but that cannot rescue a model that failed validation.
- Regime awareness has not proven robust. BTC 20W regimes are useful for audit/context, but the current regime-specific candidates do not clear the strict gate.
- Factor breadth is narrow. Learned-global weights concentrate in fundamentals and momentum; flows and factor improvement get 0%.
- Shortability is approximate. Binance USD-M availability is treated as a proxy, not a full historical borrow/listing calendar.
- Survivorship bias remains. The asset master is seeded from the current mapped universe.
- Some historical mcap/FDV coverage still relies on proxies where true historical supply data is unavailable.
- Monthly basket stability is only moderate. Sticky shorts reduce churn, but basket membership still changes materially.
- The current selector is conservative by design. It may reject baskets that look good in test, which is correct but may feel unsatisfying.

## What To Improve Next

Recommended next research work:

- Add a formal "no trade" or "cash" outcome when strict gates fail.
- Add sector/token exposure caps so large-cap infrastructure names do not dominate repeatedly.
- Build a real historical shortability/listing calendar instead of relying on current Binance USD-M availability.
- Add a turnover-aware rank objective directly into factor scoring, not only basket construction.
- Investigate why validation is weak but HYPE-excluded test is strong. Treat this as a hypothesis, not a conclusion.
- Expand robustness checks: subperiods, volatility regimes, BTC drawdown regimes, and excluding single high-impact names.
- Revisit factor improvement. It is currently 0%, but may need a different definition or slower lookback.
- Add outlier and event filters for newly listed/meme/event-driven tokens.
- Stress-test transaction costs and slippage beyond the current 50 bps assumption.
- Separate "model quality" diagnostics from "portfolio construction" diagnostics more clearly.

## What To Independently Verify

Before anyone trusts or productionizes this, independently verify:

- The HYPE short ban is enforced in all short-producing variants and all regenerated artifacts.
- Forward-return alignment: signal at `t`, entry at `t+1` close, exit at `t+h+1` close.
- BTC regime as-of rule: signal date uses the latest completed weekly BTC close and trailing 20W MA only.
- No test-period labels influence signal selection, dedupe, weights, hyperparameters, model choice, or basket rules.
- Rolling folds have a purge gap of `horizon + 1` days and never overlap the untouched test period.
- `$10M` universe membership uses historical trailing volume, not current volume backfills.
- Historical mcap/volume/FDV fields are not silently backfilled from current snapshots.
- Stable/wrapped/pegged/tokenized treasury/basket/commodity-like exclusions remain active.
- Shortability proxy limitations are clearly disclosed.
- Monthly basket returns match constituent entry/exit prices and cost assumptions.
- BTC-relative metrics are recomputed independently from raw returns.
- Notebook outputs match the CSV/JSON artifacts exactly.

## Monthly HYPE-Excluded L/S Basket Snapshot

The latest 14D monthly diagnostic basket table is:

| Month | Longs | Shorts | Net return |
|---|---|---|---:|
| 2025-11 | ADA, BTC, BNB, AVAX, BTT | GRASS, UNI, AERO, AAVE, PUMP | +0.6% |
| 2025-12 | WBT, BTC, BNB, BTT, CRO | UNI, AAVE, 0G, ENA, APT | +4.6% |
| 2026-01 | BCH, WBT, TRX, BNB, BTC | UNI, AAVE, ENA, APT, JTO | +2.8% |
| 2026-02 | WBT, CC, BNB, BTC, TRX | UNI, AAVE, ENA, ATH, PUMP | +0.7% |
| 2026-03 | WBT, BNB, CRO, TRX, RAIN | ENA, ATH, ONDO, ADA, AVAX | +0.3% |
| 2026-04 | TRX, WBT, CC, BNB, ADA | ENA, AVAX, ARB, AAVE, UNI | +1.1% |

Interpretation: HYPE is no longer shorted. The replacement shorts make the test-period diagnostic basket look better, but that is not valid selection evidence because the validation and rolling gates still failed.

## Key Commands

Run tests:

```bash
.venv/bin/python -m unittest test_research_walkforward.py
```

Rebuild from cached/audited parquet without hitting upstream APIs:

```bash
.venv/bin/python -m crypto_factor_model.research.run_walkforward --from-dataset
```

Full data refresh, slower and API-dependent:

```bash
.venv/bin/python -m crypto_factor_model.research.run_walkforward --max-assets 0
```

## Latest Verification

These checks were run after the latest update:

- `py_compile` for research modules and tests passed.
- `python -m unittest test_research_walkforward.py` passed: 11 tests OK.
- Research outputs and notebook were regenerated with `--from-dataset`.
- HYPE short scan returned 0 HYPE short rows in constituents and monthly basket outputs.

## Current Bottom Line

Keep the research layer, keep the HYPE short ban, and keep the `$10M` sticky L/S basket as a diagnostic object. Do not call the model selected yet. The right next step is robustness and independent verification, not production integration.
