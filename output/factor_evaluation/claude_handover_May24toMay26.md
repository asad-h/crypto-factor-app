# Claude Handover: May24toMay26 Crypto Factor Research

## Context

This repo now has an independent research-only crypto factor evaluation layer. It is intentionally separate from Streamlit/dashboard production code.

Do not modify dashboard files unless the user explicitly asks for production integration. The relevant code is under:

- `crypto_factor_model/research/`
- `test_research_walkforward.py`
- `notebooks/factor_model_walkforward_May24toMay26.ipynb`
- `cache/research/factor_research_May24toMay26.parquet`
- `output/factor_evaluation/`

The CoinGecko key used for the refresh was handled through runtime configuration and was not persisted in repo files.

## Current Research State

Main dataset:

- `cache/research/factor_research_May24toMay26.parquet`
- Shape from latest verification: `150675` rows x `130` columns
- Coverage: `2024-05-01` through `2026-05-05`
- Universe: 205 tokens

Executed notebook:

- `notebooks/factor_model_walkforward_May24toMay26.ipynb`
- 40 cells
- 22 code cells with outputs
- Includes monthly L/S basket table section

Plain-English summary:

- `output/factor_evaluation/factor_model_walkforward_May24toMay26_plain_english_writeup.md`

Monthly L/S basket table:

- `output/factor_evaluation/ls_basket_monthly_May24toMay26.csv`

## What Changed Most Recently

The latest update implemented the user-requested steps:

1. Stricter selector:
   - Requires positive validation return.
   - Requires positive validation Sharpe.
   - Requires positive average rolling-fold return.
   - Requires at least 3 positive rolling folds.
   - If no candidate passes, it reports `NO_STRICT_PASS` instead of forcing a weak recommendation.

2. L/S-first selection:
   - `always_l_s`, `sticky_l_s`, and `regime_aware_l_s` are the main strategy variants.
   - Long-only, short-only, and regime-switch candidates are still visible in diagnostics, but they cannot win weakly.

3. `$10M` universe as a formal candidate:
   - `eligible_sensitivity` is evaluated in validation/model selection, not just appendix sensitivity.

4. Sticky L/S basket:
   - Added `sticky_l_s`.
   - Previous shorts are retained if they remain inside the short rank buffer, reducing churn.

## Current Result

The current stricter selector says:

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
- Rolling average return: `-9.0%`

Do not describe this as selected for production. It is a diagnostic basket only.

## Monthly L/S Basket Snapshot

The latest 14D monthly diagnostic basket table shows:

| Month | Longs | Shorts | Net return |
|---|---|---|---:|
| 2025-11 | ADA, BTC, BNB, AVAX, BTT | GRASS, UNI, AERO, AAVE, PUMP | +0.6% |
| 2025-12 | WBT, BTC, BNB, BTT, CRO | UNI, AAVE, 0G, HYPE, ENA | +4.0% |
| 2026-01 | BCH, WBT, TRX, BNB, BTC | UNI, AAVE, HYPE, ENA, JTO | +0.3% |
| 2026-02 | WBT, CC, BNB, BTC, TRX | UNI, AAVE, HYPE, ENA, ATH | -0.9% |
| 2026-03 | WBT, BNB, CRO, TRX, RAIN | HYPE, ENA, ATH, ONDO, ADA | -1.7% |
| 2026-04 | TRX, WBT, CC, BNB, ADA | HYPE, ENA, ARB, AAVE, UNI | +0.1% |

Interpretation: longs are moderately stable; shorts are now intentionally stickier but still rotate when bottom ranks change.

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
- `python -m unittest test_research_walkforward.py` passed: 10 tests OK.
- Notebook was regenerated and contains non-empty monthly L/S section.
- Secret scan for the CoinGecko key fragments returned no matches.

## Important Caveats

- The stricter gate intentionally refused to recommend a model.
- The positive untouched-test performance of the diagnostic sticky L/S basket must not be used to select or tune the model.
- Full refresh can hang or slow down in upstream data stages. Prefer `--from-dataset` for code/evaluation iteration.
- `cache/research/` contains local raw CoinGecko cache files, but only the main long-form research parquet needs to be treated as the research artifact.
- There are unrelated dirty files in the working tree from outside this research task; do not revert or include them unless the user asks.

## Good Next Research Steps

- Study why validation and rolling folds are weak while test was better.
- Add a "no trade" or "cash" option when validation gates fail.
- Explore turnover-aware ranking directly in the score rather than only sticky basket construction.
- Add token/sector exposure caps to reduce repeated large-cap dominance.
- Revisit shortability history; current Binance USD-M availability is still a proxy.
