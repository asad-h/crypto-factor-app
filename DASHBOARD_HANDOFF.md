# Regime Dashboard Handoff Document

Machine-readable context for a new Claude session to continue development on the crypto regime classification dashboard.

---

## Architecture Overview

### File Tree

```
Crypto Factor Model/
  dashboard/
    app.py                     # Streamlit UI: layout, CSS, display components, CoinGecko watchlist
    data_adapter.py            # Orchestrator: fetches raw data, calls indicator functions, feeds classifier
    regime_indicators.py       # 17 indicator functions across 5 families (BTC trend, breadth, stablecoin, leverage, macro, valuation)
    regime_etf_dat.py          # 9 indicator functions for the ETF + DAT flows family
    regime_classifier.py       # Scoring engine: family aggregation, weighted regime scores, final classification
    validate_contracts.py      # Standalone contract validation with synthetic data
    data/
      dat_mnav.csv             # Weekly DAT mNAV scrape cache (MSTR, BMNR, PURR)
      watchlist.csv            # Token/equity watchlist with relationship metadata
      etf_flows.csv.example    # Schema example for ETF flow CSV
      leverage_overrides.csv.example
      onchain_overrides.csv.example
  crypto_factor_model/
    config.py                  # Project-wide config: paths, CACHE_DIR, SLUG_TO_BINANCE mapping, universe filters
    clients/
      binance.py               # Public API client: klines, funding rates (/fapi/v1/fundingRate), open interest (/fapi/v1/openInterest)
      cryptoquant.py           # MVRV and NUPL via Bearer token auth (CRYPTOQUANT_API_KEY env)
      blockworks.py            # Blockworks Research API client (not used by dashboard directly)
      defillama.py             # DefiLlama client (dashboard uses its own inline fetch in data_adapter.py)
```

### Data Flow

```
External APIs (Binance, CoinGecko, CryptoQuant, DefiLlama, FRED, yfinance, Stooq)
  + Local CSVs (dat_mnav.csv, watchlist.csv, etf_flows.csv, leverage_overrides.csv, onchain_overrides.csv)
      |
      v
  data_adapter.py
    - fetch_btc_daily(), fetch_btc_weekly(), fetch_token_weekly_closes()
    - fetch_stablecoin_supply(), fetch_dxy_data(), fetch_real_yields_4w_change()
    - fetch_ai_basket_return_8w(), fetch_m2_mom(), fetch_equity_weekly()
    - fetch_binance_funding_rate(), fetch_binance_oi_pct(), fetch_cryptoquant_onchain()
    - load_leverage_overrides(), load_etf_flows(), load_dat_mnav(), load_portfolio_drawdown()
      |
      v
  regime_indicators.py (17 functions) + regime_etf_dat.py (9 functions)
    - Each returns a standardized indicator dict
      |
      v
  regime_classifier.py
    - classify_regime(indicators) -> dict with regime, scores, DataFrames
    - classify_weekly_history(btc_weekly, indicator_func) -> historical regime DataFrame
      |
      v
  app.py (Streamlit)
    - Renders hero card, KPI metrics, family scores table, indicator audit, charts, watchlist
```

---

## Data Contracts

### classify_regime() Return Dict

```python
{
    "regime": str,              # One of: "Risk-on", "Choppy", "Risk-off", "Local top", "Local bottom"
    "risk_on_score": float,     # 0-100, weighted across families
    "choppy_score": float,      # 0-100
    "risk_off_score": float,    # 0-100
    "local_top_score": float,   # 0-100
    "local_bottom_score": float,# 0-100
    "family_scores_df": pd.DataFrame,
    "indicator_audit_df": pd.DataFrame,
    "summary_text": str,
}
```

compute_current_regime() adds `"btc_weekly": pd.Series` to this dict.

### family_scores_df

| Column | Type | Description |
|--------|------|-------------|
| family | str | One of 7 family names |
| weight | int | Family weight (5-30) |
| indicators_met | int | Count of indicators with score=2 |
| indicators_total | int | Count of indicators in family |
| family_score | float | 100 * sum(scores) / (2 * n_indicators) |
| risk_on | float | Average risk_on vote across family indicators |
| choppy | float | Average choppy vote |
| risk_off | float | Average risk_off vote |
| local_top | float | Average local_top vote |
| local_bottom | float | Average local_bottom vote |
| read | str | One-sentence family assessment |

### indicator_audit_df

| Column | Type | Description |
|--------|------|-------------|
| family | str | Family name |
| indicator | str | Indicator display name |
| value | str | Formatted current value |
| criteria | str | Threshold description |
| status | str | "Met" / "Partial" / "Not met" |
| score | int | 0 (not met), 1 (partial), 2 (met) |
| meaning | str | Plain-english interpretation |
| regime_fit | str | Which regime this reading supports |
| source | str | Data source label |
| asof | str | Date string of data |

Note: regime_votes is stripped from the audit df (it stays on the raw indicator dicts only).

### weekly_regime_df (from classify_weekly_history)

| Column | Type | Description |
|--------|------|-------------|
| date | datetime | Week start |
| btc_close | float | BTC weekly close price |
| regime | str | Classified regime for that week |
| risk_on_score | float | 0-100 |
| choppy_score | float | 0-100 |
| risk_off_score | float | 0-100 |
| local_top_score | float | 0-100 |
| local_bottom_score | float | 0-100 |
| end_date | datetime | Next week start (for chart band rendering) |

### market_kpis_df (from build_market_kpis_df)

| Column | Type |
|--------|------|
| KPI | str |
| Value | str |
| Source | str |
| Status | str |
| Why it matters | str |

KPI rows: DXY, Gold, M2 MoM, STRC price, STRC next payout.

### dat_mnav_df (from load_dat_mnav)

| Column | Type |
|--------|------|
| DAT | str (MSTR, BMNR, PURR) |
| Metric | str |
| mNAV | float |
| Source | str (URL) |
| Source status | str |
| Fetch detail | str |
| Note | str |
| asof | str |

### watchlist_df (from load_watchlist_prices in app.py)

| Column | Type |
|--------|------|
| Type | str (token/equity) |
| Name | str |
| Symbol | str |
| Price | float |
| WoW % | float |
| 7d sparkline | list[float] |
| Source | str |
| Chain | str (optional) |
| Portco | str (optional) |
| Relationship | str (optional) |
| Note | str (optional) |

---

## Indicator System

### Standardized Indicator Dict (returned by every indicator function)

```python
{
    "family": str,
    "indicator": str,
    "value": str,
    "criteria": str,
    "status": str,          # "Met" | "Partial" | "Not met"
    "score": int,           # 0 | 1 | 2
    "meaning": str,
    "regime_fit": str,
    "source": str,
    "asof": str,
    "regime_votes": {
        "risk_on": float,       # 0-100, how much this reading supports risk-on
        "choppy": float,
        "risk_off": float,
        "local_top": float,
        "local_bottom": float,
    },
}
```

### 7 Families with Weights

| # | Family | Weight | Indicator Count |
|---|--------|--------|-----------------|
| 1 | BTC trend / momentum | 20% | 4 |
| 2 | Market breadth | 15% | 3 |
| 3 | ETF + DAT flows | 30% | 9 |
| 4 | Stablecoin liquidity | 10% | 1 |
| 5 | Leverage / volatility | 10% | 4 |
| 6 | Macro / AI | 10% | 4 |
| 7 | Valuation / sentiment | 5% | 3 |

Total: 28 indicators.

### All Indicators by Family

**BTC trend / momentum (20%, regime_indicators.py)**
1. `btc_vs_20w_sma` -- BTC close vs 20-week SMA with slope
2. `sma_20w_slope` -- 20-week SMA slope direction
3. `btc_4w_return` -- BTC 4-week return
4. `btc_12w_return` -- BTC 12-week return

**Market breadth (15%, regime_indicators.py)**
5. `pct_tokens_above_20w_sma` -- % of eligible tokens above their 20w SMA
6. `median_token_vs_btc_8w` -- Median token 8w return vs BTC 8w return
7. `new_highs_vs_lows` -- Tokens at 20w high vs 20w low

**ETF + DAT flows (30%, regime_etf_dat.py)**
8. `btc_etf_flows` -- BTC ETF 5d and 20d net flows
9. `eth_etf_flows` -- ETH ETF 5d and 20d net flows
10. `etf_price_response` -- Do ETF inflows lift BTC price?
11. `mstr_mnav` -- MSTR mNAV premium vs 1.20x threshold
12. `bmnr_mnav` -- BMNR mNAV premium vs 1.15x threshold
13. `purr_mnav` -- PURR multiple to adjusted NAV vs 1.15x threshold
14. `dat_issuance_capacity` -- DAT net issuance 30d (manual)
15. `dat_treasury_purchases` -- DAT net treasury purchases 30d (manual)
16. `strc_credit_stress` -- STRC/DAT yield spread 4w change (manual)

**Stablecoin liquidity (10%, regime_indicators.py)**
17. `stablecoin_supply_growth` -- Stablecoin supply 30d and 90d growth

**Leverage / volatility (10%, regime_indicators.py)**
18. `realised_vol_8w` -- 8-week realised vol direction vs price
19. `funding_rates_indicator` -- BTC perp funding rate annualized
20. `oi_to_mcap_indicator` -- Open interest as % of market cap
21. `basis_indicator` -- CME/perp basis spread

**Macro / AI (10%, regime_indicators.py)**
22. `spy_qqq_trend` -- SPY and QQQ vs 20w SMAs
23. `dxy_trend_indicator` -- DXY level and 4w change
24. `real_yields_indicator` -- 10Y TIPS real yield 4w change (bps)
25. `ai_macro_bias` -- AI equity basket (NVDA, MSFT, GOOGL, AMZN, META, AVGO, AMD) 8w return

**Valuation / sentiment (5%, regime_indicators.py)**
26. `mvrv_nupl_indicator` -- MVRV Z-score and NUPL
27. `retail_froth_indicator` -- Coinbase app rank proxy
28. `portfolio_drawdown_indicator` -- Portfolio drawdown defensive override

---

## Classification Logic

Located in `regime_classifier.py`, function `classify_regime()`.

**Family score** = `100 * sum(indicator_scores_in_family) / (2 * n_indicators_in_family)`

**Weighted regime scores**: For each of 5 regime dimensions (risk_on, choppy, risk_off, local_top, local_bottom), compute weighted average of family-level votes using FAMILY_WEIGHTS (sum to 100).

**Classification rules** (evaluated in order):
1. Portfolio drawdown override: if drawdown <= -15% -> "Risk-off"
2. Risk-off if `risk_off_score >= 65`
3. Risk-on if `risk_on_score >= 65` AND `(risk_on - choppy) >= 10` AND `(risk_on - risk_off) >= 10`
4. Otherwise -> "Choppy"

**Local overlays** (applied after base classification):
- Local top if `local_top_score >= 40` AND `risk_on_score >= 50`
- Local bottom if `local_bottom_score >= 40` AND `risk_off_score >= 40`

Default thresholds are configurable via sidebar sliders in app.py.

---

## External Data Sources

| Source | Auth | What it provides | Endpoints/Series |
|--------|------|-----------------|-----------------|
| **Binance** | None (public) | BTC daily/weekly OHLCV, token daily closes, funding rates, open interest | `/api/v3/klines`, `/fapi/v1/fundingRate`, `/fapi/v1/openInterest` |
| **CoinGecko** | Pro key (CG-... prefix) | Watchlist token prices, 7d sparklines, 7d % change | `/coins/markets?sparkline=true` via `pro-api.coingecko.com` |
| **CryptoQuant** | Bearer token (CRYPTOQUANT_API_KEY env) | MVRV Z-score, NUPL | `/v1/btc/market-data/mvrv`, `/v1/btc/market-data/nupl` |
| **DefiLlama** | None | Aggregate stablecoin supply | `stablecoins.llama.fi/stablecoincharts/all` |
| **FRED** | None (CSV endpoint) | M2 money supply (WM2NS), 10Y real yields (DFII10) | `fred.stlouisfed.org/graph/fredgraph.csv?id=WM2NS` and `DFII10` |
| **yfinance** | None | SPY, QQQ, DXY (DX-Y.NYB), Gold (GC=F), AI basket equities, STRC | Python library, period/interval params |
| **Stooq** | None | Fallback for DXY (dx.f), Gold (xauusd), STRC (strc.us) | `stooq.com/q/l/?s={symbol}&f=sd2t2ohlcv&h&e=csv` |

**Local CSVs** (in `dashboard/data/`):
- `dat_mnav.csv` -- Required for DAT mNAV indicators. Manual weekly update.
- `watchlist.csv` -- Token/equity list with chain, portco, relationship metadata. 34 tokens + equities.
- `etf_flows.csv` (optional) -- Columns: date, btc_net_flow_usd, eth_net_flow_usd. Without this, ETF flow indicators use hardcoded fallback values.
- `leverage_overrides.csv` (optional) -- Columns: funding_ann, oi_pct, basis_pct. Fallback for anything Binance API doesn't provide (mainly basis).
- `onchain_overrides.csv` (optional) -- Columns: mvrv_z, nupl, coinbase_rank. Fallback if CryptoQuant API fails.
- `portfolio_nav.csv` (optional) -- Columns: date, nav. Used for portfolio drawdown indicator.

---

## CoinGecko Watchlist Integration

- `_TOKEN_TO_CG_ID` dict in `app.py` maps 34 token names to CoinGecko IDs
- Pro key detected by `CG-` prefix, routes to `pro-api.coingecko.com` with `x-cg-pro-api-key` header
- Demo keys route to `api.coingecko.com` with `x-cg-demo-api-key` header
- Batch fetch via `/coins/markets` with `sparkline=true` and `price_change_percentage=7d`
- Batch size: 100 per request
- Cached via `@st.cache_data(ttl=5*60)` (5 minute TTL)
- Some CoinGecko IDs may be wrong and need verification: `wet`, `plasma-2`, `canton-coin`, `2z`, `derive-2`, `turtle`

---

## Recent Bug Fixes (completed)

1. **Historical regime chart was always "Choppy"**: Fixed by expanding `compute_weekly_regime_df()` to use all 7 families via `multi_family_indicators()` instead of just 4 BTC trend indicators. Historical chart now passes a full indicator set per week to `classify_weekly_history()`.

2. **Funding rates now fetched live from Binance**: `fetch_binance_funding_rate()` calls `BinanceClient.get_current_funding_annualized()` which averages the last 7 days of 8h rates from `/fapi/v1/fundingRate` and annualizes (rate * 3 * 365 * 100). CSV fallback still works.

3. **MVRV/NUPL now fetched from CryptoQuant API**: `fetch_cryptoquant_onchain()` tries `CryptoQuantClient` first, falls back to `onchain_overrides.csv`. Client uses Bearer auth via `CRYPTOQUANT_API_KEY` env var.

4. **Watchlist now fetches live prices from CoinGecko**: Token prices come from CoinGecko batch API with 7d sparklines and WoW %. Equities still via yfinance/Stooq. The `_TOKEN_TO_CG_ID` mapping was built for the 34-token watchlist.

---

## Known Issues / Things to Watch

- Some CoinGecko ID mappings may be incorrect for newer/smaller tokens (wet, plasma-2, canton-coin, 2z, derive-2, turtle)
- Historical regime chart uses current-snapshot values for: ETF flows, DAT mNAV, leverage (funding/OI/basis), on-chain (MVRV/NUPL), and some macro data (DXY, real yields, AI basket) since these don't have deep history in the system yet. Only BTC trend, market breadth, realised vol, stablecoin supply, and SPY/QQQ trend vary historically.
- `compute_weekly_regime_df()` is slow on first run (many API calls per week); parquet caches help on subsequent runs
- ETF flow data requires manual CSV updates (no free API source wired). Without the CSV, btc_etf_flows and eth_etf_flows return hardcoded fallback values that always score as "Met".
- DAT mNAV requires manual CSV updates (scraping not automated)
- `validate_contracts.py` passes all checks but uses synthetic data, not live API responses
- `logger` is referenced in `app.py` line 515 (`_fetch_coingecko_batch`) but never defined in that file -- will throw NameError on CoinGecko failure. Fix: add `import logging; logger = logging.getLogger(__name__)` to app.py.
- ETF fallback indicators (btc_etf_flows, eth_etf_flows, dat_issuance_capacity, dat_treasury_purchases, strc_credit_stress) return `_votes(choppy=100)` which biases historical classification toward Choppy when no CSVs are present
- BinanceClient caches parquet files in `cache/binance/` keyed by symbol+interval+start+end. Changing the start date creates new cache entries rather than reusing.

---

## How to Run

```bash
cd "Crypto Factor Model"

# Minimal (no CoinGecko, no CryptoQuant):
streamlit run dashboard/app.py

# With CoinGecko Pro key for watchlist pricing:
COINGECKO_API_KEY=CG-xxxxx streamlit run dashboard/app.py

# With CryptoQuant for on-chain metrics:
CRYPTOQUANT_API_KEY=xxxxx COINGECKO_API_KEY=CG-xxxxx streamlit run dashboard/app.py

# Auto-reload on save:
streamlit run dashboard/app.py --server.runOnSave true

# Validate contracts:
cd "Crypto Factor Model"
python3 -m dashboard.validate_contracts
```

Environment variables:
- `COINGECKO_API_KEY` -- CoinGecko API key. Pro keys start with `CG-`. Also configurable in sidebar.
- `CRYPTOQUANT_API_KEY` -- CryptoQuant Bearer token for MVRV/NUPL.
- `WATCHLIST_CSV` -- Override path to watchlist CSV (default: `dashboard/data/watchlist.csv`).

---

## How to Continue Development

### Adding a new indicator

1. Write the function in `regime_indicators.py` (or `regime_etf_dat.py` for ETF/DAT family). Must return the standardized dict with all keys including `regime_votes`.
2. Wire it into `compute_current_regime()` in `data_adapter.py` by appending to the `indicators` list.
3. If it needs new raw data, add a fetcher function to `data_adapter.py`.
4. For historical regime chart support, also add it to `multi_family_indicators()` in `compute_weekly_regime_df()`.
5. Run `python3 -m dashboard.validate_contracts` after changes.

### Adding a new data source

1. Create a client in `crypto_factor_model/clients/` following the pattern of `binance.py` or `cryptoquant.py` (cache support, error handling, logging).
2. Add a fetch wrapper in `data_adapter.py`.
3. Wire the fetched data into the appropriate indicator function.

### Adding a new family

1. Add the family name and weight to `FAMILIES_ORDERED` and `FAMILY_WEIGHTS` in `regime_classifier.py`. Weights must sum to 100.
2. Add the family to `FAMILY_EXPLAINERS` in `app.py` for the UI table.
3. Create indicator functions for the new family.
4. Wire them into `compute_current_regime()`.

### Key caching behavior

- Binance klines: parquet files in `cache/binance/`, keyed by symbol+interval+start+end
- CryptoQuant: parquet files in `cache/cryptoquant/`
- Streamlit: `@st.cache_data` with TTL (5-15 min) for CoinGecko, yfinance, Stooq calls
- To force refresh: delete the cache directory or clear Streamlit cache via the UI

### Config reference

`crypto_factor_model/config.py` contains:
- `CACHE_DIR` = project_root / "cache"
- `SLUG_TO_BINANCE` -- maps chain slugs to Binance pairs (used by `fetch_token_weekly_closes`)
- Universe filters (MIN_MCAP_USD=$100M, MIN_DAILY_VOLUME_USD=$10M, etc.) used by the broader factor model, not directly by the dashboard
