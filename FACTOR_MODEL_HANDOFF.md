# Crypto Factor Model & Screener Handoff

Machine-readable context for a new Claude/Codex session to continue development.

---

## Project Goal

Build a quantitative crypto token screener and factor model. The pipeline ingests chain-level fundamentals, price momentum, and flow data across 29 L1/L2 chains, computes ~29 factor signals, scores them into a composite, and ranks tokens for a concentrated portfolio (top 8).

The user (Asad) is a liquid token trader at a crypto hedge fund. He wants this to function as a production screener with live data, not a research toy.

---

## Architecture

```
Crypto Factor Model/
  crypto_factor_model/
    main.py                 # 11-step pipeline orchestrator
    config.py               # All parameters: universe, weights, thresholds, API keys, metric IDs
    composite.py            # Cross-sectional z-scoring, family weighting, composite blending
    regime.py               # BTC trend + vol target + drawdown overlay on composite scores
    data/
      universe.py           # Universe screening: mcap, volume, age, history filters
    signals/
      fundamentals.py       # 18 fundamental signals from Blockworks chain metrics
      momentum.py           # 7 momentum signals from Binance price data
      flows.py              # 4 flow signals (stablecoins, DEX vol, bridges, OI)
      utils.py              # Winsorisation, rank z-scoring, IC computation
    clients/
      blockworks.py         # Blockworks Research API (chain + project metrics)
      binance.py            # Binance public API (OHLCV, funding, OI)
      defillama.py          # DefiLlama bridge flows
      cryptoquant.py        # CryptoQuant on-chain (MVRV, NUPL) -- not yet used in main pipeline
  cache/                    # Parquet cache for all API responses
  output/                   # Pipeline outputs (rankings, composites, diagnostics)
```

---

## Pipeline Steps (main.py)

1. Init clients (Blockworks, Binance, DefiLlama)
2. Define chain universe (29 chains from config.BW_CHAIN_SLUGS)
3. Fetch Blockworks chain data: 13 core metrics x 29 chains as date-indexed DataFrames
4. Fetch BTC daily close from Binance
5. Derive market cap panel (price * supply)
6. Fetch Binance daily closes for 18 chains with tickers (SLUG_TO_BINANCE)
7. Universe screening per rebalance date (monthly): mcap >= $100M, volume >= $10M, age >= 30d, history >= 13w
8. Compute 3 signal families: fundamentals (18), momentum (7), flows (4)
9. Signal diagnostics: rank IC vs 30d forward returns, IC IR, pairwise correlation matrix
10. Regime overlay: multiply composite by (btc_trend * vol_scalar * dd_scalar)
11. Rank and select top 8

---

## Data Sources

| Source | Auth | What's Fetched | Used In |
|--------|------|----------------|---------|
| Blockworks Research API | `BLOCKWORKS_API_KEY` env var, `x-api-key` header | 31 chain-level metrics (revenue, fees, active addresses, staking, DEX vol, stablecoin supply, lending, issuance, burn, etc.) across 29 chains | fundamentals.py, flows.py |
| Binance Public API | None | Daily OHLCV for 18 chains, BTC reference price | momentum.py |
| Binance Futures API | None (public) | Funding rates (`/fapi/v1/fundingRate`), open interest (`/fapi/v1/openInterest`) | flows.py (oi_change), regime dashboard |
| DefiLlama Bridges | None | Bridge deposits/withdrawals per chain (`bridges.llama.fi/transactions/{chain}`) | flows.py (bridge_net_flow) |
| CryptoQuant | `CRYPTOQUANT_API_KEY`, Bearer token | MVRV ratio, NUPL | Client built but NOT integrated into main pipeline yet |
| CoinGecko | `COINGECKO_API_KEY` (Pro: CG-...) | Token prices, 7d sparklines, WoW % for watchlist | Dashboard watchlist only |

### Blockworks Metric IDs (config.py: BW_CHAIN_METRICS)

```
price: token-price-usd          revenue: rev-usd
supply: token-supply             trading_fees: transaction-fee-total-usd
fdv: token-fdv-usd              base_fees: transaction-base-fee-total-usd
active_addresses: active-address-total   priority_fees: transaction-priority-fee-total-usd
issuance: issuance-usd          other_fees: transaction-other-fee-total-usd
burn: burn-usd                  mev_tips: mev-tips-fees-usd
staked_supply: token-supply-staked       dex_volume: dex-spot-volume-total-usd
stablecoin_supply: stablecoin-supply-total-usd
lending_tvl: lending-tvl-total-usd       lending_deposits: lending-deposit-total-usd
lending_borrows: lending-borrow-total-usd lending_revenue: lending-revenue-total-usd
app_revenue: app-revenue-total-usd       staking_rate: staking-rate
```

Also has project-level metrics (BW_PROJECT_METRICS) for 136 dApps: revenue-total-usd, dex-revenue-usd, dex-spot-volume-usd, dex-fees-usd, dex-memecoin-volume-share.

### Chain Universe (29)

aptos, arbitrum, avalanche, base, berachain, bitcoin, bnb, bob, boba, celestia, ethereum, fogo, hyperevm, ink, katana, megaeth, mode, monad, optimism, plasma, polygon, shape, solana, superseed, tron, unichain, worldchain, zksync, zora

18 of these have Binance tickers for price/momentum data. See SLUG_TO_BINANCE in config.py.

---

## Signal Inventory

### Fundamentals (18 signals, 55% weight)

| Signal | Formula | Blockworks Metric |
|--------|---------|-------------------|
| revenue | 7d rolling mean of daily revenue | rev-usd |
| revenue_growth_2w | pct_change(7d-smoothed revenue, 14d lag) | rev-usd |
| revenue_growth_4w | pct_change(7d-smoothed revenue, 28d lag) | rev-usd |
| revenue_stability | 1 / std(13w weekly revenue growth) | rev-usd |
| revenue_btc_corr | -rolling_corr(revenue, BTC, 91d) | rev-usd |
| inverted_active_rev_share | -(passive_rev / total_rev) | derived |
| gross_profit | 7d rolling mean | derived |
| protocol_margin | revenue - issuance | rev-usd, issuance-usd |
| trading_fees | 7d rolling mean | transaction-fee-total-usd |
| mc_fees_mean_reversion | -z(mcap / annualized_fees, 182d) | transaction-fee-total-usd |
| fdv_revenue | -(FDV / annualized_revenue) | token-fdv-usd, rev-usd |
| payback_period | -(mcap / annualized_revenue) | token-price-usd * token-supply, rev-usd |
| implied_growth_rate | -(multiple / cross_sectional_median) | derived |
| dau_growth_2w | pct_change(7d-smoothed DAU, 14d) | active-address-total |
| dau_growth_4w | pct_change(7d-smoothed DAU, 28d) | active-address-total |
| growth_composite | z(rev_growth) + z(dau_growth) | combined |
| revenue_per_address | smoothed_revenue / smoothed_DAU | rev-usd, active-address-total |
| buyback_vs_issuance | burn / issuance (>1 = deflationary) | burn-usd, issuance-usd |
| staked_supply_ratio | staked / total supply | token-supply-staked, token-supply |

### Momentum (7 signals, 17.5% weight)

| Signal | Formula | Source |
|--------|---------|--------|
| vol_adj_momentum_3w | 3w return / annualized vol | Binance daily |
| rel_strength_vs_btc | token 3w return - BTC 3w return | Binance daily |
| drawdown_from_ath | (price - ATH) / ATH | Binance daily |
| momentum_breadth | % of trailing 13w with positive weekly returns | Binance daily |
| short_term_reversal | -1w return (mean reversion) | Binance daily |
| realised_vol_rank | -annualized vol (28d) | Binance daily |
| size_signal | -mcap (small cap premium) | Derived |

### Flows (4 signals, 12.5% weight)

| Signal | Formula | Source |
|--------|---------|--------|
| stablecoin_supply_growth | pct_change(7d-smoothed supply, 14d) | Blockworks stablecoin-supply-total-usd |
| dex_volume_growth | pct_change(7d-smoothed DEX vol, 14d) | Blockworks dex-spot-volume-total-usd |
| bridge_net_flow | rolling_mean(deposits - withdrawals, 14d) | DefiLlama bridges |
| oi_change | pct_change(OI, 7d) | Binance Futures |

### Team (15% weight) -- manual input, no computed signals yet

---

## Signal Processing Pipeline

```
Raw metric panel (dates x tokens)
  -> Winsorise at 1st/99th percentile (cross-sectional per date)
  -> Rank z-score (per date: rank 1..N, then to z(0,1))
  -> Equal-weight within family
  -> Blend families: fundamentals 55%, momentum 17.5%, flows 12.5%, team 15%
  -> Regime overlay: composite *= (btc_trend * vol_scalar * dd_scalar)
  -> Rank & select top 8
```

---

## Universe Screening (config.py)

- MIN_MCAP_USD: $100M
- MIN_DAILY_VOLUME_USD: $10M (placeholder -- volume panel not fully wired)
- MIN_LISTING_AGE_DAYS: 30
- MIN_HISTORY_WEEKS: 13
- EXCLUDE_CATEGORIES: stablecoin, wrapped, pegged, tokenised-treasury, basket, infrastructure-aggregate

---

## Portfolio Constraints

- MAX_POSITIONS: 8
- MIN_POSITIONS: 3
- MAX_SINGLE_NAME_PCT: 15%
- MAX_CHAIN_NARRATIVE_PCT: 40%
- MAX_SECTOR_PCT: 40%

---

## Known Gaps / Issues

1. **Volume data is a placeholder** (main.py line ~183): `volume = price_panel * 0`. Daily volume from Binance is collected but not wired into universe screening.
2. **Bridge flows not called in main pipeline**: DefiLlama client exists but `fetch_chain_data()` doesn't invoke it. Flow signals only computed if `bridge_flows` key present.
3. **CryptoQuant not integrated into main pipeline**: Client built, MVRV/NUPL available, but never instantiated in main.py.
4. **Team score is manual**: 15% family weight with no automated signal. Requires manual input.
5. **Signal diagnostics fixed at 30d forward returns**: Only one horizon tested. Should add 7d, 14d, 60d.
6. **Backtest engine not built**: Config has TX_COST_BPS and sensitivity params, but no walk-forward backtest exists yet.
7. **Signal redundancy flag**: revenue x trading_fees correlation = 1.000 in some chains (same underlying metric). MAX_PAIRWISE_CORR = 0.12 threshold exists but dedup logic may not be applied.
8. **No project-level screener**: BW_PROJECT_METRICS defined for 136 dApps but pipeline only runs chain-level. Could expand to screen individual protocols.

---

## Screener Enhancements Requested by User

The user wants the following additions to the screener:

1. **Change in revenue** (WoW, MoM): already partially covered by revenue_growth_2w / revenue_growth_4w but user wants explicit WoW and MoM deltas displayed
2. **Change in fees** (WoW, MoM): trading_fees exists as a signal but no explicit change metric surfaced
3. **Change in deposits** (lending deposits): Blockworks has `lending-deposit-total-usd`, not currently used as a change signal
4. **Change in OI**: oi_change signal exists in flows.py but uses Binance Futures only. Could expand to more venues.
5. **Top movers by market cap bucket**: User wants tokens bucketed by mcap/FDV:
   - < $100M
   - $100M - $500M
   - $500M - $1B
   - $1B - $5B
   - $5B+
   Then show top movers (by price change, composite score change, or factor improvement) within each bucket.

These are NET NEW features to build on top of the existing pipeline.

---

## How to Run

```bash
cd "Crypto Factor Model"

# Set API keys
export BLOCKWORKS_API_KEY="your-key"
export CRYPTOQUANT_API_KEY="your-key"       # optional
export COINGECKO_API_KEY="CG-..."           # for watchlist pricing

# Run the factor pipeline
python3 -m crypto_factor_model.main

# Run the regime dashboard
streamlit run dashboard/app.py
```

---

## Key Files to Read First

To understand the system, read in this order:
1. `crypto_factor_model/config.py` -- all parameters and metric IDs
2. `crypto_factor_model/main.py` -- pipeline flow
3. `crypto_factor_model/signals/fundamentals.py` -- largest signal family
4. `crypto_factor_model/signals/momentum.py`
5. `crypto_factor_model/signals/flows.py`
6. `crypto_factor_model/signals/utils.py` -- signal processing mechanics
7. `crypto_factor_model/composite.py` -- scoring and blending
8. `crypto_factor_model/clients/blockworks.py` -- primary data source
9. `crypto_factor_model/clients/binance.py` -- price data
