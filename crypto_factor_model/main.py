"""
Main orchestration script.

Ties together data ingestion, universe screening, signal computation,
composite scoring, and outputs ranked token lists.

Usage:
    python -m crypto_factor_model.main

Flow:
    1. Fetch raw data from Blockworks + Binance
    2. Screen universe at each rebalance date
    3. Compute signals per family
    4. Z-score and combine into composite
    5. Rank and select top positions
    6. Output to CSV / parquet
"""
import logging
from pathlib import Path

import pandas as pd

from crypto_factor_model.config import (
    BACKTEST_START,
    CACHE_DIR,
    BW_CHAIN_METRICS,
    BW_CHAIN_SLUGS,
    SLUG_TO_BINANCE,
    FAMILY_WEIGHTS,
    MAX_POSITIONS,
)
from crypto_factor_model.clients.blockworks import BlockworksClient
from crypto_factor_model.clients.binance import BinanceClient
from crypto_factor_model.clients.defillama import DefiLlamaClient
from crypto_factor_model.data.universe import screen_universe, build_rebalance_dates
from crypto_factor_model.signals.fundamentals import compute_all_fundamentals
from crypto_factor_model.signals.momentum import compute_all_momentum
from crypto_factor_model.signals.flows import compute_all_flows
from crypto_factor_model.signals.utils import signal_diagnostics, pairwise_signal_correlation
from crypto_factor_model.composite import compute_family_score, compute_composite, rank_and_select
from crypto_factor_model.regime import RegimeOverlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def fetch_chain_data(
    bw: BlockworksClient,
    chains: list[str],
    start: str = BACKTEST_START,
) -> dict[str, pd.DataFrame]:
    """
    Fetch chain-level Blockworks metrics for all chains.
    Returns dict of metric_key -> DataFrame (dates x chains).
    """
    # Core chain metrics for the factor model
    metric_map = {
        "price": BW_CHAIN_METRICS["price"],
        "supply": BW_CHAIN_METRICS["supply"],
        "fdv": BW_CHAIN_METRICS["fdv"],
        "revenue": BW_CHAIN_METRICS["revenue"],
        "trading_fees": BW_CHAIN_METRICS["trading_fees"],
        "active_addresses": BW_CHAIN_METRICS["active_addresses"],
        "issuance": BW_CHAIN_METRICS["issuance"],
        "burn": BW_CHAIN_METRICS["burn"],
        "staked_supply": BW_CHAIN_METRICS["staked_supply"],
        "mev_tips": BW_CHAIN_METRICS["mev_tips"],
        "priority_fees": BW_CHAIN_METRICS["priority_fees"],
        "other_fees": BW_CHAIN_METRICS["other_fees"],
        "stablecoin_supply": BW_CHAIN_METRICS["stablecoin_supply"],
    }

    # Optional metrics (not all chains have these)
    optional = {
        "app_revenue": BW_CHAIN_METRICS.get("app_revenue"),
        "dex_volume": BW_CHAIN_METRICS.get("dex_volume"),
        "staking_rate": BW_CHAIN_METRICS.get("staking_rate"),
        "lending_tvl": BW_CHAIN_METRICS.get("lending_tvl"),
        "lending_revenue": BW_CHAIN_METRICS.get("lending_revenue"),
    }

    data = {}
    for key, bw_metric in metric_map.items():
        logger.info(f"Fetching {key} ({bw_metric}) for {len(chains)} chains...")
        try:
            df = bw.get_bulk_metric(bw_metric, chains, start=start)
            data[key] = df
            logger.info(f"  -> {df.shape[0]} dates, {df.shape[1]} chains with data")
        except Exception as e:
            logger.warning(f"  -> Failed: {e}")
            data[key] = pd.DataFrame()

    for key, bw_metric in optional.items():
        if bw_metric:
            logger.info(f"Fetching optional {key} ({bw_metric})...")
            try:
                df = bw.get_bulk_metric(bw_metric, chains, start=start)
                data[key] = df
                logger.info(f"  -> {df.shape[0]} dates, {df.shape[1]} chains with data")
            except Exception as e:
                logger.debug(f"  -> Optional metric {key} not available: {e}")
                data[key] = pd.DataFrame()

    return data


def fetch_binance_prices(
    bn: BinanceClient,
    chains: list[str],
    start: str = BACKTEST_START,
) -> pd.DataFrame:
    """
    Fetch daily close prices for chains using the slug-to-Binance mapping.
    Chains without a Binance ticker are skipped.
    Returns DataFrame: DatetimeIndex, columns = chain slugs.
    """
    symbol_to_slug = {}
    for slug in chains:
        ticker = SLUG_TO_BINANCE.get(slug)
        if ticker:
            symbol_to_slug[ticker] = slug

    if not symbol_to_slug:
        logger.warning("No Binance tickers found for the given chains")
        return pd.DataFrame()

    logger.info(f"Fetching Binance prices for {len(symbol_to_slug)} chains "
                f"(skipped {len(chains) - len(symbol_to_slug)} without tickers)")

    df = bn.get_multiple_daily(list(symbol_to_slug.keys()), start=start)
    df.columns = [symbol_to_slug.get(c, c) for c in df.columns]
    return df


def run_pipeline():
    """Execute the full pipeline."""
    logger.info("=" * 60)
    logger.info("CRYPTO FACTOR MODEL PIPELINE")
    logger.info("=" * 60)

    # ── 1. Initialise clients ──────────────────────────────────
    bw = BlockworksClient()
    bn = BinanceClient()
    dl = DefiLlamaClient()

    # ── 2. Define chain universe ──────────────────────────────
    chains = BW_CHAIN_SLUGS.copy()
    logger.info(f"Chain universe: {len(chains)} chains")

    # ── 3. Fetch raw data ──────────────────────────────────────
    logger.info("Fetching Blockworks chain-level data...")
    bw_data = fetch_chain_data(bw, chains)

    logger.info("Fetching BTC price...")
    btc_price = bn.get_btc_price(start=BACKTEST_START)

    # Compute mcap from price * supply if not directly available
    price_ts = bw_data.get("price", pd.DataFrame())
    supply_ts = bw_data.get("supply", pd.DataFrame())
    if not price_ts.empty and not supply_ts.empty:
        mcap = (price_ts * supply_ts).dropna(how="all", axis=1)
        bw_data["mcap"] = mcap
        logger.info(f"Computed mcap for {mcap.shape[1]} chains from price * supply")
    else:
        mcap = pd.DataFrame()
        bw_data["mcap"] = mcap

    # Fetch Binance prices for chains with any data
    chains_with_data = list(bw_data.get("revenue", pd.DataFrame()).columns)
    if not chains_with_data:
        chains_with_data = chains
    logger.info(f"Fetching Binance prices for {len(chains_with_data)} chains...")
    price_panel = fetch_binance_prices(bn, chains_with_data)

    # ── 4. Universe screening ──────────────────────────────────
    logger.info("Building rebalance dates...")
    rebalance_dates = build_rebalance_dates(BACKTEST_START)

    mcap = bw_data.get("mcap", pd.DataFrame())
    volume = price_panel * 0  # placeholder; replace with actual volume data
    # Use quote_volume from Binance if available, or Blockworks DEX volume
    first_seen = mcap.apply(lambda col: col.first_valid_index())

    eligible_by_date = {}
    for dt in rebalance_dates:
        if dt > mcap.index.max():
            break
        eligible = screen_universe(mcap, mcap, first_seen, as_of=dt)  # volume placeholder
        eligible_by_date[dt] = eligible

    # ── 5. Compute signals ─────────────────────────────────────
    logger.info("Computing fundamentals signals...")
    fund_signals = compute_all_fundamentals(bw_data, btc_price)
    logger.info(f"  -> {len(fund_signals)} fundamentals signals computed")

    logger.info("Computing momentum signals...")
    mom_signals = compute_all_momentum(price_panel, btc_price, mcap)
    logger.info(f"  -> {len(mom_signals)} momentum signals computed")

    logger.info("Computing flow signals...")
    flow_signals = compute_all_flows(bw_data)
    logger.info(f"  -> {len(flow_signals)} flow signals computed")

    # ── 6. Signal diagnostics ──────────────────────────────────
    # Compute forward returns for IC calculation
    fwd_ret = price_panel.pct_change(periods=30).shift(-30)  # 30-day forward return

    logger.info("\nSignal diagnostics (fundamentals):")
    for name, panel in fund_signals.items():
        diag = signal_diagnostics(panel, fwd_ret)
        logger.info(f"  {name:35s} IC={diag['mean_ic']:+.4f}  "
                     f"IR={diag['ic_ir']:+.4f}  "
                     f"pos%={diag['pct_positive']:.1%}  "
                     f"n={diag['n_periods']}")

    logger.info("\nSignal diagnostics (momentum):")
    for name, panel in mom_signals.items():
        diag = signal_diagnostics(panel, fwd_ret)
        logger.info(f"  {name:35s} IC={diag['mean_ic']:+.4f}  "
                     f"IR={diag['ic_ir']:+.4f}  "
                     f"pos%={diag['pct_positive']:.1%}  "
                     f"n={diag['n_periods']}")

    # ── 7. Pairwise correlation check ──────────────────────────
    all_signals = {**fund_signals, **mom_signals, **flow_signals}
    logger.info("\nComputing pairwise signal correlations...")
    corr_matrix = pairwise_signal_correlation(all_signals)
    corr_matrix.to_csv(OUTPUT_DIR / "signal_correlations.csv")
    logger.info(f"  -> Saved to {OUTPUT_DIR / 'signal_correlations.csv'}")

    # Flag high correlations
    high_corr = []
    for i, s1 in enumerate(corr_matrix.columns):
        for j, s2 in enumerate(corr_matrix.columns):
            if i < j and abs(corr_matrix.loc[s1, s2]) > 0.12:
                high_corr.append((s1, s2, corr_matrix.loc[s1, s2]))
    if high_corr:
        logger.warning(f"  {len(high_corr)} signal pairs above 0.12 correlation threshold:")
        for s1, s2, c in sorted(high_corr, key=lambda x: -abs(x[2]))[:10]:
            logger.warning(f"    {s1} x {s2} = {c:.3f}")

    # ── 8. Composite scoring ───────────────────────────────────
    logger.info("\nComputing family scores...")
    family_scores = {
        "fundamentals": compute_family_score(fund_signals),
        "momentum": compute_family_score(mom_signals),
        "flows": compute_family_score(flow_signals),
        # "team" scores would be added manually or from a separate input
    }

    logger.info("Computing composite scores...")
    composite = compute_composite(family_scores)

    # ── 9. Regime overlay ─────────────────────────────────────
    logger.info("\nApplying regime overlay...")
    overlay = RegimeOverlay(
        btc_price=btc_price,
        portfolio_returns=None,  # uses BTC returns as proxy until backtest exists
        portfolio_nav=None,      # uses normalised BTC price as drawdown proxy
        use_binary_trend=False,
    )

    regime_summary = overlay.summary()
    logger.info(f"  BTC trend (current):   {regime_summary['current_btc_trend']:.3f}")
    logger.info(f"  Vol scalar (current):  {regime_summary['current_vol_scalar']:.3f}")
    logger.info(f"  DD scalar (current):   {regime_summary['current_dd_scalar']:.3f}")
    logger.info(f"  Combined exposure:     {regime_summary['current_exposure']:.3f}")
    logger.info(f"  Median exposure:       {regime_summary['median_exposure']:.3f}")
    logger.info(f"  Days below 1.0:        {regime_summary['pct_below_1']:.1%}")

    # Save regime diagnostics
    regime_diag = overlay.diagnostics()
    regime_diag.to_csv(OUTPUT_DIR / "regime_diagnostics.csv")
    logger.info(f"  -> Saved to {OUTPUT_DIR / 'regime_diagnostics.csv'}")

    # Apply overlay to composite scores
    composite_adjusted = overlay.apply_to_composite(composite)

    # ── 10. Rank and select ────────────────────────────────────
    logger.info("Ranking tokens...")
    rankings = rank_and_select(composite_adjusted, n_positions=MAX_POSITIONS)
    rankings.to_csv(OUTPUT_DIR / "rankings.csv", index=False)
    logger.info(f"  -> Saved to {OUTPUT_DIR / 'rankings.csv'}")

    # Also save raw (unadjusted) rankings for comparison
    rankings_raw = rank_and_select(composite, n_positions=MAX_POSITIONS)
    rankings_raw.to_csv(OUTPUT_DIR / "rankings_raw.csv", index=False)

    # Output selected positions at each rebalance
    selected = rankings[rankings["selected"]]
    logger.info(f"\n{'='*60}")
    logger.info("SELECTED POSITIONS BY DATE (REGIME-ADJUSTED)")
    logger.info(f"{'='*60}")
    for date in selected["date"].unique():
        positions = selected[selected["date"] == date].sort_values("rank")
        tokens = ", ".join(f"{r['token']}({r['composite_score']:.2f})" for _, r in positions.iterrows())
        logger.info(f"  {date.date()}: {tokens}")

    # ── 11. Save composite panels ──────────────────────────────
    composite.to_parquet(OUTPUT_DIR / "composite_scores.parquet")
    composite.to_csv(OUTPUT_DIR / "composite_scores.csv")
    composite_adjusted.to_parquet(OUTPUT_DIR / "composite_adjusted.parquet")
    composite_adjusted.to_csv(OUTPUT_DIR / "composite_adjusted.csv")
    logger.info(f"\nComposite scores saved to {OUTPUT_DIR}")

    logger.info("\nPipeline complete.")
    return composite_adjusted, rankings


if __name__ == "__main__":
    run_pipeline()
