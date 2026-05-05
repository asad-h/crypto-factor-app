"""
Fundamentals signal computation.

Takes raw Blockworks data (revenue, fees, activity, supply) and computes
the fundamentals signal panel for all tokens at each rebalance date.

Each function returns a DataFrame: rows = rebalance dates, columns = tokens.
"""
import numpy as np
import pandas as pd

from crypto_factor_model.signals.utils import winsorise


def revenue(rev_panel: pd.DataFrame) -> pd.DataFrame:
    """Total protocol revenue. Trailing 7d average to smooth daily noise."""
    return rev_panel.rolling(7, min_periods=3).mean()


def revenue_growth(rev_panel: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Revenue growth over trailing window (default 14 days = 2 weeks).
    Percentage change of trailing-7d-average revenue.
    """
    smoothed = rev_panel.rolling(7, min_periods=3).mean()
    return smoothed.pct_change(periods=window)


def revenue_growth_4w(rev_panel: pd.DataFrame) -> pd.DataFrame:
    """Revenue growth over 4 weeks (28 days)."""
    return revenue_growth(rev_panel, window=28)


def revenue_stability(rev_panel: pd.DataFrame, window: int = 91) -> pd.DataFrame:
    """
    Inverse volatility of revenue growth over trailing 13 weeks.
    Higher = more stable = better.
    """
    growth = revenue_growth(rev_panel, window=7)  # weekly growth
    vol = growth.rolling(window, min_periods=30).std()
    # Invert: lower vol -> higher score. Add small epsilon to avoid division by zero.
    return 1.0 / (vol + 1e-8)


def revenue_btc_correlation(
    rev_panel: pd.DataFrame,
    btc_price: pd.Series,
    window: int = 91,
) -> pd.DataFrame:
    """
    Rolling correlation between revenue and BTC price.
    Lower correlation = more defensible = better.
    Returned inverted so higher = better.
    """
    corrs = {}
    for token in rev_panel.columns:
        aligned = pd.DataFrame({
            "rev": rev_panel[token],
            "btc": btc_price,
        }).dropna()
        if len(aligned) < window:
            corrs[token] = pd.Series(dtype=float)
            continue
        corrs[token] = aligned["rev"].rolling(window).corr(aligned["btc"])

    corr_panel = pd.DataFrame(corrs)
    return -corr_panel  # invert: lower corr = higher score


def inverted_active_revenue_share(
    rev_panel: pd.DataFrame,
    passive_rev_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Passive revenue (MEV, liquidations, priority fees) as share of total.
    Higher passive share = worse quality. Returned inverted so higher = better.

    passive_rev_panel = mev_tips + priority_fees + other_fees
    """
    ratio = passive_rev_panel / (rev_panel + 1e-8)
    return -ratio  # invert


def gross_profit(gross_profit_panel: pd.DataFrame) -> pd.DataFrame:
    """Gross profit, smoothed 7d."""
    return gross_profit_panel.rolling(7, min_periods=3).mean()


def protocol_margin(
    rev_panel: pd.DataFrame,
    issuance_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Revenue minus issuance. The real P&L line."""
    return rev_panel - issuance_panel


def mc_fees_mean_reversion(
    mcap_panel: pd.DataFrame,
    fees_panel: pd.DataFrame,
    lookback: int = 182,
) -> pd.DataFrame:
    """
    Z-score of MC/Fees ratio vs its own trailing history.
    Negative z-score = cheap relative to own history = better.
    Returned inverted so higher = better.
    """
    ratio = mcap_panel / (fees_panel.rolling(7).mean() * 365 + 1e-8)  # annualised fees
    rolling_mean = ratio.rolling(lookback, min_periods=60).mean()
    rolling_std = ratio.rolling(lookback, min_periods=60).std()
    z = (ratio - rolling_mean) / (rolling_std + 1e-8)
    return -z  # lower z (cheaper) = higher score


def fdv_revenue(
    fdv_panel: pd.DataFrame,
    rev_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    FDV / annualised revenue. Lower = cheaper.
    Returned inverted so higher = better.
    """
    annual_rev = rev_panel.rolling(7).mean() * 365
    ratio = fdv_panel / (annual_rev + 1e-8)
    return -ratio


def payback_period(
    mcap_panel: pd.DataFrame,
    rev_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Market cap / annualised revenue.
    Years for cumulative revenue to repay current market cap.
    Lower = better. Returned inverted.
    """
    annual_rev = rev_panel.rolling(7).mean() * 365
    years = mcap_panel / (annual_rev + 1e-8)
    return -years  # lower payback = higher score


def implied_growth_rate(
    mcap_panel: pd.DataFrame,
    rev_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    How far a token's MC/Revenue multiple exceeds its cross-sectional sector median.
    High implied growth = market expects a lot = riskier.
    Returned inverted so lower implied growth (more reasonable) = higher score.
    """
    annual_rev = rev_panel.rolling(7).mean() * 365
    multiple = mcap_panel / (annual_rev + 1e-8)
    # Cross-sectional median at each date
    sector_median = multiple.median(axis=1)
    # Ratio to median: >1 means market prices in above-median growth
    implied = multiple.div(sector_median, axis=0)
    return -implied  # lower implied growth = higher score


def dau_growth(active_addr_panel: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """DAU growth over trailing window."""
    smoothed = active_addr_panel.rolling(7, min_periods=3).mean()
    return smoothed.pct_change(periods=window)


def dau_growth_4w(active_addr_panel: pd.DataFrame) -> pd.DataFrame:
    """DAU growth over 4 weeks."""
    return dau_growth(active_addr_panel, window=28)


def growth_composite(
    rev_panel: pd.DataFrame,
    active_addr_panel: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """
    Combined fee growth + user growth signal.
    Growth as a distinct factor dimension.
    z(revenue_growth) + z(dau_growth), then re-ranked.
    """
    from crypto_factor_model.signals.utils import rank_zscore_panel

    rev_g = revenue_growth(rev_panel, window=window)
    dau_g = dau_growth(active_addr_panel, window=window)

    rev_z = rank_zscore_panel(rev_g, winsorise_first=True)
    dau_z = rank_zscore_panel(dau_g, winsorise_first=True)

    return rev_z + dau_z


def revenue_per_address(
    rev_panel: pd.DataFrame,
    active_addr_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Revenue per active address. Unit economics."""
    smoothed_rev = rev_panel.rolling(7, min_periods=3).mean()
    smoothed_addr = active_addr_panel.rolling(7, min_periods=3).mean()
    return smoothed_rev / (smoothed_addr + 1e-8)


def token_buyback_vs_issuance(
    burn_panel: pd.DataFrame,
    issuance_panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Burns / Issuance ratio. >1 = net deflationary.
    Higher = better.
    """
    return burn_panel / (issuance_panel + 1e-8)


def staked_supply_ratio(
    staked_panel: pd.DataFrame,
    supply_panel: pd.DataFrame,
) -> pd.DataFrame:
    """% of supply locked in staking."""
    return staked_panel / (supply_panel + 1e-8)


# ── Aggregate: compute all fundamentals signals ────────────────────────

def compute_all_fundamentals(
    data: dict[str, pd.DataFrame],
    btc_price: pd.Series,
) -> dict[str, pd.DataFrame]:
    """
    Compute all fundamentals signals from raw data panels.

    Args:
        data: dict of raw data panels from Blockworks. Expected keys:
            "revenue", "mcap", "fdv", "gross_profit", "trading_fees",
            "active_addresses", "issuance", "burn", "staked_supply",
            "supply", "mev_tips", "priority_fees", "other_fees"
        btc_price: BTC daily close series

    Returns:
        dict mapping signal_name -> DataFrame (dates x tokens)
    """
    signals = {}

    rev = data.get("revenue", pd.DataFrame())
    mcap = data.get("mcap", pd.DataFrame())
    fdv = data.get("fdv", pd.DataFrame())
    gp = data.get("gross_profit", pd.DataFrame())
    fees = data.get("trading_fees", pd.DataFrame())
    addr = data.get("active_addresses", pd.DataFrame())
    iss = data.get("issuance", pd.DataFrame())
    burn = data.get("burn", pd.DataFrame())
    staked = data.get("staked_supply", pd.DataFrame())
    supply = data.get("supply", pd.DataFrame())

    # Passive revenue = MEV + priority + other fees
    passive = (
        data.get("mev_tips", pd.DataFrame()).fillna(0)
        + data.get("priority_fees", pd.DataFrame()).fillna(0)
        + data.get("other_fees", pd.DataFrame()).fillna(0)
    )

    if not rev.empty:
        signals["revenue"] = revenue(rev)
        signals["revenue_growth_2w"] = revenue_growth(rev, window=14)
        signals["revenue_growth_4w"] = revenue_growth_4w(rev)
        signals["revenue_stability"] = revenue_stability(rev)

    if not rev.empty and len(btc_price) > 0:
        signals["revenue_btc_corr"] = revenue_btc_correlation(rev, btc_price)

    if not rev.empty and not passive.empty:
        signals["inverted_active_rev_share"] = inverted_active_revenue_share(rev, passive)

    if not gp.empty:
        signals["gross_profit"] = gross_profit(gp)

    if not rev.empty and not iss.empty:
        signals["protocol_margin"] = protocol_margin(rev, iss)

    if not fees.empty:
        signals["trading_fees"] = fees.rolling(7, min_periods=3).mean()

    if not mcap.empty and not fees.empty:
        signals["mc_fees_mean_reversion"] = mc_fees_mean_reversion(mcap, fees)

    if not fdv.empty and not rev.empty:
        signals["fdv_revenue"] = fdv_revenue(fdv, rev)

    if not mcap.empty and not rev.empty:
        signals["payback_period"] = payback_period(mcap, rev)
        signals["implied_growth_rate"] = implied_growth_rate(mcap, rev)

    if not addr.empty:
        signals["dau_growth_2w"] = dau_growth(addr, window=14)
        signals["dau_growth_4w"] = dau_growth_4w(addr)

    if not rev.empty and not addr.empty:
        signals["growth_composite"] = growth_composite(rev, addr)
        signals["revenue_per_address"] = revenue_per_address(rev, addr)

    if not burn.empty and not iss.empty:
        signals["buyback_vs_issuance"] = token_buyback_vs_issuance(burn, iss)

    if not staked.empty and not supply.empty:
        signals["staked_supply_ratio"] = staked_supply_ratio(staked, supply)

    return signals
