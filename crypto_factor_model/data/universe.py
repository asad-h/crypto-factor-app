"""
Universe screening and eligibility.

Filters the raw protocol list down to eligible tokens
based on market cap, volume, listing age, and category exclusions.
"""
import logging
from datetime import timedelta

import pandas as pd

from crypto_factor_model.config import (
    MIN_MCAP_USD,
    MIN_DAILY_VOLUME_USD,
    MIN_LISTING_AGE_DAYS,
    MIN_HISTORY_WEEKS,
    EXCLUDE_CATEGORIES,
)

logger = logging.getLogger(__name__)


def screen_universe(
    mcap: pd.DataFrame,
    volume: pd.DataFrame,
    first_seen: pd.Series,
    categories: pd.Series | None = None,
    as_of: pd.Timestamp | None = None,
) -> list[str]:
    """
    Screen tokens for eligibility at a given date.

    Args:
        mcap: DataFrame, rows=dates, columns=tokens, values=market cap USD
        volume: DataFrame, rows=dates, columns=tokens, values=daily volume USD
        first_seen: Series, index=token, values=first date with data
        categories: Series, index=token, values=category string (optional)
        as_of: date to screen at (defaults to latest available)

    Returns:
        List of eligible token identifiers.
    """
    if as_of is None:
        as_of = mcap.index.max()

    eligible = set(mcap.columns) & set(volume.columns)
    results = []

    for token in eligible:
        # ── Market cap check ──
        if token not in mcap.columns:
            continue
        latest_mcap = mcap.loc[:as_of, token].dropna()
        if latest_mcap.empty or latest_mcap.iloc[-1] < MIN_MCAP_USD:
            continue

        # ── Volume check (trailing 7d average) ──
        if token not in volume.columns:
            continue
        recent_vol = volume.loc[:as_of, token].dropna().tail(7)
        if recent_vol.empty or recent_vol.mean() < MIN_DAILY_VOLUME_USD:
            continue

        # ── Listing age check ──
        if token in first_seen.index:
            token_start = pd.Timestamp(first_seen[token])
            if (as_of - token_start).days < MIN_LISTING_AGE_DAYS:
                continue
        else:
            continue  # no history = skip

        # ── History length check ──
        history_days = len(mcap.loc[:as_of, token].dropna())
        if history_days < MIN_HISTORY_WEEKS * 7:
            continue

        # ── Category exclusion ──
        if categories is not None and token in categories.index:
            cat = str(categories[token]).lower()
            if any(exc in cat for exc in EXCLUDE_CATEGORIES):
                continue

        results.append(token)

    logger.info(f"Universe screen at {as_of.date()}: {len(results)} eligible from {len(eligible)} total")
    return sorted(results)


def build_rebalance_dates(
    start: str,
    end: str | None = None,
    freq: str = "M",
) -> pd.DatetimeIndex:
    """
    Generate rebalance dates.

    Args:
        start: ISO date string
        end: ISO date string (defaults to today)
        freq: "M" for month-end, "W" for weekly

    Returns:
        DatetimeIndex of rebalance dates.
    """
    end = end or pd.Timestamp.now().strftime("%Y-%m-%d")
    if freq == "M":
        return pd.date_range(start, end, freq="ME")
    elif freq == "W":
        return pd.date_range(start, end, freq="W-FRI")
    else:
        return pd.date_range(start, end, freq=freq)
