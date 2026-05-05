"""
Flow signal computation.

Captures capital movement, liquidity, and supply dynamics.
"""
import numpy as np
import pandas as pd


def stablecoin_supply_growth(
    stablecoin_panel: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """
    Stablecoin supply growth per chain/ecosystem.
    Liquidity proxy and demand signal.
    """
    smoothed = stablecoin_panel.rolling(7, min_periods=3).mean()
    return smoothed.pct_change(periods=window)


def dex_volume_signal(
    volume_panel: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """
    DEX spot volume growth. Fee generation and activity proxy.
    """
    smoothed = volume_panel.rolling(7, min_periods=3).mean()
    return smoothed.pct_change(periods=window)


def bridge_net_flow(
    bridge_flow_panel: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """
    Trailing net bridge flow (smoothed).
    Positive = capital inflow to chain = bullish.
    """
    return bridge_flow_panel.rolling(window, min_periods=5).mean()


def open_interest_change(
    oi_panel: pd.DataFrame,
    window: int = 7,
) -> pd.DataFrame:
    """
    Open interest growth over trailing week.
    Rising OI = new leveraged interest building.
    """
    return oi_panel.pct_change(periods=window)


# ── Aggregate ──────────────────────────────────────────────────────────

def compute_all_flows(
    data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Compute all flow signals.

    Args:
        data: dict with optional keys:
            "stablecoin_supply", "dex_volume", "bridge_flows", "open_interest"

    Returns:
        dict mapping signal_name -> DataFrame
    """
    signals = {}

    sc = data.get("stablecoin_supply", pd.DataFrame())
    if not sc.empty:
        signals["stablecoin_supply_growth"] = stablecoin_supply_growth(sc)

    vol = data.get("dex_volume", pd.DataFrame())
    if not vol.empty:
        signals["dex_volume_growth"] = dex_volume_signal(vol)

    bridge = data.get("bridge_flows", pd.DataFrame())
    if not bridge.empty:
        signals["bridge_net_flow"] = bridge_net_flow(bridge)

    oi = data.get("open_interest", pd.DataFrame())
    if not oi.empty:
        signals["oi_change"] = open_interest_change(oi)

    return signals
