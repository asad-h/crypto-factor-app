"""
Price / Momentum signal computation.

All signals are computed from Binance OHLCV data.
Each function returns a DataFrame: rows = dates, columns = tokens.
"""
import numpy as np
import pandas as pd


def vol_adjusted_momentum(
    close_panel: pd.DataFrame,
    window: int = 21,
) -> pd.DataFrame:
    """
    3-week vol-adjusted momentum.
    Return over window / realised vol over window.
    Rewards smooth trends, penalises noisy moves.
    """
    ret = close_panel.pct_change(periods=window)
    daily_ret = close_panel.pct_change()
    vol = daily_ret.rolling(window, min_periods=10).std() * np.sqrt(365)
    return ret / (vol + 1e-8)


def relative_strength_vs_btc(
    close_panel: pd.DataFrame,
    btc_close: pd.Series,
    window: int = 21,
) -> pd.DataFrame:
    """
    Token return minus BTC return over trailing window.
    Outperformance relative to market.
    """
    token_ret = close_panel.pct_change(periods=window)
    btc_ret = btc_close.pct_change(periods=window)
    return token_ret.sub(btc_ret, axis=0)


def drawdown_from_ath(close_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Current price vs expanding all-time high.
    Returned inverted: deeper drawdown = higher score (mean reversion).
    Only useful when combined with intact fundamentals filter.
    """
    ath = close_panel.expanding().max()
    dd = (close_panel - ath) / (ath + 1e-8)
    return dd  # already negative, deeper = more negative = more potential


def momentum_breadth(
    close_panel: pd.DataFrame,
    window: int = 13,
) -> pd.DataFrame:
    """
    % of trailing weeks with positive returns.
    Trend consistency, not just magnitude.
    """
    weekly_ret = close_panel.pct_change(periods=7)
    # For each day, look back `window` weeks worth of weekly samples
    positive = (weekly_ret > 0).astype(float)
    return positive.rolling(window * 7, min_periods=window * 3).mean()


def short_term_reversal(
    close_panel: pd.DataFrame,
    window: int = 7,
) -> pd.DataFrame:
    """
    Prior week return, inverted. Mean reversion at short horizons.
    """
    ret = close_panel.pct_change(periods=window)
    return -ret


def realised_vol_rank(
    close_panel: pd.DataFrame,
    window: int = 28,
) -> pd.DataFrame:
    """
    Trailing realised volatility.
    Returned inverted: lower vol = higher score (low-volatility anomaly).
    """
    daily_ret = close_panel.pct_change()
    vol = daily_ret.rolling(window, min_periods=14).std() * np.sqrt(365)
    return -vol  # invert: lower vol = higher score


def size_signal(mcap_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Market cap rank. Smaller = higher score (size premium).
    Returned inverted: lower mcap = higher score.
    """
    return -mcap_panel


# ── Aggregate ──────────────────────────────────────────────────────────

def compute_all_momentum(
    close_panel: pd.DataFrame,
    btc_close: pd.Series,
    mcap_panel: pd.DataFrame = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute all price/momentum signals.

    Args:
        close_panel: daily close prices, columns = tokens
        btc_close: BTC daily close series
        mcap_panel: market cap panel (optional, for size signal)

    Returns:
        dict mapping signal_name -> DataFrame
    """
    signals = {}

    signals["vol_adj_momentum_3w"] = vol_adjusted_momentum(close_panel, window=21)
    signals["rel_strength_vs_btc"] = relative_strength_vs_btc(close_panel, btc_close)
    signals["drawdown_from_ath"] = drawdown_from_ath(close_panel)
    signals["momentum_breadth"] = momentum_breadth(close_panel)
    signals["short_term_reversal"] = short_term_reversal(close_panel)
    signals["realised_vol_rank"] = realised_vol_rank(close_panel)

    if mcap_panel is not None and not mcap_panel.empty:
        signals["size_signal"] = size_signal(mcap_panel)

    return signals
