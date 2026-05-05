"""
Individual regime indicator computations.

Each function takes raw data and returns a dict with the exact fields
required by the indicator_audit_df contract:
    indicator, value, criteria, status, score, meaning, regime_fit,
    source, asof,
    regime_votes: {risk_on, choppy, risk_off, local_top, local_bottom}

Scores: 2 = Met, 1 = Partial, 0 = Not met.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_price(v: float) -> str:
    if abs(v) >= 1_000:
        return f"${v / 1000:.1f}k"
    return f"${v:,.2f}"


def _fmt_pct(v: float, decimals: int = 1) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{decimals}f}%"


def _votes(risk_on=0, choppy=0, risk_off=0, local_top=0, local_bottom=0):
    return {
        "risk_on": risk_on, "choppy": choppy, "risk_off": risk_off,
        "local_top": local_top, "local_bottom": local_bottom,
    }


# ---------------------------------------------------------------------------
# BTC Trend / Momentum family
# ---------------------------------------------------------------------------

def btc_vs_20w_sma(btc_weekly: pd.Series) -> dict:
    """BTC close vs 20-week SMA with slope check."""
    sma = btc_weekly.rolling(20, min_periods=10).mean()
    price = btc_weekly.iloc[-1]
    sma_val = sma.iloc[-1]
    slope = sma.iloc[-1] - sma.iloc[-2] if len(sma) >= 2 else 0.0
    pct_above = (price / sma_val - 1) * 100

    if price > sma_val and slope > 0:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=80, choppy=20)
    elif price > sma_val and slope <= 0:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(risk_on=30, choppy=60, risk_off=10)
    elif price <= sma_val and slope < 0:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=70, choppy=30)
    else:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=60, risk_on=20, risk_off=20)

    return {
        "family": "BTC trend / momentum",
        "indicator": "BTC vs 20-week SMA",
        "value": f"{_fmt_price(price)} vs {_fmt_price(sma_val)}",
        "criteria": "Risk-on if BTC closes above a rising 20w SMA. Risk-off if below a falling 20w SMA.",
        "status": status, "score": score,
        "meaning": f"BTC is {_fmt_pct(pct_above)} {'above' if pct_above >= 0 else 'below'} the SMA; slope is {'rising' if slope > 0 else 'falling'}.",
        "regime_fit": fit,
        "source": "Binance BTCUSDT",
        "asof": str(btc_weekly.index[-1].date()),
        "regime_votes": votes,
    }


def sma_20w_slope(btc_weekly: pd.Series) -> dict:
    """20-week SMA slope direction."""
    sma = btc_weekly.rolling(20, min_periods=10).mean()
    price = btc_weekly.iloc[-1]
    slope = sma.iloc[-1] - sma.iloc[-2] if len(sma) >= 2 else 0.0

    if slope > 0:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=70, choppy=30)
    elif slope < 0 and price < sma.iloc[-1]:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=70, choppy=20, local_bottom=10)
    else:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=60, risk_on=20, risk_off=20)

    return {
        "family": "BTC trend / momentum",
        "indicator": "20-week SMA slope",
        "value": f"{'+' if slope > 0 else ''}{slope:,.0f}/wk",
        "criteria": "Risk-on if slope is positive. Risk-off if slope is negative and price is below SMA.",
        "status": status, "score": score,
        "meaning": f"The longer trend is {'rising' if slope > 0 else 'still repairing'}.",
        "regime_fit": fit,
        "source": "Binance BTCUSDT",
        "asof": str(btc_weekly.index[-1].date()),
        "regime_votes": votes,
    }


def btc_4w_return(btc_weekly: pd.Series) -> dict:
    """BTC 4-week return."""
    ret = (btc_weekly.iloc[-1] / btc_weekly.iloc[-5] - 1) * 100 if len(btc_weekly) >= 5 else 0.0

    if ret > 0:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=70, choppy=20, local_top=10 if ret > 20 else 0)
    else:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=60, choppy=30, local_bottom=10 if ret < -15 else 0)

    return {
        "family": "BTC trend / momentum",
        "indicator": "BTC 4-week return",
        "value": _fmt_pct(ret),
        "criteria": "Risk-on if positive. Risk-off if negative.",
        "status": status, "score": score,
        "meaning": f"Short-term momentum is {'positive' if ret > 0 else 'negative'}.",
        "regime_fit": fit,
        "source": "Binance BTCUSDT",
        "asof": str(btc_weekly.index[-1].date()),
        "regime_votes": votes,
    }


def btc_12w_return(btc_weekly: pd.Series) -> dict:
    """BTC 12-week return."""
    ret = (btc_weekly.iloc[-1] / btc_weekly.iloc[-13] - 1) * 100 if len(btc_weekly) >= 13 else 0.0

    if ret > 3:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=70, choppy=20, local_top=10 if ret > 30 else 0)
    elif ret >= -3:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=70, risk_on=15, risk_off=15)
    else:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=65, choppy=25, local_bottom=10 if ret < -20 else 0)

    return {
        "family": "BTC trend / momentum",
        "indicator": "BTC 12-week return",
        "value": _fmt_pct(ret),
        "criteria": "Risk-on if > +3%. Choppy if -3% to +3%. Risk-off if < -3%.",
        "status": status, "score": score,
        "meaning": f"Medium-term momentum is {'positive' if ret > 3 else 'flat' if ret >= -3 else 'negative'}.",
        "regime_fit": fit,
        "source": "Binance BTCUSDT",
        "asof": str(btc_weekly.index[-1].date()),
        "regime_votes": votes,
    }


# ---------------------------------------------------------------------------
# Market breadth family
# ---------------------------------------------------------------------------

def pct_tokens_above_20w_sma(token_weekly_closes: pd.DataFrame) -> dict:
    """Percentage of eligible tokens above their 20-week SMA."""
    if token_weekly_closes.empty:
        return _breadth_fallback("% tokens above 20w SMA", "43%", 1)

    sma = token_weekly_closes.rolling(20, min_periods=10).mean()
    latest_prices = token_weekly_closes.iloc[-1]
    latest_sma = sma.iloc[-1]
    valid = latest_prices.dropna().index.intersection(latest_sma.dropna().index)
    if len(valid) == 0:
        return _breadth_fallback("% tokens above 20w SMA", "n/a", 0)

    above = (latest_prices[valid] > latest_sma[valid]).sum()
    pct = above / len(valid) * 100

    if pct > 60:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=75, choppy=20, local_top=5 if pct > 85 else 0)
    elif pct >= 40:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=65, risk_on=20, risk_off=15)
    else:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=60, choppy=25, local_bottom=15 if pct < 20 else 0)

    return {
        "family": "Market breadth",
        "indicator": "% tokens above 20w SMA",
        "value": f"{pct:.0f}%",
        "criteria": "Risk-on if >60%. Choppy if 40-60%. Risk-off if <40%.",
        "status": status, "score": score,
        "meaning": f"{'Broad' if pct > 60 else 'Mixed' if pct >= 40 else 'Narrow'} participation across the eligible universe.",
        "regime_fit": fit,
        "source": "Binance eligible universe",
        "asof": str(token_weekly_closes.index[-1].date()),
        "regime_votes": votes,
    }


def median_token_vs_btc_8w(token_weekly_closes: pd.DataFrame, btc_weekly: pd.Series) -> dict:
    """Median token 8-week return vs BTC 8-week return."""
    if token_weekly_closes.empty or len(btc_weekly) < 9:
        return _breadth_fallback("Median token vs BTC, 8w", "n/a", 0)

    token_rets = (token_weekly_closes.iloc[-1] / token_weekly_closes.iloc[-9] - 1) * 100
    btc_ret = (btc_weekly.iloc[-1] / btc_weekly.iloc[-9] - 1) * 100
    median_ret = token_rets.median()
    diff = median_ret - btc_ret

    if diff > 0:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=70, choppy=25, local_top=5 if diff > 15 else 0)
    elif diff >= -5:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=60, risk_off=25, risk_on=15)
    else:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=60, choppy=25, local_bottom=15 if diff < -15 else 0)

    return {
        "family": "Market breadth",
        "indicator": "Median token vs BTC, 8w",
        "value": f"{_fmt_pct(diff)} vs BTC",
        "criteria": "Risk-on if median token beats BTC. Risk-off if underperformance is worse than -5%.",
        "status": status, "score": score,
        "meaning": f"Median token {'outperforms' if diff > 0 else 'underperforms'} BTC by {abs(diff):.1f}pp over 8 weeks.",
        "regime_fit": fit,
        "source": "Binance eligible universe",
        "asof": str(token_weekly_closes.index[-1].date()),
        "regime_votes": votes,
    }


def new_highs_vs_lows(token_weekly_closes: pd.DataFrame) -> dict:
    """Count of tokens at 20-week high vs 20-week low."""
    if token_weekly_closes.empty or len(token_weekly_closes) < 20:
        return _breadth_fallback("New highs vs new lows", "n/a", 0)

    rolling_max = token_weekly_closes.rolling(20).max().iloc[-1]
    rolling_min = token_weekly_closes.rolling(20).min().iloc[-1]
    latest = token_weekly_closes.iloc[-1]

    highs = (latest >= rolling_max * 0.99).sum()  # within 1% of 20w high
    lows = (latest <= rolling_min * 1.01).sum()    # within 1% of 20w low

    if highs > lows:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=70, choppy=20, local_top=10 if highs > lows * 3 else 0)
    elif lows > highs and (len(token_weekly_closes) >= 21 and
          (token_weekly_closes.iloc[-2] <= token_weekly_closes.rolling(20).min().iloc[-2] * 1.01).sum() > lows):
        score, status, fit = 1, "Partial", "Bottom watch"
        votes = _votes(choppy=40, local_bottom=40, risk_off=20)
    else:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=50, risk_off=30, local_bottom=20)

    return {
        "family": "Market breadth",
        "indicator": "New highs vs new lows",
        "value": f"{highs} / {lows}",
        "criteria": "Risk-on if new highs exceed new lows. Bottom watch if new lows stop expanding.",
        "status": status, "score": score,
        "meaning": f"{'Broad strength' if highs > lows else 'Weak internals, but downside breadth may be stabilizing' if 'Bottom' in fit else 'Market internals are weak'}.",
        "regime_fit": fit,
        "source": "Binance eligible universe",
        "asof": str(token_weekly_closes.index[-1].date()),
        "regime_votes": votes,
    }


def _breadth_fallback(name, value, score):
    return {
        "family": "Market breadth", "indicator": name, "value": value,
        "criteria": "Insufficient data", "status": "Partial" if score == 1 else "Not met",
        "score": score, "meaning": "Insufficient token price data for breadth computation.",
        "regime_fit": "Choppy", "source": "Fallback", "asof": str(datetime.utcnow().date()),
        "regime_votes": _votes(choppy=100),
    }


# ---------------------------------------------------------------------------
# Stablecoin liquidity family
# ---------------------------------------------------------------------------

def stablecoin_supply_growth(supply_series: pd.Series | None) -> dict:
    """Stablecoin supply 30d and 90d growth."""
    if supply_series is None or len(supply_series) < 31:
        return {
            "family": "Stablecoin liquidity",
            "indicator": "Stablecoin supply growth",
            "value": "n/a",
            "criteria": "Risk-on if both 30d and 90d supply growth are positive. Risk-off if contracting.",
            "status": "Not met", "score": 0,
            "meaning": "Stablecoin supply data unavailable.",
            "regime_fit": "Choppy",
            "source": "DefiLlama (unavailable)",
            "asof": str(datetime.utcnow().date()),
            "regime_votes": _votes(choppy=100),
        }

    latest = supply_series.iloc[-1]
    d30_ago = supply_series.iloc[-31] if len(supply_series) >= 31 else supply_series.iloc[0]
    d90_ago = supply_series.iloc[-91] if len(supply_series) >= 91 else supply_series.iloc[0]
    g30 = (latest / d30_ago - 1) * 100
    g90 = (latest / d90_ago - 1) * 100

    if g30 > 0 and g90 > 0:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=75, choppy=25)
    elif g30 > 0 or g90 > 0:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=60, risk_on=25, risk_off=15)
    else:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=65, choppy=25, local_bottom=10)

    return {
        "family": "Stablecoin liquidity",
        "indicator": "Stablecoin supply growth",
        "value": f"{_fmt_pct(g30)} 30d; {_fmt_pct(g90)} 90d",
        "criteria": "Risk-on if both 30d and 90d supply growth are positive. Risk-off if contracting.",
        "status": status, "score": score,
        "meaning": f"Crypto-native liquidity is {'expanding' if g30 > 0 and g90 > 0 else 'mixed' if g30 > 0 or g90 > 0 else 'contracting'}.",
        "regime_fit": fit,
        "source": "DefiLlama stablecoins",
        "asof": str(supply_series.index[-1].date()) if hasattr(supply_series.index[-1], 'date') else str(supply_series.index[-1]),
        "regime_votes": votes,
    }


# ---------------------------------------------------------------------------
# Leverage / volatility family
# ---------------------------------------------------------------------------

def realised_vol_8w(btc_daily: pd.Series) -> dict:
    """8-week realised volatility and direction."""
    if len(btc_daily) < 57:
        return _lev_fallback("8-week realised volatility", "n/a", 1)

    rets = btc_daily.pct_change().dropna()
    vol_now = rets.iloc[-56:].std() * np.sqrt(365) * 100
    vol_prev = rets.iloc[-112:-56].std() * np.sqrt(365) * 100 if len(rets) >= 112 else vol_now
    vol_chg = vol_now - vol_prev
    price_chg = (btc_daily.iloc[-1] / btc_daily.iloc[-57] - 1) * 100

    if vol_chg < 0 and price_chg >= 0:
        score, status, fit = 2, "Met", "Local bottom support"
        votes = _votes(risk_on=40, choppy=30, local_bottom=30)
    elif vol_chg > 0 and price_chg < 0:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=60, choppy=25, local_top=15)
    else:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=60, risk_on=20, risk_off=20)

    return {
        "family": "Leverage / volatility",
        "indicator": "8-week realised volatility",
        "value": f"{vol_now:.0f}%; 8w change {vol_chg:+.0f} pp",
        "criteria": "Risk-on if vol falls while price holds/rises. Risk-off if vol rises on downside.",
        "status": status, "score": score,
        "meaning": f"{'Stress is compressing' if vol_chg < 0 else 'Volatility is expanding'} {'while price holds' if price_chg >= 0 else 'on the downside'}.",
        "regime_fit": fit,
        "source": "Binance BTCUSDT",
        "asof": str(btc_daily.index[-1].date()),
        "regime_votes": votes,
    }


def funding_rates_indicator(funding_ann: float | None) -> dict:
    """BTC/ETH perp funding rate assessment."""
    if funding_ann is None:
        return _lev_fallback("Funding rates", "n/a", 1)

    if 0 < funding_ann < 15:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=60, choppy=30, local_top=10 if funding_ann > 12 else 0)
    elif funding_ann >= 15:
        score, status, fit = 1, "Partial", "Local top watch"
        votes = _votes(local_top=50, choppy=30, risk_on=20)
    elif funding_ann < 0:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=50, choppy=30, local_bottom=20)
    else:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=70, risk_on=30)

    return {
        "family": "Leverage / volatility",
        "indicator": "Funding rates",
        "value": f"{funding_ann:+.1f}% annualized",
        "criteria": "Healthy if positive but below 15%. Top warning if extremely positive.",
        "status": status, "score": score,
        "meaning": f"Positioning is {'constructive' if 0 < funding_ann < 15 else 'euphoric' if funding_ann >= 15 else 'bearish'}.",
        "regime_fit": fit,
        "source": "Manual / CoinGlass CSV",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": votes,
    }


def oi_to_mcap_indicator(oi_pct: float | None) -> dict:
    """Open interest as % of market cap."""
    if oi_pct is None:
        return _lev_fallback("Open interest / market cap", "n/a", 1)

    if 3 <= oi_pct <= 5:
        score, status, fit = 2, "Met", "Choppy to risk-on"
        votes = _votes(risk_on=40, choppy=50, local_top=10 if oi_pct > 4.5 else 0)
    elif oi_pct > 5:
        score, status, fit = 1, "Partial", "Local top watch"
        votes = _votes(local_top=50, choppy=30, risk_off=20)
    else:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=60, risk_on=20, risk_off=20)

    return {
        "family": "Leverage / volatility",
        "indicator": "Open interest / market cap",
        "value": f"{oi_pct:.1f}%",
        "criteria": "Healthy if 3-5%. Top warning if elevated and rising fast.",
        "status": status, "score": score,
        "meaning": f"Leverage is {'normal' if 3 <= oi_pct <= 5 else 'elevated' if oi_pct > 5 else 'light'}.",
        "regime_fit": fit,
        "source": "Manual / CoinGlass CSV",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": votes,
    }


def basis_indicator(basis_pct: float | None) -> dict:
    """CME / perp basis spread."""
    if basis_pct is None:
        return _lev_fallback("CME / perp basis", "n/a", 1)

    if 5 <= basis_pct <= 12:
        score, status, fit = 2, "Met", "Choppy to risk-on"
        votes = _votes(risk_on=40, choppy=50, local_top=10 if basis_pct > 10 else 0)
    elif basis_pct > 15:
        score, status, fit = 0, "Not met", "Local top watch"
        votes = _votes(local_top=60, choppy=20, risk_off=20)
    elif basis_pct < 5:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=50, choppy=30, local_bottom=20)
    else:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=50, risk_on=30, local_top=20)

    return {
        "family": "Leverage / volatility",
        "indicator": "CME / perp basis",
        "value": f"{basis_pct:.1f}%",
        "criteria": "Healthy if 5-12%. Risk-off if basis collapses. Top warning if above 15%.",
        "status": status, "score": score,
        "meaning": f"Basis is {'normal' if 5 <= basis_pct <= 12 else 'compressed' if basis_pct < 5 else 'elevated'}.",
        "regime_fit": fit,
        "source": "Manual / CoinGlass CSV",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": votes,
    }


def _lev_fallback(name, value, score):
    return {
        "family": "Leverage / volatility", "indicator": name, "value": value,
        "criteria": "Data unavailable", "status": "Partial" if score == 1 else "Not met",
        "score": score, "meaning": "Manual input required.",
        "regime_fit": "Choppy", "source": "Manual / fallback",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": _votes(choppy=100),
    }


# ---------------------------------------------------------------------------
# Macro / AI family
# ---------------------------------------------------------------------------

def spy_qqq_trend(spy_weekly: pd.Series | None, qqq_weekly: pd.Series | None) -> dict:
    """SPY and QQQ vs their 20-week SMAs."""
    if spy_weekly is None or qqq_weekly is None or len(spy_weekly) < 20:
        return _macro_fallback("SPY / QQQ trend")

    spy_sma = spy_weekly.rolling(20).mean()
    qqq_sma = qqq_weekly.rolling(20).mean()
    spy_pct = (spy_weekly.iloc[-1] / spy_sma.iloc[-1] - 1) * 100
    qqq_pct = (qqq_weekly.iloc[-1] / qqq_sma.iloc[-1] - 1) * 100
    spy_slope = spy_sma.iloc[-1] - spy_sma.iloc[-2]
    qqq_slope = qqq_sma.iloc[-1] - qqq_sma.iloc[-2]

    both_above = spy_pct > 0 and qqq_pct > 0
    both_rising = spy_slope > 0 and qqq_slope > 0

    if both_above and both_rising:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=70, choppy=30)
    elif both_above:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=50, risk_on=40, risk_off=10)
    elif spy_pct < 0 and qqq_pct < 0:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=60, choppy=30, local_bottom=10)
    else:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=60, risk_on=20, risk_off=20)

    return {
        "family": "Macro / AI",
        "indicator": "SPY / QQQ trend",
        "value": f"SPY {_fmt_pct(spy_pct)}; QQQ {_fmt_pct(qqq_pct)} vs 20w SMA",
        "criteria": "Risk-on if both are above rising 20w SMAs. Risk-off if both break below.",
        "status": status, "score": score,
        "meaning": f"Traditional risk assets are {'supportive' if both_above else 'mixed' if spy_pct > 0 or qqq_pct > 0 else 'weak'}.",
        "regime_fit": fit,
        "source": "yfinance",
        "asof": str(spy_weekly.index[-1].date()),
        "regime_votes": votes,
    }


def dxy_trend_indicator(dxy_latest: float | None, dxy_4w_chg_pct: float | None) -> dict:
    """DXY trend assessment."""
    if dxy_latest is None:
        return _macro_fallback("DXY trend")

    chg = dxy_4w_chg_pct or 0.0
    if chg <= 0:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=60, choppy=35, local_bottom=5)
    elif chg <= 1.5:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=60, risk_off=25, risk_on=15)
    else:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=60, choppy=30, local_top=10)

    return {
        "family": "Macro / AI",
        "indicator": "DXY trend",
        "value": f"{dxy_latest:.1f}; 4w change {_fmt_pct(chg)}",
        "criteria": "Risk-on if DXY is flat/falling. Risk-off if DXY rises sharply.",
        "status": status, "score": score,
        "meaning": f"A {'weaker' if chg <= 0 else 'stronger'} dollar is {'supportive' if chg <= 0 else 'a headwind'} for crypto.",
        "regime_fit": fit,
        "source": "yfinance DX-Y.NYB / Stooq DX.F",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": votes,
    }


def real_yields_indicator(change_4w_bps: float | None) -> dict:
    """10Y TIPS real yield 4-week change."""
    if change_4w_bps is None:
        return _macro_fallback("Real yields")

    if change_4w_bps <= 0:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=55, choppy=40, local_bottom=5)
    elif change_4w_bps <= 10:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=60, risk_off=25, risk_on=15)
    else:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=55, choppy=30, local_top=15)

    return {
        "family": "Macro / AI",
        "indicator": "Real yields",
        "value": f"{change_4w_bps:+.0f} bps 4w",
        "criteria": "Risk-on if flat/falling. Risk-off if rising materially.",
        "status": status, "score": score,
        "meaning": f"{'Falling' if change_4w_bps <= 0 else 'Rising'} real yields {'support' if change_4w_bps <= 0 else 'pressure'} speculative assets.",
        "regime_fit": fit,
        "source": "FRED DFII10",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": votes,
    }


def ai_macro_bias(ai_basket_return_8w: float | None) -> dict:
    """AI equity basket 8-week return."""
    if ai_basket_return_8w is None:
        return _macro_fallback("AI macro bias")

    ret = ai_basket_return_8w
    if ret > 0:
        score, status, fit = 2, "Met", "Risk-on tilt"
        votes = _votes(risk_on=60, choppy=35, local_top=5 if ret > 20 else 0)
    elif ret >= -5:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=60, risk_on=20, risk_off=20)
    else:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=50, choppy=35, local_bottom=15)

    return {
        "family": "Macro / AI",
        "indicator": "AI macro bias",
        "value": f"{_fmt_pct(ret)} 8w AI basket",
        "criteria": "Positive if AI leadership and capex sentiment are firm. Negative if AI breaks down.",
        "status": status, "score": score,
        "meaning": f"AI sector is {'supportive' if ret > 0 else 'mixed' if ret >= -5 else 'weak'}.",
        "regime_fit": fit,
        "source": "yfinance (NVDA, MSFT, GOOGL, AMZN, META, AVGO, AMD)",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": votes,
    }


def _macro_fallback(name):
    return {
        "family": "Macro / AI", "indicator": name, "value": "n/a",
        "criteria": "Data unavailable", "status": "Partial", "score": 1,
        "meaning": "Source unavailable; using neutral assumption.",
        "regime_fit": "Choppy", "source": "Fallback",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": _votes(choppy=100),
    }


# ---------------------------------------------------------------------------
# Valuation / sentiment family
# ---------------------------------------------------------------------------

def mvrv_nupl_indicator(mvrv_z: float | None, nupl: float | None) -> dict:
    """MVRV Z-score and NUPL."""
    if mvrv_z is None:
        return {
            "family": "Valuation / sentiment", "indicator": "MVRV / NUPL",
            "value": "n/a", "criteria": "Top if expensive. Bottom if cheap. Choppy if middle.",
            "status": "Partial", "score": 1, "meaning": "On-chain valuation data unavailable.",
            "regime_fit": "Choppy", "source": "Manual CSV fallback",
            "asof": str(datetime.utcnow().date()),
            "regime_votes": _votes(choppy=100),
        }

    nupl_val = nupl or 0.5
    if mvrv_z > 3.5:
        score, status, fit = 0, "Not met", "Local top"
        votes = _votes(local_top=70, choppy=20, risk_off=10)
    elif mvrv_z < 0.5:
        score, status, fit = 2, "Met", "Local bottom"
        votes = _votes(local_bottom=70, choppy=20, risk_on=10)
    else:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=70, risk_on=15, risk_off=15)

    return {
        "family": "Valuation / sentiment",
        "indicator": "MVRV / NUPL",
        "value": f"MVRV Z {mvrv_z:.1f}; NUPL {nupl_val:.2f}",
        "criteria": "Top if historically expensive. Bottom if historically cheap. Choppy if middle range.",
        "status": status, "score": score,
        "meaning": f"Valuation is {'extended' if mvrv_z > 3.5 else 'cheap' if mvrv_z < 0.5 else 'mid-range'}.",
        "regime_fit": fit,
        "source": "Manual / on-chain CSV",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": votes,
    }


def retail_froth_indicator(coinbase_rank: int | None) -> dict:
    """Retail froth via Coinbase app rank proxy."""
    if coinbase_rank is None:
        coinbase_rank = 200  # neutral assumption

    if coinbase_rank <= 10:
        score, status, fit = 0, "Not met", "Local top"
        votes = _votes(local_top=70, choppy=20, risk_off=10)
    elif coinbase_rank <= 50:
        score, status, fit = 1, "Partial", "Top watch"
        votes = _votes(local_top=40, choppy=40, risk_on=20)
    else:
        score, status, fit = 2, "Met", "No top signal"
        votes = _votes(choppy=50, risk_on=30, local_bottom=20)

    return {
        "family": "Valuation / sentiment",
        "indicator": "Retail froth",
        "value": f"COIN #{coinbase_rank} overall",
        "criteria": "Local top if crypto apps are top-ranked and low-quality alts/memecoins lead.",
        "status": status, "score": score,
        "meaning": f"{'Euphoria present' if coinbase_rank <= 10 else 'Elevated retail interest' if coinbase_rank <= 50 else 'No retail euphoria signal'}.",
        "regime_fit": fit,
        "source": "Manual / app store rank",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": _votes(choppy=60, risk_on=20, risk_off=20) if coinbase_rank > 50 else votes,
    }


def portfolio_drawdown_indicator(drawdown_pct: float | None) -> dict:
    """Portfolio drawdown defensive override."""
    if drawdown_pct is None:
        drawdown_pct = 0.0

    if drawdown_pct > -15:
        score, status, fit = 2, "Met", "Normal sizing"
        votes = _votes(risk_on=30, choppy=60, risk_off=10)
    elif drawdown_pct > -20:
        score, status, fit = 1, "Partial", "Defensive"
        votes = _votes(risk_off=50, choppy=30, local_bottom=20)
    else:
        score, status, fit = 0, "Not met", "Capital preservation"
        votes = _votes(risk_off=80, local_bottom=20)

    return {
        "family": "Valuation / sentiment",
        "indicator": "Portfolio drawdown",
        "value": f"{drawdown_pct:.1f}%",
        "criteria": "Defensive override if drawdown exceeds -15%. Capital preservation mode above -20%.",
        "status": status, "score": score,
        "meaning": f"{'No defensive override active' if drawdown_pct > -15 else 'Defensive mode' if drawdown_pct > -20 else 'Capital preservation active'}.",
        "regime_fit": fit,
        "source": "Portfolio equity curve",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": votes,
    }
