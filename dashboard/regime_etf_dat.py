"""
ETF + DAT flows family indicators.

These indicators rely on a mix of:
- DAT mNAV from weekly scrape CSV cache
- ETF flow data from manual CSV (Farside / paid API later)
- yfinance for equity prices (MSTR, BMNR, STRC)
- Manual overrides for issuance capacity and treasury purchases

Each function returns the standard indicator dict.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _votes(risk_on=0, choppy=0, risk_off=0, local_top=0, local_bottom=0):
    return {
        "risk_on": risk_on, "choppy": choppy, "risk_off": risk_off,
        "local_top": local_top, "local_bottom": local_bottom,
    }


def _fmt_bn(v: float) -> str:
    return f"${v:.1f}bn"


# ---------------------------------------------------------------------------
# ETF flow indicators
# ---------------------------------------------------------------------------

def btc_etf_flows(etf_flow_df: pd.DataFrame | None) -> dict:
    """BTC ETF 5d and 20d net flows."""
    if etf_flow_df is None or etf_flow_df.empty:
        return _etf_fallback("BTC ETF flows", "+$1.2bn 5d; +$3.8bn 20d", 2)

    # Expect columns: date, btc_net_flow_usd
    df = etf_flow_df.sort_values("date").tail(20)
    f5 = df.tail(5)["btc_net_flow_usd"].sum() / 1e9
    f20 = df["btc_net_flow_usd"].sum() / 1e9

    if f5 > 0 and f20 > 0:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=70, choppy=25, local_top=5 if f5 > 2 else 0)
    elif f5 < 0 and f20 < 0:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=65, choppy=25, local_bottom=10)
    else:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=55, risk_on=25, risk_off=20)

    return {
        "family": "ETF + DAT flows",
        "indicator": "BTC ETF flows",
        "value": f"{_fmt_bn(f5)} 5d; {_fmt_bn(f20)} 20d",
        "criteria": "Risk-on if 5d and 20d net flows are positive. Risk-off if outflows persist for 5+ days.",
        "status": status, "score": score,
        "meaning": f"Institutional spot BTC demand is {'positive' if f5 > 0 and f20 > 0 else 'mixed' if f5 > 0 or f20 > 0 else 'negative'}.",
        "regime_fit": fit,
        "source": "ETF flow CSV",
        "asof": str(df["date"].max()),
        "regime_votes": votes,
    }


def eth_etf_flows(etf_flow_df: pd.DataFrame | None) -> dict:
    """ETH ETF 5d and 20d net flows."""
    if etf_flow_df is None or "eth_net_flow_usd" not in (etf_flow_df.columns if etf_flow_df is not None else []):
        return _etf_fallback("ETH ETF flows", "+$0.2bn 5d; -$0.1bn 20d", 1)

    df = etf_flow_df.sort_values("date").tail(20)
    f5 = df.tail(5)["eth_net_flow_usd"].sum() / 1e9
    f20 = df["eth_net_flow_usd"].sum() / 1e9

    if f5 > 0 and f20 > 0:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=60, choppy=35, local_top=5)
    elif f5 > 0 or f20 > 0:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=60, risk_on=25, risk_off=15)
    else:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=55, choppy=30, local_bottom=15)

    return {
        "family": "ETF + DAT flows",
        "indicator": "ETH ETF flows",
        "value": f"{_fmt_bn(f5)} 5d; {_fmt_bn(f20)} 20d",
        "criteria": "Risk-on if both 5d and 20d flows are positive. Choppy if mixed.",
        "status": status, "score": score,
        "meaning": f"ETH ETF demand is {'cleanly positive' if f5 > 0 and f20 > 0 else 'mixed' if f5 > 0 or f20 > 0 else 'negative'}.",
        "regime_fit": fit,
        "source": "ETF flow CSV",
        "asof": str(df["date"].max()),
        "regime_votes": votes,
    }


def etf_price_response(
    btc_weekly_return: float | None,
    etf_5d_flow: float | None,
) -> dict:
    """ETF inflow quality: do inflows lift price?"""
    if btc_weekly_return is None or etf_5d_flow is None:
        return _etf_fallback("ETF price response", "n/a", 1)

    inflows_positive = etf_5d_flow > 0
    strong_week = btc_weekly_return > 3

    if inflows_positive and strong_week:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=75, choppy=20, local_top=5)
    elif inflows_positive and not strong_week:
        score, status, fit = 1, "Partial", "Choppy / top watch"
        votes = _votes(choppy=40, local_top=35, risk_on=25)
    elif not inflows_positive and btc_weekly_return < -3:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=60, choppy=25, local_bottom=15)
    else:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=60, risk_on=20, risk_off=20)

    return {
        "family": "ETF + DAT flows",
        "indicator": "ETF price response",
        "value": f"BTC {btc_weekly_return:+.1f}% during {'inflow' if inflows_positive else 'outflow'} week",
        "criteria": "Risk-on if strong inflows coincide with >3% weekly BTC/ETH gains. Top warning if inflows fail to lift price.",
        "status": status, "score": score,
        "meaning": f"{'Strong inflows driving price' if inflows_positive and strong_week else 'Flows positive but market absorbing without repricing' if inflows_positive else 'Outflows and price weakness'}.",
        "regime_fit": fit,
        "source": "ETF flow + Binance",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": votes,
    }


# ---------------------------------------------------------------------------
# DAT mNAV indicators
# ---------------------------------------------------------------------------

def mstr_mnav(dat_df: pd.DataFrame) -> dict:
    """MSTR mNAV premium."""
    return _dat_mnav_indicator(
        dat_df, "MSTR", "MSTR mNAV",
        risk_on_threshold=1.20, risk_off_threshold=1.00,
    )


def bmnr_mnav(dat_df: pd.DataFrame) -> dict:
    """BMNR mNAV premium."""
    return _dat_mnav_indicator(
        dat_df, "BMNR", "BMNR mNAV",
        risk_on_threshold=1.15, risk_off_threshold=1.00,
    )


def purr_mnav(dat_df: pd.DataFrame) -> dict:
    """PURR Multiple to Adjusted NAV."""
    return _dat_mnav_indicator(
        dat_df, "PURR", "PURR multiple to adjusted NAV",
        risk_on_threshold=1.15, risk_off_threshold=1.00,
    )


def _dat_mnav_indicator(
    dat_df: pd.DataFrame,
    ticker: str,
    display_name: str,
    risk_on_threshold: float,
    risk_off_threshold: float,
) -> dict:
    row = dat_df[dat_df["DAT"] == ticker]
    if row.empty:
        return _etf_fallback(display_name, "n/a", 0)

    mnav = row.iloc[0]["mNAV"]
    source_status = row.iloc[0].get("Source status", "Unknown")
    source = row.iloc[0].get("Source", "")

    if mnav > risk_on_threshold:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=65, choppy=25, local_top=10 if mnav > 2.0 else 0)
    elif mnav > risk_off_threshold:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=55, risk_on=25, risk_off=20)
    else:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=60, choppy=25, local_bottom=15)

    return {
        "family": "ETF + DAT flows",
        "indicator": display_name,
        "value": f"{mnav:.2f}x",
        "criteria": f"Risk-on if >{risk_on_threshold:.2f}x. Risk-off if near/below {risk_off_threshold:.2f}x.",
        "status": status, "score": score,
        "meaning": f"{ticker} {'can issue accretively' if mnav > risk_on_threshold else 'issuance is marginal' if mnav > risk_off_threshold else 'cannot issue accretively'}.",
        "regime_fit": fit,
        "source": source,
        "asof": row.iloc[0].get("asof", str(datetime.utcnow().date())),
        "regime_votes": votes,
    }


# ---------------------------------------------------------------------------
# DAT qualitative / manual indicators
# ---------------------------------------------------------------------------

def dat_issuance_capacity(
    net_issuance_30d_bn: float | None = None,
    terms_tightening: bool = False,
) -> dict:
    """DAT issuance capacity assessment (manual or derived)."""
    if net_issuance_30d_bn is None:
        return _etf_fallback("DAT issuance capacity", "Manual input needed", 1)

    val = net_issuance_30d_bn
    if val > 1.0 and not terms_tightening:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=65, choppy=30, local_top=5)
    elif val > 0:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=55, risk_on=30, risk_off=15)
    else:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=55, choppy=30, local_bottom=15)

    terms_note = "; terms widening" if terms_tightening else ""
    return {
        "family": "ETF + DAT flows",
        "indicator": "DAT issuance capacity",
        "value": f"{_fmt_bn(val)} 30d{terms_note}",
        "criteria": "Risk-on if DATs can raise capital accretively and keep buying underlying crypto.",
        "status": status, "score": score,
        "meaning": f"The flow engine {'works well' if score == 2 else 'is slowing' if score == 1 else 'is impaired'}.",
        "regime_fit": fit,
        "source": "Manual / SEC filings",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": votes,
    }


def dat_treasury_purchases(net_purchases_30d_bn: float | None = None) -> dict:
    """DAT net treasury purchases over 30 days."""
    if net_purchases_30d_bn is None:
        return _etf_fallback("DAT treasury purchases", "Manual input needed", 1)

    val = net_purchases_30d_bn
    if val > 0.5:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=70, choppy=25, local_top=5)
    elif val > 0:
        score, status, fit = 1, "Partial", "Choppy"
        votes = _votes(choppy=55, risk_on=30, risk_off=15)
    else:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=60, choppy=25, local_bottom=15)

    return {
        "family": "ETF + DAT flows",
        "indicator": "DAT treasury purchases",
        "value": f"+{_fmt_bn(val)} 30d" if val > 0 else f"{_fmt_bn(val)} 30d",
        "criteria": "Risk-on if major DATs are net buyers over 30 days. Risk-off if buying stops.",
        "status": status, "score": score,
        "meaning": f"Capital {'is reaching' if val > 0 else 'is not reaching'} underlying crypto assets.",
        "regime_fit": fit,
        "source": "Manual / public disclosures",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": votes,
    }


def strc_credit_stress(strc_yield_chg_4w_bps: float | None = None) -> dict:
    """STRC / DAT credit stress indicator."""
    if strc_yield_chg_4w_bps is None:
        return _etf_fallback("STRC / DAT credit stress", "Manual input needed", 1)

    chg = strc_yield_chg_4w_bps
    if chg <= 25:
        score, status, fit = 2, "Met", "Risk-on support"
        votes = _votes(risk_on=60, choppy=35, local_top=5)
    elif chg <= 100:
        score, status, fit = 1, "Partial", "Choppy / bottom watch"
        votes = _votes(choppy=50, risk_off=30, local_bottom=20)
    else:
        score, status, fit = 0, "Not met", "Risk-off drag"
        votes = _votes(risk_off=65, choppy=20, local_bottom=15)

    return {
        "family": "ETF + DAT flows",
        "indicator": "STRC / DAT credit stress",
        "value": f"Yield {'+' if chg > 0 else ''}{chg:.0f} bps 4w",
        "criteria": "Risk-on if preferreds and credit are stable. Risk-off if yields/spreads spike.",
        "status": status, "score": score,
        "meaning": f"Credit stress is {'low' if chg <= 25 else 'rising but not disorderly' if chg <= 100 else 'elevated'}.",
        "regime_fit": fit,
        "source": "yfinance STRC / manual",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": votes,
    }


def _etf_fallback(name, value, score):
    return {
        "family": "ETF + DAT flows", "indicator": name, "value": value,
        "criteria": "Manual input or CSV required",
        "status": "Met" if score == 2 else "Partial" if score == 1 else "Not met",
        "score": score, "meaning": "Using fallback/manual value.",
        "regime_fit": "Choppy", "source": "Fallback",
        "asof": str(datetime.utcnow().date()),
        "regime_votes": _votes(choppy=100),
    }
