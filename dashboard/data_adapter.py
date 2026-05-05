"""
Data adapter layer.

Fetches real data from Binance, DefiLlama, yfinance, FRED, and local CSVs,
then feeds it into the regime indicator functions and classifier.

Returns the exact DataFrames the Streamlit app expects:
    weekly_regime_df, family_scores_df, indicator_audit_df,
    market_kpis_df, dat_mnav_df, watchlist_df

Fallback: if a source is unavailable, returns mock data with status marked.
"""
from __future__ import annotations

import os
import sys
import logging
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:
    yf = None

# Add project root to path so we can import crypto_factor_model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.regime_indicators import (
    btc_vs_20w_sma, sma_20w_slope, btc_4w_return, btc_12w_return,
    pct_tokens_above_20w_sma, median_token_vs_btc_8w, new_highs_vs_lows,
    stablecoin_supply_growth, realised_vol_8w,
    funding_rates_indicator, oi_to_mcap_indicator, basis_indicator,
    spy_qqq_trend, dxy_trend_indicator, real_yields_indicator, ai_macro_bias,
    mvrv_nupl_indicator, retail_froth_indicator, portfolio_drawdown_indicator,
)
from dashboard.regime_etf_dat import (
    btc_etf_flows, eth_etf_flows, etf_price_response,
    mstr_mnav, bmnr_mnav, purr_mnav,
    dat_issuance_capacity, dat_treasury_purchases, strc_credit_stress,
)
from dashboard.regime_classifier import classify_regime, classify_weekly_history

logger = logging.getLogger(__name__)

DASHBOARD_DIR = Path(__file__).resolve().parent
DATA_DIR = DASHBOARD_DIR / "data"
DAT_MNAV_PATH = DATA_DIR / "dat_mnav.csv"

# Manual override CSV paths (user can drop these in dashboard/data/)
ETF_FLOW_CSV = DATA_DIR / "etf_flows.csv"
LEVERAGE_CSV = DATA_DIR / "leverage_overrides.csv"
ONCHAIN_CSV = DATA_DIR / "onchain_overrides.csv"
PORTFOLIO_NAV_CSV = DATA_DIR / "portfolio_nav.csv"


# ---------------------------------------------------------------------------
# Raw data fetchers
# ---------------------------------------------------------------------------

def fetch_btc_daily(start: str = "2024-01-01") -> pd.Series:
    """Fetch BTC daily closes from Binance."""
    try:
        from crypto_factor_model.clients.binance import BinanceClient
        bn = BinanceClient()
        return bn.get_daily_close("BTCUSDT", start=start)
    except Exception as e:
        logger.warning(f"Binance fetch failed: {e}; trying yfinance")
        if yf:
            df = yf.Ticker("BTC-USD").history(start=start, interval="1d", auto_adjust=False)
            return df["Close"].rename("BTCUSDT")
        return pd.Series(dtype=float, name="BTCUSDT")


def fetch_btc_weekly(start: str = "2024-01-01") -> pd.Series:
    """Resample BTC daily to weekly (Monday start, last close of week)."""
    daily = fetch_btc_daily(start)
    if daily.empty:
        return daily
    weekly = daily.resample("W-MON", label="left", closed="left").last()
    return weekly.dropna()


def fetch_token_weekly_closes(start: str = "2024-06-01") -> pd.DataFrame:
    """Fetch weekly closes for eligible tokens from Binance."""
    try:
        from crypto_factor_model.clients.binance import BinanceClient
        from crypto_factor_model.config import SLUG_TO_BINANCE
        bn = BinanceClient()
        symbols = [v for v in SLUG_TO_BINANCE.values() if v and v != "BTCUSDT"]
        df = bn.get_multiple_daily(symbols[:15], start=start)  # cap to avoid rate limits
        weekly = df.resample("W-MON", label="left", closed="left").last()
        return weekly.dropna(how="all")
    except Exception as e:
        logger.warning(f"Token weekly fetch failed: {e}")
        return pd.DataFrame()


def fetch_stablecoin_supply() -> pd.Series | None:
    """Fetch aggregate stablecoin supply from DefiLlama."""
    try:
        resp = requests.get(
            "https://stablecoins.llama.fi/stablecoincharts/all",
            params={"stablecoin": "1"},  # USDT as proxy, or use aggregate
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        records = [
            {"date": pd.Timestamp(d["date"], unit="s"), "supply": d.get("totalCirculating", {}).get("peggedUSD", 0)}
            for d in data
        ]
        df = pd.DataFrame(records).set_index("date").sort_index()
        return df["supply"]
    except Exception as e:
        logger.warning(f"DefiLlama stablecoin fetch failed: {e}")
        # Try aggregate endpoint
        try:
            resp = requests.get("https://stablecoins.llama.fi/stablecoincharts/all", timeout=15)
            resp.raise_for_status()
            data = resp.json()
            records = [
                {"date": pd.Timestamp(d["date"], unit="s"),
                 "supply": sum(v.get("peggedUSD", 0) for v in d.get("totalCirculating", {}).values())
                           if isinstance(d.get("totalCirculating"), dict)
                           else d.get("totalCirculating", {}).get("peggedUSD", 0)}
                for d in data
            ]
            df = pd.DataFrame(records).set_index("date").sort_index()
            return df["supply"]
        except Exception:
            return None


def fetch_equity_weekly(symbol: str, start: str = "2024-06-01") -> pd.Series | None:
    """Fetch weekly closes for an equity via yfinance."""
    if yf is None:
        return None
    try:
        df = yf.Ticker(symbol).history(start=start, interval="1wk", auto_adjust=False)
        return df["Close"].rename(symbol)
    except Exception:
        return None


def fetch_ai_basket_return_8w() -> float | None:
    """Compute 8-week return of AI equity basket."""
    tickers = ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "AVGO", "AMD"]
    if yf is None:
        return None
    try:
        rets = []
        for t in tickers:
            hist = yf.Ticker(t).history(period="3mo", interval="1wk", auto_adjust=False)
            if len(hist) >= 9:
                r = (hist["Close"].iloc[-1] / hist["Close"].iloc[-9] - 1) * 100
                rets.append(r)
        return np.mean(rets) if rets else None
    except Exception:
        return None


def fetch_dxy_data() -> tuple[float | None, float | None]:
    """Fetch DXY latest and 4-week change."""
    if yf:
        try:
            hist = yf.Ticker("DX-Y.NYB").history(period="2mo", interval="1wk", auto_adjust=False)
            if len(hist) >= 5:
                latest = hist["Close"].iloc[-1]
                prior = hist["Close"].iloc[-5]
                chg = (latest / prior - 1) * 100
                return float(latest), float(chg)
        except Exception:
            pass
    # Stooq fallback
    try:
        resp = requests.get(
            "https://stooq.com/q/l/?s=dx.f&f=sd2t2ohlcv&h&e=csv", timeout=10
        )
        df = pd.read_csv(StringIO(resp.text))
        close = float(df.iloc[0]["Close"])
        return close, None
    except Exception:
        return None, None


def fetch_real_yields_4w_change() -> float | None:
    """Fetch 10Y TIPS real yield 4-week change from FRED."""
    try:
        resp = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10", timeout=12
        )
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df["DFII10"] = pd.to_numeric(df["DFII10"], errors="coerce")
        df = df.dropna().tail(30)
        if len(df) >= 20:
            latest = df.iloc[-1]["DFII10"]
            prior = df.iloc[-20]["DFII10"]
            return (latest - prior) * 100  # bps
        return None
    except Exception:
        return None


def fetch_m2_mom() -> dict:
    """Fetch M2 money supply MoM from FRED."""
    try:
        resp = requests.get(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WM2NS", timeout=12
        )
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df["WM2NS"] = pd.to_numeric(df["WM2NS"], errors="coerce")
        df = df.dropna().tail(8)
        latest = df.iloc[-1]["WM2NS"]
        prior = df.iloc[-5]["WM2NS"] if len(df) >= 5 else df.iloc[0]["WM2NS"]
        mom = (latest / prior - 1) * 100
        return {"value": mom, "status": "Live", "date": str(df.iloc[-1].get("observation_date", ""))}
    except Exception:
        return {"value": 0.0, "status": "Fallback", "date": "unavailable"}


def load_dat_mnav(path: Path = DAT_MNAV_PATH) -> pd.DataFrame:
    """Load DAT mNAV from weekly scrape cache CSV."""
    if path.exists():
        try:
            df = pd.read_csv(path)
            if "asof" not in df.columns:
                df["asof"] = str(datetime.utcnow().date())
            return df
        except Exception:
            pass
    # Fallback
    return pd.DataFrame([
        {"DAT": "MSTR", "Metric": "mNAV", "mNAV": 1.34, "Source": "https://www.strategy.com/",
         "Source status": "Fallback", "Fetch detail": "Cache not found", "Note": "", "asof": str(datetime.utcnow().date())},
        {"DAT": "BMNR", "Metric": "mNAV", "mNAV": 1.02, "Source": "https://bmnr.rocks/",
         "Source status": "Fallback", "Fetch detail": "Cache not found", "Note": "", "asof": str(datetime.utcnow().date())},
        {"DAT": "PURR", "Metric": "Multiple to Adjusted Net Asset Value", "mNAV": 0.94, "Source": "https://www.hypestrat.xyz/dashboard",
         "Source status": "Fallback", "Fetch detail": "Cache not found", "Note": "", "asof": str(datetime.utcnow().date())},
    ])


def fetch_binance_funding_rate() -> float | None:
    """Fetch live annualized funding rate from Binance Futures."""
    try:
        from crypto_factor_model.clients.binance import BinanceClient
        bn = BinanceClient()
        return bn.get_current_funding_annualized("BTCUSDT", lookback_days=7)
    except Exception as e:
        logger.warning(f"Binance funding rate fetch failed: {e}")
        return None


def fetch_binance_oi_pct(btc_price: float | None = None) -> float | None:
    """Fetch BTC futures OI as % of market cap from Binance."""
    try:
        from crypto_factor_model.clients.binance import BinanceClient
        bn = BinanceClient()
        oi_contracts = bn.get_open_interest("BTCUSDT")
        if oi_contracts is None or btc_price is None:
            return None
        # OI is in BTC contracts; convert to USD
        oi_usd = oi_contracts * btc_price
        # BTC circulating supply ~19.8M (hardcoded approximation, good enough for %)
        btc_mcap = btc_price * 19_800_000
        return (oi_usd / btc_mcap) * 100
    except Exception as e:
        logger.warning(f"Binance OI fetch failed: {e}")
        return None


def fetch_onchain_overrides() -> dict:
    """Load MVRV, NUPL, Coinbase rank from CSV only."""
    result = {"mvrv_z": None, "nupl": None, "coinbase_rank": None}

    if ONCHAIN_CSV.exists():
        try:
            df = pd.read_csv(ONCHAIN_CSV)
            row = df.iloc[-1]
            if pd.notna(row.get("mvrv_z")):
                result["mvrv_z"] = float(row["mvrv_z"])
            if pd.notna(row.get("nupl")):
                result["nupl"] = float(row["nupl"])
            if pd.notna(row.get("coinbase_rank")):
                result["coinbase_rank"] = int(row["coinbase_rank"])
        except Exception:
            pass

    return result


def load_leverage_overrides(btc_price: float | None = None) -> dict:
    """
    Load leverage/vol data. Priority:
    1. Binance live funding rate
    2. Binance live OI
    3. CSV overrides for anything missing (basis, etc.)
    """
    result = {"funding_ann": None, "oi_pct": None, "basis_pct": None}

    # Live: Binance funding rate
    funding = fetch_binance_funding_rate()
    if funding is not None:
        result["funding_ann"] = funding

    # Live: Binance OI
    oi_pct = fetch_binance_oi_pct(btc_price)
    if oi_pct is not None:
        result["oi_pct"] = oi_pct

    # CSV fallback for anything still missing (especially basis)
    if LEVERAGE_CSV.exists():
        try:
            df = pd.read_csv(LEVERAGE_CSV)
            row = df.iloc[-1]
            for key in ["funding_ann", "oi_pct", "basis_pct"]:
                if result[key] is None and pd.notna(row.get(key)):
                    result[key] = float(row[key])
        except Exception:
            pass

    return result


def load_etf_flows() -> pd.DataFrame | None:
    """Load ETF flow data from CSV."""
    if ETF_FLOW_CSV.exists():
        try:
            return pd.read_csv(ETF_FLOW_CSV, parse_dates=["date"])
        except Exception:
            pass
    return None


def load_portfolio_drawdown() -> float | None:
    """Load portfolio drawdown from CSV."""
    if PORTFOLIO_NAV_CSV.exists():
        try:
            df = pd.read_csv(PORTFOLIO_NAV_CSV, parse_dates=["date"])
            nav = df.set_index("date")["nav"]
            peak = nav.expanding().max()
            dd = ((nav - peak) / peak * 100).iloc[-1]
            return float(dd)
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Main adapter: compute all indicators and classify
# ---------------------------------------------------------------------------

def compute_current_regime() -> dict:
    """
    Run the full pipeline: fetch data, compute all indicators, classify.

    Returns dict with:
        regime, risk_on_score, choppy_score, risk_off_score,
        local_top_score, local_bottom_score,
        family_scores_df, indicator_audit_df, summary_text,
        btc_weekly (the series, for historical chart)
    """
    # Fetch raw data
    btc_weekly = fetch_btc_weekly(start="2024-01-01")
    btc_daily = fetch_btc_daily(start="2024-01-01")
    token_weekly = fetch_token_weekly_closes()
    stablecoin = fetch_stablecoin_supply()
    dat_df = load_dat_mnav()
    etf_flows_df = load_etf_flows()
    # Get BTC price for OI % calc
    btc_price = float(btc_daily.iloc[-1]) if not btc_daily.empty else None
    lev = load_leverage_overrides(btc_price=btc_price)
    onchain = fetch_onchain_overrides()
    dd = load_portfolio_drawdown()

    # Macro
    spy_w = fetch_equity_weekly("SPY")
    qqq_w = fetch_equity_weekly("QQQ")
    dxy_latest, dxy_4w_chg = fetch_dxy_data()
    real_yield_chg = fetch_real_yields_4w_change()
    ai_ret = fetch_ai_basket_return_8w()

    # ETF flow derived values
    btc_weekly_ret = None
    etf_5d_flow = None
    if not btc_weekly.empty and len(btc_weekly) >= 2:
        btc_weekly_ret = (btc_weekly.iloc[-1] / btc_weekly.iloc[-2] - 1) * 100
    if etf_flows_df is not None and not etf_flows_df.empty:
        etf_5d_flow = etf_flows_df.sort_values("date").tail(5)["btc_net_flow_usd"].sum() / 1e9

    # Compute all indicators
    indicators = []

    # BTC trend / momentum
    if not btc_weekly.empty and len(btc_weekly) >= 21:
        indicators.extend([
            btc_vs_20w_sma(btc_weekly),
            sma_20w_slope(btc_weekly),
            btc_4w_return(btc_weekly),
            btc_12w_return(btc_weekly),
        ])

    # Market breadth
    if not token_weekly.empty:
        indicators.extend([
            pct_tokens_above_20w_sma(token_weekly),
            median_token_vs_btc_8w(token_weekly, btc_weekly),
            new_highs_vs_lows(token_weekly),
        ])

    # ETF + DAT flows
    indicators.extend([
        btc_etf_flows(etf_flows_df),
        eth_etf_flows(etf_flows_df),
        etf_price_response(btc_weekly_ret, etf_5d_flow),
        mstr_mnav(dat_df),
        bmnr_mnav(dat_df),
        purr_mnav(dat_df),
        dat_issuance_capacity(),  # manual for now
        dat_treasury_purchases(),  # manual for now
        strc_credit_stress(),  # manual for now
    ])

    # Stablecoin liquidity
    indicators.append(stablecoin_supply_growth(stablecoin))

    # Leverage / volatility
    if not btc_daily.empty:
        indicators.append(realised_vol_8w(btc_daily))
    indicators.extend([
        funding_rates_indicator(lev.get("funding_ann")),
        oi_to_mcap_indicator(lev.get("oi_pct")),
        basis_indicator(lev.get("basis_pct")),
    ])

    # Macro / AI
    indicators.extend([
        spy_qqq_trend(spy_w, qqq_w),
        dxy_trend_indicator(dxy_latest, dxy_4w_chg),
        real_yields_indicator(real_yield_chg),
        ai_macro_bias(ai_ret),
    ])

    # Valuation / sentiment
    indicators.extend([
        mvrv_nupl_indicator(onchain.get("mvrv_z"), onchain.get("nupl")),
        retail_froth_indicator(onchain.get("coinbase_rank")),
        portfolio_drawdown_indicator(dd),
    ])

    # Classify
    result = classify_regime(indicators)
    result["btc_weekly"] = btc_weekly
    return result


def compute_weekly_regime_df(btc_weekly: pd.Series | None = None) -> pd.DataFrame:
    """
    Build historical weekly regime DataFrame for chart shading.

    Uses all historically-computable indicators so the regime actually varies:
    - BTC trend / momentum (4 indicators, 20% weight)
    - Realised vol from daily data (1 indicator, part of Leverage/vol 10%)
    - Stablecoin supply from DefiLlama cache (1 indicator, 10% weight)
    - Macro: SPY/QQQ trend, DXY, real yields, AI basket (4 indicators, 10%)
    - ETF + DAT: uses current-snapshot values as constants (9 indicators, 30%)
    - Market breadth: from token data if available (3 indicators, 15%)
    - Valuation/sentiment: from CSV overrides/defaults (3 indicators, 5%)

    This gives all 7 families a voice in historical classification.
    """
    if btc_weekly is None:
        btc_weekly = fetch_btc_weekly(start="2024-01-01")

    if btc_weekly.empty:
        return pd.DataFrame()

    # Pre-fetch data used across all weeks
    btc_daily = fetch_btc_daily(start="2024-01-01")
    stablecoin = fetch_stablecoin_supply()
    token_weekly = fetch_token_weekly_closes(start="2024-01-01")
    dat_df = load_dat_mnav()
    etf_flows_df = load_etf_flows()

    # Macro data (weekly series for historical lookback)
    spy_w = fetch_equity_weekly("SPY", start="2024-01-01")
    qqq_w = fetch_equity_weekly("QQQ", start="2024-01-01")

    # Current-snapshot values used as constants for historical (imperfect but
    # far better than omitting these families entirely)
    btc_price = float(btc_daily.iloc[-1]) if not btc_daily.empty else None
    lev = load_leverage_overrides(btc_price=btc_price)
    onchain = fetch_onchain_overrides()
    dd = load_portfolio_drawdown()

    # Pre-fetch macro snapshot values (expensive HTTP calls, do once)
    _dxy_latest, _dxy_4w_chg = fetch_dxy_data()
    _real_yield_chg = fetch_real_yields_4w_change()
    _ai_ret = fetch_ai_basket_return_8w()

    def multi_family_indicators(btc_slice):
        """Compute indicators from multiple families for a historical week."""
        inds = []

        # ── BTC trend / momentum (Family 1, weight 20%) ──
        inds.extend([
            btc_vs_20w_sma(btc_slice),
            sma_20w_slope(btc_slice),
            btc_4w_return(btc_slice),
            btc_12w_return(btc_slice),
        ])

        # ── Market breadth (Family 2, weight 15%) ──
        # Use token data up to the same date as the btc_slice end
        if not token_weekly.empty:
            end_date = btc_slice.index[-1]
            tok_slice = token_weekly.loc[:end_date]
            btc_for_breadth = btc_slice
            if not tok_slice.empty and len(tok_slice) >= 20:
                inds.extend([
                    pct_tokens_above_20w_sma(tok_slice),
                    median_token_vs_btc_8w(tok_slice, btc_for_breadth),
                    new_highs_vs_lows(tok_slice),
                ])

        # ── ETF + DAT flows (Family 3, weight 30%) ──
        # Use current snapshot values (ETF flows and DAT mNAV don't have deep
        # history in our CSVs, but including them at current values is better
        # than having the 30%-weight family be empty)
        btc_weekly_ret = None
        etf_5d_flow = None
        if len(btc_slice) >= 2:
            btc_weekly_ret = (btc_slice.iloc[-1] / btc_slice.iloc[-2] - 1) * 100
        if etf_flows_df is not None and not etf_flows_df.empty:
            etf_5d_flow = etf_flows_df.sort_values("date").tail(5)["btc_net_flow_usd"].sum() / 1e9

        inds.extend([
            btc_etf_flows(etf_flows_df),
            eth_etf_flows(etf_flows_df),
            etf_price_response(btc_weekly_ret, etf_5d_flow),
            mstr_mnav(dat_df),
            bmnr_mnav(dat_df),
            purr_mnav(dat_df),
            dat_issuance_capacity(),
            dat_treasury_purchases(),
            strc_credit_stress(),
        ])

        # ── Stablecoin liquidity (Family 4, weight 10%) ──
        inds.append(stablecoin_supply_growth(stablecoin))

        # ── Leverage / volatility (Family 5, weight 10%) ──
        # Realised vol computed from daily data up to this week's date
        if not btc_daily.empty:
            end_date = btc_slice.index[-1]
            daily_slice = btc_daily.loc[:end_date]
            if len(daily_slice) >= 56:
                inds.append(realised_vol_8w(daily_slice))

        inds.extend([
            funding_rates_indicator(lev.get("funding_ann")),
            oi_to_mcap_indicator(lev.get("oi_pct")),
            basis_indicator(lev.get("basis_pct")),
        ])

        # ── Macro / AI (Family 6, weight 10%) ──
        # SPY/QQQ: use data up to this week
        spy_slice = spy_w.loc[:btc_slice.index[-1]] if spy_w is not None else None
        qqq_slice = qqq_w.loc[:btc_slice.index[-1]] if qqq_w is not None else None
        inds.append(spy_qqq_trend(spy_slice, qqq_slice))

        # DXY, real yields, AI basket: use pre-fetched snapshot
        inds.extend([
            dxy_trend_indicator(_dxy_latest, _dxy_4w_chg),
            real_yields_indicator(_real_yield_chg),
            ai_macro_bias(_ai_ret),
        ])

        # ── Valuation / sentiment (Family 7, weight 5%) ──
        inds.extend([
            mvrv_nupl_indicator(onchain.get("mvrv_z"), onchain.get("nupl")),
            retail_froth_indicator(onchain.get("coinbase_rank")),
            portfolio_drawdown_indicator(dd),
        ])

        return inds

    return classify_weekly_history(
        btc_weekly,
        indicator_func=multi_family_indicators,
        start_date="2025-01-06",
    )


def build_market_kpis_df() -> pd.DataFrame:
    """Build market KPIs DataFrame."""
    dxy_latest, _ = fetch_dxy_data()
    m2 = fetch_m2_mom()

    # Gold
    gold_close = None
    gold_status = "Unavailable"
    if yf:
        try:
            hist = yf.Ticker("GC=F").history(period="5d", interval="1d", auto_adjust=False)
            gold_close = hist["Close"].iloc[-1]
            gold_status = "Live via yfinance"
        except Exception:
            pass
    if gold_close is None:
        try:
            resp = requests.get("https://stooq.com/q/l/?s=xauusd&f=sd2t2ohlcv&h&e=csv", timeout=10)
            df = pd.read_csv(StringIO(resp.text))
            gold_close = float(df.iloc[0]["Close"])
            gold_status = "Live via Stooq"
        except Exception:
            gold_close = float("nan")
            gold_status = "Unavailable"

    # STRC
    strc_close = None
    strc_status = "Unavailable"
    try:
        resp = requests.get("https://stooq.com/q/l/?s=strc.us&f=sd2t2ohlcv&h&e=csv", timeout=10)
        df = pd.read_csv(StringIO(resp.text))
        c = df.iloc[0]["Close"]
        if pd.notna(c) and str(c) != "N/D":
            strc_close = float(c)
            strc_status = "Live via Stooq"
    except Exception:
        pass
    if strc_close is None and yf:
        try:
            hist = yf.Ticker("STRC").history(period="5d", auto_adjust=False)
            strc_close = hist["Close"].iloc[-1]
            strc_status = "Live via yfinance"
        except Exception:
            pass

    rows = [
        {"KPI": "DXY", "Value": f"{dxy_latest:.2f}" if dxy_latest else "n/a",
         "Source": "yfinance / Stooq", "Status": "Live" if dxy_latest else "Unavailable",
         "Why it matters": "Rising dollar is usually a headwind for crypto beta."},
        {"KPI": "Gold", "Value": f"${gold_close:,.0f}" if pd.notna(gold_close) else "n/a",
         "Source": "yfinance / Stooq", "Status": gold_status,
         "Why it matters": "Gold strength can indicate debasement demand or macro stress."},
        {"KPI": "M2 MoM", "Value": f"{m2['value']:.2f}%",
         "Source": "FRED WM2NS", "Status": m2["status"],
         "Why it matters": "Broad liquidity growth supports risk assets with a lag."},
        {"KPI": "STRC price", "Value": f"${strc_close:.2f}" if strc_close else "n/a",
         "Source": "Stooq / yfinance", "Status": strc_status,
         "Why it matters": "STRC trading near par suggests the Strategy credit stack is orderly."},
        {"KPI": "STRC next payout", "Value": "2026-04-30",
         "Source": "strategy.com/stretch", "Status": "Reference",
         "Why it matters": "Payout timing and rate help monitor stress in Strategy's preferred stack."},
    ]
    return pd.DataFrame(rows)
