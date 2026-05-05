"""Market update tab for the screener dashboard."""
from __future__ import annotations

import json
import math
import os
import pickle
import re
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from crypto_factor_model.config import COINGECKO_API_KEY, COINGECKO_BASE_URL

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional runtime dependency
    yf = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFILLAMA_CACHE_DIR = PROJECT_ROOT / "cache" / "defillama"
MARKET_UPDATE_CACHE_DIR = PROJECT_ROOT / "cache" / "market_update"
MARKET_UPDATE_SNAPSHOT_PATH = MARKET_UPDATE_CACHE_DIR / "latest.pkl"
MARKET_UPDATE_MAX_AGE_HOURS = 30

FRED_ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"
DEFAULT_FRED_API_KEY = "REDACTED_FRED_API_KEY"
COINGECKO_GLOBAL_ENDPOINT = f"{COINGECKO_BASE_URL.rstrip('/')}/global"
CMC_CYCLE_URL = "https://coinmarketcap.com/charts/crypto-market-cycle-indicators/"
SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_COMPANY_CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{tag}.json"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "CryptoFactorModel/1.0 asad.hussain@example.com")
SEC_REQUEST_TIMEOUT_SECONDS = 30
SEC_REQUEST_RETRIES = 2
BITCOIN_COM_CHARTS_BASE_URL = "https://charts.bitcoin.com/api/v1/charts"

REQUEST_HEADERS = {
    "User-Agent": "CryptoFactorModel/1.0 (+local dashboard)",
    "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
}
COINGECKO_HEADERS = {**REQUEST_HEADERS, "x-cg-pro-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else REQUEST_HEADERS

SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate",
}

FRED_SERIES = [
    ("DGS10", "10-Year Treasury Yield", "%"),
    ("DGS2", "2-Year Treasury Yield", "%"),
    ("T10Y2Y", "10Y-2Y Spread", "%"),
    ("DFII10", "10-Year Real Yield (TIPS)", "%"),
    ("CPIAUCSL", "CPI (YoY % Change)", "%"),
    ("M2SL", "M2 Money Supply", ""),
    ("DTWEXBGS", "USD Index (Broad, Trade-Weighted)", ""),
]

BENCHMARK_ASSETS = [
    ("Bitcoin", "crypto", "BTCUSDT", "#ff9f1a"),
    ("Ethereum", "crypto", "ETHUSDT", "#9aa6c8"),
    ("GLD", "equity", "GLD", "#ffd21f"),
    ("SPY", "equity", "SPY", "#e60035"),
    ("QQQ", "equity", "QQQ", "#20a6c8"),
    ("Coinbase Global, Inc", "equity", "COIN", "#1565ff"),
    ("Robinhood", "equity", "HOOD", "#caff00"),
]

CRYPTO_EQUITIES = ["MSTR", "STRC", "BMNR", "COIN", "HOOD", "CRCL"]
CRYPTO_EQUITY_LOGOS = {
    "MSTR": "https://logo.clearbit.com/strategy.com",
    "STRC": "https://logo.clearbit.com/strategy.com",
    "BMNR": "https://logo.clearbit.com/bitminetech.io",
    "COIN": "https://logo.clearbit.com/coinbase.com",
    "HOOD": "https://logo.clearbit.com/robinhood.com",
    "CRCL": "https://logo.clearbit.com/circle.com",
}
HYPERSCALER_CAPEX_COMPANIES = [
    {
        "Ticker": "AMZN",
        "Company": "Amazon",
        "CIK": 1_018_724,
        "Tag": "PaymentsToAcquireProductiveAssets",
        "Color": "#4E9CDF",
    },
    {
        "Ticker": "GOOGL",
        "Company": "Alphabet",
        "CIK": 1_652_044,
        "Tag": "PaymentsToAcquirePropertyPlantAndEquipment",
        "Color": "#7A2E3B",
    },
    {
        "Ticker": "META",
        "Company": "Meta",
        "CIK": 1_326_801,
        "Tag": "PaymentsToAcquirePropertyPlantAndEquipment",
        "Color": "#C7B56B",
    },
    {
        "Ticker": "MSFT",
        "Company": "Microsoft",
        "CIK": 789_019,
        "Tag": "PaymentsToAcquirePropertyPlantAndEquipment",
        "Color": "#625BD6",
    },
    {
        "Ticker": "ORCL",
        "Company": "Oracle",
        "CIK": 1_341_439,
        "Tag": "PaymentsToAcquirePropertyPlantAndEquipment",
        "Color": "#2F5B4C",
    },
    {
        "Ticker": "AAPL",
        "Company": "Apple",
        "CIK": 320_193,
        "Tag": "PaymentsToAcquirePropertyPlantAndEquipment",
        "Color": "#D8D2C0",
    },
]
TOKEN_MOVER_MIN_FDV = 100_000_000
TOKEN_MOVER_MIN_30D_VOLUME = 10_000_000

BTC_MACRO_RATIO_ASSETS = [
    ("BTC/Gold", "GC=F", "GLD", "Gold futures / GLD fallback", "#f2c75c"),
    ("BTC/SPX", "^GSPC", "SPY", "S&P 500 / SPY fallback", "#5f9cff"),
    ("BTC/NDX", "^NDX", "QQQ", "Nasdaq 100 / QQQ fallback", "#caff00"),
]
INDEX_BENCHMARK_START_DATE = "2026-01-01"

BITCOIN_COM_CHART_SPECS = [
    ("pi-cycle-top", "Pi Cycle Top", "pi-cycle-top"),
    ("rainbow", "Rainbow Chart", "rainbow"),
    ("stock-to-flow", "Stock-to-Flow", "s2f"),
    ("golden-ratio", "Golden Ratio", "golden-ratio"),
    ("mayer-multiple", "Mayer Multiple", "mayer"),
    ("m2", "Global M2 vs Bitcoin", "m2"),
]

SEC_NOTABLE_FORMS = {
    "8-K": "Current report",
    "10-Q": "Quarterly report",
    "10-K": "Annual report",
    "S-1": "Registration statement",
    "S-1/A": "Registration amendment",
    "S-3": "Shelf registration",
    "S-3ASR": "Automatic shelf registration",
    "424B2": "Prospectus supplement",
    "424B3": "Prospectus supplement",
    "424B4": "Prospectus supplement",
    "424B5": "Prospectus supplement",
    "DEF 14A": "Proxy statement",
    "DEFA14A": "Additional proxy materials",
    "SC 13D": "Beneficial ownership",
    "SC 13D/A": "Beneficial ownership amendment",
    "SC 13G": "Passive ownership",
    "SC 13G/A": "Passive ownership amendment",
}

BRIDGE_FLOW_CHAINS = [
    "Ethereum",
    "Base",
    "Arbitrum",
    "Optimism",
    "Polygon",
    "Solana",
    "Avalanche",
    "BSC",
    "Tron",
    "Hyperliquid",
    "World Chain",
    "Celo",
    "Blast",
    "Mantle",
    "zkSync Era",
    "Linea",
    "edgeX",
    "Berachain",
    "Starknet",
    "Ink",
    "TON",
    "Sui",
    "Plasma",
    "Sei",
    "Aptos",
    "Monad",
]

STABLECOIN_SUPPLY_CHAINS = [
    "Ethereum",
    "Tron",
    "Solana",
    "Arbitrum",
    "Base",
    "BSC",
    "Polygon",
    "Avalanche",
    "Optimism",
    "Sui",
    "Aptos",
    "TON",
    "Sei",
    "Plasma",
    "Hyperliquid",
    "Mantle",
    "Celo",
    "Blast",
    "Ink",
    "Berachain",
    "Starknet",
    "zkSync Era",
    "Linea",
    "World Chain",
    "MegaETH",
]

DEFILLAMA_BRIDGE_CHAIN_PAGES = [
    "https://defillama.com/bridges/chains",
    "https://defillama2.llamao.fi/bridges/chains",
]

CHAIN_NAME_OVERRIDES = {
    "bsc": "BNB Chain",
    "hyperliquid l1": "Hyperliquid",
    "op mainnet": "OP Mainnet",
    "polygon pos": "Polygon PoS",
    "zksync era": "zkSync Era",
    "edgex l1": "edgeX",
}


def _fred_api_key() -> str:
    """Prefer env/secrets, falling back to the key supplied for this local app."""
    if os.getenv("FRED_API_KEY"):
        return str(os.environ["FRED_API_KEY"])
    try:
        secret = st.secrets.get("FRED_API_KEY")
        if secret:
            return str(secret)
    except Exception:
        pass
    return DEFAULT_FRED_API_KEY


def _parse_value(value: Any) -> float:
    if value in {None, "", "."}:
        raise ValueError("missing observation value")
    return float(value)


def _num(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        out = float(value)
        return out if math.isfinite(out) else float("nan")
    except Exception:
        return float("nan")


def _sum_nested(value: Any) -> float:
    if isinstance(value, dict):
        return float(np.nansum([_sum_nested(v) for v in value.values()]))
    if isinstance(value, list):
        return float(np.nansum([_sum_nested(v) for v in value]))
    return _num(value)


def _format_macro_value(value: float, suffix: str) -> str:
    if suffix:
        return f"{value:.2f}{suffix}"
    return f"{value:,.2f}"


def _compact_usd(value: Any) -> str:
    value = _num(value)
    if math.isnan(value):
        return "n/a"
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000:
        return f"{sign}${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{sign}${value / 1_000:.1f}K"
    return f"{sign}${value:,.0f}"


def _format_token_price(value: Any) -> str:
    value = _num(value)
    if math.isnan(value):
        return "n/a"
    if abs(value) < 1:
        return f"${value:,.4f}"
    return f"${value:,.2f}"


def _format_pct(value: Any) -> str:
    value = _num(value)
    return "n/a" if math.isnan(value) else f"{value:+.1f}%"


def _format_ratio(value: Any) -> str:
    value = _num(value)
    return "n/a" if math.isnan(value) else f"{value:.4f}"


def _naive_datetime_index(index: Any) -> pd.DatetimeIndex:
    out = pd.to_datetime(index)
    if not isinstance(out, pd.DatetimeIndex):
        out = pd.DatetimeIndex(out)
    if out.tz is not None:
        return out.tz_localize(None)
    return out


def _canonical_chain_name(name: Any) -> str:
    text = str(name or "").strip()
    if not text:
        return "Unknown"
    return CHAIN_NAME_OVERRIDES.get(text.lower(), text[:1].upper() + text[1:])


def _requests_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: int = 15,
    headers: dict[str, str] | None = None,
) -> Any:
    resp = requests.get(url, params=params, headers=headers or REQUEST_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _fred_latest_observation(
    series_id: str,
    api_key: str,
    observation_end: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "series_id": series_id,
        "api_key": api_key,
        "sort_order": "desc",
        "limit": 1,
        "file_type": "json",
    }
    if observation_end:
        params["observation_end"] = observation_end

    resp = requests.get(FRED_ENDPOINT, params=params, headers=REQUEST_HEADERS, timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    observations = payload.get("observations", [])
    if not observations:
        raise ValueError("no observations returned")
    obs = observations[0]
    return {"date": obs.get("date"), "value": _parse_value(obs.get("value"))}


@st.cache_data(ttl=3600, show_spinner=False)
def load_macro_monitor() -> tuple[pd.DataFrame, datetime, list[str]]:
    """Fetch and cache the latest public FRED observations for the macro tab."""
    api_key = _fred_api_key()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for series_id, label, suffix in FRED_SERIES:
        try:
            latest = _fred_latest_observation(series_id, api_key)
            value = latest["value"]
            detail = ""
            if series_id == "CPIAUCSL":
                prior_target = (pd.Timestamp(latest["date"]) - pd.DateOffset(months=12)).strftime("%Y-%m-%d")
                prior = _fred_latest_observation(series_id, api_key, observation_end=prior_target)
                value = (value / prior["value"] - 1.0) * 100.0
                detail = f"YoY from {prior['date']}"

            rows.append(
                {
                    "Series ID": series_id,
                    "Label": label,
                    "Value": value,
                    "Display Value": _format_macro_value(value, suffix),
                    "Observation Date": latest["date"],
                    "Detail": detail,
                }
            )
        except Exception as exc:
            errors.append(f"{label} ({series_id}) failed: {exc}")

    return pd.DataFrame(rows), datetime.now(timezone.utc), errors


def _fetch_binance_ohlcv(symbol: str, start: str) -> pd.DataFrame:
    try:
        from crypto_factor_model.clients.binance import BinanceClient

        df = BinanceClient().get_klines(symbol, interval="1d", start=start, use_cache=False)
        if df.empty:
            return pd.DataFrame()
        out = df.rename(columns={"close": "Close", "quote_volume": "Volume"})
        out = out[["Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
        out.index = _naive_datetime_index(out.index)
        return out
    except Exception:
        return pd.DataFrame()


def _fetch_yfinance_ohlcv(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        hist = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=False)
        if hist.empty or "Close" not in hist:
            return pd.DataFrame()
        cols = ["Close"] + (["Volume"] if "Volume" in hist else [])
        out = hist[cols].apply(pd.to_numeric, errors="coerce")
        out.index = _naive_datetime_index(out.index)
        return out
    except Exception:
        return pd.DataFrame()


def _fetch_stooq_ohlcv(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    try:
        start_dt = pd.Timestamp(start).strftime("%Y%m%d")
        end_dt = pd.Timestamp(end or pd.Timestamp.now()).strftime("%Y%m%d")
        stooq_symbol = f"{symbol.lower()}.us"
        resp = requests.get(
            "https://stooq.com/q/d/l/",
            params={"s": stooq_symbol, "d1": start_dt, "d2": end_dt, "i": "d"},
            headers=REQUEST_HEADERS,
            timeout=12,
        )
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        if df.empty or "Close" not in df:
            return pd.DataFrame()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        keep = ["Close"] + (["Volume"] if "Volume" in df else [])
        out = df[keep].apply(pd.to_numeric, errors="coerce")
        out.index = _naive_datetime_index(out.index)
        return out
    except Exception:
        return pd.DataFrame()


def _fetch_equity_ohlcv(symbol: str, start: str, end: str | None = None) -> tuple[pd.DataFrame, str]:
    df = _fetch_yfinance_ohlcv(symbol, start, end)
    if not df.empty:
        return df, "yfinance"
    df = _fetch_stooq_ohlcv(symbol, start, end)
    if not df.empty:
        return df, "Stooq"
    return pd.DataFrame(), "Unavailable"


@st.cache_data(ttl=3600, show_spinner=False)
def load_index_benchmark() -> tuple[pd.DataFrame, list[str]]:
    start = INDEX_BENCHMARK_START_DATE
    start_ts = pd.Timestamp(start)
    frames = []
    errors: list[str] = []

    for label, source_type, symbol, _color in BENCHMARK_ASSETS:
        if source_type == "crypto":
            df = _fetch_binance_ohlcv(symbol, start)
            source = "Binance"
        else:
            df, source = _fetch_equity_ohlcv(symbol, start)
        if df.empty or df["Close"].dropna().shape[0] < 2:
            errors.append(f"{label} unavailable")
            continue
        series = df["Close"].dropna().sort_index()
        series.index = _naive_datetime_index(series.index)
        series = series[series.index >= start_ts]
        if len(series) < 2:
            errors.append(f"{label} insufficient YTD history")
            continue
        indexed = (series / series.iloc[0]) * 100.0
        frames.append(indexed.rename(label))

    ratio_payload = load_btc_macro_ratios()
    ratio_data = ratio_payload.get("data", pd.DataFrame()) if isinstance(ratio_payload, dict) else pd.DataFrame()
    if isinstance(ratio_data, pd.DataFrame) and not ratio_data.empty and "date" in ratio_data:
        ratio_data = ratio_data.copy()
        ratio_data["date"] = pd.to_datetime(ratio_data["date"], errors="coerce")
        ratio_data = ratio_data.dropna(subset=["date"]).set_index("date").sort_index()
        for label, _primary, _fallback, _source, _color in BTC_MACRO_RATIO_ASSETS:
            if label not in ratio_data:
                continue
            series = pd.to_numeric(ratio_data[label], errors="coerce").dropna()
            series = series[series.index >= start_ts]
            if len(series) < 2:
                continue
            frames.append((series / series.iloc[0] * 100.0).rename(label))
    if isinstance(ratio_payload, dict):
        errors.extend([f"BTC ratio: {err}" for err in ratio_payload.get("errors", [])])

    if not frames:
        return pd.DataFrame(), errors
    out = pd.concat(frames, axis=1).sort_index().ffill().dropna(how="all")
    out = out.dropna()
    if out.empty:
        return pd.DataFrame(), errors
    out = out.div(out.iloc[0]).mul(100.0)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.reset_index(names="date"), errors


@st.cache_data(ttl=900, show_spinner=False)
def load_market_structure() -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    rows: dict[str, Any] = {
        "eth_btc": float("nan"),
        "eth_btc_7d_pct": float("nan"),
        "btc_d": float("nan"),
        "eth_d": float("nan"),
        "btc_d_source": "Unavailable",
    }
    start = (pd.Timestamp.now().normalize() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    btc = _fetch_binance_ohlcv("BTCUSDT", start)
    eth = _fetch_binance_ohlcv("ETHUSDT", start)
    if not btc.empty and not eth.empty:
        aligned = pd.concat([eth["Close"].rename("eth"), btc["Close"].rename("btc")], axis=1).dropna()
        if len(aligned) >= 8:
            ratio = aligned["eth"] / aligned["btc"]
            rows["eth_btc"] = float(ratio.iloc[-1])
            rows["eth_btc_7d_pct"] = float((ratio.iloc[-1] / ratio.iloc[-8] - 1.0) * 100.0)
    else:
        errors.append("ETH/BTC unavailable")

    try:
        payload = _requests_json(COINGECKO_GLOBAL_ENDPOINT, timeout=12, headers=COINGECKO_HEADERS)
        percentages = payload.get("data", {}).get("market_cap_percentage", {})
        rows["btc_d"] = _num(percentages.get("btc"))
        rows["eth_d"] = _num(percentages.get("eth"))
        rows["btc_d_source"] = "CoinGecko Pro global endpoint" if COINGECKO_API_KEY else "CoinGecko free global endpoint"
    except Exception as exc:
        errors.append(f"BTC.D unavailable: {exc}")

    return rows, errors


@st.cache_data(ttl=1800, show_spinner=False)
def load_crypto_equity_changes() -> pd.DataFrame:
    start = (pd.Timestamp.now().normalize() - pd.Timedelta(days=70)).strftime("%Y-%m-%d")
    rows = []
    for symbol in CRYPTO_EQUITIES:
        df, source = _fetch_equity_ohlcv(symbol, start)
        close = df["Close"].dropna().astype(float) if not df.empty and "Close" in df else pd.Series(dtype=float)
        volume = df["Volume"].dropna().astype(float) if not df.empty and "Volume" in df else pd.Series(dtype=float)
        latest = close.iloc[-1] if len(close) else float("nan")
        price_1d = (latest / close.iloc[-2] - 1.0) * 100.0 if len(close) >= 2 else float("nan")
        price_5d = (latest / close.iloc[-6] - 1.0) * 100.0 if len(close) >= 6 else float("nan")
        latest_volume = volume.iloc[-1] if len(volume) else float("nan")
        latest_5d_vol = volume.tail(5).mean() if len(volume) >= 5 else float("nan")
        prior_5d_vol = volume.iloc[-10:-5].mean() if len(volume) >= 10 else float("nan")
        volume_5d = (latest_5d_vol / prior_5d_vol - 1.0) * 100.0 if prior_5d_vol and not math.isnan(prior_5d_vol) else float("nan")
        rows.append(
            {
                "Ticker": symbol,
                "Last": latest,
                "1D Price": price_1d,
                "5D Price": price_5d,
                "5D Volume": volume_5d,
                "Latest Volume": latest_volume,
                "Source": source,
                "As Of": str(close.index[-1].date()) if len(close) else "n/a",
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def load_btc_macro_ratios() -> dict[str, Any]:
    start = (pd.Timestamp.now().normalize() - pd.Timedelta(days=365 * 5 + 14)).strftime("%Y-%m-%d")
    btc = _fetch_binance_ohlcv("BTCUSDT", start)
    if btc.empty:
        return {"data": pd.DataFrame(), "sources": {}, "errors": ["BTC price unavailable"]}
    btc_close = btc["Close"].dropna().astype(float).rename("BTC")
    frames = []
    sources: dict[str, str] = {}
    errors: list[str] = []

    for label, primary_symbol, fallback_symbol, source_label, _color in BTC_MACRO_RATIO_ASSETS:
        asset, source = _fetch_yfinance_ohlcv(primary_symbol, start), "yfinance"
        if asset.empty:
            asset, source = _fetch_equity_ohlcv(fallback_symbol, start)
            if not asset.empty:
                source = f"{source} fallback {fallback_symbol}"
        if asset.empty:
            errors.append(f"{label}: reference asset unavailable")
            continue
        ref = asset["Close"].dropna().astype(float).rename("Reference")
        aligned = pd.concat([btc_close, ref], axis=1).sort_index().ffill().dropna()
        if aligned.empty:
            errors.append(f"{label}: no overlapping history")
            continue
        ratio = (aligned["BTC"] / aligned["Reference"]).dropna()
        ratio = ratio.tail(365 * 5)
        if len(ratio) < 2:
            errors.append(f"{label}: insufficient history")
            continue
        indexed = (ratio / ratio.iloc[0] * 100.0).rename(label)
        frames.append(indexed)
        sources[label] = f"{source_label}; {source}"

    if not frames:
        return {"data": pd.DataFrame(), "sources": sources, "errors": errors}
    out = pd.concat(frames, axis=1).sort_index().ffill().dropna(how="all")
    out = out.reset_index(names="date")
    return {"data": out, "sources": sources, "errors": errors}


def _bitcoin_com_chart_url(endpoint: str) -> str:
    return f"{BITCOIN_COM_CHARTS_BASE_URL}/{endpoint}"


def _timestamp_to_datetime(values: pd.Series) -> pd.Series:
    nums = pd.to_numeric(values, errors="coerce")
    unit = "ms" if nums.dropna().median() > 10_000_000_000 else "s"
    return pd.to_datetime(nums, unit=unit, errors="coerce")


def _points_frame(points: list[dict[str, Any]], columns: dict[str, str], max_points: int = 1825) -> pd.DataFrame:
    if not isinstance(points, list) or not points:
        return pd.DataFrame(columns=["date", *columns.values()])
    df = pd.DataFrame(points)
    if "timestamp" not in df:
        return pd.DataFrame(columns=["date", *columns.values()])
    out = pd.DataFrame({"date": _timestamp_to_datetime(df["timestamp"])})
    for source, target in columns.items():
        out[target] = pd.to_numeric(df.get(source), errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    now = pd.Timestamp.now() + pd.Timedelta(days=1)
    out = out[out["date"] <= now]
    return out.tail(max_points).reset_index(drop=True)


def _merge_chart_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.DataFrame()
    for frame in frames:
        if frame.empty or "date" not in frame:
            continue
        merged = frame if merged.empty else merged.merge(frame, on="date", how="outer")
    if merged.empty:
        return merged
    return merged.sort_values("date").reset_index(drop=True)


def _parse_bitcoin_com_payload(chart_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data", {}) if isinstance(payload, dict) else {}
    parsed: dict[str, Any] = {
        "title": next((name for cid, name, _endpoint in BITCOIN_COM_CHART_SPECS if cid == chart_id), chart_id),
        "description": payload.get("description", ""),
        "current": payload.get("current") or data.get("currentZone") or payload.get("currentZone") or {},
        "metadata": payload.get("metadata", {}),
        "data": pd.DataFrame(),
    }

    if chart_id == "pi-cycle-top":
        parsed["data"] = _merge_chart_frames(
            [
                _points_frame(data.get("price", []), {"price": "BTC Price"}),
                _points_frame(data.get("ma111", []), {"value": "111DMA"}),
                _points_frame(data.get("ma350x2", []), {"value": "350DMA x2"}),
            ]
        )
    elif chart_id == "rainbow":
        parsed["data"] = _points_frame(data.get("price", []), {"price": "BTC Price"})
        bands = []
        for zone in data.get("zonesTimeSeries", []) or []:
            frame = _points_frame(zone.get("upper", []), {"value": str(zone.get("name") or "Band")})
            if not frame.empty:
                bands.append({"name": str(zone.get("name") or "Band"), "color": zone.get("color"), "data": frame})
        parsed["bands"] = bands
        parsed["current"] = data.get("currentZone", {})
    elif chart_id == "stock-to-flow":
        parsed["data"] = _points_frame(
            data.get("s2fData", []),
            {
                "actualPrice": "BTC Price",
                "predictedPrice": "S2F Model",
                "stockToFlowRatio": "S2F Ratio",
            },
        )
    elif chart_id == "golden-ratio":
        rows = []
        for point in data.get("goldenRatioData", []) or []:
            levels = point.get("levels", {}) if isinstance(point, dict) else {}
            rows.append(
                {
                    "date": _timestamp_to_datetime(pd.Series([point.get("timestamp")])).iloc[0],
                    "BTC Price": _num(point.get("price")),
                    "350DMA": _num(point.get("dma350")),
                    "1.6x": _num(levels.get("level_1_6") or levels.get("level1_6")),
                    "2x": _num(levels.get("level_2") or levels.get("level3_3")),
                    "3x": _num(levels.get("level_3") or levels.get("level5_0")),
                    "5x": _num(levels.get("level_5")),
                }
            )
        parsed["data"] = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date").tail(1825).reset_index(drop=True)
    elif chart_id == "mayer-multiple":
        parsed["data"] = _points_frame(
            data.get("mayerData", []),
            {
                "price": "BTC Price",
                "dma200": "200DMA",
                "mayerMultiple": "Mayer Multiple",
            },
        )
    elif chart_id == "m2":
        parsed["data"] = _points_frame(
            data.get("combined", []),
            {
                "btcNormalized": "BTC Indexed",
                "m2Normalized": "M2 Indexed",
            },
        )
        parsed["normalization"] = payload.get("normalization", {})
    return parsed


@st.cache_data(ttl=86400, show_spinner=False)
def load_bitcoin_com_charts() -> dict[str, Any]:
    charts: dict[str, Any] = {}
    errors: list[str] = []
    for chart_id, _title, endpoint in BITCOIN_COM_CHART_SPECS:
        try:
            payload = _requests_json(
                _bitcoin_com_chart_url(endpoint),
                params={"interval": "daily", "timespan": "5y", "limit": 2000},
                timeout=20,
            )
            if not payload.get("success", False):
                raise ValueError(payload.get("error") or "API returned success=false")
            charts[chart_id] = _parse_bitcoin_com_payload(chart_id, payload)
        except Exception as exc:
            errors.append(f"{chart_id}: {exc}")
    return {"as_of": datetime.now(timezone.utc).isoformat(), "charts": charts, "errors": errors, "source": "charts.bitcoin.com API"}


def _sec_json(url: str) -> Any:
    last_exc: Exception | None = None
    for attempt in range(SEC_REQUEST_RETRIES + 1):
        try:
            resp = requests.get(url, headers=SEC_HEADERS, timeout=SEC_REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= SEC_REQUEST_RETRIES:
                raise
            time.sleep(0.75 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("SEC request failed without an exception")


def _valid_sec_ytd_duration(fp: Any, duration_days: Any) -> bool:
    try:
        days = int(duration_days)
    except Exception:
        return False
    fp_text = str(fp or "").upper()
    windows = {
        "Q1": (70, 110),
        "Q2": (140, 220),
        "Q3": (230, 310),
        "FY": (340, 390),
    }
    low_high = windows.get(fp_text)
    if not low_high:
        return False
    low, high = low_high
    return low <= days <= high


def _hyperscaler_companyconcept_frame(payload: dict[str, Any], spec: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for fact in payload.get("units", {}).get("USD", []):
        if fact.get("form") not in {"10-K", "10-Q"}:
            continue
        start = pd.to_datetime(fact.get("start"), errors="coerce")
        end = pd.to_datetime(fact.get("end"), errors="coerce")
        filed = pd.to_datetime(fact.get("filed"), errors="coerce")
        value = _num(fact.get("val"))
        if pd.isna(start) or pd.isna(end) or pd.isna(filed) or math.isnan(value):
            continue
        duration_days = int((end - start).days + 1)
        rows.append(
            {
                "Ticker": spec["Ticker"],
                "Company": spec["Company"],
                "CIK": f"{int(spec['CIK']):010d}",
                "Tag": spec["Tag"],
                "Start": start.normalize(),
                "End": end.normalize(),
                "Filed": filed.normalize(),
                "FY": fact.get("fy"),
                "FP": str(fact.get("fp") or ""),
                "Form": fact.get("form"),
                "Accession": fact.get("accn"),
                "Frame": fact.get("frame"),
                "Duration Days": duration_days,
                "Value": value,
                "Value ($bn)": value / 1e9,
                "Source Type": "reported_ytd",
                "Color": spec["Color"],
            }
        )
    return pd.DataFrame(rows)


def _latest_hyperscaler_facts(facts: pd.DataFrame) -> pd.DataFrame:
    if facts.empty:
        return facts
    return (
        facts.sort_values(["Ticker", "Start", "End", "Filed", "Accession"])
        .drop_duplicates(["Ticker", "Start", "End"], keep="last")
        .sort_values(["Ticker", "Start", "End"])
        .reset_index(drop=True)
    )


def _derive_hyperscaler_quarterly_capex(facts: pd.DataFrame) -> pd.DataFrame:
    if facts.empty:
        return pd.DataFrame()

    cumulative = facts[facts.apply(lambda row: _valid_sec_ytd_duration(row["FP"], row["Duration Days"]), axis=1)].copy()
    cumulative = _latest_hyperscaler_facts(cumulative)
    if cumulative.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    standalone = cumulative[cumulative["Duration Days"].between(70, 110)].copy()
    for _, row in standalone.iterrows():
        item = row.to_dict()
        item["Source Type"] = "reported_quarter"
        rows.append(item)

    for (_ticker, _year_start), group in cumulative.groupby(["Ticker", "Start"], sort=False):
        group = group.sort_values("End")
        for _, current in group.iterrows():
            if int(current["Duration Days"]) <= 110:
                continue
            previous = group[group["End"] < current["End"]].sort_values("End")
            if previous.empty:
                continue
            previous_row = previous.iloc[-1]
            value = _num(current["Value"]) - _num(previous_row["Value"])
            if math.isnan(value) or value < 0:
                continue
            item = current.to_dict()
            item["Start"] = pd.Timestamp(previous_row["End"]) + pd.Timedelta(days=1)
            item["End"] = current["End"]
            item["Value"] = value
            item["Value ($bn)"] = value / 1e9
            item["Source Type"] = "derived_from_ytd_delta"
            item["Duration Days"] = int((pd.Timestamp(item["End"]) - pd.Timestamp(item["Start"])).days + 1)
            rows.append(item)

    quarterly = pd.DataFrame(rows)
    if quarterly.empty:
        return quarterly
    quarterly["Quarter"] = pd.to_datetime(quarterly["End"], errors="coerce").dt.to_period("Q").astype(str)
    quarterly = quarterly.replace([np.inf, -np.inf], np.nan).dropna(subset=["Quarter", "Value ($bn)"])
    if quarterly.empty:
        return quarterly
    quarterly = (
        quarterly.sort_values(["Ticker", "Start", "End", "Source Type", "Filed", "Accession"])
        .drop_duplicates(["Ticker", "Start", "End"], keep="last")
        .sort_values(["Quarter", "Ticker"])
        .reset_index(drop=True)
    )
    latest_quarters = sorted(quarterly["Quarter"].dropna().unique())[-4:]
    return quarterly[quarterly["Quarter"].isin(latest_quarters)].reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False)
def load_hyperscaler_capex() -> dict[str, Any]:
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    year_start = pd.Timestamp(year=today.year, month=1, day=1)
    trailing_cutoff = today - pd.Timedelta(days=90)
    selected_rows: list[pd.Series] = []
    all_facts: list[pd.DataFrame] = []
    errors: list[str] = []

    for spec in HYPERSCALER_CAPEX_COMPANIES:
        ticker = str(spec["Ticker"])
        try:
            payload = _sec_json(SEC_COMPANY_CONCEPT_URL.format(cik=int(spec["CIK"]), tag=spec["Tag"]))
            facts = _hyperscaler_companyconcept_frame(payload, spec)
            if facts.empty:
                errors.append(f"{ticker}: no USD companyconcept facts found")
                continue
            all_facts.append(facts)

            facts = facts[facts.apply(lambda row: _valid_sec_ytd_duration(row["FP"], row["Duration Days"]), axis=1)].copy()
            facts = facts[facts["End"].ge(year_start)]
            facts = facts[facts["End"].ge(trailing_cutoff) | facts["Filed"].ge(trailing_cutoff)]
            if facts.empty:
                errors.append(f"{ticker}: no current-year YTD fact found in the trailing 90-day window")
                continue

            facts = (
                facts.sort_values(["Ticker", "Start", "End", "Filed", "Accession"])
                .drop_duplicates(["Ticker", "Start", "End"], keep="last")
                .sort_values(["End", "Filed", "Duration Days"])
            )
            selected_rows.append(facts.iloc[-1])
        except Exception as exc:
            errors.append(f"{ticker}: {exc}")

    quarterly = _derive_hyperscaler_quarterly_capex(pd.concat(all_facts, ignore_index=True)) if all_facts else pd.DataFrame()
    quarterly_total = float(quarterly["Value ($bn)"].sum()) if not quarterly.empty else float("nan")

    data = pd.DataFrame(selected_rows)
    if not data.empty:
        data = data.sort_values("Value ($bn)", ascending=False).reset_index(drop=True)
        total = float(data["Value ($bn)"].sum())
        data["% of Total"] = np.where(total > 0, data["Value ($bn)"] / total * 100.0, np.nan)
    else:
        total = float("nan")

    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "data": data,
        "quarterly": quarterly,
        "quarterly_total_bn": quarterly_total,
        "total_bn": total,
        "errors": errors,
        "source": "SEC EDGAR XBRL companyconcept API",
    }


def _sec_filing_url(cik: int, accession: Any, primary_doc: Any) -> str:
    accession_text = str(accession or "").replace("-", "")
    primary = str(primary_doc or "")
    if not accession_text or not primary:
        return ""
    return f"{SEC_ARCHIVES_BASE_URL}/{int(cik)}/{accession_text}/{primary}"


def _sec_recent_frame(submissions: dict[str, Any], ticker: str, company: str, cik: int) -> pd.DataFrame:
    recent = submissions.get("filings", {}).get("recent", {})
    if not isinstance(recent, dict) or not recent:
        return pd.DataFrame()
    length = max((len(v) for v in recent.values() if isinstance(v, list)), default=0)
    rows = []
    for idx in range(length):
        row = {key: values[idx] if isinstance(values, list) and idx < len(values) else None for key, values in recent.items()}
        form = str(row.get("form") or "")
        filed = pd.to_datetime(row.get("filingDate"), errors="coerce")
        report_date = pd.to_datetime(row.get("reportDate"), errors="coerce")
        rows.append(
            {
                "Ticker": ticker,
                "Company": company,
                "CIK": cik,
                "Form": form,
                "Event": SEC_NOTABLE_FORMS.get(form, form),
                "Filed": filed,
                "Period": report_date,
                "Accession": row.get("accessionNumber"),
                "Description": row.get("primaryDocDescription") or row.get("primaryDocument") or "",
                "Link": _sec_filing_url(cik, row.get("accessionNumber"), row.get("primaryDocument")),
            }
        )
    return pd.DataFrame(rows)


def _sec_quarter_ends(fiscal_year_end: str, start_year: int, end_year: int) -> list[pd.Timestamp]:
    text = re.sub(r"\D", "", str(fiscal_year_end or "1231"))
    if len(text) != 4:
        text = "1231"
    month = min(max(int(text[:2]), 1), 12)
    day = min(max(int(text[2:]), 1), 31)
    fye_dates = []
    for year in range(start_year, end_year + 1):
        days_in_month = pd.Timestamp(year=year, month=month, day=1).days_in_month
        fye_dates.append(pd.Timestamp(year=year, month=month, day=min(day, days_in_month)).normalize())
    quarter_ends: list[pd.Timestamp] = []
    for fye in fye_dates:
        quarter_ends.extend(
            [
                (fye - pd.DateOffset(months=9)).normalize(),
                (fye - pd.DateOffset(months=6)).normalize(),
                (fye - pd.DateOffset(months=3)).normalize(),
                fye.normalize(),
            ]
        )
    return sorted(set(pd.Timestamp(d).normalize() for d in quarter_ends))


def _median_filing_lag_days(recent: pd.DataFrame, form: str, fallback: int) -> int:
    if recent.empty:
        return fallback
    rows = recent[recent["Form"].eq(form)].copy()
    rows = rows.dropna(subset=["Filed", "Period"])
    if rows.empty:
        return fallback
    lags = (rows["Filed"] - rows["Period"]).dt.days
    lags = lags[(lags > 0) & (lags < 150)]
    if lags.empty:
        return fallback
    return int(round(float(lags.median())))


def _expected_sec_event(submissions: dict[str, Any], recent: pd.DataFrame, ticker: str, company: str) -> dict[str, Any] | None:
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    fiscal_year_end = str(submissions.get("fiscalYearEnd") or "1231")
    periodic = recent[recent["Form"].isin(["10-Q", "10-K"])].dropna(subset=["Period"]).copy()
    if periodic.empty:
        return None
    last_period = pd.Timestamp(periodic["Period"].max()).normalize()
    quarter_ends = _sec_quarter_ends(fiscal_year_end, last_period.year, today.year + 1)
    next_periods = [date for date in quarter_ends if date > last_period and date <= today + pd.Timedelta(days=140)]
    if not next_periods:
        return None
    period = next_periods[0]
    form = "10-K" if period.strftime("%m%d") == re.sub(r"\D", "", fiscal_year_end or "1231").zfill(4)[-4:] else "10-Q"
    lag = _median_filing_lag_days(recent, form, 75 if form == "10-K" else 45)
    expected_date = period + pd.Timedelta(days=lag)
    days_until = (expected_date - today).days
    if days_until < -30:
        status = "Past estimate"
    elif days_until < 0:
        status = "Due/Watch"
    elif days_until <= 14:
        status = "Upcoming"
    else:
        status = "Scheduled"
    return {
        "Ticker": ticker,
        "Company": company,
        "Event": f"Expected {form}",
        "Period": period,
        "Expected Filing Date": expected_date,
        "Days": days_until,
        "Status": status,
        "Basis": f"Estimated from recent EDGAR {form} filing cadence ({lag} days after period end).",
    }


@st.cache_data(ttl=86400, show_spinner=False)
def load_sec_equity_events() -> dict[str, Any]:
    mapping_payload = _sec_json(SEC_COMPANY_TICKERS_URL)
    fields = mapping_payload.get("fields", [])
    rows = [dict(zip(fields, row)) for row in mapping_payload.get("data", [])]
    by_ticker = {str(row.get("ticker", "")).upper(): row for row in rows}

    recent_frames: list[pd.DataFrame] = []
    upcoming_rows: list[dict[str, Any]] = []
    issuer_rows: list[dict[str, Any]] = []
    cik_groups: dict[int, dict[str, Any]] = {}
    errors: list[str] = []

    for ticker in CRYPTO_EQUITIES:
        sec_row = by_ticker.get(ticker)
        if not sec_row:
            errors.append(f"{ticker}: SEC ticker mapping not found")
            continue
        cik = int(sec_row["cik"])
        company = str(sec_row.get("name") or ticker)
        issuer_rows.append({"Ticker": ticker, "Company": company, "CIK": f"{cik:010d}", "Exchange": sec_row.get("exchange")})
        group = cik_groups.setdefault(cik, {"tickers": [], "company": company})
        group["tickers"].append(ticker)

    for cik, group in cik_groups.items():
        ticker_label = " / ".join(group["tickers"])
        company = str(group["company"])
        try:
            submissions = _sec_json(SEC_SUBMISSIONS_URL.format(cik=cik))
            recent = _sec_recent_frame(submissions, ticker_label, company, cik)
            if not recent.empty:
                recent_frames.append(recent)
                expected = _expected_sec_event(submissions, recent, ticker_label, company)
                if expected:
                    upcoming_rows.append(expected)
        except Exception as exc:
            errors.append(f"{ticker_label}: {exc}")

    recent_all = pd.concat(recent_frames, ignore_index=True) if recent_frames else pd.DataFrame()
    if not recent_all.empty:
        recent_all = recent_all[recent_all["Form"].isin(SEC_NOTABLE_FORMS)].copy()
        cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=45)
        recent_all = recent_all[recent_all["Filed"].ge(cutoff)]
        recent_all = recent_all.sort_values("Filed", ascending=False).head(30).reset_index(drop=True)

    upcoming = pd.DataFrame(upcoming_rows)
    if not upcoming.empty:
        upcoming = upcoming.sort_values(["Expected Filing Date", "Ticker"]).reset_index(drop=True)

    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "issuers": pd.DataFrame(issuer_rows),
        "upcoming": upcoming,
        "recent": recent_all,
        "errors": errors,
        "source": "SEC EDGAR public submissions API",
    }


def _project_ts_signature() -> int:
    try:
        from dashboard.screener_data import PROJECT_TS_PATH

        return PROJECT_TS_PATH.stat().st_mtime_ns if PROJECT_TS_PATH.exists() else 0
    except Exception:
        return 0


def _protocol_change(row: pd.Series) -> float:
    direct = _num(row.get("change_7dover7d"))
    if not math.isnan(direct):
        return direct
    latest = _num(row.get("total7d"))
    prior = _num(row.get("total14dto7d"))
    if not math.isnan(latest) and not math.isnan(prior) and prior > 0:
        return (latest / prior - 1.0) * 100.0
    latest = _num(row.get("total24h"))
    prior = _num(row.get("total48hto24h"))
    if not math.isnan(latest) and not math.isnan(prior) and prior > 0:
        return (latest / prior - 1.0) * 100.0
    return float("nan")


def _overview_movers(payload: dict[str, Any], latest_floor: float = 10_000.0) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    rows = pd.DataFrame(payload.get("protocols") or [])
    if rows.empty:
        return None
    rows["Change"] = rows.apply(_protocol_change, axis=1)
    rows["Latest"] = pd.to_numeric(rows.get("total7d"), errors="coerce").combine_first(
        pd.to_numeric(rows.get("total24h"), errors="coerce")
    )
    rows["Project"] = rows.get("displayName", rows.get("name", pd.Series(index=rows.index))).fillna(rows.get("name"))
    rows["Ticker"] = rows.get("slug", rows["Project"]).fillna(rows["Project"]).astype(str)
    rows = rows.replace([np.inf, -np.inf], np.nan).dropna(subset=["Change", "Latest"])
    rows = rows[rows["Latest"] >= latest_floor]
    if rows.empty:
        return None
    keep = ["Ticker", "Project", "Change", "Latest"]
    gainers = rows.sort_values("Change", ascending=False)[keep].head(5).reset_index(drop=True)
    losers = rows.sort_values("Change", ascending=True)[keep].head(5).reset_index(drop=True)
    return gainers, losers


@st.cache_data(ttl=1800, show_spinner=False)
def load_token_movers(cache_signature: int) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    del cache_signature
    try:
        from dashboard.screener_data import CG_MARKETS_PATH, CoinGeckoClient, load_project_timeseries

        ts = load_project_timeseries("")
        if ts.empty or not {"ticker", "coingecko_id"}.issubset(ts.columns):
            return {}

        volume_30d = pd.Series(dtype=float, name="30D Volume")
        if "volume_24h" in ts.columns:
            volume_frame = ts.dropna(subset=["coingecko_id"]).copy()
            volume_frame["volume_24h"] = pd.to_numeric(volume_frame["volume_24h"], errors="coerce")
            volume_30d = volume_frame.groupby("coingecko_id", observed=True)["volume_24h"].sum().rename("30D Volume")

        meta_cols = ["ticker", "coingecko_id"] + (["project"] if "project" in ts.columns else [])
        meta = ts.sort_values("date").dropna(subset=["ticker", "coingecko_id"]).drop_duplicates("coingecko_id", keep="last")
        meta = meta[meta_cols].copy()
        if not volume_30d.empty:
            meta = meta.merge(volume_30d, left_on="coingecko_id", right_index=True, how="left")
        if meta.empty:
            return {}

        ids = meta["coingecko_id"].dropna().astype(str).unique()
        try:
            markets = CoinGeckoClient().get_markets(ids, use_cache=False)
        except Exception:
            markets = pd.DataFrame()
            if CG_MARKETS_PATH.exists():
                cached = pd.read_parquet(CG_MARKETS_PATH)
                if "id" in cached:
                    markets = cached[cached["id"].astype(str).isin(ids)].copy()
        if markets.empty or "id" not in markets:
            return {}

        keep = [
            "id",
            "current_price",
            "market_cap",
            "fully_diluted_valuation",
            "total_volume",
            "price_change_percentage_7d_in_currency",
            "price_change_percentage_30d_in_currency",
        ]
        keep = [col for col in keep if col in markets]
        merged = meta.merge(markets[keep], left_on="coingecko_id", right_on="id", how="inner")
        if "project" not in merged:
            merged["project"] = merged["ticker"]
        merged["market_cap"] = pd.to_numeric(
            merged["market_cap"] if "market_cap" in merged else pd.Series(np.nan, index=merged.index),
            errors="coerce",
        )
        merged["fdv"] = pd.to_numeric(
            merged["fully_diluted_valuation"] if "fully_diluted_valuation" in merged else pd.Series(np.nan, index=merged.index),
            errors="coerce",
        )
        merged["Price"] = pd.to_numeric(
            merged["current_price"] if "current_price" in merged else pd.Series(np.nan, index=merged.index),
            errors="coerce",
        )
        fallback_30d_volume = (
            pd.to_numeric(
                merged["total_volume"] if "total_volume" in merged else pd.Series(np.nan, index=merged.index),
                errors="coerce",
            )
            * 30.0
        )
        merged["30D Volume"] = pd.to_numeric(
            merged["30D Volume"] if "30D Volume" in merged else pd.Series(np.nan, index=merged.index),
            errors="coerce",
        ).combine_first(fallback_30d_volume)
        universe = merged[
            (merged["fdv"] >= TOKEN_MOVER_MIN_FDV)
            & (merged["30D Volume"] >= TOKEN_MOVER_MIN_30D_VOLUME)
        ].copy()
        if universe.empty:
            return {}

        universe["Ticker"] = universe["ticker"].astype(str).str.upper()
        universe["Project"] = universe["project"].fillna(universe["ticker"]).astype(str)
        universe["FDV"] = universe["fdv"]
        out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        timeframe_cols = {
            "7D Price": "price_change_percentage_7d_in_currency",
            "30D Price": "price_change_percentage_30d_in_currency",
        }
        for label, change_col in timeframe_cols.items():
            if change_col not in universe:
                continue
            frame = universe.copy()
            frame["Change"] = pd.to_numeric(frame[change_col], errors="coerce")
            frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["Change"])
            if frame.empty:
                continue
            keep_cols = ["Ticker", "Project", "Change", "Price", "FDV", "30D Volume"]
            gainers = frame.sort_values("Change", ascending=False)[keep_cols].head(5).reset_index(drop=True)
            losers = frame.sort_values("Change", ascending=True)[keep_cols].head(5).reset_index(drop=True)
            out[label] = (gainers, losers)
        return out
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def load_sector_performance(cache_signature: int, period_days: int = 30) -> pd.DataFrame:
    del cache_signature, period_days
    try:
        from dashboard.screener_data import load_project_timeseries

        ts = load_project_timeseries("")
    except Exception:
        ts = pd.DataFrame()
    required = {"ticker", "category", "coingecko_id"}
    if ts.empty or not required.issubset(ts.columns):
        return pd.DataFrame(columns=["Sector", "Change", "Latest FDV", "Base FDV", "Tokens", "Token List"])

    meta = ts.sort_values("date").groupby("ticker", as_index=False, observed=True).tail(1)
    meta = meta.dropna(subset=["coingecko_id", "category"]).copy()
    if meta.empty:
        return pd.DataFrame(columns=["Sector", "Change", "Latest FDV", "Base FDV", "Tokens", "Token List"])

    try:
        from dashboard.screener_data import CoinGeckoClient

        markets = CoinGeckoClient().get_markets(meta["coingecko_id"].dropna().astype(str).unique(), use_cache=False)
    except Exception:
        markets = pd.DataFrame()
    if markets.empty or "id" not in markets:
        return pd.DataFrame(columns=["Sector", "Change", "Latest FDV", "Base FDV", "Tokens", "Token List"])

    keep = [
        "id",
        "fully_diluted_valuation",
        "market_cap",
        "price_change_percentage_30d_in_currency",
    ]
    keep = [col for col in keep if col in markets]
    merged = meta[["ticker", "category", "coingecko_id"]].merge(
        markets[keep],
        left_on="coingecko_id",
        right_on="id",
        how="inner",
    )
    latest_fdv = pd.to_numeric(
        merged["fully_diluted_valuation"] if "fully_diluted_valuation" in merged else pd.Series(np.nan, index=merged.index),
        errors="coerce",
    )
    market_cap = pd.to_numeric(
        merged["market_cap"] if "market_cap" in merged else pd.Series(np.nan, index=merged.index),
        errors="coerce",
    )
    merged["latest_fdv"] = latest_fdv.combine_first(market_cap)
    merged["Change"] = pd.to_numeric(
        merged["price_change_percentage_30d_in_currency"]
        if "price_change_percentage_30d_in_currency" in merged
        else pd.Series(np.nan, index=merged.index),
        errors="coerce",
    )
    merged["base_fdv"] = merged["latest_fdv"] / (1.0 + merged["Change"] / 100.0)
    merged = merged[(merged["latest_fdv"] > 0) & (merged["base_fdv"] > 0)].dropna(subset=["Change"])
    if merged.empty:
        return pd.DataFrame(columns=["Sector", "Change", "Latest FDV", "Base FDV", "Tokens", "Token List"])

    def _token_list(values: pd.Series) -> str:
        tokens = sorted({str(value).upper() for value in values if str(value).strip()})
        if len(tokens) > 35:
            return f"{', '.join(tokens[:35])}, ..."
        return ", ".join(tokens)

    grouped = (
        merged.groupby("category", observed=True)
        .agg(
            latest_fdv=("latest_fdv", "sum"),
            base_fdv=("base_fdv", "sum"),
            tokens=("ticker", "nunique"),
            token_list=("ticker", _token_list),
        )
        .reset_index()
    )
    grouped["Change"] = (grouped["latest_fdv"] / grouped["base_fdv"] - 1.0) * 100.0
    grouped = grouped.replace([np.inf, -np.inf], np.nan).dropna(subset=["Change"])
    grouped = grouped.rename(
        columns={
            "category": "Sector",
            "latest_fdv": "Latest FDV",
            "base_fdv": "Base FDV",
            "tokens": "Tokens",
            "token_list": "Token List",
        }
    )
    return grouped.sort_values("Change", ascending=False).reset_index(drop=True)


def _safe_defillama_overview(data_type: str) -> dict:
    from crypto_factor_model.clients.defillama import DefiLlamaClient

    dl = DefiLlamaClient()
    return dl.get_fees_overview(data_type=data_type, use_cache=False)


@st.cache_data(ttl=1800, show_spinner=False)
def load_chain_fees_24h() -> pd.DataFrame:
    try:
        data = _safe_defillama_overview("dailyFees")
    except Exception:
        return pd.DataFrame(columns=["Chain", "Fees"])

    name_lookup = {str(c).lower(): str(c) for c in data.get("allChains", []) if c}
    totals: dict[str, float] = {}
    for protocol in data.get("protocols", []) or []:
        breakdown = protocol.get("breakdown24h") or {}
        if not isinstance(breakdown, dict):
            continue
        for chain, value in breakdown.items():
            chain_name = name_lookup.get(str(chain).lower(), _canonical_chain_name(chain))
            totals[chain_name] = totals.get(chain_name, 0.0) + _sum_nested(value)
    rows = [{"Chain": chain, "Fees": value} for chain, value in totals.items() if pd.notna(value) and value > 0]
    return pd.DataFrame(rows).sort_values("Fees", ascending=False).head(20).reset_index(drop=True)


def _stablecoin_entry_value(entry: dict[str, Any]) -> float:
    for key in ["totalCirculatingUSD", "totalCirculating"]:
        payload = entry.get(key)
        if isinstance(payload, dict):
            value = _num(payload.get("peggedUSD"))
            if not math.isnan(value):
                return value
    return float("nan")


@st.cache_data(ttl=3600, show_spinner=False)
def load_stablecoin_supply_changes() -> pd.DataFrame:
    rows = []
    for chain in STABLECOIN_SUPPLY_CHAINS:
        try:
            encoded = quote(chain, safe="")
            data = _requests_json(f"https://stablecoins.llama.fi/stablecoincharts/{encoded}", timeout=15)
            if not isinstance(data, list) or len(data) < 2:
                continue
            latest = data[-1]
            prior = data[-2]
            latest_value = _stablecoin_entry_value(latest)
            prior_value = _stablecoin_entry_value(prior)
            if math.isnan(latest_value) or math.isnan(prior_value):
                continue
            latest_ts = latest.get("date")
            as_of = (
                datetime.fromtimestamp(float(latest_ts), tz=timezone.utc).replace(tzinfo=None)
                if latest_ts is not None
                else pd.NaT
            )
            rows.append(
                {
                    "Chain": _canonical_chain_name(chain),
                    "Change": latest_value - prior_value,
                    "Supply": latest_value,
                    "As Of": as_of,
                }
            )
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["Chain", "Change", "Supply", "As Of"])
    df = pd.DataFrame(rows)
    positives = df[df["Change"] > 0].sort_values("Change", ascending=False).head(8)
    negatives = df[df["Change"] < 0].sort_values("Change", ascending=True).head(8)
    return pd.concat([positives, negatives], ignore_index=True).sort_values("Change", ascending=False).reset_index(drop=True)


def _fetch_bridge_volume(chain: str) -> dict[str, Any] | None:
    encoded = quote(chain, safe="")
    candidates = [
        f"https://bridges.llama.fi/bridgevolume/{encoded}",
        f"https://bridges.llama.fi/bridgevolume/{quote(chain.lower(), safe='')}",
    ]
    for url in candidates:
        try:
            data = _requests_json(url, timeout=12)
            records = data.get("data", data) if isinstance(data, dict) else data
            if not isinstance(records, list) or not records:
                continue
            records = [r for r in records if isinstance(r, dict)]
            if not records:
                continue
            latest = records[-1]
            deposits = _num(latest.get("depositUSD"))
            withdrawals = _num(latest.get("withdrawUSD"))
            if math.isnan(deposits) and math.isnan(withdrawals):
                continue
            return {
                "deposits": 0.0 if math.isnan(deposits) else deposits,
                "withdrawals": 0.0 if math.isnan(withdrawals) else withdrawals,
                "date": latest.get("date"),
            }
        except Exception:
            continue
    return None


def _load_bridge_rows_from_public_page() -> pd.DataFrame:
    for url in DEFILLAMA_BRIDGE_CHAIN_PAGES:
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
            resp.raise_for_status()
            match = re.search(
                r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
                resp.text,
                flags=re.DOTALL,
            )
            if not match:
                continue
            payload = json.loads(match.group(1))
            table = payload.get("props", {}).get("pageProps", {}).get("tableData", [])
            if not isinstance(table, list) or not table:
                continue
            rows = []
            for row in table:
                if not isinstance(row, dict):
                    continue
                rows.append(
                    {
                        "Chain": _canonical_chain_name(row.get("name")),
                        "Net Flow": _num(row.get("prevDayNetFlow")),
                        "Deposits": _num(row.get("prevDayUsdDeposits")),
                        "Withdrawals": _num(row.get("prevDayUsdWithdrawals")),
                        "As Of": "Previous UTC day",
                    }
                )
            out = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=["Net Flow"])
            if not out.empty:
                return out
        except Exception:
            continue
    return pd.DataFrame(columns=["Chain", "Net Flow", "Deposits", "Withdrawals", "As Of"])


@st.cache_data(ttl=1800, show_spinner=False)
def load_bridge_net_flows() -> pd.DataFrame:
    free_page = _load_bridge_rows_from_public_page()
    if not free_page.empty:
        positives = free_page[free_page["Net Flow"] > 0].sort_values("Net Flow", ascending=False).head(8)
        negatives = free_page[free_page["Net Flow"] < 0].sort_values("Net Flow", ascending=True).head(8)
        return pd.concat([positives, negatives], ignore_index=True).sort_values("Net Flow", ascending=False).reset_index(drop=True)

    rows = []
    for chain in BRIDGE_FLOW_CHAINS:
        data = _fetch_bridge_volume(chain)
        if not data:
            continue
        net = data["deposits"] - data["withdrawals"]
        rows.append(
            {
                "Chain": _canonical_chain_name(chain),
                "Net Flow": net,
                "Deposits": data["deposits"],
                "Withdrawals": data["withdrawals"],
                "As Of": pd.to_datetime(data.get("date"), unit="s", errors="coerce"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Chain", "Net Flow", "Deposits", "Withdrawals", "As Of"])
    df = pd.DataFrame(rows)
    positives = df[df["Net Flow"] > 0].sort_values("Net Flow", ascending=False).head(8)
    negatives = df[df["Net Flow"] < 0].sort_values("Net Flow", ascending=True).head(8)
    return pd.concat([positives, negatives], ignore_index=True).sort_values("Net Flow", ascending=False).reset_index(drop=True)


def _compute_pi_cycle_row() -> dict[str, Any]:
    start = (pd.Timestamp.now().normalize() - pd.Timedelta(days=430)).strftime("%Y-%m-%d")
    btc = _fetch_binance_ohlcv("BTCUSDT", start)
    close = btc["Close"].dropna().astype(float) if not btc.empty else pd.Series(dtype=float)
    if len(close) < 350:
        return {
            "Indicator": "Pi Cycle Top",
            "Current": "n/a",
            "24h %": "n/a",
            "Reference": "111DMA vs 350DMA x2",
            "Triggered": "n/a",
            "Source": "CMC public page label; BTC history unavailable",
        }
    dma_111 = close.rolling(111).mean().iloc[-1]
    dma_350x2 = close.rolling(350).mean().iloc[-1] * 2.0
    distance = (dma_111 / dma_350x2 - 1.0) * 100.0 if dma_350x2 else float("nan")
    prior_distance = (
        close.rolling(111).mean().iloc[-2] / (close.rolling(350).mean().iloc[-2] * 2.0) - 1.0
    ) * 100.0
    return {
        "Indicator": "Pi Cycle Top",
        "Current": f"{distance:+.1f}% below/above trigger",
        "24h %": f"{distance - prior_distance:+.1f} pp",
        "Reference": "Triggered when 111DMA crosses above 350DMA x2",
        "Triggered": "Yes" if dma_111 >= dma_350x2 else "No",
        "Source": "Computed from Binance BTC daily closes; CMC free page reference",
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_cycle_indicators() -> pd.DataFrame:
    rows = [_compute_pi_cycle_row()]
    cmc_status = "CMC public/free page"
    try:
        resp = requests.get(CMC_CYCLE_URL, headers=REQUEST_HEADERS, timeout=12)
        if resp.status_code >= 400:
            cmc_status = f"CMC public page returned HTTP {resp.status_code}"
    except Exception:
        cmc_status = "CMC public page unavailable"

    rows.extend(
        [
            {
                "Indicator": "Puell Multiple",
                "Current": "n/a",
                "24h %": "n/a",
                "Reference": "Daily BTC issuance value / 365D average issuance value",
                "Triggered": "n/a",
                "Source": f"{cmc_status}; no paid CMC API key used",
            },
            {
                "Indicator": "Bitcoin Rainbow Chart",
                "Current": "n/a",
                "24h %": "n/a",
                "Reference": "Long-term BTC log-regression valuation bands",
                "Triggered": "n/a",
                "Source": f"{cmc_status}; no paid CMC API key used",
            },
        ]
    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    signal_cols = [col for col in ["Current", "24h %", "Triggered"] if col in out]
    if signal_cols:
        unavailable = out[signal_cols].apply(
            lambda row: all(str(value).strip().lower() in {"", "n/a", "nan", "none"} for value in row),
            axis=1,
        )
        out = out[~unavailable].reset_index(drop=True)
    return out


def clear_market_update_live_caches() -> None:
    for fn in [
        load_macro_monitor,
        load_index_benchmark,
        load_market_structure,
        load_crypto_equity_changes,
        load_btc_macro_ratios,
        load_bitcoin_com_charts,
        load_sec_equity_events,
        load_hyperscaler_capex,
        load_token_movers,
        load_sector_performance,
        load_chain_fees_24h,
        load_bridge_net_flows,
        load_stablecoin_supply_changes,
        load_cycle_indicators,
    ]:
        try:
            fn.clear()
        except Exception:
            pass


def _empty_movers() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    return {}


def _snapshot_signature() -> int:
    try:
        return MARKET_UPDATE_SNAPSHOT_PATH.stat().st_mtime_ns if MARKET_UPDATE_SNAPSHOT_PATH.exists() else 0
    except Exception:
        return 0


def _snapshot_created_at(snapshot: dict[str, Any]) -> datetime | None:
    value = snapshot.get("created_at")
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _snapshot_age_hours(snapshot: dict[str, Any]) -> float | None:
    created_at = _snapshot_created_at(snapshot)
    if created_at is None:
        return None
    return (datetime.now(timezone.utc) - created_at).total_seconds() / 3600.0


def build_market_update_snapshot() -> dict[str, Any]:
    """Build the daily Market Update snapshot from current public/live sources."""
    MARKET_UPDATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    clear_market_update_live_caches()
    errors: list[str] = []

    try:
        macro_data, macro_refreshed_at, macro_errors = load_macro_monitor()
    except Exception as exc:
        macro_data, macro_refreshed_at, macro_errors = pd.DataFrame(), datetime.now(timezone.utc), [str(exc)]
    errors.extend([f"Macro: {err}" for err in macro_errors])

    try:
        benchmark_data, benchmark_errors = load_index_benchmark()
    except Exception as exc:
        benchmark_data, benchmark_errors = pd.DataFrame(), [str(exc)]
    errors.extend([f"Benchmark: {err}" for err in benchmark_errors])

    try:
        market_structure, market_structure_errors = load_market_structure()
    except Exception as exc:
        market_structure, market_structure_errors = {}, [str(exc)]
    errors.extend([f"Market structure: {err}" for err in market_structure_errors])

    section_loaders = {
        "crypto_equities": (load_crypto_equity_changes, pd.DataFrame()),
        "bitcoin_com_charts": (load_bitcoin_com_charts, {}),
        "sec_equity_events": (load_sec_equity_events, {}),
        "hyperscaler_capex": (load_hyperscaler_capex, {}),
        "token_movers": (lambda: load_token_movers(_project_ts_signature()), _empty_movers()),
        "sector_performance": (lambda: load_sector_performance(_project_ts_signature()), pd.DataFrame()),
        "chain_fees": (load_chain_fees_24h, pd.DataFrame()),
        "bridge_net_flows": (load_bridge_net_flows, pd.DataFrame()),
        "stablecoin_supply_changes": (load_stablecoin_supply_changes, pd.DataFrame()),
        "cycle_indicators": (load_cycle_indicators, pd.DataFrame()),
    }
    sections: dict[str, Any] = {}
    for name, (loader, fallback) in section_loaders.items():
        try:
            sections[name] = loader()
        except Exception as exc:
            sections[name] = fallback
            errors.append(f"{name.replace('_', ' ').title()}: {exc}")

    snapshot: dict[str, Any] = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "macro": {"data": macro_data, "refreshed_at": macro_refreshed_at, "errors": macro_errors},
        "index_benchmark": {"data": benchmark_data, "errors": benchmark_errors},
        "market_structure": {"data": market_structure, "errors": market_structure_errors},
        "errors": errors,
        **sections,
    }
    tmp_path = MARKET_UPDATE_SNAPSHOT_PATH.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(snapshot, f)
    tmp_path.replace(MARKET_UPDATE_SNAPSHOT_PATH)
    return snapshot


@st.cache_data(show_spinner=False)
def load_market_update_snapshot(cache_signature: int) -> tuple[dict[str, Any] | None, str | None]:
    del cache_signature
    if not MARKET_UPDATE_SNAPSHOT_PATH.exists():
        return None, f"Missing Market Update snapshot: {MARKET_UPDATE_SNAPSHOT_PATH}"
    try:
        with open(MARKET_UPDATE_SNAPSHOT_PATH, "rb") as f:
            snapshot = pickle.load(f)
        if not isinstance(snapshot, dict):
            return None, "Market Update snapshot is not a valid dictionary."
        return snapshot, None
    except Exception as exc:
        return None, f"Could not load Market Update snapshot: {exc}"


def _style_plotly(fig: go.Figure, height: int = 360, showlegend: bool = True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="#121410",
        plot_bgcolor="#121410",
        font=dict(color="#f2efe4"),
        margin=dict(l=16, r=18, t=18, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="left", x=0),
        showlegend=showlegend,
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#2b3128", zerolinecolor="#707866")
    fig.update_yaxes(gridcolor="#2b3128", zerolinecolor="#707866")
    return fig


def _benchmark_chart(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    data = data.copy()
    if "date" in data:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"])
        data = data[data["date"] >= pd.Timestamp(INDEX_BENCHMARK_START_DATE)].sort_values("date")
        value_cols = [col for col in data.columns if col != "date"]
        if value_cols and not data.empty:
            data[value_cols] = data[value_cols].div(data[value_cols].iloc[0]).mul(100.0)
    color_map = {label: color for label, _kind, _symbol, color in BENCHMARK_ASSETS}
    color_map.update({label: color for label, _primary, _fallback, _source, color in BTC_MACRO_RATIO_ASSETS})
    for col in data.columns:
        if col == "date":
            continue
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data[col],
                mode="lines",
                name=col,
                line=dict(width=2.2, color=color_map.get(col)),
            )
        )
    fig.add_hline(y=100, line_color="#d8d2c0", line_width=1, opacity=0.6)
    fig.update_yaxes(title="Index (start=100)")
    fig.update_xaxes(title=None)
    return _style_plotly(fig, height=430)


def _bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color_signed: bool = False,
    height: int = 390,
) -> go.Figure:
    chart = data[[y, x]].dropna().copy()
    if chart.empty:
        return _style_plotly(go.Figure(), height=height, showlegend=False)
    chart = chart.sort_values(x, ascending=True)
    colors = np.where(chart[x] >= 0, "#31a354", "#ef3b35") if color_signed else "#8eed8a"
    fig = go.Figure(
        go.Bar(
            x=chart[x],
            y=chart[y],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>%{x:$,.0f}<extra></extra>",
        )
    )
    fig.update_layout(title=dict(text=title, font=dict(size=16)), showlegend=False)
    fig.update_xaxes(tickprefix="$", separatethousands=True)
    return _style_plotly(fig, height=height, showlegend=False)


def _hyperscaler_capex_chart(payload: dict[str, Any]) -> go.Figure:
    quarterly = payload.get("quarterly", pd.DataFrame()) if isinstance(payload, dict) else pd.DataFrame()
    if isinstance(quarterly, pd.DataFrame) and not quarterly.empty:
        chart = quarterly.copy()
        chart["Quarter"] = chart["Quarter"].astype(str)
        total_bn = _num(payload.get("quarterly_total_bn")) if isinstance(payload, dict) else float("nan")
        if math.isnan(total_bn):
            total_bn = float(chart["Value ($bn)"].sum())
        title = f"Hyperscaler Reported Capex Proxy: Latest 4 Quarters (${total_bn:,.1f}bn total)"

        fig = go.Figure()
        quarter_order = sorted(chart["Quarter"].dropna().unique())
        for ticker, group in chart.sort_values(["Ticker", "Quarter"]).groupby("Ticker", sort=False):
            group = group.sort_values("Quarter")
            custom = group[["Start", "End", "Filed", "Tag", "Source Type"]].copy()
            for col in ["Start", "End", "Filed"]:
                custom[col] = custom[col].map(_format_date_cell)
            fig.add_trace(
                go.Bar(
                    x=group["Quarter"],
                    y=group["Value ($bn)"],
                    name=str(ticker),
                    marker_color=group["Color"].dropna().iloc[0] if "Color" in group and group["Color"].notna().any() else None,
                    customdata=custom.to_numpy(dtype=object),
                    hovertemplate=(
                        "%{fullData.name}<br>"
                        "Quarter: %{x}<br>"
                        "Capex: $%{y:.1f}bn<br>"
                        "Period: %{customdata[0]} to %{customdata[1]}<br>"
                        "Filed: %{customdata[2]}<br>"
                        "SEC tag: %{customdata[3]}<br>"
                        "Source: %{customdata[4]}<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            barmode="stack",
            bargap=0.24,
            hovermode="closest",
        )
        fig.update_yaxes(title="Quarterly capex ($bn)", ticksuffix="bn")
        fig.update_xaxes(title=None, categoryorder="array", categoryarray=quarter_order)
        return _style_plotly(fig, height=460, showlegend=True).update_layout(hovermode="closest")

    data = payload.get("data", pd.DataFrame()) if isinstance(payload, dict) else pd.DataFrame()
    if not isinstance(data, pd.DataFrame) or data.empty:
        return _style_plotly(go.Figure(), height=430, showlegend=False)

    chart = data.copy()
    total_bn = _num(payload.get("total_bn")) if isinstance(payload, dict) else float("nan")
    if math.isnan(total_bn):
        total_bn = float(chart["Value ($bn)"].sum())
    title = f"Hyperscaler Reported Capex Proxy: YTD (${total_bn:,.1f}bn total)"

    fig = go.Figure()
    for _, row in chart.sort_values("Value ($bn)", ascending=False).iterrows():
        custom = np.array(
            [
                [
                    _format_date_cell(row.get("Start")),
                    _format_date_cell(row.get("End")),
                    _format_date_cell(row.get("Filed")),
                    _num(row.get("% of Total")),
                    row.get("Tag", ""),
                    row.get("Source Type", "reported_ytd"),
                ]
            ],
            dtype=object,
        )
        fig.add_trace(
            go.Bar(
                x=["Latest reported fiscal YTD"],
                y=[_num(row.get("Value ($bn)"))],
                name=str(row.get("Ticker", "")),
                marker_color=row.get("Color"),
                customdata=custom,
                hovertemplate=(
                    "%{fullData.name}<br>"
                    "Fiscal YTD: %{customdata[0]} to %{customdata[1]}<br>"
                    "Filed: %{customdata[2]}<br>"
                    "YTD capex: $%{y:.1f}bn<br>"
                    "Share of total: %{customdata[3]:.1f}%<br>"
                    "SEC tag: %{customdata[4]}<br>"
                    "Source: %{customdata[5]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        barmode="stack",
        bargap=0.55,
        hovermode="closest",
    )
    fig.update_yaxes(title="YTD capex ($bn)", ticksuffix="bn")
    fig.update_xaxes(title=None)
    return _style_plotly(fig, height=430, showlegend=True).update_layout(hovermode="closest")


def _sector_performance_chart(data: pd.DataFrame) -> go.Figure:
    required = ["Sector", "Change", "Latest FDV", "Tokens"]
    chart = data[[col for col in required + ["Token List"] if col in data]].dropna(subset=["Sector", "Change"]).copy()
    if chart.empty:
        return _style_plotly(go.Figure(), height=430, showlegend=False)
    if "Token List" not in chart:
        chart["Token List"] = ""
    chart = chart.sort_values("Change", ascending=False)
    colors = np.where(chart["Change"] >= 0, "#8eed8a", "#ff746d")
    fig = go.Figure(
        go.Bar(
            x=chart["Change"],
            y=chart["Sector"],
            orientation="h",
            marker_color=colors,
            customdata=chart[["Latest FDV", "Tokens", "Token List"]].to_numpy(),
            text=[f"{v:.1f}%" for v in chart["Change"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>"
                "1M FDV change: %{x:+.1f}%<br>"
                "Latest FDV: %{customdata[0]:$,.0f}<br>"
                "Tokens: %{customdata[1]:.0f}<br>"
                "Token list: %{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_color="#d8d2c0", line_width=1, opacity=0.65)
    fig.update_layout(title=dict(text="Sector Performance", font=dict(size=16)), showlegend=False)
    fig.update_xaxes(
        title=None,
        ticksuffix="%",
    )
    fig.update_yaxes(
        title=None,
        tickfont=dict(size=10),
        categoryorder="array",
        categoryarray=chart["Sector"].tolist(),
        autorange="reversed",
    )
    return _style_plotly(fig, height=max(620, 120 + len(chart) * 22), showlegend=False)


def _bitcoin_com_line_fig(title: str, data: pd.DataFrame, columns: list[str], *, log_y: bool = False) -> go.Figure:
    fig = go.Figure()
    if isinstance(data, pd.DataFrame) and not data.empty:
        for col in columns:
            if col in data:
                fig.add_trace(go.Scatter(x=data["date"], y=data[col], mode="lines", name=col, line=dict(width=2)))
    fig.update_layout(title=dict(text=title, font=dict(size=16)))
    fig.update_yaxes(type="log" if log_y else None, title=None)
    fig.update_xaxes(title=None)
    return _style_plotly(fig, height=430, showlegend=True)


def _bitcoin_com_rainbow_fig(chart: dict[str, Any]) -> go.Figure:
    data = chart.get("data", pd.DataFrame())
    fig = go.Figure()
    if isinstance(data, pd.DataFrame) and not data.empty and "BTC Price" in data:
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["BTC Price"],
                mode="lines",
                name="BTC Price",
                line=dict(width=2.6, color="#f2efe4"),
            )
        )
    for band in chart.get("bands", []) or []:
        frame = band.get("data")
        name = band.get("name", "Band")
        if isinstance(frame, pd.DataFrame) and not frame.empty and name in frame:
            fig.add_trace(
                go.Scatter(
                    x=frame["date"],
                    y=frame[name],
                    mode="lines",
                    name=name,
                    line=dict(width=1.2, color=band.get("color")),
                    opacity=0.72,
                )
            )
    current = chart.get("current", {}) or {}
    suffix = f" - Current zone: {current.get('name')}" if current.get("name") else ""
    fig.update_layout(title=dict(text=f"Rainbow Chart{suffix}", font=dict(size=16)))
    fig.update_yaxes(type="log", title=None)
    fig.update_xaxes(title=None)
    return _style_plotly(fig, height=430, showlegend=True)


def _bitcoin_com_chart_fig(chart_id: str, chart: dict[str, Any]) -> go.Figure:
    data = chart.get("data", pd.DataFrame())
    if chart_id == "pi-cycle-top":
        return _bitcoin_com_line_fig("Pi Cycle Top", data, ["BTC Price", "111DMA", "350DMA x2"], log_y=True)
    if chart_id == "rainbow":
        return _bitcoin_com_rainbow_fig(chart)
    if chart_id == "stock-to-flow":
        return _bitcoin_com_line_fig("Stock-to-Flow", data, ["BTC Price", "S2F Model"], log_y=True)
    if chart_id == "golden-ratio":
        return _bitcoin_com_line_fig("Golden Ratio Multiplier", data, ["BTC Price", "350DMA", "1.6x", "2x", "3x", "5x"], log_y=True)
    if chart_id == "mayer-multiple":
        fig = _bitcoin_com_line_fig("Mayer Multiple", data, ["Mayer Multiple"], log_y=False)
        for y, label in [(0.8, "Undervalued"), (1.0, "Fair value"), (2.4, "Overvalued")]:
            fig.add_hline(y=y, line_dash="dot", line_width=1, line_color="#d8d2c0", annotation_text=label)
        return fig
    if chart_id == "m2":
        return _bitcoin_com_line_fig("Global M2 vs Bitcoin", data, ["BTC Indexed", "M2 Indexed"], log_y=False)
    return _bitcoin_com_line_fig(chart.get("title", chart_id), data, [c for c in data.columns if c != "date"], log_y=False)


def _display_mover_table(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()
    if out.empty:
        cols = ["Ticker", "Project", "Change", "Price", "FDV", "30D Volume"]
        return pd.DataFrame(columns=[col for col in cols if col in out.columns] or cols)
    for col in ["Ticker", "Project", "Change"]:
        if col not in out:
            out[col] = "n/a"
    out["Change"] = out["Change"].map(_format_pct)
    if {"Price", "FDV", "30D Volume"}.issubset(out.columns):
        out["Price"] = out["Price"].map(_format_token_price)
        out["FDV"] = out["FDV"].map(_compact_usd)
        out["30D Volume"] = out["30D Volume"].map(_compact_usd)
        return out[["Ticker", "Project", "Change", "Price", "FDV", "30D Volume"]]
    if "Latest" not in out:
        out["Latest"] = np.nan
    out["Latest"] = out["Latest"].map(_compact_usd)
    return out[["Ticker", "Project", "Change", "Latest"]]


def _render_macro_cards(data: pd.DataFrame) -> None:
    cols_per_row = 4
    for start in range(0, len(data), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, (_, row) in zip(cols, data.iloc[start : start + cols_per_row].iterrows()):
            col.metric(str(row["Label"]), str(row["Display Value"]))
            date_text = f"Obs: {row['Observation Date']}"
            if row.get("Detail"):
                date_text = f"{date_text} | {row['Detail']}"
            col.caption(date_text)


def _render_market_structure(data: dict[str, Any] | None = None, errors: list[str] | None = None) -> None:
    st.subheader("Market Structure")
    if data is None:
        data, errors = load_market_structure()
    errors = errors or []
    cols = st.columns(3)
    cols[0].metric("ETH/BTC", _format_ratio(data.get("eth_btc")), _format_pct(data.get("eth_btc_7d_pct")))
    cols[1].metric("BTC.D", f"{_num(data.get('btc_d')):.1f}%" if pd.notna(data.get("btc_d")) else "n/a")
    cols[2].metric("ETH Dominance", f"{_num(data.get('eth_d')):.1f}%" if pd.notna(data.get("eth_d")) else "n/a")
    for error in errors:
        st.warning(error)


def _render_token_movers(movers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None = None) -> None:
    if movers is None:
        movers = load_token_movers(_project_ts_signature())
    st.subheader("Token Movers (>$100M FDV, >$10M 30D volume)")
    st.caption(
        "Top 5 gainers and losers for 7D and 30D price changes. "
        "Universe uses the current CoinGecko markets snapshot, filtered for FDV over $100M "
        "and 30D USD volume over $10M from the existing screener volume history."
    )
    if not movers:
        st.info("Token mover cache is unavailable. Refresh the screener cache to populate this section.")
        return
    tabs = st.tabs(list(movers.keys()))
    for tab, label in zip(tabs, movers.keys()):
        gainers, losers = movers[label]
        with tab:
            c1, c2 = st.columns(2)
            c1.markdown("**Top 5 gainers**")
            c1.dataframe(_display_mover_table(gainers), hide_index=True, width="stretch")
            c2.markdown("**Top 5 losers**")
            c2.dataframe(_display_mover_table(losers), hide_index=True, width="stretch")


def _render_sector_performance(sectors: pd.DataFrame | None = None) -> None:
    st.subheader("Sector Performance")
    st.caption(
        "FDV-weighted token performance by category over the last 1M. "
        "Uses current CoinGecko market snapshots mapped to the existing screener categories."
    )
    if sectors is None:
        sectors = load_sector_performance(_project_ts_signature(), period_days=30)
    if sectors.empty:
        st.info("Sector performance data is unavailable.")
        return
    st.plotly_chart(_sector_performance_chart(sectors), width="stretch")


def _render_crypto_equities(table: pd.DataFrame | None = None) -> None:
    st.subheader("Crypto Equity Price and Volume Changes")
    if table is None:
        table = load_crypto_equity_changes()
    if table.empty:
        st.info("Crypto equity data is unavailable.")
        return
    display = table.copy()
    display["Last"] = display["Last"].map(lambda v: "n/a" if pd.isna(v) else f"${float(v):,.2f}")
    for col in ["1D Price", "5D Price", "5D Volume"]:
        display[col] = display[col].map(_format_pct)
    display["Latest Volume"] = display["Latest Volume"].map(lambda v: "n/a" if pd.isna(v) else f"{float(v):,.0f}")
    display = display.drop(columns=["Source"], errors="ignore")
    st.dataframe(display, hide_index=True, width="stretch")


def _format_date_cell(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    return "n/a" if pd.isna(ts) else str(ts.date())


def _equity_logo_url(ticker_label: Any) -> str:
    tickers = [part.strip().upper() for part in str(ticker_label or "").split("/") if part.strip()]
    for ticker in tickers:
        if ticker in CRYPTO_EQUITY_LOGOS:
            return CRYPTO_EQUITY_LOGOS[ticker]
    return ""


def _sec_event_summary_metrics(upcoming: pd.DataFrame, recent: pd.DataFrame) -> None:
    due = 0
    next_two_weeks = 0
    if isinstance(upcoming, pd.DataFrame) and not upcoming.empty and "Days" in upcoming:
        days = pd.to_numeric(upcoming["Days"], errors="coerce")
        due = int((days <= 0).sum())
        next_two_weeks = int(((days > 0) & (days <= 14)).sum())
    recent_count = int(len(recent)) if isinstance(recent, pd.DataFrame) else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Due / watch", due)
    c2.metric("Next 14 days", next_two_weeks)
    c3.metric("Recent filings", recent_count)


def _render_hyperscaler_capex(payload: dict[str, Any] | None = None) -> None:
    st.subheader("Hyperscaler Reported Capex Proxy: Latest 4 Quarters")
    st.caption(
        "Reported SEC capex/productive asset spend. Useful as an AI infrastructure capex proxy, "
        "but not a pure AI capex disclosure."
    )
    if payload is None:
        payload = load_hyperscaler_capex()
    if not isinstance(payload, dict):
        st.info("Hyperscaler capex data is unavailable.")
        return

    quarterly = payload.get("quarterly", pd.DataFrame())
    if not isinstance(quarterly, pd.DataFrame) or quarterly.empty:
        live_payload = load_hyperscaler_capex()
        live_quarterly = live_payload.get("quarterly", pd.DataFrame()) if isinstance(live_payload, dict) else pd.DataFrame()
        if isinstance(live_quarterly, pd.DataFrame) and not live_quarterly.empty:
            payload = live_payload
            quarterly = live_quarterly

    for error in payload.get("errors", []):
        st.warning(error)

    if (
        (not isinstance(quarterly, pd.DataFrame) or quarterly.empty)
    ):
        st.info("Latest four-quarter hyperscaler capex observations are unavailable.")
        return

    total_bn = _num(payload.get("quarterly_total_bn"))
    if math.isnan(total_bn):
        total_bn = _num(payload.get("total_bn"))
    st.metric(
        "Latest four-quarter total",
        "n/a" if math.isnan(total_bn) else f"${total_bn:,.1f}bn",
    )
    st.plotly_chart(_hyperscaler_capex_chart(payload), width="stretch")
    st.caption(
        "Source: SEC EDGAR XBRL companyconcept API. Reported actuals only; excludes analyst forecasts. "
        "Quarterly values use reported quarterly facts where available and YTD deltas where companies report cumulative cash-flow periods."
    )


def _render_sec_equity_events(sec_events: dict[str, Any] | None = None) -> None:
    st.subheader("Crypto Equity Events (from SEC Filings)")
    st.caption(
        "Daily snapshot from SEC EDGAR public submissions. "
        "Upcoming items are estimated periodic filing windows, not company event guidance."
    )
    if sec_events is None:
        sec_events = load_sec_equity_events()
    if not isinstance(sec_events, dict):
        st.info("SEC EDGAR event data is unavailable.")
        return

    for error in sec_events.get("errors", []):
        st.warning(error)

    upcoming = sec_events.get("upcoming", pd.DataFrame())
    recent = sec_events.get("recent", pd.DataFrame())
    _sec_event_summary_metrics(upcoming, recent)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Expected periodic filings**")
        if isinstance(upcoming, pd.DataFrame) and not upcoming.empty:
            display = upcoming.copy()
            display.insert(0, "Logo", display["Ticker"].map(_equity_logo_url))
            for col in ["Period", "Expected Filing Date"]:
                display[col] = display[col].map(_format_date_cell)
            keep = ["Logo", "Ticker", "Event", "Expected Filing Date", "Days", "Status"]
            st.dataframe(
                display[[c for c in keep if c in display]],
                hide_index=True,
                width="stretch",
                column_config={"Logo": st.column_config.ImageColumn("Company", width="small")},
            )
        else:
            st.info("No expected periodic filing windows found.")
    with c2:
        st.markdown("**Recent notable filings**")
        if isinstance(recent, pd.DataFrame) and not recent.empty:
            display = recent.copy()
            display.insert(0, "Logo", display["Ticker"].map(_equity_logo_url))
            for col in ["Filed", "Period"]:
                display[col] = display[col].map(_format_date_cell)
            keep = ["Logo", "Ticker", "Filed", "Form", "Event", "Description", "Link"]
            st.dataframe(
                display[[c for c in keep if c in display]],
                hide_index=True,
                width="stretch",
                column_config={
                    "Logo": st.column_config.ImageColumn("Company", width="small"),
                    "Link": st.column_config.LinkColumn("SEC filing"),
                },
            )
        else:
            st.info("No notable SEC filings in the last 45 days.")


def _render_bitcoin_com_charts(payload: dict[str, Any] | None = None) -> None:
    st.subheader("Bitcoin Charts")
    st.caption("Daily snapshot from the public charts.bitcoin.com API.")
    if payload is None:
        payload = load_bitcoin_com_charts()
    if not isinstance(payload, dict):
        st.info("Bitcoin.com chart data is unavailable.")
        return
    for error in payload.get("errors", []):
        st.warning(error)
    charts = payload.get("charts", {})
    if not charts:
        st.info("Bitcoin.com chart data is unavailable.")
        return
    chart_ids = [chart_id for chart_id, _title, _endpoint in BITCOIN_COM_CHART_SPECS if chart_id in charts]
    tabs = st.tabs([charts[chart_id].get("title", chart_id) for chart_id in chart_ids])
    for tab, chart_id in zip(tabs, chart_ids):
        chart = charts[chart_id]
        with tab:
            description = chart.get("description")
            if description:
                st.caption(description)
            st.plotly_chart(_bitcoin_com_chart_fig(chart_id, chart), width="stretch")


def _render_defillama_flow_charts(
    fees: pd.DataFrame | None = None,
    bridges: pd.DataFrame | None = None,
    stablecoins: pd.DataFrame | None = None,
) -> None:
    st.subheader("On-Chain Flow Charts")
    if fees is None:
        fees = load_chain_fees_24h()
    if bridges is None:
        bridges = load_bridge_net_flows()
    if stablecoins is None:
        stablecoins = load_stablecoin_supply_changes()
    c1, c2, c3 = st.columns(3)
    with c1:
        if fees.empty:
            st.info("Chain fee data unavailable.")
        else:
            st.plotly_chart(_bar_chart(fees, "Fees", "Chain", "Top Chains by Fees (Last 24H)"), width="stretch")
    with c2:
        if bridges.empty:
            st.info("Bridge net-flow data unavailable from the free DefiLlama endpoint.")
        else:
            st.plotly_chart(
                _bar_chart(bridges, "Net Flow", "Chain", "Top Bridged Net Flows (Last 24H)", color_signed=True),
                width="stretch",
            )
    with c3:
        if stablecoins.empty:
            st.info("Stablecoin supply data unavailable.")
        else:
            st.plotly_chart(
                _bar_chart(
                    stablecoins,
                    "Change",
                    "Chain",
                    "Top Stablecoin Supply Changes (Last 24H)",
                    color_signed=True,
                ),
                width="stretch",
            )
    st.caption("Fees, bridge flows, and stablecoin changes are loaded from the daily Market Update snapshot.")


def _render_cycle_indicators(cycle: pd.DataFrame | None = None) -> None:
    st.subheader("Crypto Market Cycle Indicators")
    if cycle is None:
        cycle = load_cycle_indicators()
    st.caption("Free/public mode: no paid CoinMarketCap API key. Rows with no current value are hidden.")
    if cycle.empty:
        st.info("No current free/public cycle indicator values are available.")
        return
    st.dataframe(cycle, hide_index=True, width="stretch")


def render_macro_monitor() -> None:
    """Render the Market Update tab."""
    header, reload_col, rebuild_col = st.columns([4, 1, 1])
    header.markdown(
        '<div class="hero-card"><h3>Market Update</h3>'
        "<p>Daily cached macro, crypto market structure, token movers, equities, cycle indicators, and on-chain flows.</p></div>",
        unsafe_allow_html=True,
    )
    if reload_col.button("Reload", key="macro_monitor_reload"):
        load_market_update_snapshot.clear()
        st.rerun()
    if rebuild_col.button("Rebuild now", key="macro_monitor_rebuild"):
        with st.spinner("Building Market Update snapshot..."):
            build_market_update_snapshot()
        load_market_update_snapshot.clear()
        st.rerun()

    snapshot, snapshot_error = load_market_update_snapshot(_snapshot_signature())
    if snapshot_error:
        st.warning(snapshot_error)
        st.info("Run `scripts/refresh_market_update_cache.py` to create the daily snapshot.")
        return
    if snapshot is None:
        st.warning("No Market Update snapshot is available.")
        return

    snapshot_created_at = _snapshot_created_at(snapshot)
    snapshot_age = _snapshot_age_hours(snapshot)
    if snapshot_created_at is None:
        st.error("Market Update snapshot has no valid creation timestamp. Rebuild the snapshot before using this tab.")
        return
    st.caption(
        f"Snapshot built {snapshot_created_at.strftime('%Y-%m-%d %H:%M:%S UTC')} "
        f"({snapshot_age:.1f}h old). Scheduled refresh: daily at 09:00 GMT."
    )
    if snapshot_age is not None and snapshot_age > MARKET_UPDATE_MAX_AGE_HOURS:
        st.error(
            f"Market Update snapshot is stale ({snapshot_age:.1f}h old). "
            "Rebuild the snapshot before relying on this tab."
        )
        return

    for error in snapshot.get("errors", []):
        st.warning(error)

    macro = snapshot.get("macro", {})
    data = macro.get("data", pd.DataFrame())
    for error in macro.get("errors", []):
        st.warning(error)

    market_structure = snapshot.get("market_structure", {})
    _render_market_structure(market_structure.get("data", {}), market_structure.get("errors", []))

    st.subheader("Index Performance Benchmark (YTD 2026)")
    benchmark_block = snapshot.get("index_benchmark", {})
    benchmark = benchmark_block.get("data", pd.DataFrame())
    benchmark_errors = benchmark_block.get("errors", [])
    if benchmark.empty:
        st.info("Benchmark data is unavailable.")
    else:
        st.plotly_chart(_benchmark_chart(benchmark), width="stretch")
    for error in benchmark_errors:
        st.caption(error)

    _render_sector_performance(snapshot.get("sector_performance", pd.DataFrame()))
    _render_token_movers(snapshot.get("token_movers", {}))
    _render_crypto_equities(snapshot.get("crypto_equities", pd.DataFrame()))
    _render_sec_equity_events(snapshot.get("sec_equity_events", {}))
    _render_cycle_indicators(snapshot.get("cycle_indicators", pd.DataFrame()))
    _render_bitcoin_com_charts(snapshot.get("bitcoin_com_charts", {}))
    _render_defillama_flow_charts(
        snapshot.get("chain_fees", pd.DataFrame()),
        snapshot.get("bridge_net_flows", pd.DataFrame()),
        snapshot.get("stablecoin_supply_changes", pd.DataFrame()),
    )

    if data.empty:
        st.warning("No FRED macro data is available right now.")
    else:
        st.subheader("Macro Reference Points")
        _render_macro_cards(data)
        table = data[["Series ID", "Label", "Display Value", "Observation Date", "Detail"]].rename(
            columns={"Display Value": "Value"}
        )
        st.dataframe(table, hide_index=True, width="stretch")

    _render_hyperscaler_capex(snapshot.get("hyperscaler_capex", {}))
