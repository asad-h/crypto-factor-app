"""Historical as-of data assembly for the May24toMay26 research layer."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from crypto_factor_model.clients.binance import BinanceClient
from crypto_factor_model.clients.blockworks import BlockworksClient
from crypto_factor_model.clients.defillama import DefiLlamaClient
from crypto_factor_model.config import BW_CHAIN_METRICS, BW_CHAIN_SLUGS, COINGECKO_API_KEY, COINGECKO_BASE_URL
from crypto_factor_model.data.asset_master import normalise_text
from crypto_factor_model.research.constants import (
    BASE_VOLUME_USD,
    DATA_END_EXCLUSIVE,
    EVALUATION_START,
    HORIZONS,
    MIN_MARKET_CAP_USD,
    MIN_PRICE_HISTORY_DAYS,
    PROJECT_ROOT,
    RESEARCH_CACHE_DIR,
    RESEARCH_NAME,
    SENSITIVITY_VOLUME_USD,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
    WARMUP_START,
)
from crypto_factor_model.signals.flows import compute_all_flows
from crypto_factor_model.signals.fundamentals import compute_all_fundamentals
from crypto_factor_model.signals.momentum import compute_all_momentum
from crypto_factor_model.signals.utils import rank_zscore_panel

logger = logging.getLogger(__name__)


STABLE_TICKERS = {
    "USDT",
    "USDC",
    "DAI",
    "FDUSD",
    "TUSD",
    "USDE",
    "USDS",
    "BUSD",
    "FRAX",
    "GHO",
    "PYUSD",
    "USDD",
    "USD1",
}

EXCLUSION_RE = re.compile(
    r"stable|wrapped|pegged|tokeni[sz]ed|treasury|t[- ]?bill|basket|"
    r"commodity|gold|silver|synthetic usd|institutional digital liquidity",
    flags=re.IGNORECASE,
)

COMMODITY_TICKERS = {"PAXG", "XAUT", "KDAU", "KAG"}


@dataclass
class ResearchPanels:
    """Wide-panel research inputs keyed by internal research_id columns."""

    master: pd.DataFrame
    price: pd.DataFrame
    volume_24h: pd.DataFrame
    market_cap: pd.DataFrame
    fdv: pd.DataFrame
    raw_metrics: dict[str, pd.DataFrame]
    raw_metrics_lag1: dict[str, pd.DataFrame]
    btc_price: pd.Series
    regime: pd.DataFrame
    eligibility: dict[str, pd.DataFrame]
    forward_returns: dict[int, pd.DataFrame]
    entry_prices: dict[int, pd.DataFrame]
    exit_prices: dict[int, pd.DataFrame]
    raw_signals: dict[str, dict[str, pd.DataFrame]]
    zscore_signals: dict[str, dict[str, pd.DataFrame]]
    source_audit: pd.DataFrame


def ensure_research_dirs() -> None:
    RESEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "notebooks").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "output" / "factor_evaluation").mkdir(parents=True, exist_ok=True)


def _daily_panel(df: pd.DataFrame | pd.Series | None) -> pd.DataFrame:
    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    out = df.copy()
    out.index = pd.to_datetime(out.index).normalize()
    out = out.groupby(level=0).last().sort_index()
    return out.apply(pd.to_numeric, errors="coerce")


def _align_panel(panel: pd.DataFrame | pd.Series | None, index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    if panel is None or (hasattr(panel, "empty") and panel.empty):
        return pd.DataFrame(index=index, columns=columns, dtype=float)
    out = _daily_panel(panel).reindex(index).ffill(limit=7)
    return out.reindex(columns=columns)


def _combine_first(left: pd.DataFrame | None, right: pd.DataFrame | None) -> pd.DataFrame:
    if left is None or left.empty:
        return right.copy() if right is not None else pd.DataFrame()
    if right is None or right.empty:
        return left.copy()
    return left.combine_first(right)


def _common_index(panels: list[pd.DataFrame]) -> pd.DatetimeIndex:
    indexes = [pd.to_datetime(p.index).normalize() for p in panels if isinstance(p, pd.DataFrame) and not p.empty]
    if not indexes:
        return pd.date_range(WARMUP_START, DATA_END_EXCLUSIVE - pd.Timedelta(days=1), freq="D")
    start = min(WARMUP_START, min(idx.min() for idx in indexes))
    end = min(DATA_END_EXCLUSIVE - pd.Timedelta(days=1), max(idx.max() for idx in indexes))
    return pd.date_range(start.normalize(), end.normalize(), freq="D")


def _is_excluded_asset(row: pd.Series) -> bool:
    ticker = str(row.get("ticker", "")).upper()
    if ticker in STABLE_TICKERS or ticker in COMMODITY_TICKERS:
        return True
    text = " ".join(
        str(row.get(col, ""))
        for col in ["ticker", "name", "category", "sector", "coingecko_id", "defillama_slug"]
        if pd.notna(row.get(col, pd.NA))
    )
    return bool(EXCLUSION_RE.search(text))


def load_research_asset_master(max_assets: int = 140) -> pd.DataFrame:
    """Load the screener asset master as a mapping seed, then filter it for research."""
    path = PROJECT_ROOT / "cache" / "screener" / "asset_master.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing asset master seed: {path}")

    master = pd.read_parquet(path).copy()
    master["research_id"] = master["blockworks_slug"].astype(str)
    master["token"] = master["ticker"].astype(str).str.upper()
    master = master[master["token"].notna() & master["coingecko_id"].notna()].copy()
    master = master[~master.apply(_is_excluded_asset, axis=1)].copy()

    markets_path = PROJECT_ROOT / "cache" / "screener" / "coingecko_markets.parquet"
    if markets_path.exists():
        markets = pd.read_parquet(markets_path)
        if "id" in markets:
            markets = markets.rename(
                columns={
                    "id": "coingecko_id",
                    "current_price": "current_price_snapshot",
                    "market_cap": "_current_market_cap",
                    "total_volume": "_current_volume_24h",
                    "fully_diluted_valuation": "_current_fdv",
                }
            )
            keep = [c for c in ["coingecko_id", "current_price_snapshot", "_current_market_cap", "_current_volume_24h", "_current_fdv"] if c in markets]
            master = master.merge(markets[keep], on="coingecko_id", how="left")
    for col in ["_current_market_cap", "_current_volume_24h", "_current_fdv"]:
        if col not in master:
            master[col] = np.nan
    if "current_price_snapshot" not in master:
        master["current_price_snapshot"] = np.nan

    master["_has_price_venue"] = master["binance_spot_symbol"].notna() | master["binance_futures_symbol"].notna()
    master["_sort_market"] = pd.to_numeric(master["_current_market_cap"], errors="coerce").fillna(0)
    master["_sort_volume"] = pd.to_numeric(master["_current_volume_24h"], errors="coerce").fillna(0)
    master["_shortable"] = master["binance_futures_symbol"].notna()
    master["_is_major"] = master["token"].isin({"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK"})

    master = master.sort_values(
        ["_is_major", "_shortable", "_has_price_venue", "_sort_market", "_sort_volume", "token"],
        ascending=[False, False, False, False, False, True],
    )
    master = master.drop_duplicates("token", keep="first")
    if max_assets and len(master) > max_assets:
        majors = master[master["_is_major"]]
        futures = master[master["_shortable"]].head(max_assets)
        top = master.head(max_assets)
        master = pd.concat([majors, futures, top], ignore_index=True).drop_duplicates("research_id", keep="first")
        master = master.sort_values(
            ["_is_major", "_shortable", "_sort_market", "_sort_volume", "token"],
            ascending=[False, False, False, False, True],
        ).head(max_assets)

    master["current_market_cap_snapshot"] = pd.to_numeric(master["_current_market_cap"], errors="coerce")
    master["current_volume_24h_snapshot"] = pd.to_numeric(master["_current_volume_24h"], errors="coerce")
    master["current_fdv_snapshot"] = pd.to_numeric(master["_current_fdv"], errors="coerce")
    return master.drop(columns=[c for c in master.columns if c.startswith("_")], errors="ignore").reset_index(drop=True)


def _coingecko_cache_path(coin_id: str, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    safe = str(coin_id).replace("/", "_")
    return RESEARCH_CACHE_DIR / f"coingecko_range_{safe}_{start.date()}_{end.date()}.parquet"


class CoinGeckoResearchClient:
    """Small CoinGecko range client with persistent research cache."""

    def __init__(self, api_key: str = COINGECKO_API_KEY, base_url: str = COINGECKO_BASE_URL):
        self.api_key = api_key or ""
        if self.api_key and "pro-api.coingecko.com" not in base_url:
            base_url = "https://pro-api.coingecko.com/api/v3"
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"x-cg-pro-api-key": self.api_key})

    def get_market_chart_range(
        self,
        coin_id: str,
        start: pd.Timestamp = WARMUP_START,
        end: pd.Timestamp = DATA_END_EXCLUSIVE,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_path = _coingecko_cache_path(coin_id, start, end)
        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        params = {
            "vs_currency": "usd",
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
        }
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                resp = self.session.get(
                    f"{self.base_url}/coins/{coin_id}/market_chart/range",
                    params=params,
                    timeout=30,
                )
                if resp.status_code == 429:
                    time.sleep(20 + 10 * attempt)
                    continue
                resp.raise_for_status()
                payload = resp.json()
                frames = {}
                for key, out_col in [
                    ("prices", "price"),
                    ("market_caps", "market_cap"),
                    ("total_volumes", "volume_24h"),
                ]:
                    values = payload.get(key, [])
                    if not values:
                        continue
                    part = pd.DataFrame(values, columns=["ts", out_col])
                    part["date"] = pd.to_datetime(part["ts"], unit="ms").dt.normalize()
                    part[out_col] = pd.to_numeric(part[out_col], errors="coerce")
                    frames[out_col] = part.groupby("date")[out_col].last()
                chart = pd.DataFrame(frames).sort_index()
                if not chart.empty:
                    chart = chart.loc[(chart.index >= start.normalize()) & (chart.index < end.normalize())]
                    chart.to_parquet(cache_path)
                return chart
            except Exception as exc:  # pragma: no cover - exercised by live API only
                last_error = exc
                time.sleep(2 + attempt * 3)
        if last_error:
            logger.warning("CoinGecko range failed for %s: %s", coin_id, last_error)
        return pd.DataFrame()


def fetch_coingecko_panels(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    client = CoinGeckoResearchClient()
    by_coin = master.dropna(subset=["coingecko_id"]).groupby("coingecko_id")["research_id"].apply(list).to_dict()
    frames: dict[str, dict[str, pd.Series]] = {"price": {}, "market_cap": {}, "volume_24h": {}}
    for i, (coin_id, research_ids) in enumerate(by_coin.items(), start=1):
        cache_path = _coingecko_cache_path(str(coin_id), WARMUP_START, DATA_END_EXCLUSIVE)
        if cache_path.exists():
            chart = pd.read_parquet(cache_path)
        elif client.api_key:
            chart = client.get_market_chart_range(str(coin_id))
        else:
            screener_path = PROJECT_ROOT / "cache" / "screener" / f"coingecko_market_chart_{coin_id}_120.parquet"
            chart = pd.read_parquet(screener_path) if screener_path.exists() else pd.DataFrame()
        if chart.empty:
            continue
        chart = _daily_panel(chart)
        for rid in research_ids:
            for col in frames:
                if col in chart:
                    frames[col][rid] = chart[col].rename(rid)
        if i % 10 == 0:
            logger.info("CoinGecko history fetched/cached for %s/%s ids", i, len(by_coin))
        if cache_path.exists() or client.api_key:
            time.sleep(0.01)
    return {name: _daily_panel(pd.DataFrame(series_map)) for name, series_map in frames.items()}


def fetch_binance_market_panels(master: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    client = BinanceClient()
    price: dict[str, pd.Series] = {}
    volume: dict[str, pd.Series] = {}
    audit_rows: list[dict[str, Any]] = []
    end = DATA_END_EXCLUSIVE.strftime("%Y-%m-%d")
    start = WARMUP_START.strftime("%Y-%m-%d")

    for _, row in master.iterrows():
        rid = str(row["research_id"])
        chosen_source = None
        for source_name, symbol, fetcher in [
            ("binance_spot", row.get("binance_spot_symbol"), client.get_klines),
            ("binance_futures", row.get("binance_futures_symbol"), client.get_futures_klines),
        ]:
            if pd.isna(symbol) or not str(symbol):
                continue
            try:
                df = fetcher(str(symbol), interval="1d", start=start, end=end, use_cache=True)
            except Exception as exc:
                logger.debug("Binance %s unavailable for %s/%s: %s", source_name, rid, symbol, exc)
                continue
            if df.empty or df["close"].dropna().empty:
                continue
            price[rid] = df["close"].rename(rid)
            volume[rid] = df["quote_volume"].rename(rid)
            chosen_source = source_name
            break
        audit_rows.append(
            {
                "research_id": rid,
                "token": row.get("token"),
                "binance_price_source": chosen_source,
                "has_binance_price_history": bool(chosen_source),
            }
        )

    return {
        "price": _daily_panel(pd.DataFrame(price)),
        "volume_24h": _daily_panel(pd.DataFrame(volume)),
    }, pd.DataFrame(audit_rows)


def _defillama_child_slugs(row: pd.Series) -> list[str]:
    raw = row.get("defillama_child_slugs", pd.NA)
    if pd.notna(raw) and str(raw).strip():
        return [part for part in str(raw).split("|") if part]
    slug = row.get("defillama_slug", pd.NA)
    return [str(slug)] if pd.notna(slug) else []


def _defillama_match_keys(row: pd.Series) -> set[str]:
    keys = {
        normalise_text(row.get("name")),
        normalise_text(str(row.get("name", "")).replace(" (Combined)", "")),
        normalise_text(row.get("defillama_slug")),
    }
    keys.update(normalise_text(slug) for slug in _defillama_child_slugs(row))
    return {key for key in keys if key}


def _flatten_defillama_breakdown(payload: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            flat.update(_flatten_defillama_breakdown(value))
        else:
            try:
                flat[str(key)] = float(value)
            except (TypeError, ValueError):
                flat[str(key)] = np.nan
    return flat


def _breakdown_chart_to_panel(breakdown: list, wanted_keys: set[str]) -> pd.DataFrame:
    records: list[dict[str, float | pd.Timestamp]] = []
    for entry in breakdown:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
            continue
        day = pd.to_datetime(entry[0], unit="s").normalize()
        record: dict[str, float | pd.Timestamp] = {"date": day}
        for name, value in _flatten_defillama_breakdown(entry[1]).items():
            key = normalise_text(name)
            if key not in wanted_keys:
                continue
            current = record.get(key, np.nan)
            record[key] = float(value) if pd.isna(current) else float(current) + float(value)
        if len(record) > 1:
            records.append(record)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).set_index("date").sort_index()


def _overview_metric_panel(master: pd.DataFrame, overview: dict) -> pd.DataFrame:
    breakdown = overview.get("totalDataChartBreakdown", []) if isinstance(overview, dict) else []
    if not breakdown:
        return pd.DataFrame()
    wanted = set().union(*[_defillama_match_keys(row) for _, row in master.iterrows()])
    source = _breakdown_chart_to_panel(breakdown, wanted)
    if source.empty:
        return pd.DataFrame()

    frames: dict[str, pd.Series] = {}
    for _, row in master[master["asset_type"].ne("chain")].iterrows():
        matched = [source[key] for key in _defillama_match_keys(row) if key in source]
        if matched:
            frames[str(row["research_id"])] = pd.concat(matched, axis=1).sum(axis=1, min_count=1)
    return _daily_panel(pd.DataFrame(frames))


def fetch_defillama_panels(master: pd.DataFrame, tvl_limit: int = 80) -> dict[str, pd.DataFrame]:
    dl = DefiLlamaClient()
    dl_cache = PROJECT_ROOT / "cache" / "defillama"
    out: dict[str, pd.DataFrame] = {}
    try:
        out["trading_fees"] = _overview_metric_panel(master, dl.get_fees_overview(data_type="dailyFees", use_cache=True))
    except Exception as exc:
        logger.warning("DefiLlama fees overview unavailable: %s", exc)
        out["trading_fees"] = pd.DataFrame()
    try:
        out["revenue"] = _overview_metric_panel(master, dl.get_fees_overview(data_type="dailyRevenue", use_cache=True))
    except Exception as exc:
        logger.warning("DefiLlama revenue overview unavailable: %s", exc)
        out["revenue"] = pd.DataFrame()
    try:
        out["dex_volume"] = _overview_metric_panel(master, dl.get_dex_overview(use_cache=True))
    except Exception as exc:
        logger.warning("DefiLlama DEX overview unavailable: %s", exc)
        out["dex_volume"] = pd.DataFrame()
    try:
        out["open_interest"] = _overview_metric_panel(master, dl.get_open_interest_overview(use_cache=True))
    except Exception as exc:
        logger.warning("DefiLlama open-interest overview unavailable: %s", exc)
        out["open_interest"] = pd.DataFrame()

    tvl_frames: dict[str, pd.Series] = {}
    tvl_rows = master.dropna(subset=["defillama_slug"]).copy()
    if "_current_market_cap" in tvl_rows:
        tvl_rows["_sort"] = pd.to_numeric(tvl_rows["_current_market_cap"], errors="coerce").fillna(0)
    else:
        tvl_rows["_sort"] = 0.0
    tvl_rows = tvl_rows.sort_values("_sort", ascending=False).head(tvl_limit)
    for _, row in tvl_rows.iterrows():
        rid = str(row["research_id"])
        try:
            if str(row.get("asset_type", "")).lower() == "chain":
                safe_chain = str(row["defillama_chain"]).replace("/", "-")
                cache_path = dl_cache / f"chain_tvl_{safe_chain}.parquet"
                series = pd.read_parquet(cache_path).iloc[:, 0].rename(rid) if cache_path.exists() else pd.Series(dtype=float)
            else:
                child_series = []
                for slug in _defillama_child_slugs(row):
                    cache_path = dl_cache / f"protocol_tvl_{slug}.parquet"
                    s = pd.read_parquet(cache_path).iloc[:, 0].rename(slug) if cache_path.exists() else pd.Series(dtype=float)
                    if not s.empty:
                        child_series.append(s)
                series = pd.concat(child_series, axis=1).sum(axis=1, min_count=1) if child_series else pd.Series(dtype=float)
            if not series.empty:
                tvl_frames[rid] = series.rename(rid)
        except Exception as exc:
            logger.debug("DefiLlama TVL unavailable for %s: %s", rid, exc)
    out["tvl"] = _daily_panel(pd.DataFrame(tvl_frames))

    stable_frames: dict[str, pd.Series] = {}
    chain_rows = master[master["asset_type"].eq("chain") & master["defillama_chain"].notna()].copy()
    for _, row in chain_rows.iterrows():
        rid = str(row["research_id"])
        try:
            safe_chain = str(row["defillama_chain"]).replace("/", "-")
            cache_path = dl_cache / f"stablecoin_supply_{safe_chain}.parquet"
            s = pd.read_parquet(cache_path).iloc[:, 0].rename(rid) if cache_path.exists() else pd.Series(dtype=float)
            if not s.empty:
                stable_frames[rid] = s.rename(rid)
        except Exception as exc:
            logger.debug("Stablecoin supply unavailable for %s: %s", rid, exc)
    out["stablecoin_supply"] = _daily_panel(pd.DataFrame(stable_frames))
    return out


def _blockworks_chain_slug(row: pd.Series) -> str | None:
    candidates = [
        row.get("blockworks_match_slug"),
        row.get("defillama_slug"),
        str(row.get("defillama_chain", "")).lower().replace(" ", "-"),
        str(row.get("coingecko_id", "")).lower(),
    ]
    for candidate in candidates:
        if pd.isna(candidate):
            continue
        value = str(candidate).lower()
        if value in BW_CHAIN_SLUGS:
            return value
    return None


def _read_blockworks_cached_series(bw: BlockworksClient, project: str, metric: str) -> pd.Series:
    for start in [
        WARMUP_START.strftime("%Y-%m-%d"),
        "2024-01-01",
        "2024-06-01",
    ]:
        cache_path = bw._cache_path(f"{project}_{metric}_{start}_{None}")
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            if not df.empty:
                series = df.iloc[:, 0]
                series.index = pd.to_datetime(series.index).normalize()
                return pd.to_numeric(series, errors="coerce").sort_index().rename(project)
    return pd.Series(dtype=float, name=project)


def fetch_blockworks_chain_panels(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    bw = BlockworksClient()
    chain_map = {
        str(row["research_id"]): slug
        for _, row in master[master["asset_type"].eq("chain")].iterrows()
        if (slug := _blockworks_chain_slug(row)) is not None
    }
    token_map: dict[str, str] = {}
    for _, row in master.iterrows():
        rid = str(row["research_id"])
        if rid in chain_map:
            token_map[rid] = chain_map[rid]
            continue
        slug = row.get("blockworks_match_slug")
        if pd.notna(slug) and str(slug).strip():
            token_map[rid] = str(slug)

    panels: dict[str, pd.DataFrame] = {}

    token_metric_map = {
        "bw_price": BW_CHAIN_METRICS["price"],
        "supply": BW_CHAIN_METRICS["supply"],
        "fdv": BW_CHAIN_METRICS["fdv"],
    }
    token_reverse: dict[str, list[str]] = {}
    for rid, bw_slug in token_map.items():
        token_reverse.setdefault(bw_slug, []).append(rid)
    for key, metric in token_metric_map.items():
        expanded: dict[str, pd.Series] = {}
        for bw_slug, rids in token_reverse.items():
            series = _read_blockworks_cached_series(bw, bw_slug, metric)
            if series.empty:
                continue
            for rid in rids:
                expanded[rid] = series.rename(rid)
        panels[key] = _daily_panel(pd.DataFrame(expanded))

    chain_metric_map = {
        "bw_price": BW_CHAIN_METRICS["price"],
        "supply": BW_CHAIN_METRICS["supply"],
        "fdv": BW_CHAIN_METRICS["fdv"],
        "revenue": BW_CHAIN_METRICS["revenue"],
        "trading_fees": BW_CHAIN_METRICS["trading_fees"],
        "active_addresses": BW_CHAIN_METRICS["active_addresses"],
        "issuance": BW_CHAIN_METRICS["issuance"],
        "burn": BW_CHAIN_METRICS["burn"],
        "stablecoin_supply": BW_CHAIN_METRICS["stablecoin_supply"],
        "dex_volume": BW_CHAIN_METRICS["dex_volume"],
    }
    reverse = {}
    for rid, bw_slug in chain_map.items():
        reverse.setdefault(bw_slug, []).append(rid)

    for key, metric in chain_metric_map.items():
        if key in panels and not panels[key].empty:
            continue
        expanded: dict[str, pd.Series] = {}
        for bw_slug, rids in reverse.items():
            series = _read_blockworks_cached_series(bw, bw_slug, metric)
            if series.empty:
                continue
            for rid in rids:
                expanded[rid] = series.rename(rid)
        panels[key] = _combine_first(panels.get(key), _daily_panel(pd.DataFrame(expanded)))
    if not panels.get("bw_price", pd.DataFrame()).empty and not panels.get("supply", pd.DataFrame()).empty:
        panels["market_cap"] = panels["bw_price"] * panels["supply"]
    return panels


def compute_btc_20w_regime(btc_price: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    btc = btc_price.dropna().sort_index()
    btc.index = pd.to_datetime(btc.index).normalize()
    weekly_close = btc.resample("W-SUN", label="right", closed="right").last()
    weekly_ma = weekly_close.rolling(20, min_periods=20).mean()
    weekly = pd.DataFrame({"btc_weekly_close": weekly_close, "btc_20w_ma": weekly_ma})
    weekly["regime"] = np.where(weekly["btc_weekly_close"] > weekly["btc_20w_ma"], "Bullish", "Bearish")
    weekly.loc[weekly["btc_20w_ma"].isna(), "regime"] = pd.NA
    daily = weekly.reindex(index, method="ffill")
    daily["btc_close"] = btc.reindex(index).ffill()
    return daily[["btc_close", "btc_weekly_close", "btc_20w_ma", "regime"]]


def compute_forward_return_panels(price: pd.DataFrame) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    returns: dict[int, pd.DataFrame] = {}
    entries: dict[int, pd.DataFrame] = {}
    exits: dict[int, pd.DataFrame] = {}
    for horizon in HORIZONS:
        entry = price.shift(-1)
        exit_ = price.shift(-(horizon + 1))
        entries[horizon] = entry
        exits[horizon] = exit_
        returns[horizon] = (exit_ / entry - 1).replace([np.inf, -np.inf], np.nan)
    return returns, entries, exits


def compute_eligibility_panels(master: pd.DataFrame, price: pd.DataFrame, volume: pd.DataFrame, market_cap: pd.DataFrame) -> dict[str, pd.DataFrame]:
    columns = price.columns.tolist()
    index = price.index
    volume_7d = volume.rolling(7, min_periods=5).mean()
    history_days = pd.DataFrame(0.0, index=index, columns=columns)
    for col in columns:
        first = price[col].first_valid_index()
        if first is not None:
            history_days[col] = (index - pd.Timestamp(first)).days
            history_days.loc[index < first, col] = 0

    excluded = master.set_index("research_id").apply(_is_excluded_asset, axis=1).reindex(columns).fillna(True)
    excluded_panel = pd.DataFrame({col: bool(excluded.get(col, True)) for col in columns}, index=index)
    base = (
        market_cap.ge(MIN_MARKET_CAP_USD)
        & volume_7d.ge(BASE_VOLUME_USD)
        & history_days.ge(MIN_PRICE_HISTORY_DAYS)
        & price.notna()
        & ~excluded_panel
    )
    sensitivity = base & volume_7d.ge(SENSITIVITY_VOLUME_USD)
    shortable_static = master.set_index("research_id")["binance_futures_symbol"].notna().reindex(columns).fillna(False)
    shortable = pd.DataFrame({col: bool(shortable_static.get(col, False)) for col in columns}, index=index) & price.notna()
    return {
        "trailing_7d_volume_24h": volume_7d,
        "price_history_days": history_days,
        "eligible_base": base.astype(bool),
        "eligible_sensitivity": sensitivity.astype(bool),
        "shortable": shortable.astype(bool),
    }


def compute_signal_panels(
    price: pd.DataFrame,
    market_cap: pd.DataFrame,
    fdv: pd.DataFrame,
    raw_metrics_lag1: dict[str, pd.DataFrame],
    btc_price: pd.Series,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, pd.DataFrame]]]:
    fund_inputs = {
        **raw_metrics_lag1,
        "mcap": market_cap,
        "fdv": fdv,
    }
    flow_inputs = {
        key: raw_metrics_lag1.get(key, pd.DataFrame())
        for key in ["stablecoin_supply", "dex_volume", "bridge_flows", "open_interest"]
    }
    raw_signals = {
        "fundamentals": compute_all_fundamentals(fund_inputs, btc_price),
        "momentum": compute_all_momentum(price, btc_price, market_cap),
        "flows": compute_all_flows(flow_inputs),
    }
    zscores = {
        family: {name: rank_zscore_panel(panel) for name, panel in signals.items() if panel is not None and not panel.empty}
        for family, signals in raw_signals.items()
    }
    return raw_signals, zscores


def build_research_panels(max_assets: int = 140) -> ResearchPanels:
    """Build all wide panels needed for walk-forward research."""
    ensure_research_dirs()
    master = load_research_asset_master(max_assets=max_assets)
    slugs = master["research_id"].astype(str).tolist()

    logger.info("Fetching/caching Binance market panels")
    binance_panels, binance_audit = fetch_binance_market_panels(master)
    logger.info("Fetching/caching CoinGecko market-cap and volume panels")
    coingecko_panels = fetch_coingecko_panels(master)
    logger.info("Fetching/caching DefiLlama historical fundamentals and flows")
    dl_panels = fetch_defillama_panels(master)
    logger.info("Fetching/caching Blockworks chain panels")
    bw_panels = fetch_blockworks_chain_panels(master)

    index = _common_index([*binance_panels.values(), *coingecko_panels.values(), *dl_panels.values(), *bw_panels.values()])
    price = _align_panel(binance_panels.get("price"), index, slugs).combine_first(_align_panel(coingecko_panels.get("price"), index, slugs))
    volume = _align_panel(binance_panels.get("volume_24h"), index, slugs).combine_first(
        _align_panel(coingecko_panels.get("volume_24h"), index, slugs)
    )
    market_cap = _align_panel(coingecko_panels.get("market_cap"), index, slugs).combine_first(
        _align_panel(bw_panels.get("market_cap"), index, slugs)
    )
    supply_proxy = (
        pd.to_numeric(master.set_index("research_id")["current_market_cap_snapshot"], errors="coerce")
        / pd.to_numeric(master.set_index("research_id")["current_price_snapshot"], errors="coerce").replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    market_cap_proxy = pd.DataFrame({rid: price[rid] * supply_proxy.get(rid, np.nan) for rid in slugs}, index=index)
    historical_market_cap = market_cap.copy()
    market_cap = historical_market_cap.combine_first(market_cap_proxy)
    fdv = _align_panel(bw_panels.get("fdv"), index, slugs)

    raw_metrics = {
        "revenue": _align_panel(dl_panels.get("revenue"), index, slugs).combine_first(_align_panel(bw_panels.get("revenue"), index, slugs)),
        "trading_fees": _align_panel(dl_panels.get("trading_fees"), index, slugs).combine_first(
            _align_panel(bw_panels.get("trading_fees"), index, slugs)
        ),
        "dex_volume": _align_panel(dl_panels.get("dex_volume"), index, slugs).combine_first(_align_panel(bw_panels.get("dex_volume"), index, slugs)),
        "stablecoin_supply": _align_panel(dl_panels.get("stablecoin_supply"), index, slugs).combine_first(
            _align_panel(bw_panels.get("stablecoin_supply"), index, slugs)
        ),
        "open_interest": _align_panel(dl_panels.get("open_interest"), index, slugs),
        "tvl": _align_panel(dl_panels.get("tvl"), index, slugs),
        "active_addresses": _align_panel(bw_panels.get("active_addresses"), index, slugs),
        "issuance": _align_panel(bw_panels.get("issuance"), index, slugs),
        "burn": _align_panel(bw_panels.get("burn"), index, slugs),
        "supply": _align_panel(bw_panels.get("supply"), index, slugs),
    }
    raw_metrics_lag1 = {key: panel.shift(1) for key, panel in raw_metrics.items()}

    btc = price["chain:bitcoin"] if "chain:bitcoin" in price and price["chain:bitcoin"].dropna().size else pd.Series(dtype=float)
    if btc.empty:
        btc_client = BinanceClient()
        btc = btc_client.get_daily_close(
            "BTCUSDT",
            start=WARMUP_START.strftime("%Y-%m-%d"),
            end=DATA_END_EXCLUSIVE.strftime("%Y-%m-%d"),
        )
        btc.index = pd.to_datetime(btc.index).normalize()
        btc = btc.reindex(index).ffill()
    btc = btc.rename("BTCUSDT")

    regime = compute_btc_20w_regime(btc, index)
    eligibility = compute_eligibility_panels(master, price, volume, market_cap)
    fwd, entry, exit_ = compute_forward_return_panels(price)
    raw_signals, zscore_signals = compute_signal_panels(price, market_cap, fdv, raw_metrics_lag1, btc)

    cg_audit = pd.DataFrame(
        {
            "research_id": slugs,
            "has_coingecko_history": [rid in _daily_panel(coingecko_panels.get("market_cap")).columns for rid in slugs],
        }
    )
    source_audit = master[["research_id", "token", "name", "coingecko_id", "defillama_slug", "binance_spot_symbol", "binance_futures_symbol"]].merge(
        binance_audit.drop(columns=["token"], errors="ignore"),
        on="research_id",
        how="left",
    ).merge(cg_audit, on="research_id", how="left")
    source_audit["has_market_cap_history"] = source_audit["research_id"].isin(market_cap.dropna(how="all", axis=1).columns)
    source_audit["has_true_historical_market_cap"] = source_audit["research_id"].isin(historical_market_cap.dropna(how="all", axis=1).columns)
    source_audit["uses_current_supply_mcap_proxy"] = source_audit["has_market_cap_history"] & ~source_audit["has_true_historical_market_cap"]
    source_audit["has_revenue_history"] = source_audit["research_id"].isin(raw_metrics["revenue"].dropna(how="all", axis=1).columns)
    source_audit["has_fees_history"] = source_audit["research_id"].isin(raw_metrics["trading_fees"].dropna(how="all", axis=1).columns)

    return ResearchPanels(
        master=master,
        price=price,
        volume_24h=volume,
        market_cap=market_cap,
        fdv=fdv,
        raw_metrics=raw_metrics,
        raw_metrics_lag1=raw_metrics_lag1,
        btc_price=btc,
        regime=regime,
        eligibility=eligibility,
        forward_returns=fwd,
        entry_prices=entry,
        exit_prices=exit_,
        raw_signals=raw_signals,
        zscore_signals=zscore_signals,
        source_audit=source_audit,
    )


def _period_label(date: pd.Timestamp) -> str:
    dt = pd.Timestamp(date)
    if TRAIN_START <= dt <= TRAIN_END:
        return "Train"
    if VALIDATION_START <= dt <= VALIDATION_END:
        return "Validation"
    if dt >= TEST_START:
        return "Test"
    return "OutOfSampleGap"


def build_long_dataset(
    panels: ResearchPanels,
    family_scores: dict[str, pd.DataFrame],
    composite_panels: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Convert wide research panels into one long-form auditable parquet frame."""
    slugs = panels.master["research_id"].astype(str).tolist()
    index = panels.price.index[panels.price.index >= EVALUATION_START]
    meta = panels.master.set_index("research_id")
    composite_panels = composite_panels or {}

    metric_panels: dict[str, pd.DataFrame] = {
        "price": panels.price,
        "volume_24h": panels.volume_24h,
        "market_cap": panels.market_cap,
        "fdv": panels.fdv,
        "trailing_7d_volume_24h": panels.eligibility["trailing_7d_volume_24h"],
        "price_history_days": panels.eligibility["price_history_days"],
        **{key: value for key, value in panels.raw_metrics.items()},
        **{f"lag1_{key}": value for key, value in panels.raw_metrics_lag1.items()},
        **{f"family_score_{key}": value for key, value in family_scores.items()},
        **{f"composite_{key}": value for key, value in composite_panels.items()},
    }
    bool_panels = {
        "eligible_base": panels.eligibility["eligible_base"],
        "eligible_sensitivity": panels.eligibility["eligible_sensitivity"],
        "shortable": panels.eligibility["shortable"],
    }
    aligned_metric_panels = {
        col: _align_panel(panel, panels.price.index, slugs).loc[index]
        for col, panel in metric_panels.items()
    }
    aligned_bool_panels = {
        col: panel.reindex(index).reindex(columns=slugs)
        for col, panel in bool_panels.items()
    }
    aligned_regime = panels.regime.reindex(index)
    aligned_raw_signals = {
        (family, name): _align_panel(panel, panels.price.index, slugs).loc[index]
        for family, signals in panels.raw_signals.items()
        for name, panel in signals.items()
    }
    aligned_zscore_signals = {
        (family, name): _align_panel(panel, panels.price.index, slugs).loc[index]
        for family, signals in panels.zscore_signals.items()
        for name, panel in signals.items()
    }
    aligned_forward = {
        horizon: {
            "entry": panels.entry_prices[horizon].reindex(index).reindex(columns=slugs),
            "exit": panels.exit_prices[horizon].reindex(index).reindex(columns=slugs),
            "return": panels.forward_returns[horizon].reindex(index).reindex(columns=slugs),
        }
        for horizon in HORIZONS
    }
    period_values = np.array([_period_label(dt) for dt in index], dtype=object)

    def _panel_values(panel: pd.DataFrame, rid: str, default: float | bool = np.nan) -> np.ndarray:
        if rid in panel.columns:
            return panel[rid].to_numpy()
        return np.full(len(index), default)

    rows: list[pd.DataFrame] = []
    for rid in slugs:
        if rid not in meta.index:
            continue
        m = meta.loc[rid]
        token_data: dict[str, Any] = {
            "date": index,
            "token": m.get("token", m.get("ticker")),
            "research_id": rid,
            "asset_id": m.get("asset_id", pd.NA),
            "entity_key": m.get("entity_key", pd.NA),
            "project": m.get("name", pd.NA),
            "category": m.get("category", pd.NA),
            "sector": m.get("sector", pd.NA),
            "asset_type": m.get("asset_type", pd.NA),
            "coingecko_id": m.get("coingecko_id", pd.NA),
            "defillama_slug": m.get("defillama_slug", pd.NA),
            "blockworks_match_slug": m.get("blockworks_match_slug", pd.NA),
            "binance_spot_symbol": m.get("binance_spot_symbol", pd.NA),
            "binance_futures_symbol": m.get("binance_futures_symbol", pd.NA),
            "period": period_values,
        }

        for col, aligned in aligned_metric_panels.items():
            token_data[col] = _panel_values(aligned, rid)
        for col, aligned in aligned_bool_panels.items():
            token_data[col] = _panel_values(aligned, rid, default=False)
        for col in aligned_regime.columns:
            token_data[col] = aligned_regime[col].to_numpy()

        for (family, name), aligned in aligned_raw_signals.items():
            token_data[f"signal_raw_{family}_{name}"] = _panel_values(aligned, rid)
        for (family, name), aligned in aligned_zscore_signals.items():
            token_data[f"signal_z_{family}_{name}"] = _panel_values(aligned, rid)

        for horizon in HORIZONS:
            ret_values = _panel_values(aligned_forward[horizon]["return"], rid)
            token_data[f"entry_price_{horizon}d"] = _panel_values(aligned_forward[horizon]["entry"], rid)
            token_data[f"exit_price_{horizon}d"] = _panel_values(aligned_forward[horizon]["exit"], rid)
            token_data[f"fwd_return_{horizon}d"] = ret_values
            token_data[f"label_complete_{horizon}d"] = pd.notna(ret_values)
        token_df = pd.DataFrame(token_data)
        rows.append(token_df)

    data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not data.empty:
        audit_cols = [
            "research_id",
            "binance_price_source",
            "has_binance_price_history",
            "has_coingecko_history",
            "has_market_cap_history",
            "has_true_historical_market_cap",
            "uses_current_supply_mcap_proxy",
            "has_revenue_history",
            "has_fees_history",
        ]
        data = data.merge(panels.source_audit[[c for c in audit_cols if c in panels.source_audit]], on="research_id", how="left")
    return data


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
