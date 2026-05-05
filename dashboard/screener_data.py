"""
Cached real-data adapter for dashboard/screener_app.py.

The dashboard should stay a fast reader. This module owns the slower source
refresh path and persists the exact frames Streamlit needs under cache/screener.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crypto_factor_model.clients.binance import BinanceClient
from crypto_factor_model.clients.blockworks import BlockworksClient
from crypto_factor_model.clients.defillama import DefiLlamaClient
from crypto_factor_model.composite import compute_composite, compute_family_score
from crypto_factor_model.config import (
    BW_CHAIN_METRICS,
    BW_CHAIN_SLUGS,
    BW_PROJECT_METRICS,
    CACHE_DIR,
    COINGECKO_API_KEY,
    COINGECKO_BASE_URL,
    EXCLUDE_CATEGORIES,
    SLUG_TO_BINANCE,
)
from crypto_factor_model.data.asset_master import (
    ASSET_MASTER_COLUMNS,
    build_defillama_asset_frame,
    normalise_symbol,
    normalise_text,
    score_name_symbol_match,
    slugify_key,
)
from crypto_factor_model.signals.flows import compute_all_flows
from crypto_factor_model.signals.fundamentals import compute_all_fundamentals
from crypto_factor_model.signals.momentum import compute_all_momentum

logger = logging.getLogger(__name__)

SCREENER_CACHE_DIR = CACHE_DIR / "screener"
SCREENER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ASSET_MASTER_PATH = SCREENER_CACHE_DIR / "asset_master.parquet"
SNAPSHOT_PATH = SCREENER_CACHE_DIR / "snapshot.parquet"
PROJECT_TS_PATH = SCREENER_CACHE_DIR / "project_timeseries.parquet"
SUMMARY_TS_PATH = SCREENER_CACHE_DIR / "summary_timeseries.parquet"
FACTOR_SCORES_PATH = SCREENER_CACHE_DIR / "factor_scores.parquet"
FACTOR_BASKETS_PATH = SCREENER_CACHE_DIR / "factor_baskets.parquet"
RAW_ASSETS_PATH = SCREENER_CACHE_DIR / "blockworks_assets.json"
CG_COINS_PATH = SCREENER_CACHE_DIR / "coingecko_coins.json"
CG_MARKETS_PATH = SCREENER_CACHE_DIR / "coingecko_markets.parquet"

DEFAULT_START_DATE = os.getenv("SCREENER_START_DATE", "2024-06-01")
DEFAULT_MIN_MARKET_CAP = float(os.getenv("SCREENER_MIN_MARKET_CAP_USD", "50000000"))
DEFAULT_MAX_CANDIDATES = int(os.getenv("SCREENER_MAX_CANDIDATES", "350"))
DEFAULT_STALE_DAYS = int(os.getenv("SCREENER_STALE_DAYS", "14"))
MAX_DEFILLAMA_TVL_FETCHES = int(os.getenv("SCREENER_MAX_DEFILLAMA_TVL_FETCHES", "80"))
MAX_BINANCE_OI_FETCHES = int(os.getenv("SCREENER_MAX_BINANCE_OI_FETCHES", "120"))
MAX_COINGECKO_CHART_FETCHES = int(os.getenv("SCREENER_MAX_COINGECKO_CHART_FETCHES", "40"))
MAX_DEFILLAMA_METRIC_FETCHES = int(os.getenv("SCREENER_MAX_DEFILLAMA_METRIC_FETCHES", "60"))
MAX_DEFILLAMA_STABLECOIN_FETCHES = int(os.getenv("SCREENER_MAX_DEFILLAMA_STABLECOIN_FETCHES", "30"))
FORCE_RAW_SOURCE_REFRESH = os.getenv("SCREENER_FORCE_RAW_SOURCE_REFRESH", "").lower() in {"1", "true", "yes"}

FDV_BUCKETS = [
    ("Micro", 0, 100_000_000),
    ("Small", 100_000_000, 500_000_000),
    ("Mid", 500_000_000, 1_000_000_000),
    ("Large", 1_000_000_000, 5_000_000_000),
    ("Mega", 5_000_000_000, float("inf")),
]
FDV_BUCKET_ORDER = [bucket[0] for bucket in FDV_BUCKETS]

CHANGE_METRICS_30D = {
    "revenue_mom_pct": "Revenue",
    "fees_mom_pct": "Fees",
    "deposits_mom_pct": "Deposits",
    "oi_mom_pct": "Open interest",
}

DEFI_CATEGORIES = {
    "dex",
    "dexs",
    "lending",
    "liquid staking",
    "yield",
    "yield / restaking",
    "derivatives",
    "perps",
    "decentralized exchanges",
}

COINGECKO_ID_OVERRIDES = {
    "aave": "aave",
    "aerodrome": "aerodrome-finance",
    "curve-finance": "curve-dao-token",
    "hyperliquid": "hyperliquid",
    "lido": "lido-dao",
    "pump": "pump-fun",
    "uniswap": "uniswap",
}

FACTOR_LAG_DAYS = 30
FACTOR_SCORE_COLUMNS = [
    "factor_score",
    "fundamentals_score",
    "momentum_score",
    "flows_score",
]

DEFILLAMA_CHAIN_SLUG_TO_BW = {
    "aptos": "aptos",
    "arbitrum": "arbitrum",
    "avalanche": "avalanche",
    "base": "base",
    "berachain": "berachain",
    "bitcoin": "bitcoin",
    "bnb": "bnb",
    "bsc": "bnb",
    "celestia": "celestia",
    "ethereum": "ethereum",
    "fogo": "fogo",
    "megaeth": "megaeth",
    "monad": "monad",
    "optimism": "optimism",
    "polygon": "polygon",
    "solana": "solana",
    "tron": "tron",
    "unichain": "unichain",
    "world-chain": "worldchain",
    "zksync-era": "zksync",
    "zora": "zora",
}

PRIORITY_PROTOCOL_TICKERS = {"UNI", "AAVE", "HYPE", "LDO", "CRV", "AERO", "GMX", "DYDX", "JUP", "CAKE"}


class CoinGeckoClient:
    """Small CoinGecko Pro/public client used only by the screener adapter."""

    def __init__(
        self,
        api_key: str = COINGECKO_API_KEY,
        base_url: str = COINGECKO_BASE_URL,
        cache_dir: Path = SCREENER_CACHE_DIR,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"x-cg-pro-api-key": api_key})

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        resp = self.session.get(f"{self.base_url}/{endpoint.lstrip('/')}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def list_coins(self, use_cache: bool = True) -> list[dict]:
        if use_cache and CG_COINS_PATH.exists():
            with open(CG_COINS_PATH) as f:
                return json.load(f)
        data = self._get("/coins/list", params={"include_platform": "true"})
        with open(CG_COINS_PATH, "w") as f:
            json.dump(data, f)
        return data

    def get_markets(self, ids: Iterable[str], use_cache: bool = True) -> pd.DataFrame:
        ids = sorted({str(i) for i in ids if i})
        if not ids:
            return pd.DataFrame()
        cached = pd.DataFrame()
        if CG_MARKETS_PATH.exists():
            cached = pd.read_parquet(CG_MARKETS_PATH)
            if "id" in cached:
                cached = cached[cached["id"].isin(ids)].copy()
                if use_cache and set(cached["id"].astype(str)) >= set(ids):
                    return cached

        rows: list[dict] = []
        cached_ids = set(cached["id"].astype(str)) if not cached.empty and "id" in cached else set()
        fetch_ids = [coin_id for coin_id in ids if coin_id not in cached_ids] if use_cache else ids
        for i in range(0, len(fetch_ids), 250):
            batch = fetch_ids[i : i + 250]
            data = self._get(
                "/coins/markets",
                params={
                    "vs_currency": "usd",
                    "ids": ",".join(batch),
                    "per_page": 250,
                    "page": 1,
                    "sparkline": "false",
                    "price_change_percentage": "7d,30d",
                },
            )
            rows.extend(data)
            time.sleep(1.2 if self.api_key else 2.0)

        df = pd.DataFrame(rows)
        if not cached.empty:
            df = pd.concat([df, cached], ignore_index=True)
            df = df.drop_duplicates("id", keep="first")
        if not df.empty:
            df.to_parquet(CG_MARKETS_PATH)
        return df

    def get_market_chart(self, coin_id: str, days: int = 90, use_cache: bool = True) -> pd.DataFrame:
        cache_path = self.cache_dir / f"coingecko_market_chart_{coin_id}_{days}.parquet"
        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)
        data = self._get(
            f"/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
        )
        frames = {}
        for key, out_col in [
            ("prices", "price"),
            ("market_caps", "market_cap"),
            ("total_volumes", "volume_24h"),
        ]:
            values = data.get(key, [])
            if values:
                part = pd.DataFrame(values, columns=["ts", out_col])
                part["date"] = pd.to_datetime(part["ts"], unit="ms").dt.normalize()
                frames[out_col] = part.set_index("date")[out_col].groupby(level=0).last()
        if not frames:
            return pd.DataFrame()
        df = pd.DataFrame(frames).sort_index()
        df.to_parquet(cache_path)
        return df


def load_screener_snapshot() -> pd.DataFrame:
    """Load current cross-sectional screener snapshot from cache."""
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(f"Missing screener cache: {SNAPSHOT_PATH}")
    return pd.read_parquet(SNAPSHOT_PATH)


def load_project_timeseries(ticker: str) -> pd.DataFrame:
    """Load 30D/90D per-project time series from cache."""
    if not PROJECT_TS_PATH.exists():
        raise FileNotFoundError(f"Missing screener cache: {PROJECT_TS_PATH}")
    ts = pd.read_parquet(PROJECT_TS_PATH)
    if ticker:
        ts = ts[ts["ticker"].astype(str).str.upper() == str(ticker).upper()]
    if not ts.empty:
        cutoff = ts["date"].max() - pd.Timedelta(days=29)
        ts = ts[ts["date"] >= cutoff]
    return ts.sort_values("date").reset_index(drop=True)


def load_summary_timeseries(metric: str) -> pd.DataFrame:
    """Load executive-summary time series for one metric column."""
    if not SUMMARY_TS_PATH.exists():
        raise FileNotFoundError(f"Missing screener cache: {SUMMARY_TS_PATH}")
    ts = pd.read_parquet(SUMMARY_TS_PATH)
    if metric:
        ts = ts[ts["metric"] == metric]
    return ts.sort_values("date").reset_index(drop=True)


def load_factor_baskets() -> pd.DataFrame:
    """Load indexed top/bottom factor basket performance."""
    if not FACTOR_BASKETS_PATH.exists():
        raise FileNotFoundError(f"Missing screener cache: {FACTOR_BASKETS_PATH}")
    return pd.read_parquet(FACTOR_BASKETS_PATH)


def load_factor_scores() -> pd.DataFrame:
    """Load historical factor scores for lagged score analysis."""
    if not FACTOR_SCORES_PATH.exists():
        raise FileNotFoundError(f"Missing screener cache: {FACTOR_SCORES_PATH}")
    return pd.read_parquet(FACTOR_SCORES_PATH)


def refresh_screener_cache(force: bool = False, start: str = DEFAULT_START_DATE) -> dict[str, Path]:
    """Build all persisted screener outputs from live clients and cached calls."""
    required = [ASSET_MASTER_PATH, SNAPSHOT_PATH, PROJECT_TS_PATH, SUMMARY_TS_PATH, FACTOR_SCORES_PATH, FACTOR_BASKETS_PATH]
    if not force and all(path.exists() for path in required):
        return _cache_outputs()

    bn = BinanceClient()
    dl = DefiLlamaClient()
    cg = CoinGeckoClient()

    protocols, chains = _load_defillama_universe(dl, force=force)
    base_master = _build_defillama_asset_master(protocols, chains)
    base_master = _normalise_master_labels(base_master)

    coingecko_matches, cg_audit = _match_coingecko(cg, base_master, force=force)
    binance_matches, bn_audit = _match_binance(bn, base_master)
    blockworks_matches, bw_audit = _match_blockworks_optional(base_master, force=force)

    asset_master = base_master.copy()
    for matches, columns in [
        (coingecko_matches, ["coingecko_id"]),
        (binance_matches, ["binance_spot_symbol", "binance_futures_symbol"]),
        (blockworks_matches, ["blockworks_match_slug", "blockworks_id", "blockworks_url"]),
    ]:
        if matches.empty:
            continue
        updates = matches.set_index("blockworks_slug")
        for column in columns:
            mapped = asset_master["blockworks_slug"].map(updates[column])
            asset_master[column] = mapped.combine_first(asset_master[column])
    asset_master = _normalise_master_labels(asset_master)
    asset_master = _attach_match_audit(asset_master, [cg_audit, bn_audit, bw_audit])

    preliminary_candidates = _select_defillama_candidates(asset_master, pd.DataFrame())
    cg_markets = _fetch_coingecko_markets(cg, preliminary_candidates, force=force)
    asset_master = _select_defillama_candidates(asset_master, cg_markets)
    logger.info("Selected %s DefiLlama-left candidates from %s rows", len(asset_master), len(base_master))

    logger.info("Fetching DefiLlama metric panels")
    defillama_panels = _fetch_defillama_metric_panels(dl, asset_master, force=force)
    defillama_tvl = defillama_panels.pop("tvl", pd.DataFrame())
    logger.info("Fetching Binance price/OI panels")
    binance_panels = _fetch_binance_panels(bn, asset_master, start=start)
    logger.info("Fetching CoinGecko fallback chart panels")
    coingecko_panels = _fetch_coingecko_chart_panels(cg, asset_master, force=force)

    market_snapshot = _build_market_snapshot(asset_master, cg_markets, defillama_panels, binance_panels)
    panels = _assemble_model_panels(asset_master, market_snapshot, defillama_panels, binance_panels, coingecko_panels, defillama_tvl)
    quality = _compute_quality_flags(panels, stale_days=DEFAULT_STALE_DAYS)
    factor_panels = _compute_factor_panels(panels, quality)

    project_ts = _build_project_timeseries(asset_master, market_snapshot, panels, factor_panels, quality)
    snapshot = _build_snapshot(project_ts, asset_master, market_snapshot, quality)
    asset_master = _update_asset_master_coverage(asset_master, snapshot)
    summary_ts = _build_summary_timeseries(project_ts)
    factor_scores = _build_factor_scores_frame(project_ts)
    factor_baskets = _build_factor_baskets(project_ts, snapshot)

    asset_master.to_parquet(ASSET_MASTER_PATH)
    snapshot.to_parquet(SNAPSHOT_PATH)
    project_ts.to_parquet(PROJECT_TS_PATH)
    summary_ts.to_parquet(SUMMARY_TS_PATH)
    factor_scores.to_parquet(FACTOR_SCORES_PATH)
    factor_baskets.to_parquet(FACTOR_BASKETS_PATH)

    return _cache_outputs()


def _source_use_cache(force: bool) -> bool:
    """Reuse raw API caches by default even when rebuilding screener parquet."""
    return not (force and FORCE_RAW_SOURCE_REFRESH)


def _cache_outputs() -> dict[str, Path]:
    return {
        "asset_master": ASSET_MASTER_PATH,
        "snapshot": SNAPSHOT_PATH,
        "project_timeseries": PROJECT_TS_PATH,
        "summary_timeseries": SUMMARY_TS_PATH,
        "factor_scores": FACTOR_SCORES_PATH,
        "factor_baskets": FACTOR_BASKETS_PATH,
    }


def _load_defillama_universe(dl: DefiLlamaClient, force: bool = False) -> tuple[list[dict], list[dict]]:
    use_cache = _source_use_cache(force)
    try:
        protocols = dl.list_protocols(use_cache=use_cache)
    except Exception as e:
        logger.warning("DefiLlama protocol list unavailable: %s", e)
        try:
            protocols = dl.list_protocols(use_cache=True)
        except Exception:
            protocols = []
    try:
        chains = dl.list_chains(use_cache=use_cache)
    except Exception as e:
        logger.warning("DefiLlama chain list unavailable: %s", e)
        try:
            chains = dl.list_chains(use_cache=True)
        except Exception:
            chains = []
    return protocols, chains


def _build_defillama_asset_master(protocols: list[dict], chains: list[dict]) -> pd.DataFrame:
    master = build_defillama_asset_frame(protocols, chains)
    if master.empty:
        return master

    for column in [
        "blockworks_price",
        "blockworks_market_cap",
        "blockworks_market_cap_rank",
        "blockworks_price_7d_pct",
        "blockworks_price_30d_pct",
    ]:
        if column not in master:
            master[column] = np.nan

    if "defillama_mcap" in master:
        master["blockworks_market_cap"] = pd.to_numeric(master["defillama_mcap"], errors="coerce")
    if "metric_project_slug" not in master:
        master["metric_project_slug"] = master["defillama_slug"]
    return master


def _select_defillama_candidates(master: pd.DataFrame, cg_markets: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()
    df = df[df["ticker"].notna() & df["blockworks_slug"].notna()].copy()

    if not cg_markets.empty and "id" in cg_markets:
        market_cols = ["id", "market_cap", "fully_diluted_valuation", "total_volume"]
        market_cols = [c for c in market_cols if c in cg_markets]
        markets = cg_markets[market_cols].rename(
            columns={
                "id": "coingecko_id",
                "market_cap": "_cg_market_cap",
                "fully_diluted_valuation": "_cg_fdv",
                "total_volume": "_cg_volume",
            }
        )
        df = df.merge(markets, on="coingecko_id", how="left")
    for column in ["_cg_market_cap", "_cg_fdv", "_cg_volume", "defillama_current_tvl", "defillama_mcap"]:
        if column not in df:
            df[column] = np.nan

    excluded = {normalise_text(x) for x in EXCLUDE_CATEGORIES}
    category_norm = df["category"].map(normalise_text)
    name_norm = df["name"].map(normalise_text)
    ticker_norm = df["ticker"].map(normalise_symbol)
    stableish = ticker_norm.isin({"usdt", "usdc", "dai", "fdusd", "tusd", "usde", "usds", "busd", "frax", "gho"})
    wrappedish = name_norm.str.contains("wrapped|pegged", na=False)
    excluded_category = category_norm.isin(excluded)

    mcap = pd.to_numeric(df["_cg_market_cap"], errors="coerce").combine_first(pd.to_numeric(df["defillama_mcap"], errors="coerce"))
    tvl = pd.to_numeric(df["defillama_current_tvl"], errors="coerce")
    volume = pd.to_numeric(df["_cg_volume"], errors="coerce")
    is_chain = df["asset_type"].eq("chain")
    liquid_market = mcap.fillna(0) >= DEFAULT_MIN_MARKET_CAP
    liquid_tvl = tvl.fillna(0) >= max(DEFAULT_MIN_MARKET_CAP, 25_000_000)
    meaningful_volume = volume.fillna(0) >= 1_000_000

    df = df[(liquid_market | liquid_tvl | meaningful_volume | is_chain) & ~stableish & ~wrappedish & ~excluded_category].copy()
    df["_sort_mcap"] = mcap.reindex(df.index).fillna(0)
    df["_sort_tvl"] = tvl.reindex(df.index).fillna(0)
    df["_sort_volume"] = volume.reindex(df.index).fillna(0)
    df["_is_chain"] = df["asset_type"].eq("chain")

    # Use CoinGecko market cap for market-data fallbacks while preserving the
    # old column name expected by downstream sort code.
    df["blockworks_market_cap"] = df["_sort_mcap"].replace(0, np.nan)
    df["blockworks_market_cap_rank"] = df["_sort_mcap"].rank(ascending=False, method="min")

    df = df.sort_values(
        ["_is_chain", "_sort_mcap", "_sort_tvl", "_sort_volume", "ticker"],
        ascending=[False, False, False, False, True],
    )
    df = df.drop_duplicates(["asset_type", "ticker", "coingecko_id"], keep="first")
    df = _dedupe_defillama_tickers(df)

    if DEFAULT_MAX_CANDIDATES > 0 and len(df) > DEFAULT_MAX_CANDIDATES:
        chains = df[df["_is_chain"]].head(50)
        others = df[~df["_is_chain"]].head(max(DEFAULT_MAX_CANDIDATES - len(chains), 0))
        df = pd.concat([chains, others], ignore_index=True)

    return df.drop(columns=[c for c in df.columns if c.startswith("_sort_") or c == "_is_chain"], errors="ignore").reset_index(drop=True)


def _dedupe_defillama_tickers(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one dashboard row per ticker so project detail and charts stay unambiguous."""
    if df.empty or "ticker" not in df:
        return df
    rows = df.copy()
    category = rows["category"].map(normalise_text)
    is_chain = rows["asset_type"].eq("chain")
    is_defi_protocol = ~is_chain & category.isin({"dex", "dexs", "lending", "liquidstaking", "yield", "yieldrestaking", "derivatives", "perps"})
    has_protocol_metrics = is_defi_protocol & (
        pd.to_numeric(rows.get("_sort_tvl"), errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(rows.get("_sort_volume"), errors="coerce").fillna(0).gt(0)
    )
    rows["_ticker_dedupe_priority"] = np.select(
        [has_protocol_metrics, is_chain, is_defi_protocol],
        [0, 1, 2],
        default=3,
    )
    rows = rows.sort_values(
        ["ticker", "_ticker_dedupe_priority", "_sort_mcap", "_sort_tvl", "_sort_volume"],
        ascending=[True, True, False, False, False],
        na_position="last",
    )
    return rows.drop_duplicates("ticker", keep="first").drop(columns=["_ticker_dedupe_priority"], errors="ignore")


def _match_blockworks_optional(master: pd.DataFrame, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RAW_ASSETS_PATH.exists():
        return pd.DataFrame(columns=["blockworks_slug", "blockworks_match_slug", "blockworks_id", "blockworks_url"]), pd.DataFrame()
    try:
        with open(RAW_ASSETS_PATH) as f:
            assets = json.load(f)
    except Exception as e:
        logger.debug("Cached Blockworks asset list unavailable: %s", e)
        return pd.DataFrame(columns=["blockworks_slug", "blockworks_match_slug", "blockworks_id", "blockworks_url"]), pd.DataFrame()

    by_slug = {str(a.get("slug", "")).lower(): a for a in assets if a.get("slug")}
    by_name_symbol = _unique_lookup(assets, lambda a: (normalise_text(a.get("title") or a.get("name")), normalise_symbol(a.get("code"))))
    by_name = _unique_lookup(assets, lambda a: normalise_text(a.get("title") or a.get("name")))

    rows = []
    audit = []
    for _, row in master.iterrows():
        key = str(row["blockworks_slug"])
        chain_name = _text(row.get("defillama_chain")) or _text(row.get("name"))
        candidates = [_text(row.get("defillama_slug")).lower(), key.lower()]
        if str(row.get("asset_type")) == "chain":
            candidates.append(DEFILLAMA_CHAIN_SLUG_TO_BW.get(slugify_key(chain_name), ""))
        match = next((by_slug[c] for c in candidates if c in by_slug), None)
        method = "blockworks_unmatched"
        confidence = 0.0
        if match is not None:
            method, confidence = "blockworks_exact_slug", 0.9
        if match is None:
            lookup_name = _text(row.get("name")).replace(" (Combined)", "")
            match = by_name_symbol.get((normalise_text(lookup_name), normalise_symbol(row.get("ticker"))))
            if match:
                method, confidence = "blockworks_exact_name_symbol", 0.86
        if match is None:
            lookup_name = _text(row.get("name")).replace(" (Combined)", "")
            match = by_name.get(normalise_text(lookup_name))
            if match:
                method, confidence = "blockworks_exact_name", 0.76
        if match:
            slug = match.get("slug")
            rows.append(
                {
                    "blockworks_slug": key,
                    "blockworks_match_slug": slug,
                    "blockworks_id": match.get("id"),
                    "blockworks_url": f"https://app.blockworksresearch.com/assets/{slug}",
                }
            )
        audit.append({"blockworks_slug": key, "source": "blockworks", "method": method, "confidence": confidence})
    return pd.DataFrame(rows), pd.DataFrame(audit)


def _load_blockworks_assets(bw: BlockworksClient, force: bool = False) -> list[dict]:
    if not force and RAW_ASSETS_PATH.exists():
        with open(RAW_ASSETS_PATH) as f:
            return json.load(f)
    assets = bw.list_all_assets(limit=100, expand=["price", "market_cap"])
    with open(RAW_ASSETS_PATH, "w") as f:
        json.dump(assets, f)
    return assets


def _nested(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _num(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _asset_metadata_frame(assets: list[dict]) -> pd.DataFrame:
    rows = []
    for row in assets:
        slug = row.get("slug")
        if not slug:
            continue
        rows.append(
            {
                "blockworks_slug": slug,
                "metric_project_slug": _nested(row, "project", "slug") or slug,
                "asset_class": row.get("asset_class"),
                "is_preview": bool(row.get("is_preview", False)),
                "blockworks_price": _num(_nested(row, "price", "usd")),
                "blockworks_market_cap": _num(_nested(row, "market_cap", "usd")),
                "blockworks_market_cap_rank": _num(_nested(row, "market_cap", "rank")),
                "blockworks_price_7d_pct": _num(_nested(row, "market_cap", "percent_change_usd_7d")),
                "blockworks_price_30d_pct": _num(_nested(row, "market_cap", "percent_change_usd_30d")),
            }
        )
    return pd.DataFrame(rows)


def _normalise_master_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "category" in df:
        df["category"] = df["category"].map(_normalise_category)
    if "sector" in df:
        df["sector"] = [
            _normalise_sector(sector, category)
            for sector, category in zip(df["sector"], df.get("category", pd.Series(index=df.index)))
        ]
    return df


def _normalise_category(value: Any) -> str:
    if pd.isna(value):
        return "Uncategorised"
    text = str(value).strip()
    lower = text.lower()
    if lower in {"l1", "l2", "l1/l2", "rollup"}:
        return "L1 / L2"
    if lower in {"derivatives", "perpetuals", "perps"}:
        return "Perps"
    if lower == "decentralized exchanges":
        return "DEX"
    if lower in {"liquid staking", "restaking"}:
        return "Yield / Restaking"
    return text


def _normalise_sector(value: Any, category: Any) -> str:
    category_text = "" if pd.isna(category) else str(category).strip().lower()
    if category_text in DEFI_CATEGORIES:
        return "DeFi"
    if category_text == "l1 / l2":
        return "Infrastructure"
    if pd.isna(value):
        return "Other"
    text = str(value).strip()
    if text.lower() == "application" and category_text in DEFI_CATEGORIES:
        return "DeFi"
    return text


def _select_screener_candidates(master: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()
    if "asset_class" in df:
        df = df[df["asset_class"].fillna("token").astype(str).str.lower().eq("token")]
    if "is_preview" in df:
        df = df[~df["is_preview"].fillna(False).astype(bool)]
    df = df[df["ticker"].notna() & df["blockworks_slug"].notna()]

    excluded = {normalise_text(x) for x in EXCLUDE_CATEGORIES}
    category_norm = df["category"].map(normalise_text)
    name_norm = df["name"].map(normalise_text)
    ticker_norm = df["ticker"].map(normalise_symbol)
    stableish = ticker_norm.isin({"usdt", "usdc", "dai", "fdusd", "tusd", "usde"})
    wrappedish = name_norm.str.contains("wrapped", na=False)
    excluded_category = category_norm.isin(excluded)

    liquid = df["blockworks_market_cap"].fillna(0) >= DEFAULT_MIN_MARKET_CAP
    ranked = df["blockworks_market_cap_rank"].fillna(np.inf) <= DEFAULT_MAX_CANDIDATES
    known_chain = df["blockworks_slug"].isin(BW_CHAIN_SLUGS)
    df = df[(liquid | ranked | known_chain) & ~stableish & ~wrappedish & ~excluded_category].copy()

    df["_sort_mcap"] = df["blockworks_market_cap"].fillna(0)
    df["_known_chain"] = df["blockworks_slug"].isin(BW_CHAIN_SLUGS)
    df = df.sort_values(["_known_chain", "_sort_mcap", "ticker"], ascending=[False, False, True])
    df = df.drop_duplicates("ticker", keep="first")

    if DEFAULT_MAX_CANDIDATES > 0 and len(df) > DEFAULT_MAX_CANDIDATES:
        chains = df[df["_known_chain"]]
        others = df[~df["_known_chain"]].head(max(DEFAULT_MAX_CANDIDATES - len(chains), 0))
        df = pd.concat([chains, others], ignore_index=True)

    return df.drop(columns=["_sort_mcap", "_known_chain"], errors="ignore").reset_index(drop=True)


def _match_coingecko(
    cg: CoinGeckoClient,
    master: pd.DataFrame,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    use_cache = _source_use_cache(force)
    try:
        coins = cg.list_coins(use_cache=use_cache)
    except Exception as e:
        logger.warning("CoinGecko coin list unavailable: %s", e)
        try:
            coins = cg.list_coins(use_cache=True)
        except Exception:
            return pd.DataFrame(columns=["blockworks_slug", "coingecko_id"]), pd.DataFrame()

    by_id = {str(c.get("id", "")).lower(): c for c in coins}
    by_name_symbol = _unique_lookup(coins, lambda c: (normalise_text(c.get("name")), normalise_symbol(c.get("symbol"))))
    by_name = _unique_lookup(coins, lambda c: normalise_text(c.get("name")))

    matches = []
    audit = []
    for _, row in master.iterrows():
        slug = str(row["blockworks_slug"])
        ticker = str(row["ticker"])
        name = str(row["name"])
        defillama_slug = _text(row.get("defillama_slug"))
        existing = row.get("coingecko_id", pd.NA)
        match = None
        method = "unmatched"
        confidence = 0.0

        override = COINGECKO_ID_OVERRIDES.get(defillama_slug.lower()) or COINGECKO_ID_OVERRIDES.get(slug.lower())
        if override and override.lower() in by_id:
            match = by_id[override.lower()]
            method, confidence = "coingecko_override", 0.98
        if match is None and pd.notna(existing) and str(existing).lower() in by_id:
            match = by_id[str(existing).lower()]
            method, confidence = "coingecko_defillama_gecko_id", 0.97
        if match is None and defillama_slug.lower() in by_id:
            match = by_id[defillama_slug.lower()]
            method, confidence = "coingecko_exact_defillama_slug", 0.9
        if match is None and slug.lower() in by_id:
            match = by_id[slug.lower()]
            method, confidence = "coingecko_exact_id", 0.9
        if match is None:
            key = (normalise_text(name.replace(" (Combined)", "")), normalise_symbol(ticker))
            match = by_name_symbol.get(key)
            if match:
                method, confidence = "coingecko_exact_name_symbol", 0.92
        if match is None:
            match = by_name.get(normalise_text(name.replace(" (Combined)", "")))
            if match:
                scorer_method, confidence = score_name_symbol_match(name, ticker, match.get("name"), match.get("symbol"))
                method = f"coingecko_{scorer_method}"

        if match and confidence >= 0.8:
            matches.append({"blockworks_slug": slug, "coingecko_id": match.get("id")})
        audit.append(
            {
                "blockworks_slug": slug,
                "source": "coingecko",
                "method": method,
                "confidence": confidence,
            }
        )

    return pd.DataFrame(matches), pd.DataFrame(audit)


def _unique_lookup(items: list[dict], key_func) -> dict[Any, dict]:
    grouped: dict[Any, list[dict]] = {}
    for item in items:
        key = key_func(item)
        if not key or key == ("", ""):
            continue
        grouped.setdefault(key, []).append(item)
    return {key: rows[0] for key, rows in grouped.items() if len(rows) == 1}


def _match_defillama(
    dl: DefiLlamaClient,
    master: pd.DataFrame,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    use_cache = _source_use_cache(force)
    try:
        protocols = dl.list_protocols(use_cache=use_cache)
    except Exception as e:
        logger.warning("DefiLlama protocol list unavailable: %s", e)
        protocols = []

    by_slug = {str(p.get("slug", "")).lower(): p for p in protocols}
    by_name_symbol = _unique_lookup(protocols, lambda p: (normalise_text(p.get("name")), normalise_symbol(p.get("symbol"))))
    by_name = _unique_lookup(protocols, lambda p: normalise_text(p.get("name")))

    matches = []
    audit = []
    for _, row in master.iterrows():
        slug = str(row["blockworks_slug"])
        ticker = str(row["ticker"])
        name = str(row["name"])
        match = None
        method = "unmatched"
        confidence = 0.0

        if str(row.get("asset_type", "")).lower() == "chain":
            matches.append({"blockworks_slug": slug, "defillama_slug": name})
            audit.append(
                {"blockworks_slug": slug, "source": "defillama", "method": "defillama_chain_name", "confidence": 0.86}
            )
            continue

        if slug.lower() in by_slug:
            match = by_slug[slug.lower()]
            method, confidence = "defillama_exact_slug", 0.88
        if match is None:
            key = (normalise_text(name), normalise_symbol(ticker))
            match = by_name_symbol.get(key)
            if match:
                method, confidence = "defillama_exact_name_symbol", 0.9
        if match is None:
            match = by_name.get(normalise_text(name))
            if match:
                method, confidence = "defillama_exact_name", 0.82

        if match and confidence >= 0.8:
            matches.append({"blockworks_slug": slug, "defillama_slug": match.get("slug")})
        audit.append(
            {
                "blockworks_slug": slug,
                "source": "defillama",
                "method": method,
                "confidence": confidence,
            }
        )

    return pd.DataFrame(matches), pd.DataFrame(audit)


def _match_binance(bn: BinanceClient, master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        spot_symbols = bn.get_exchange_symbols("USDT")
    except Exception as e:
        logger.warning("Binance spot symbol list unavailable: %s", e)
        spot_symbols = set()
    try:
        futures_symbols = bn.get_futures_symbols("USDT")
    except Exception as e:
        logger.warning("Binance futures symbol list unavailable: %s", e)
        futures_symbols = set()

    rows = []
    audit = []
    for _, row in master.iterrows():
        slug = str(row["blockworks_slug"])
        ticker = str(row["ticker"]).upper()
        chain_name = _text(row.get("defillama_chain")) or _text(row.get("name"))
        lookup_slugs = [
            slug,
            _text(row.get("defillama_slug")),
            slugify_key(chain_name),
        ]
        if str(row.get("asset_type")) == "chain":
            lookup_slugs.extend(
                [
                    DEFILLAMA_CHAIN_SLUG_TO_BW.get(slugify_key(chain_name), ""),
                    slug.replace("chain:", ""),
                ]
            )
        configured = next((SLUG_TO_BINANCE.get(s) for s in lookup_slugs if s in SLUG_TO_BINANCE), None)
        exact = f"{ticker}USDT"
        spot = configured if configured in spot_symbols else exact if exact in spot_symbols else pd.NA
        futures = configured if configured in futures_symbols else exact if exact in futures_symbols else pd.NA
        method = "binance_unmatched"
        confidence = 0.0
        if pd.notna(spot) or pd.notna(futures):
            method = "binance_config_slug" if configured and (configured == spot or configured == futures) else "binance_exact_symbol"
            confidence = 0.95
        rows.append({"blockworks_slug": slug, "binance_spot_symbol": spot, "binance_futures_symbol": futures})
        audit.append({"blockworks_slug": slug, "source": "binance", "method": method, "confidence": confidence})
    return pd.DataFrame(rows), pd.DataFrame(audit)


def _attach_match_audit(master: pd.DataFrame, audits: list[pd.DataFrame]) -> pd.DataFrame:
    audit_frames = [df for df in audits if df is not None and not df.empty]
    audit = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    if audit.empty:
        master["match_method"] = "blockworks_only"
        master["match_confidence"] = 1.0
        return master
    grouped = (
        audit.sort_values(["blockworks_slug", "confidence"], ascending=[True, False])
        .groupby("blockworks_slug", as_index=False)
        .first()
        .rename(columns={"method": "match_method", "confidence": "match_confidence"})
    )
    master = master.drop(columns=["match_method", "match_confidence"], errors="ignore").merge(
        grouped[["blockworks_slug", "match_method", "match_confidence"]],
        on="blockworks_slug",
        how="left",
    )
    master["match_method"] = master["match_method"].fillna("blockworks_only")
    master["match_confidence"] = master["match_confidence"].fillna(1.0)
    return master


def _fetch_coingecko_markets(cg: CoinGeckoClient, master: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    ids = master["coingecko_id"].dropna().astype(str).unique().tolist()
    try:
        return cg.get_markets(ids, use_cache=_source_use_cache(force))
    except Exception as e:
        logger.warning("CoinGecko markets unavailable: %s", e)
        if CG_MARKETS_PATH.exists():
            try:
                cached = pd.read_parquet(CG_MARKETS_PATH)
                if "id" in cached:
                    return cached[cached["id"].isin(ids)].copy()
            except Exception:
                pass
        return pd.DataFrame()


def _fetch_blockworks_panels(
    bw: BlockworksClient,
    master: pd.DataFrame,
    start: str,
) -> dict[str, pd.DataFrame]:
    supported_chain = master["metric_project_slug"].isin(BW_CHAIN_SLUGS)
    chain_rows = master[master["asset_type"].eq("chain") & supported_chain]
    project_rows = master[~master["asset_type"].eq("chain")]
    metric_to_slugs = master.groupby("metric_project_slug")["blockworks_slug"].apply(list).to_dict()

    chain_metrics = {
        "revenue": BW_CHAIN_METRICS["revenue"],
        "trading_fees": BW_CHAIN_METRICS["trading_fees"],
        "active_addresses": BW_CHAIN_METRICS["active_addresses"],
        "issuance": BW_CHAIN_METRICS["issuance"],
        "burn": BW_CHAIN_METRICS["burn"],
        "stablecoin_supply": BW_CHAIN_METRICS["stablecoin_supply"],
        "dex_volume": BW_CHAIN_METRICS["dex_volume"],
        "lending_deposits": BW_CHAIN_METRICS["lending_deposits"],
        "lending_tvl": BW_CHAIN_METRICS["lending_tvl"],
    }
    project_metrics = {
        "revenue": BW_PROJECT_METRICS["revenue"],
        "dex_revenue": BW_PROJECT_METRICS["dex_revenue"],
        "trading_fees": BW_PROJECT_METRICS["dex_fees"],
        "dex_volume": BW_PROJECT_METRICS["dex_volume"],
    }

    panels: dict[str, pd.DataFrame] = {}
    chain_slugs = chain_rows["metric_project_slug"].dropna().astype(str).unique().tolist()
    project_slugs = project_rows["metric_project_slug"].dropna().astype(str).unique().tolist()
    chain_available = _discover_metric_availability(bw, chain_slugs, list(chain_metrics.values()))
    project_available = _discover_metric_availability(bw, project_slugs, list(project_metrics.values()))

    for key, metric in chain_metrics.items():
        panels[key] = _fetch_one_blockworks_metric(bw, metric, chain_available.get(metric, []), metric_to_slugs, start)

    for key, metric in project_metrics.items():
        project_panel = _fetch_one_blockworks_metric(bw, metric, project_available.get(metric, []), metric_to_slugs, start)
        if key == "dex_revenue":
            panels["revenue"] = _combine_panels(panels.get("revenue"), project_panel)
        else:
            panels[key] = _combine_panels(panels.get(key), project_panel)

    return {key: _daily_panel(panel) for key, panel in panels.items()}


def _discover_metric_availability(
    bw: BlockworksClient,
    metric_slugs: list[str],
    metric_ids: list[str],
) -> dict[str, list[str]]:
    """Map Blockworks metric identifiers to project slugs that advertise them."""
    wanted = set(metric_ids)
    available = {metric_id: [] for metric_id in wanted}
    for slug in metric_slugs:
        try:
            catalog = bw.list_project_metrics(slug)
        except Exception as e:
            logger.debug("Metric catalog unavailable for %s: %s", slug, e)
            continue
        identifiers = {str(entry.get("identifier")) for entry in catalog if entry.get("identifier")}
        for metric_id in wanted & identifiers:
            available[metric_id].append(slug)
    return available


def _fetch_one_blockworks_metric(
    bw: BlockworksClient,
    metric: str,
    metric_slugs: list[str],
    metric_to_slugs: dict[str, list[str]],
    start: str,
) -> pd.DataFrame:
    if not metric_slugs:
        return pd.DataFrame()
    try:
        raw = bw.get_bulk_metric(metric, metric_slugs, start=start, batch_size=10)
    except Exception as e:
        logger.warning("Blockworks metric %s unavailable: %s", metric, e)
        return pd.DataFrame()
    return _expand_metric_slug_columns(raw, metric_to_slugs)


def _expand_metric_slug_columns(df: pd.DataFrame, metric_to_slugs: dict[str, list[str]]) -> pd.DataFrame:
    if df.empty:
        return df
    frames = {}
    for metric_slug in df.columns:
        for blockworks_slug in metric_to_slugs.get(metric_slug, [metric_slug]):
            frames[blockworks_slug] = df[metric_slug]
    return pd.DataFrame(frames).sort_index()


def _fetch_binance_panels(
    bn: BinanceClient,
    master: pd.DataFrame,
    start: str,
) -> dict[str, pd.DataFrame]:
    symbol_to_slug = (
        master.dropna(subset=["binance_spot_symbol"])
        .drop_duplicates("binance_spot_symbol")
        .set_index("binance_spot_symbol")["blockworks_slug"]
        .astype(str)
        .to_dict()
    )
    ohlcv = bn.get_multiple_daily_ohlcv(list(symbol_to_slug), start=start) if symbol_to_slug else {}
    close = {}
    quote_volume = {}
    for symbol, df in ohlcv.items():
        slug = symbol_to_slug.get(symbol)
        if slug and not df.empty:
            close[slug] = df["close"]
            quote_volume[slug] = df["quote_volume"]

    futures_to_slug = (
        master.dropna(subset=["binance_futures_symbol"])
        .assign(
            _sort_liquidity=lambda x: pd.to_numeric(x.get("blockworks_market_cap"), errors="coerce")
            .combine_first(pd.to_numeric(x.get("defillama_current_tvl"), errors="coerce"))
        )
        .sort_values("_sort_liquidity", ascending=False, na_position="last")
        .head(MAX_BINANCE_OI_FETCHES)
        .drop_duplicates("binance_futures_symbol")
        .set_index("binance_futures_symbol")["blockworks_slug"]
        .astype(str)
        .to_dict()
    )
    futures_ohlcv = bn.get_multiple_futures_daily_ohlcv(list(futures_to_slug), start=start) if futures_to_slug else {}
    for symbol, df in futures_ohlcv.items():
        slug = futures_to_slug.get(symbol)
        if not slug or df.empty:
            continue
        if slug not in close or _series_looks_flat(close[slug]):
            close[slug] = df["close"]
        if slug not in quote_volume or quote_volume[slug].dropna().tail(30).sum() <= 0:
            quote_volume[slug] = df["quote_volume"]

    oi_raw = bn.get_multiple_open_interest_history(list(futures_to_slug), start=None) if futures_to_slug else pd.DataFrame()
    if not oi_raw.empty:
        oi_raw = oi_raw.rename(columns=futures_to_slug)

    return {
        "price": _daily_panel(pd.DataFrame(close)),
        "volume_24h": _daily_panel(pd.DataFrame(quote_volume)),
        "open_interest": _daily_panel(oi_raw),
    }


def _series_looks_flat(series: pd.Series, window: int = 30) -> bool:
    clean = pd.to_numeric(series, errors="coerce").dropna().tail(window)
    if len(clean) < 7:
        return False
    return clean.nunique() <= 2 or clean.pct_change().abs().fillna(0).sum() < 1e-8


def _fetch_coingecko_chart_panels(
    cg: CoinGeckoClient,
    master: pd.DataFrame,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily CoinGecko market charts for liquid tokens lacking Binance spot.

    This is primarily the fallback for assets such as HYPE that have a current
    market-data match and a futures venue, but no spot OHLCV on Binance.
    """
    rows = master[master["coingecko_id"].notna()].copy()
    rows = rows[rows["binance_spot_symbol"].isna() | rows["asset_type"].eq("chain")]
    if "blockworks_market_cap" not in rows:
        rows["blockworks_market_cap"] = 0.0
    rows["_sort_market"] = pd.to_numeric(rows["blockworks_market_cap"], errors="coerce").combine_first(
        pd.to_numeric(rows.get("defillama_current_tvl"), errors="coerce")
    )
    rows["_chart_priority"] = (
        rows["binance_spot_symbol"].isna().astype(int) * 2
        + rows["binance_futures_symbol"].notna().astype(int)
    )
    rows = (
        rows.sort_values(["_chart_priority", "_sort_market"], ascending=[False, False], na_position="last")
        .drop_duplicates("coingecko_id")
        .head(MAX_COINGECKO_CHART_FETCHES)
    )
    frames: dict[str, dict[str, pd.Series]] = {"price": {}, "market_cap": {}, "volume_24h": {}}
    use_cache = _source_use_cache(force)
    for _, row in rows.iterrows():
        slug = str(row["blockworks_slug"])
        coin_id = str(row["coingecko_id"])
        try:
            chart = cg.get_market_chart(coin_id, days=120, use_cache=use_cache)
        except Exception as e:
            logger.debug("CoinGecko chart unavailable for %s/%s: %s", slug, coin_id, e)
            continue
        if chart.empty:
            continue
        for column in frames:
            if column in chart:
                frames[column][slug] = chart[column].sort_index().groupby(level=0).last()
    return {column: _daily_panel(pd.DataFrame(series_map)) for column, series_map in frames.items()}


def _fetch_defillama_tvl_panel(
    dl: DefiLlamaClient,
    master: pd.DataFrame,
    force: bool = False,
) -> pd.DataFrame:
    use_cache = _source_use_cache(force)
    rows = master[master["defillama_slug"].notna()].copy()
    rows["_sort_liquidity"] = pd.to_numeric(rows.get("blockworks_market_cap"), errors="coerce").combine_first(
        pd.to_numeric(rows.get("defillama_current_tvl"), errors="coerce")
    ).fillna(0)
    priority = rows[rows["ticker"].astype(str).str.upper().isin(PRIORITY_PROTOCOL_TICKERS)]
    top = rows.sort_values("_sort_liquidity", ascending=False).head(MAX_DEFILLAMA_TVL_FETCHES)
    rows = pd.concat([priority, top], ignore_index=True).drop_duplicates("blockworks_slug")
    frames = {}
    for _, row in rows.iterrows():
        slug = str(row["blockworks_slug"])
        try:
            if str(row.get("asset_type", "")).lower() == "chain":
                series = dl.get_chain_tvl(str(row["defillama_slug"]), use_cache=use_cache)
            else:
                series = _fetch_defillama_protocol_series(
                    row,
                    lambda child_slug: dl.get_protocol_tvl(child_slug, use_cache=use_cache),
                )
            if len(series) > 0:
                frames[slug] = series
        except Exception as e:
            logger.debug("DefiLlama TVL unavailable for %s: %s", slug, e)
    return _daily_panel(pd.DataFrame(frames))


def _fetch_defillama_metric_panels(
    dl: DefiLlamaClient,
    master: pd.DataFrame,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    use_cache = _source_use_cache(force)
    tvl = _fetch_defillama_tvl_panel(dl, master, force=force)
    fees = _fetch_defillama_overview_metric_panel(
        dl,
        master,
        lambda: dl.get_fees_overview(data_type="dailyFees", use_cache=use_cache),
        "fees",
    )
    revenue = _fetch_defillama_overview_metric_panel(
        dl,
        master,
        lambda: dl.get_fees_overview(data_type="dailyRevenue", use_cache=use_cache),
        "revenue",
    )
    dex_volume = _fetch_defillama_overview_metric_panel(
        dl,
        master,
        lambda: dl.get_dex_overview(use_cache=use_cache),
        "dex_volume",
    )
    priority = master[master["ticker"].astype(str).str.upper().isin(PRIORITY_PROTOCOL_TICKERS)].copy()
    fees = _combine_panels(
        fees,
        _fetch_defillama_protocol_metric_panel(
            dl,
            _missing_metric_rows(priority, fees),
            lambda slug: dl.get_fees_series(slug, data_type="dailyFees", use_cache=use_cache),
            "fees_fallback",
        ),
    )
    revenue = _combine_panels(
        revenue,
        _fetch_defillama_protocol_metric_panel(
            dl,
            _missing_metric_rows(priority, revenue),
            lambda slug: dl.get_fees_series(slug, data_type="dailyRevenue", use_cache=use_cache),
            "revenue_fallback",
        ),
    )
    dex_volume = _combine_panels(
        dex_volume,
        _fetch_defillama_protocol_metric_panel(
            dl,
            _missing_metric_rows(priority, dex_volume),
            lambda slug: dl.get_dex_volume_series(slug, use_cache=use_cache),
            "dex_volume_fallback",
        ),
    )
    stablecoin_supply = _fetch_defillama_stablecoin_supply_panel(dl, master, force=force)
    open_interest = _fetch_defillama_open_interest_panel(dl, master, force=force)

    lending_like = master["category"].astype(str).str.lower().isin(
        {"lending", "yield / restaking", "liquid staking", "rwa lending", "l1 / l2"}
    )
    deposit_slugs = master.loc[lending_like, "blockworks_slug"].astype(str).tolist()
    lending_deposits = tvl.reindex(columns=deposit_slugs) if not tvl.empty else pd.DataFrame()

    return {
        "tvl": tvl,
        "revenue": revenue,
        "trading_fees": fees,
        "dex_volume": dex_volume,
        "stablecoin_supply": stablecoin_supply,
        "open_interest": open_interest,
        "lending_deposits": lending_deposits,
    }


def _missing_metric_rows(rows: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    if panel is None or panel.empty:
        return rows
    missing = [slug for slug in rows["blockworks_slug"].astype(str) if slug not in panel or panel[slug].dropna().empty]
    return rows[rows["blockworks_slug"].astype(str).isin(missing)].copy()


def _fetch_defillama_overview_metric_panel(
    dl: DefiLlamaClient,
    master: pd.DataFrame,
    overview_fetcher,
    metric_name: str,
) -> pd.DataFrame:
    try:
        data = overview_fetcher()
    except Exception as e:
        logger.debug("DefiLlama %s overview unavailable: %s", metric_name, e)
        return pd.DataFrame()
    breakdown = data.get("totalDataChartBreakdown", []) if isinstance(data, dict) else []
    if not breakdown:
        return pd.DataFrame()

    wanted_keys = set().union(*[_defillama_metric_match_keys(row) for _, row in master.iterrows()])
    source_panel = _breakdown_chart_to_panel(breakdown, wanted_keys=wanted_keys)
    if source_panel.empty:
        return pd.DataFrame()

    frames = {}
    for _, row in master[master["asset_type"].ne("chain")].iterrows():
        keys = _defillama_metric_match_keys(row)
        matched = [source_panel[key] for key in keys if key in source_panel]
        if matched:
            frames[str(row["blockworks_slug"])] = pd.concat(matched, axis=1).sum(axis=1, min_count=1)
    return _daily_panel(pd.DataFrame(frames))


def _breakdown_chart_to_panel(breakdown: list, wanted_keys: set[str] | None = None) -> pd.DataFrame:
    records = []
    for entry in breakdown:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
            continue
        day = pd.to_datetime(entry[0], unit="s")
        flat = _flatten_defillama_breakdown(entry[1])
        record: dict[str, float | pd.Timestamp] = {"date": day}
        for name, value in flat.items():
            key = normalise_text(name)
            if not key or (wanted_keys is not None and key not in wanted_keys):
                continue
            current = record.get(key, np.nan)
            record[key] = _num(value) if pd.isna(current) else _num(current) + _num(value)
        if len(record) > 1:
            records.append(record)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).set_index("date").sort_index()


def _defillama_metric_match_keys(row: pd.Series) -> set[str]:
    keys = {
        normalise_text(row.get("name")),
        normalise_text(_text(row.get("name")).replace(" (Combined)", "")),
        normalise_text(row.get("defillama_slug")),
    }
    keys.update(normalise_text(slug) for slug in _defillama_child_slugs(row))
    return {key for key in keys if key}


def _fetch_defillama_protocol_metric_panel(
    dl: DefiLlamaClient,
    master: pd.DataFrame,
    fetcher,
    metric_name: str,
) -> pd.DataFrame:
    rows = master[master["asset_type"].ne("chain") & master["defillama_slug"].notna()].copy()
    rows["_sort_liquidity"] = pd.to_numeric(rows.get("blockworks_market_cap"), errors="coerce").combine_first(
        pd.to_numeric(rows.get("defillama_current_tvl"), errors="coerce")
    ).fillna(0)
    rows = rows.sort_values("_sort_liquidity", ascending=False).head(MAX_DEFILLAMA_METRIC_FETCHES)
    frames = {}
    for _, row in rows.iterrows():
        key = str(row["blockworks_slug"])
        try:
            series = _fetch_defillama_protocol_series(row, fetcher, try_canonical=True)
            if not series.empty:
                frames[key] = series.rename(key)
        except Exception as e:
            logger.debug("DefiLlama %s unavailable for %s: %s", metric_name, key, e)
    return _daily_panel(pd.DataFrame(frames))


def _fetch_defillama_protocol_series(row: pd.Series, fetcher, try_canonical: bool = False) -> pd.Series:
    canonical = _text(row.get("defillama_slug"))
    child_slugs = _defillama_child_slugs(row)
    if try_canonical and canonical and canonical not in child_slugs:
        try:
            series = fetcher(canonical)
            if not series.empty:
                return series
        except Exception:
            pass
    frames = []
    for child_slug in child_slugs or [canonical]:
        if not child_slug:
            continue
        try:
            series = fetcher(child_slug)
            if not series.empty:
                frames.append(series.rename(child_slug))
        except Exception:
            continue
    if not frames:
        return pd.Series(dtype=float, name=canonical)
    return pd.concat(frames, axis=1).sum(axis=1, min_count=1).sort_index().rename(canonical)


def _defillama_child_slugs(row: pd.Series) -> list[str]:
    raw = row.get("defillama_child_slugs", pd.NA)
    if pd.notna(raw) and str(raw).strip():
        return [part for part in str(raw).split("|") if part]
    slug = row.get("defillama_slug", pd.NA)
    return [str(slug)] if pd.notna(slug) else []


def _fetch_defillama_stablecoin_supply_panel(
    dl: DefiLlamaClient,
    master: pd.DataFrame,
    force: bool = False,
) -> pd.DataFrame:
    use_cache = _source_use_cache(force)
    chains = master[master["asset_type"].eq("chain") & master["defillama_chain"].notna()].copy()
    chains["_sort_liquidity"] = pd.to_numeric(chains.get("blockworks_market_cap"), errors="coerce").combine_first(
        pd.to_numeric(chains.get("defillama_current_tvl"), errors="coerce")
    ).fillna(0)
    chains = chains.sort_values("_sort_liquidity", ascending=False).head(MAX_DEFILLAMA_STABLECOIN_FETCHES)
    frames = {}
    for _, row in chains.iterrows():
        key = str(row["blockworks_slug"])
        try:
            series = dl.get_stablecoin_chain_supply(str(row["defillama_chain"]), use_cache=use_cache)
            if not series.empty:
                frames[key] = series.rename(key)
        except Exception as e:
            logger.debug("DefiLlama stablecoin supply unavailable for %s: %s", key, e)
    return _daily_panel(pd.DataFrame(frames))


def _fetch_defillama_open_interest_panel(
    dl: DefiLlamaClient,
    master: pd.DataFrame,
    force: bool = False,
) -> pd.DataFrame:
    try:
        data = dl.get_open_interest_overview(use_cache=_source_use_cache(force))
    except Exception as e:
        logger.debug("DefiLlama open-interest overview unavailable: %s", e)
        return pd.DataFrame()
    breakdown = data.get("totalDataChartBreakdown", []) if isinstance(data, dict) else []
    if not breakdown:
        return pd.DataFrame()

    wanted_keys = set().union(*[_defillama_metric_match_keys(row) for _, row in master.iterrows()])
    source_panel = _breakdown_chart_to_panel(breakdown, wanted_keys=wanted_keys)
    if source_panel.empty:
        return pd.DataFrame()

    frames = {}
    for _, row in master.iterrows():
        keys = _defillama_metric_match_keys(row)
        matched = [source_panel[key] for key in keys if key in source_panel]
        if matched:
            frames[str(row["blockworks_slug"])] = pd.concat(matched, axis=1).sum(axis=1, min_count=1)
    return _daily_panel(pd.DataFrame(frames))


def _flatten_defillama_breakdown(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in payload.items():
        label = str(key)
        if isinstance(value, dict):
            flat.update(_flatten_defillama_breakdown(value, label))
        else:
            flat[label or prefix] = _num(value)
    return flat


def _build_market_snapshot(
    master: pd.DataFrame,
    cg_markets: pd.DataFrame,
    bw_panels: dict[str, pd.DataFrame],
    binance_panels: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    master = master.copy()
    for column in ["blockworks_price", "blockworks_market_cap", "blockworks_price_7d_pct", "blockworks_price_30d_pct", "defillama_mcap"]:
        if column not in master:
            master[column] = np.nan
    snapshot = master[[
        "blockworks_slug",
        "coingecko_id",
        "blockworks_price",
        "blockworks_market_cap",
        "blockworks_price_7d_pct",
        "blockworks_price_30d_pct",
        "defillama_mcap",
    ]].copy()
    if not cg_markets.empty:
        cg = cg_markets.rename(
            columns={
                "id": "coingecko_id",
                "current_price": "cg_price",
                "market_cap": "cg_market_cap",
                "fully_diluted_valuation": "cg_fdv",
                "total_volume": "cg_volume_24h",
                "price_change_percentage_7d_in_currency": "cg_price_7d_pct",
                "price_change_percentage_30d_in_currency": "cg_price_30d_pct",
            }
        )
        keep = [c for c in ["coingecko_id", "cg_price", "cg_market_cap", "cg_fdv", "cg_volume_24h", "cg_price_7d_pct", "cg_price_30d_pct"] if c in cg]
        snapshot = snapshot.merge(cg[keep], on="coingecko_id", how="left")

    latest_binance_price = _latest_row(binance_panels.get("price"))
    latest_binance_volume = _latest_row(binance_panels.get("volume_24h"))
    latest_bw_fdv = _latest_row(bw_panels.get("fdv"))
    snapshot = snapshot.merge(latest_binance_price.rename("binance_price"), left_on="blockworks_slug", right_index=True, how="left")
    snapshot = snapshot.merge(latest_binance_volume.rename("binance_volume_24h"), left_on="blockworks_slug", right_index=True, how="left")
    snapshot = snapshot.merge(latest_bw_fdv.rename("bw_fdv"), left_on="blockworks_slug", right_index=True, how="left")

    snapshot["price"] = _coalesce(snapshot, ["cg_price", "binance_price", "blockworks_price"])
    snapshot["market_cap"] = _coalesce(snapshot, ["cg_market_cap", "blockworks_market_cap", "defillama_mcap"])
    snapshot["fdv"] = _coalesce(snapshot, ["cg_fdv", "bw_fdv"])
    snapshot["volume_24h"] = _coalesce(snapshot, ["binance_volume_24h", "cg_volume_24h"])
    snapshot["price_7d_pct"] = _coalesce(snapshot, ["cg_price_7d_pct", "blockworks_price_7d_pct"])
    snapshot["price_30d_pct"] = _coalesce(snapshot, ["cg_price_30d_pct", "blockworks_price_30d_pct"])
    return snapshot.set_index("blockworks_slug")


def _assemble_model_panels(
    master: pd.DataFrame,
    market_snapshot: pd.DataFrame,
    bw_panels: dict[str, pd.DataFrame],
    binance_panels: dict[str, pd.DataFrame],
    coingecko_panels: dict[str, pd.DataFrame],
    defillama_tvl: pd.DataFrame,
) -> dict[str, pd.DataFrame | pd.Series]:
    index = _common_daily_index([*bw_panels.values(), *binance_panels.values(), *coingecko_panels.values(), defillama_tvl])
    slugs = master["blockworks_slug"].astype(str).tolist()
    static_mcap = _static_panel(market_snapshot["market_cap"], index, slugs)
    static_fdv = _static_panel(market_snapshot["fdv"], index, slugs)
    static_price = _static_panel(market_snapshot["price"], index, slugs)
    static_volume = _static_panel(market_snapshot["volume_24h"], index, slugs)

    bw_price = _align_panel(bw_panels.get("bw_price"), index, slugs)
    supply = _align_panel(bw_panels.get("supply"), index, slugs)
    dynamic_mcap = bw_price * supply if not bw_price.empty and not supply.empty else pd.DataFrame(index=index, columns=slugs)

    cg_price = _align_panel(coingecko_panels.get("price"), index, slugs)
    cg_volume = _align_panel(coingecko_panels.get("volume_24h"), index, slugs)
    cg_mcap = _align_panel(coingecko_panels.get("market_cap"), index, slugs)

    price = _align_panel(binance_panels.get("price"), index, slugs).combine_first(cg_price).combine_first(bw_price).combine_first(static_price)
    volume = _align_panel(binance_panels.get("volume_24h"), index, slugs).combine_first(cg_volume).combine_first(static_volume)
    fdv = _align_panel(bw_panels.get("fdv"), index, slugs).combine_first(static_fdv)
    mcap = dynamic_mcap.combine_first(cg_mcap).combine_first(static_mcap)
    blockworks_tvl = _align_panel(bw_panels.get("lending_tvl"), index, slugs)
    dl_tvl = _align_panel(defillama_tvl, index, slugs)
    tvl = dl_tvl.combine_first(blockworks_tvl)
    binance_oi = _align_panel(binance_panels.get("open_interest"), index, slugs)
    defillama_oi = _align_panel(bw_panels.get("open_interest"), index, slugs)
    open_interest = binance_oi.combine_first(defillama_oi)

    btc = price.get("bitcoin")
    if btc is None or btc.dropna().empty:
        btc = price.get("ethereum", pd.Series(index=index, dtype=float))

    panels: dict[str, pd.DataFrame | pd.Series] = {
        "price": price,
        "volume_24h": volume,
        "mcap": mcap,
        "fdv": fdv,
        "revenue": _align_panel(bw_panels.get("revenue"), index, slugs),
        "trading_fees": _align_panel(bw_panels.get("trading_fees"), index, slugs),
        "lending_deposits": _align_panel(bw_panels.get("lending_deposits"), index, slugs),
        "blockworks_tvl": blockworks_tvl,
        "defillama_tvl": dl_tvl,
        "tvl": tvl,
        "open_interest": open_interest,
        "binance_open_interest": binance_oi,
        "defillama_open_interest": defillama_oi,
        "active_addresses": _align_panel(bw_panels.get("active_addresses"), index, slugs),
        "issuance": _align_panel(bw_panels.get("issuance"), index, slugs),
        "burn": _align_panel(bw_panels.get("burn"), index, slugs),
        "stablecoin_supply": _align_panel(bw_panels.get("stablecoin_supply"), index, slugs),
        "dex_volume": _align_panel(bw_panels.get("dex_volume"), index, slugs),
        "btc_price": btc.reindex(index).ffill(),
    }
    return panels


def _compute_quality_flags(
    panels: dict[str, pd.DataFrame | pd.Series],
    stale_days: int,
) -> pd.DataFrame:
    revenue = panels["revenue"]
    fees = panels["trading_fees"]
    tvl = panels.get("defillama_tvl", pd.DataFrame())
    rows = []
    today = pd.Timestamp.now().normalize()
    slugs = revenue.columns if isinstance(revenue, pd.DataFrame) else []
    for slug in slugs:
        rev = revenue[slug].dropna()
        fee = fees[slug].dropna() if isinstance(fees, pd.DataFrame) and slug in fees else pd.Series(dtype=float)
        tvl_series = tvl[slug].dropna() if isinstance(tvl, pd.DataFrame) and slug in tvl else pd.Series(dtype=float)
        last_rev = rev.index.max() if len(rev) else pd.NaT
        last_fee = fee.index.max() if len(fee) else pd.NaT
        last_tvl = tvl_series.index.max() if len(tvl_series) else pd.NaT
        latest_metric_date = max([d for d in [last_rev, last_fee, last_tvl] if pd.notna(d)], default=pd.NaT)
        corr = np.nan
        aligned = pd.DataFrame({"revenue": revenue.get(slug), "fees": fees.get(slug)}).tail(120).dropna()
        if len(aligned) >= 30 and aligned["revenue"].std() > 0 and aligned["fees"].std() > 0:
            corr = float(aligned["revenue"].corr(aligned["fees"]))
        stale_metric = bool(
            pd.isna(latest_metric_date) or (today - pd.Timestamp(latest_metric_date).normalize()).days > stale_days
        )
        rows.append(
            {
                "blockworks_slug": slug,
                "blockworks_latest_metric_date": latest_metric_date,
                "blockworks_stale_metric": stale_metric,
                "defillama_latest_metric_date": latest_metric_date,
                "defillama_stale_metric": stale_metric,
                "missing_defillama_fees_revenue": bool(rev.empty and fee.empty),
                "missing_defillama_tvl": bool(tvl_series.empty),
                "revenue_fees_corr_90d": corr,
                "revenue_fees_redundant": bool(pd.notna(corr) and abs(corr) >= 0.98),
            }
        )
    quality = pd.DataFrame(rows)
    if not quality.empty:
        quality["fees_signal_deduped"] = quality["revenue_fees_redundant"]
    return quality


def _compute_factor_panels(
    panels: dict[str, pd.DataFrame | pd.Series],
    quality: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    signal_data = {
        "revenue": panels["revenue"],
        "mcap": panels["mcap"],
        "fdv": panels["fdv"],
        "trading_fees": panels["trading_fees"].copy(),
        "active_addresses": panels["active_addresses"],
        "issuance": panels["issuance"],
        "burn": panels["burn"],
        "stablecoin_supply": panels["stablecoin_supply"],
        "dex_volume": panels["dex_volume"],
        "open_interest": panels["open_interest"],
    }
    deduped = quality.loc[quality.get("fees_signal_deduped", False).astype(bool), "blockworks_slug"].tolist() if not quality.empty else []
    fees = signal_data["trading_fees"]
    if isinstance(fees, pd.DataFrame) and deduped:
        fees.loc[:, [slug for slug in deduped if slug in fees.columns]] = np.nan
        signal_data["trading_fees"] = fees

    btc = panels["btc_price"]
    btc_series = btc if isinstance(btc, pd.Series) else pd.Series(dtype=float)
    price = panels["price"] if isinstance(panels["price"], pd.DataFrame) else pd.DataFrame()
    mcap = panels["mcap"] if isinstance(panels["mcap"], pd.DataFrame) else pd.DataFrame()

    fund = compute_all_fundamentals(signal_data, btc_series)
    mom = compute_all_momentum(price, btc_series, mcap)
    flows = compute_all_flows(signal_data)
    family_scores = {
        "fundamentals_score": compute_family_score(fund),
        "momentum_score": compute_family_score(mom),
        "flows_score": compute_family_score(flows),
    }
    composite = compute_composite(
        {
            "fundamentals": family_scores["fundamentals_score"],
            "momentum": family_scores["momentum_score"],
            "flows": family_scores["flows_score"],
        },
        min_families=1,
    )
    return {"factor_score": composite, **family_scores}


def _build_project_timeseries(
    master: pd.DataFrame,
    market_snapshot: pd.DataFrame,
    panels: dict[str, pd.DataFrame | pd.Series],
    factor_panels: dict[str, pd.DataFrame],
    quality: pd.DataFrame,
) -> pd.DataFrame:
    slugs = master["blockworks_slug"].astype(str).tolist()
    index = _common_daily_index([p for p in panels.values() if isinstance(p, pd.DataFrame)])
    if len(index) > 120:
        index = index[-120:]

    price = _align_panel(panels["price"], index, slugs)
    volume = _align_panel(panels["volume_24h"], index, slugs)
    mcap = _align_panel(panels["mcap"], index, slugs)
    fdv = _align_panel(panels["fdv"], index, slugs)
    revenue = _align_panel(panels["revenue"], index, slugs)
    fees = _align_panel(panels["trading_fees"], index, slugs)
    deposits = _align_panel(panels["lending_deposits"], index, slugs)
    tvl = _align_panel(panels["tvl"], index, slugs)
    oi = _align_panel(panels["open_interest"], index, slugs)
    active = _align_panel(panels["active_addresses"], index, slugs)
    burn = _align_panel(panels["burn"], index, slugs)
    issuance = _align_panel(panels["issuance"], index, slugs)
    btc = panels["btc_price"].reindex(index).ffill() if isinstance(panels["btc_price"], pd.Series) else pd.Series(index=index, dtype=float)

    rev_7d = revenue.rolling(7, min_periods=3).mean()
    rev_30d = revenue.rolling(30, min_periods=7).sum()
    ann_rev = rev_30d * (365 / 30)
    fees_7d = fees.rolling(7, min_periods=3).mean()
    fees_30d = fees.rolling(30, min_periods=7).sum()
    ann_fees = fees_30d * (365 / 30)
    active_7d = active.rolling(7, min_periods=3).mean()

    metric_panels: dict[str, pd.DataFrame] = {
        "price": price,
        "market_cap": mcap,
        "fdv": fdv,
        "volume_24h": volume,
        "price_7d_pct": _pct_change_panel(price, 7),
        "revenue_7d_avg": rev_7d,
        "revenue_30d": rev_30d,
        "annualised_revenue_30d": ann_rev,
        "revenue_wow_pct": _ratio_change_panel(rev_7d, rev_7d.shift(7)),
        "revenue_mom_pct": _ratio_change_panel(rev_7d, rev_7d.shift(30)),
        "revenue_2w_4w_ratio": _growth_ratio_panel(revenue),
        "fees_7d_avg": fees_7d,
        "fees_30d": fees_30d,
        "annualised_fees_30d": ann_fees,
        "fees_wow_pct": _ratio_change_panel(fees_7d, fees_7d.shift(7)),
        "fees_mom_pct": _ratio_change_panel(fees_7d, fees_7d.shift(30)),
        "fees_2w_4w_ratio": _growth_ratio_panel(fees),
        "deposits": deposits,
        "deposits_wow_pct": _ratio_change_panel(deposits, deposits.shift(7)),
        "deposits_mom_pct": _ratio_change_panel(deposits, deposits.shift(30)),
        "tvl": tvl,
        "tvl_wow_pct": _ratio_change_panel(tvl, tvl.shift(7)),
        "tvl_mom_pct": _ratio_change_panel(tvl, tvl.shift(30)),
        "open_interest": oi,
        "oi_wow_pct": _ratio_change_panel(oi, oi.shift(7)),
        "oi_mom_pct": _ratio_change_panel(oi, oi.shift(30)),
        "daily_active_users": active_7d,
        "fdv_annualised_revenue": _safe_divide(fdv, ann_rev),
        "fdv_annualised_fees": _safe_divide(fdv, ann_fees),
        "payback_period_fees": _safe_divide(fdv, ann_fees),
        "revenue_per_active_address": _safe_divide(rev_7d, active_7d),
        "buyback_versus_issuance": _safe_divide(burn, issuance),
        "spot_volume_oi_ratio": _safe_divide(volume, oi),
        "daily_pct_over_btc": _daily_pct_over_btc(price, btc),
        "price_btc_corr": _rolling_corr_to_btc(price.pct_change(), btc.pct_change(), window=30),
        "revenue_btc_corr": _rolling_corr_to_btc(revenue.pct_change(), btc.pct_change(), window=91),
    }
    metric_panels["implied_growth_rate"] = _implied_growth(metric_panels["fdv_annualised_revenue"], master)
    for score_col, panel in factor_panels.items():
        metric_panels[score_col] = _align_panel(panel, index, slugs)
    metric_panels["factor_4w_change"] = metric_panels["factor_score"] - metric_panels["factor_score"].shift(28)
    for score_col in FACTOR_SCORE_COLUMNS:
        if score_col in metric_panels:
            metric_panels[f"{score_col}_lag_{FACTOR_LAG_DAYS}d"] = metric_panels[score_col].shift(FACTOR_LAG_DAYS)

    rows = []
    master_idx = master.set_index("blockworks_slug")
    quality_idx = quality.set_index("blockworks_slug") if not quality.empty else pd.DataFrame()
    for slug in slugs:
        if slug not in master_idx.index:
            continue
        meta = master_idx.loc[slug]
        token_df = pd.DataFrame({"date": index})
        token_df["blockworks_slug"] = slug
        token_df["ticker"] = meta["ticker"]
        token_df["project"] = meta["name"]
        token_df["category"] = meta["category"]
        token_df["sector"] = meta["sector"]
        token_df["asset_type"] = meta["asset_type"]
        token_df["coingecko_id"] = meta.get("coingecko_id", pd.NA)
        token_df["defillama_slug"] = meta.get("defillama_slug", pd.NA)
        token_df["entity_key"] = meta.get("entity_key", slug)
        token_df["defillama_url"] = meta.get("defillama_url", pd.NA)
        token_df["defillama_unlocks_url"] = meta.get("defillama_unlocks_url", pd.NA)
        token_df["blockworks_match_slug"] = meta.get("blockworks_match_slug", pd.NA)
        token_df["blockworks_url"] = meta.get("blockworks_url", pd.NA)
        token_df["binance_spot_symbol"] = meta.get("binance_spot_symbol", pd.NA)
        token_df["binance_futures_symbol"] = meta.get("binance_futures_symbol", pd.NA)
        token_df["factor_lag_date"] = token_df["date"] - pd.Timedelta(days=FACTOR_LAG_DAYS)
        for metric, panel in metric_panels.items():
            token_df[metric] = panel[slug].to_numpy() if slug in panel else np.nan
        if not quality_idx.empty and slug in quality_idx.index:
            for col in [
                "blockworks_latest_metric_date",
                "blockworks_stale_metric",
                "defillama_latest_metric_date",
                "defillama_stale_metric",
                "missing_defillama_fees_revenue",
                "missing_defillama_tvl",
                "revenue_fees_corr_90d",
                "revenue_fees_redundant",
                "fees_signal_deduped",
            ]:
                token_df[col] = quality_idx.loc[slug, col]
        rows.append(token_df)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out["fdv_bucket"] = out["fdv"].map(assign_fdv_bucket)
    out["mcap_bucket"] = out["market_cap"].map(assign_value_bucket)
    out["fdv_bucket"] = pd.Categorical(out["fdv_bucket"], categories=FDV_BUCKET_ORDER, ordered=True)
    out["mcap_bucket"] = pd.Categorical(out["mcap_bucket"], categories=FDV_BUCKET_ORDER, ordered=True)
    return out


def _build_snapshot(
    project_ts: pd.DataFrame,
    master: pd.DataFrame,
    market_snapshot: pd.DataFrame,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    if project_ts.empty:
        return pd.DataFrame()
    snapshot = (
        project_ts.sort_values("date")
        .groupby("ticker", as_index=False, observed=True)
        .tail(1)
        .reset_index(drop=True)
    )
    static_cols = [
        "asset_id",
        "entity_key",
        "blockworks_id",
        "blockworks_slug",
        "blockworks_match_slug",
        "defillama_url",
        "defillama_unlocks_url",
        "blockworks_url",
        "universe_source",
        "match_method",
        "match_confidence",
        "primary_tvl_source",
    ]
    snapshot = snapshot.drop(columns=[c for c in static_cols if c != "blockworks_slug" and c in snapshot], errors="ignore").merge(
        master[[c for c in static_cols if c in master.columns]],
        on="blockworks_slug",
        how="left",
    )
    quality_cols = [
        "blockworks_slug",
        "blockworks_latest_metric_date",
        "blockworks_stale_metric",
        "defillama_latest_metric_date",
        "defillama_stale_metric",
        "missing_defillama_fees_revenue",
        "missing_defillama_tvl",
        "revenue_fees_corr_90d",
        "revenue_fees_redundant",
        "fees_signal_deduped",
    ]
    if not quality.empty:
        snapshot = snapshot.drop(columns=[c for c in quality_cols if c != "blockworks_slug" and c in snapshot], errors="ignore").merge(
            quality[[c for c in quality_cols if c in quality]],
            on="blockworks_slug",
            how="left",
        )
    snapshot = snapshot.sort_values(["factor_score", "market_cap"], ascending=[False, False], na_position="last")
    return snapshot.reset_index(drop=True)


def _update_asset_master_coverage(master: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    out = master.copy()
    if snapshot.empty:
        for column in ASSET_MASTER_COLUMNS:
            if column not in out:
                out[column] = pd.NA
        return out[ASSET_MASTER_COLUMNS]

    coverage_cols = {
        "has_price": "price",
        "has_market_cap": "market_cap",
        "has_fdv": "fdv",
        "has_revenue": "revenue_7d_avg",
        "has_fees": "fees_7d_avg",
        "has_lending_deposits": "deposits",
        "has_defillama_tvl": "tvl",
        "has_open_interest": "open_interest",
    }
    snap = snapshot.set_index("blockworks_slug")
    for flag, metric in coverage_cols.items():
        values = snap[metric].notna() if metric in snap else pd.Series(dtype=bool)
        out[flag] = out["blockworks_slug"].map(values).fillna(False).astype(bool)
    out["has_blockworks_tvl"] = False
    out.loc[out["has_defillama_tvl"], "primary_tvl_source"] = "defillama"
    out.loc[~out["has_defillama_tvl"] & out["has_blockworks_tvl"], "primary_tvl_source"] = "blockworks"
    for column in ASSET_MASTER_COLUMNS:
        if column not in out:
            out[column] = pd.NA
    return out[ASSET_MASTER_COLUMNS]


def _build_summary_timeseries(project_ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not project_ts.empty:
        cutoff = project_ts["date"].max() - pd.Timedelta(days=29)
        project_ts = project_ts[project_ts["date"] >= cutoff].copy()
    for metric, label in CHANGE_METRICS_30D.items():
        if metric not in project_ts:
            continue
        base = project_ts.dropna(subset=[metric]).copy()
        if base.empty:
            continue
        for group_type, group_col in [
            ("aggregate", None),
            ("category", "category"),
            ("fdv_bucket", "fdv_bucket"),
            ("mcap_bucket", "mcap_bucket"),
        ]:
            groups = [("Aggregate", base)] if group_col is None else base.groupby(group_col, observed=True)
            for group_name, group in groups:
                for day, day_df in group.groupby("date", observed=True):
                    values = day_df[metric].dropna()
                    if values.empty:
                        continue
                    rows.append(
                        {
                            "date": day,
                            "metric": metric,
                            "metric_label": label,
                            "view": "change",
                            "group_type": group_type,
                            "Group": str(group_name),
                            "Statistic": "Average",
                            "Value": values.mean(),
                            "Eligible Projects": len(values),
                        }
                    )
                    for stat, value in [
                        ("Average", values.mean()),
                        ("Median", values.median()),
                        ("75th percentile", values.quantile(0.75)),
                        ("99th percentile", values.quantile(0.99)),
                    ]:
                        rows.append(
                            {
                                "date": day,
                                "metric": metric,
                                "metric_label": label,
                                "view": "dispersion",
                                "group_type": group_type,
                                "Group": str(group_name),
                                "Statistic": stat,
                                "Value": value,
                                "Eligible Projects": len(values),
                            }
                        )
    return pd.DataFrame(rows)


def _build_factor_scores_frame(project_ts: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "date",
        "ticker",
        "project",
        "category",
        "fdv_bucket",
        "factor_score",
        "fundamentals_score",
        "momentum_score",
        "flows_score",
        "factor_4w_change",
        "factor_lag_date",
    ]
    cols.extend(f"{col}_lag_{FACTOR_LAG_DAYS}d" for col in FACTOR_SCORE_COLUMNS)
    return project_ts[[c for c in cols if c in project_ts]].copy()


def _build_factor_baskets(project_ts: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    if project_ts.empty or snapshot.empty:
        return pd.DataFrame(columns=["date", "Factor", "Basket", "Index"])
    frames = []
    labels = {
        "factor_score": "Total",
        "fundamentals_score": "Fundamentals",
        "momentum_score": "Momentum",
        "flows_score": "Flows",
    }
    for score_col, factor_label in labels.items():
        lag_col = f"{score_col}_lag_{FACTOR_LAG_DAYS}d"
        if lag_col not in snapshot:
            continue
        ranked = snapshot.dropna(subset=[lag_col]).sort_values(lag_col, ascending=False)
        if ranked.empty:
            continue
        top = ranked.head(5)["ticker"].tolist()
        bottom = ranked.tail(5)["ticker"].tolist()
        top_frame = _basket_index(project_ts, snapshot, top, "Top factor basket")
        bottom_frame = _basket_index(project_ts, snapshot, bottom, "Bottom factor basket")
        for frame in [top_frame, bottom_frame]:
            if not frame.empty:
                frame["Factor"] = factor_label
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["date", "Factor", "Basket", "Index"])
    return pd.concat(frames, ignore_index=True)[["date", "Factor", "Basket", "Index"]]


def _basket_index(project_ts: pd.DataFrame, snapshot: pd.DataFrame, tickers: list[str], label: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["date", "Basket", "Index"])
    frames = []
    if not project_ts.empty:
        cutoff = project_ts["date"].max() - pd.Timedelta(days=29)
        project_ts = project_ts[project_ts["date"] >= cutoff].copy()
    current_weights = snapshot[snapshot["ticker"].isin(tickers)].set_index("ticker")
    weight_values: dict[str, float] = {}
    for ticker in tickers:
        token_ts = project_ts[project_ts["ticker"] == ticker].sort_values("date") if not project_ts.empty else pd.DataFrame()
        price_rows = token_ts.dropna(subset=["price"]) if "price" in token_ts else pd.DataFrame()
        start_row = price_rows.iloc[0] if not price_rows.empty else pd.Series(dtype=float)
        weight = pd.NA
        for weight_col in ["fdv", "market_cap"]:
            if weight_col in start_row and pd.notna(start_row[weight_col]):
                weight = start_row[weight_col]
                break
        if (pd.isna(weight) or float(weight) <= 0) and ticker in current_weights.index:
            for weight_col in ["fdv", "market_cap"]:
                if weight_col in current_weights and pd.notna(current_weights.at[ticker, weight_col]):
                    weight = current_weights.at[ticker, weight_col]
                    break
        weight_values[ticker] = float(weight) if pd.notna(weight) and float(weight) > 0 else 1.0
        series = price_rows.set_index("date")["price"] if not price_rows.empty else pd.Series(dtype=float)
        if len(series) < 2:
            continue
        indexed = series / series.iloc[0] * 100
        frames.append(indexed.rename(ticker))
    if not frames:
        return pd.DataFrame(columns=["date", "Basket", "Index"])
    indexed_prices = pd.concat(frames, axis=1).sort_index()
    weights = pd.Series(weight_values, dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(tickers), index=tickers)
    aligned_weights = weights.reindex(indexed_prices.columns).fillna(0.0)
    available_weights = indexed_prices.notna().mul(aligned_weights, axis=1)
    weight_sum = available_weights.sum(axis=1).replace(0, np.nan)
    basket_index = indexed_prices.mul(aligned_weights, axis=1).sum(axis=1, min_count=1) / weight_sum
    basket_index = basket_index.dropna()
    if basket_index.empty:
        return pd.DataFrame(columns=["date", "Basket", "Index"])
    basket_index = basket_index / basket_index.iloc[0] * 100
    combined = basket_index.reset_index()
    combined.columns = ["date", "Index"]
    combined["Basket"] = label
    return combined[["date", "Basket", "Index"]]


def _daily_panel(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index = pd.to_datetime(out.index).normalize()
    out = out.groupby(level=0).last().sort_index()
    return out.apply(pd.to_numeric, errors="coerce")


def _common_daily_index(panels: list[pd.DataFrame]) -> pd.DatetimeIndex:
    indexes = [pd.to_datetime(p.index).normalize() for p in panels if isinstance(p, pd.DataFrame) and not p.empty]
    if not indexes:
        return pd.date_range(end=pd.Timestamp.now().normalize(), periods=120, freq="D")
    start = min(idx.min() for idx in indexes)
    end = max(idx.max() for idx in indexes)
    return pd.date_range(start=start, end=end, freq="D")


def _align_panel(panel: pd.DataFrame | pd.Series | None, index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    if panel is None or (hasattr(panel, "empty") and panel.empty):
        return pd.DataFrame(index=index, columns=columns, dtype=float)
    if isinstance(panel, pd.Series):
        panel = panel.to_frame()
    out = _daily_panel(panel)
    out = out.reindex(index).ffill(limit=7)
    return out.reindex(columns=columns)


def _static_panel(values: pd.Series, index: pd.DatetimeIndex, columns: list[str]) -> pd.DataFrame:
    values = pd.to_numeric(values, errors="coerce").reindex(columns)
    return pd.DataFrame({col: values.get(col, np.nan) for col in columns}, index=index)


def _combine_panels(left: pd.DataFrame | None, right: pd.DataFrame | None) -> pd.DataFrame:
    if left is None or left.empty:
        return right.copy() if right is not None else pd.DataFrame()
    if right is None or right.empty:
        return left.copy()
    return left.combine_first(right)


def _latest_row(panel: pd.DataFrame | None) -> pd.Series:
    if panel is None or panel.empty:
        return pd.Series(dtype=float)
    return panel.sort_index().ffill().iloc[-1]


def _coalesce(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [c for c in columns if c in df]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].bfill(axis=1).iloc[:, 0]


def _pct_change_panel(panel: pd.DataFrame, periods: int) -> pd.DataFrame:
    return _ratio_change_panel(panel, panel.shift(periods))


def _ratio_change_panel(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    out = numerator / denominator.replace(0, np.nan) - 1
    return out.replace([np.inf, -np.inf], np.nan) * 100


def _growth_ratio_panel(panel: pd.DataFrame) -> pd.DataFrame:
    smoothed = panel.rolling(7, min_periods=3).mean()
    growth_2w = smoothed.pct_change(14)
    growth_4w = smoothed.pct_change(28)
    return (growth_2w / growth_4w.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _safe_divide(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _daily_pct_over_btc(price: pd.DataFrame, btc: pd.Series) -> pd.DataFrame:
    token_ret = price.pct_change() * 100
    btc_ret = btc.pct_change() * 100
    return token_ret.sub(btc_ret, axis=0)


def _rolling_corr_to_btc(panel: pd.DataFrame, btc_ret: pd.Series, window: int) -> pd.DataFrame:
    frames = {}
    for col in panel.columns:
        aligned = pd.DataFrame({"token": panel[col], "btc": btc_ret}).dropna()
        if len(aligned) < max(10, window // 2):
            frames[col] = pd.Series(index=panel.index, dtype=float)
            continue
        frames[col] = aligned["token"].rolling(window, min_periods=max(10, window // 2)).corr(aligned["btc"])
    return pd.DataFrame(frames).reindex(panel.index)


def _implied_growth(fdv_revenue: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=fdv_revenue.index, columns=fdv_revenue.columns, dtype=float)
    categories = master.set_index("blockworks_slug")["category"].to_dict()
    for category in sorted(set(categories.values())):
        slugs = [slug for slug, cat in categories.items() if cat == category and slug in fdv_revenue.columns]
        if not slugs:
            continue
        median = fdv_revenue[slugs].median(axis=1)
        out[slugs] = fdv_revenue[slugs].div(median.replace(0, np.nan), axis=0).sub(1) * 100
    return out


def assign_value_bucket(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    value = float(value)
    for label, lower, upper in FDV_BUCKETS:
        if lower <= value < upper:
            return label
    return "Unknown"


def assign_fdv_bucket(value: Any) -> str:
    return assign_value_bucket(value)
