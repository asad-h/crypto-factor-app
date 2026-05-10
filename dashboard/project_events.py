"""Project event, governance, and external context helpers for the dashboard."""
from __future__ import annotations

import html
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from crypto_factor_model.config import NANSEN_API_KEY, _get_secret
from dashboard.screener_data import CG_COINS_PATH, SCREENER_CACHE_DIR


PROJECT_EVENTS_PATH = SCREENER_CACHE_DIR / "project_events.parquet"
NANSEN_CONTEXT_PATH = SCREENER_CACHE_DIR / "nansen_token_context.parquet"
RAW_EVENT_CACHE_DIR = SCREENER_CACHE_DIR / "raw_project_events"
RAW_EVENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFILLAMA_UNLOCK_BASE_URL = "https://defillama.com/protocol/unlocks"
DEFILLAMA_GOVERNANCE_BASE_URL = "https://defillama.com/governance"
SNAPSHOT_GRAPHQL_URL = "https://hub.snapshot.org/graphql"
HYPURRSCAN_UNSTAKING_URL = "https://hypurrscan.io/staking#unstaking"
HYPERLIQUID_STAKING_DOCS_URL = "https://hyperliquid.gitbook.io/hyperliquid-docs/hypercore/staking"
NANSEN_BASE_URL = "https://api.nansen.ai"

EVENT_COLUMNS = [
    "event_type",
    "date",
    "end_date",
    "project",
    "ticker",
    "defillama_slug",
    "recipient",
    "bucket",
    "token_amount",
    "token_symbol",
    "value_usd",
    "pct_supply",
    "pct_float",
    "state",
    "title",
    "notes",
    "source",
    "source_url",
]

NANSEN_ENDPOINTS = {
    "token_information": "/api/v1/tgm/token-information",
    "indicators": "/api/v1/tgm/indicators",
    "token_ohlcv": "/api/v1/tgm/token-ohlcv",
    "holders": "/api/v1/tgm/holders",
    "flow_intelligence": "/api/v1/tgm/flow-intelligence",
    "flows": "/api/v1/tgm/flows",
    "who_bought_sold": "/api/v1/tgm/who-bought-sold",
    "dex_trades": "/api/v1/tgm/dex-trades",
    "transfers": "/api/v1/tgm/transfers",
    "pnl_leaderboard": "/api/v1/tgm/pnl-leaderboard",
    "token_screener": "/api/v1/tgm/token-screener",
    "perp_screener": "/api/v1/tgm/perp-screener",
    "smart_money_netflow": "/api/v1/smart-money/netflow",
    "smart_money_holdings": "/api/v1/smart-money/holdings",
    "smart_money_dex_trades": "/api/v1/smart-money/dex-trades",
    "smart_money_perp_trades": "/api/v1/smart-money/perp-trades",
}

SUPPORTED_NANSEN_CHAINS = {
    "ethereum",
    "solana",
    "base",
    "arbitrum",
    "bnb",
    "hyperevm",
    "hyperliquid",
}

COINGECKO_NANSEN_CHAIN_MAP = {
    "ethereum": "ethereum",
    "solana": "solana",
    "base": "base",
    "arbitrum-one": "arbitrum",
    "arbitrum": "arbitrum",
    "binance-smart-chain": "bnb",
    "bnb": "bnb",
    "hyperevm": "hyperevm",
    "hyperliquid": "hyperliquid",
}

NANSEN_PLATFORM_PRIORITY = (
    "ethereum",
    "base",
    "arbitrum-one",
    "solana",
    "binance-smart-chain",
    "hyperevm",
    "hyperliquid",
)

DEFAULT_PROTOCOL_SOURCES: dict[str, dict[str, Any]] = {
    "hyperliquid": {
        "defillama_slug": "hyperliquid",
        "unlock_url": f"{DEFILLAMA_UNLOCK_BASE_URL}/hyperliquid",
        "governance_url": f"{DEFILLAMA_GOVERNANCE_BASE_URL}/hyperliquid",
        "flow_adapter": "hyperliquid",
        "nansen_chain": "hyperevm",
        "nansen_perp_chain": "hyperliquid",
    },
    "lido": {
        "defillama_slug": "lido",
        "unlock_url": f"{DEFILLAMA_UNLOCK_BASE_URL}/lido",
        "governance_url": f"{DEFILLAMA_GOVERNANCE_BASE_URL}/lido",
        "snapshot_space": "lido-snapshot.eth",
        "nansen_chain": "ethereum",
    },
    "uniswap": {
        "defillama_slug": "uniswap",
        "unlock_url": f"{DEFILLAMA_UNLOCK_BASE_URL}/uniswap",
        "governance_url": f"{DEFILLAMA_GOVERNANCE_BASE_URL}/uniswap",
        "snapshot_space": "uniswapgovernance.eth",
        "nansen_chain": "ethereum",
    },
    "aave": {
        "defillama_slug": "aave",
        "unlock_url": f"{DEFILLAMA_UNLOCK_BASE_URL}/aave",
        "governance_url": f"{DEFILLAMA_GOVERNANCE_BASE_URL}/aave",
        "snapshot_space": "aave.eth",
        "nansen_chain": "ethereum",
    },
}

RECIPIENT_LABELS = [
    "Core Contributors",
    "Contributors",
    "Team",
    "Investors",
    "Private Sale",
    "Seed Investors",
    "Advisors",
    "Founders",
    "Foundation",
    "Ecosystem",
    "Community",
    "Treasury",
    "Rewards",
    "Liquidity",
    "Airdrop",
]

POTENTIAL_SELLING_KEYWORDS = (
    "advisor",
    "contributor",
    "core contributor",
    "employee",
    "founder",
    "investor",
    "private sale",
    "seed",
    "team",
)

MONTH_PATTERN = (
    "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    "January|February|March|April|June|July|August|September|October|November|December"
)
DATE_RE = re.compile(rf"\b({MONTH_PATTERN})\s+\d{{1,2}},\s+\d{{4}}\b", re.I)
USD_RE = re.compile(r"\$([0-9][0-9,]*(?:\.[0-9]+)?)\s*([kmbt])?\b", re.I)
PERCENT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")
FLOAT_PERCENT_RE = re.compile(r"\(?\s*([0-9]+(?:\.[0-9]+)?)\s*%\s+of\s+float\s*\)?", re.I)


def empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def normalise_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def source_config_for_project(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    slug = normalise_key(_first_present(row, ["defillama_slug", "coingecko_id", "project"]))
    ticker = str(_first_present(row, ["ticker"]) or "").upper()
    config = dict(DEFAULT_PROTOCOL_SOURCES.get(slug, {}))
    if not config:
        config["defillama_slug"] = slug
        config["unlock_url"] = f"{DEFILLAMA_UNLOCK_BASE_URL}/{slug}" if slug else ""
        config["governance_url"] = f"{DEFILLAMA_GOVERNANCE_BASE_URL}/{slug}" if slug else ""

    secret_slug = slug.upper().replace("-", "_")
    address = _get_secret(f"NANSEN_{ticker}_TOKEN_ADDRESS") or _get_secret(f"NANSEN_{secret_slug}_TOKEN_ADDRESS")
    if address:
        config["nansen_token_address"] = address
    config.setdefault("nansen_chain", _infer_nansen_chain(row))
    if not config.get("nansen_token_address"):
        inferred = _coingecko_nansen_address(_first_present(row, ["coingecko_id"]), config.get("nansen_chain"))
        if inferred:
            config["nansen_chain"], config["nansen_token_address"] = inferred
    return config


def emission_bucket(recipient: Any) -> str:
    text = str(recipient or "").strip()
    lower = text.lower()
    if any(keyword in lower for keyword in POTENTIAL_SELLING_KEYWORDS):
        return "Potential Selling"
    return text or "Other"


def parse_number_with_suffix(value: str, suffix: str | None = None) -> float | None:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    scale = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "t": 1_000_000_000_000}.get(str(suffix or "").lower(), 1)
    return number * scale


def html_to_visible_text(raw: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", raw or "")
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|section|article|li|tr|h[1-6])>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    return normalise_whitespace(text)


def normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_defillama_unlock_text(
    text: str,
    *,
    slug: str,
    source_url: str,
    project: str | None = None,
    ticker: str | None = None,
) -> pd.DataFrame:
    clean = normalise_whitespace(text)
    if not clean or "Just a moment" in clean[:200]:
        return empty_events()

    matches = list(DATE_RE.finditer(clean))
    if not matches:
        return empty_events()

    rows: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        previous_date_start = matches[idx - 1].start() if idx > 0 else -1
        previous_unlock_start = clean.rfind("Unlock Value", 0, match.start())
        if previous_unlock_start > previous_date_start:
            chunk_start = previous_unlock_start
        else:
            chunk_start = max(0, match.start() - 180)
        chunk_end = matches[idx + 1].start() if idx + 1 < len(matches) else min(len(clean), match.end() + 420)
        chunk = clean[chunk_start:chunk_end]
        event_date = _parse_unlock_date(match.group(0))
        if event_date is None:
            continue

        recipient = _extract_recipient(chunk)
        value_usd = _extract_usd_value(chunk)
        token_symbol = str(ticker or "").upper() or _extract_token_symbol(chunk)
        token_amount = _extract_token_amount(chunk, token_symbol)
        pct_supply, pct_float = _extract_percentages(chunk)
        if recipient is None and value_usd is None and token_amount is None:
            continue

        rows.append(
            {
                "event_type": "unlock",
                "date": pd.Timestamp(event_date.date()),
                "end_date": pd.NaT,
                "project": project or slug,
                "ticker": ticker or token_symbol or "",
                "defillama_slug": slug,
                "recipient": recipient or "Unknown",
                "bucket": emission_bucket(recipient),
                "token_amount": token_amount,
                "token_symbol": token_symbol,
                "value_usd": value_usd,
                "pct_supply": pct_supply,
                "pct_float": pct_float,
                "state": "",
                "title": "Token unlock",
                "notes": "Parsed from DefiLlama unlock page text.",
                "source": "DefiLlama unlock page scrape",
                "source_url": source_url,
            }
        )

    if not rows:
        return empty_events()
    return pd.DataFrame(rows, columns=EVENT_COLUMNS).drop_duplicates(
        subset=["date", "recipient", "token_amount", "value_usd"],
        keep="first",
    )


def fetch_defillama_unlock_events(
    slug: str,
    *,
    project: str | None = None,
    ticker: str | None = None,
    source_url: str | None = None,
    use_cache: bool = True,
    max_age_hours: int = 24,
) -> pd.DataFrame:
    slug = normalise_key(slug)
    if not slug:
        return empty_events()
    url = source_url or f"{DEFILLAMA_UNLOCK_BASE_URL}/{slug}"
    cache_path = RAW_EVENT_CACHE_DIR / f"defillama_unlocks_{slug}.txt"
    text = ""
    if use_cache and _cache_is_fresh(cache_path, max_age_hours):
        text = cache_path.read_text(encoding="utf-8", errors="ignore")
    else:
        try:
            response = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 CryptoFactorScreener/1.0",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
                timeout=20,
            )
            response.raise_for_status()
            text = response.text
            cache_path.write_text(text, encoding="utf-8")
        except Exception:
            if cache_path.exists():
                text = cache_path.read_text(encoding="utf-8", errors="ignore")
            else:
                return empty_events()

    visible = html_to_visible_text(text)
    events = parse_defillama_unlock_text(visible, slug=slug, source_url=url, project=project, ticker=ticker)
    cache_project_events(events)
    return events


def cache_project_events(events: pd.DataFrame) -> None:
    if events.empty:
        return
    try:
        current = pd.read_parquet(PROJECT_EVENTS_PATH) if PROJECT_EVENTS_PATH.exists() else empty_events()
        combined = pd.concat([current, events], ignore_index=True)
        for col in EVENT_COLUMNS:
            if col not in combined:
                combined[col] = pd.NA
        combined = combined[EVENT_COLUMNS].drop_duplicates(
            subset=["event_type", "date", "project", "ticker", "recipient", "title", "source_url"],
            keep="last",
        )
        combined.to_parquet(PROJECT_EVENTS_PATH)
    except Exception:
        return


def hyperliquid_unstaking_context(project: str = "Hyperliquid", ticker: str = "HYPE") -> pd.DataFrame:
    rules = [
        ("Delegation lock", "Delegations are locked for 1 day before they can be undelegated."),
        ("Unstaking queue", "Unstaked HYPE enters a 7-day staking-to-spot withdrawal queue."),
        ("Pending withdrawal limit", "Each address can have at most 5 pending withdrawals."),
        ("Live queue source", "Use Hypurrscan unstaking page when parsable; static pages may not expose queue rows."),
    ]
    rows = []
    for title, note in rules:
        rows.append(
            {
                "event_type": "unstaking",
                "date": pd.NaT,
                "end_date": pd.NaT,
                "project": project,
                "ticker": ticker,
                "defillama_slug": "hyperliquid",
                "recipient": "",
                "bucket": "Flows",
                "token_amount": None,
                "token_symbol": ticker,
                "value_usd": None,
                "pct_supply": None,
                "pct_float": None,
                "state": "",
                "title": title,
                "notes": note,
                "source": "Hyperliquid staking docs / Hypurrscan",
                "source_url": HYPURRSCAN_UNSTAKING_URL,
            }
        )
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    cache_project_events(events)
    return events


def fetch_snapshot_governance_events(
    space_id: str | None,
    *,
    project: str,
    ticker: str,
    defillama_slug: str,
    governance_url: str,
    use_cache: bool = True,
    max_age_hours: int = 6,
) -> pd.DataFrame:
    if not space_id:
        return empty_events()
    cache_path = RAW_EVENT_CACHE_DIR / f"snapshot_{normalise_key(space_id)}.json"
    payload: dict[str, Any] | None = None
    if use_cache and _cache_is_fresh(cache_path, max_age_hours):
        try:
            payload = json.loads(cache_path.read_text())
        except Exception:
            payload = None
    if payload is None:
        query = """
        query Proposals($spaces: [String]) {
          proposals(
            first: 20,
            skip: 0,
            where: { space_in: $spaces },
            orderBy: "created",
            orderDirection: desc
          ) {
            id
            title
            start
            end
            state
            scores_total
            choices
            space { id name }
          }
        }
        """
        try:
            response = requests.post(
                SNAPSHOT_GRAPHQL_URL,
                json={"query": query, "variables": {"spaces": [space_id]}},
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            return empty_events()

    events = parse_snapshot_governance_payload(
        payload,
        project=project,
        ticker=ticker,
        defillama_slug=defillama_slug,
        governance_url=governance_url,
    )
    cache_project_events(events)
    return events


def parse_snapshot_governance_payload(
    payload: dict[str, Any],
    *,
    project: str,
    ticker: str,
    defillama_slug: str,
    governance_url: str,
) -> pd.DataFrame:
    proposals = ((payload or {}).get("data") or {}).get("proposals") or []
    rows = []
    for proposal in proposals:
        space = proposal.get("space") or {}
        space_id = space.get("id") or ""
        proposal_id = proposal.get("id") or ""
        rows.append(
            {
                "event_type": "governance",
                "date": _timestamp_to_date(proposal.get("start")),
                "end_date": _timestamp_to_date(proposal.get("end")),
                "project": project,
                "ticker": ticker,
                "defillama_slug": defillama_slug,
                "recipient": "",
                "bucket": "Governance",
                "token_amount": None,
                "token_symbol": ticker,
                "value_usd": proposal.get("scores_total"),
                "pct_supply": None,
                "pct_float": None,
                "state": proposal.get("state") or "",
                "title": proposal.get("title") or "Snapshot proposal",
                "notes": f"Snapshot space: {space.get('name') or space_id}",
                "source": "Snapshot GraphQL",
                "source_url": f"https://snapshot.box/#/{space_id}/proposal/{proposal_id}" if space_id and proposal_id else governance_url,
            }
        )
    if not rows:
        return empty_events()
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def build_nansen_token_context_requests(
    chain: str,
    token_address: str,
    *,
    date_range: dict[str, str] | None = None,
    lookback_period: str = "7d",
) -> list[dict[str, Any]]:
    chain = str(chain or "").lower()
    token_address = str(token_address or "").strip()
    if not chain or not token_address:
        return []
    if chain not in SUPPORTED_NANSEN_CHAINS:
        return []
    flow_range = date_range or {"from": "7D_AGO", "to": "NOW"}
    base = {"chain": chain, "token_address": token_address}
    requests_to_make: list[dict[str, Any]] = [
        {"name": "token_information", "endpoint": NANSEN_ENDPOINTS["token_information"], "payload": base.copy()},
        {"name": "indicators", "endpoint": NANSEN_ENDPOINTS["indicators"], "payload": base.copy()},
        {"name": "token_ohlcv", "endpoint": NANSEN_ENDPOINTS["token_ohlcv"], "payload": {**base, "date_range": flow_range}},
        {"name": "flow_intelligence", "endpoint": NANSEN_ENDPOINTS["flow_intelligence"], "payload": {**base, "lookback_period": lookback_period}},
        {"name": "who_bought_sold_buy", "endpoint": NANSEN_ENDPOINTS["who_bought_sold"], "payload": {**base, "buy_or_sell": "BUY", "time_range": flow_range, "pagination": {"page": 1, "pageSize": 25}, "order_by": "bought_volume_usd", "order_by_direction": "desc"}},
        {"name": "who_bought_sold_sell", "endpoint": NANSEN_ENDPOINTS["who_bought_sold"], "payload": {**base, "buy_or_sell": "SELL", "time_range": flow_range, "pagination": {"page": 1, "pageSize": 25}, "order_by": "sold_volume_usd", "order_by_direction": "desc"}},
        {"name": "dex_trades", "endpoint": NANSEN_ENDPOINTS["dex_trades"], "payload": {**base, "filters": {"value_usd": {"from": 50_000}}, "pagination": {"page": 1, "pageSize": 25}, "order_by": "timestamp", "order_by_direction": "desc"}},
        {"name": "transfers", "endpoint": NANSEN_ENDPOINTS["transfers"], "payload": {**base, "filters": {"value_usd": {"from": 100_000}}, "pagination": {"page": 1, "pageSize": 25}, "order_by": "timestamp", "order_by_direction": "desc"}},
    ]
    for label_type in ["smart_money", "whale", "exchange", "top_100_holders"]:
        requests_to_make.append(
            {
                "name": f"holders_{label_type}",
                "endpoint": NANSEN_ENDPOINTS["holders"],
                "payload": {**base, "label_type": label_type, "pagination": {"page": 1, "pageSize": 25}, "order_by": "amount", "order_by_direction": "desc"},
            }
        )
    for segment in ["smart_money", "whale", "exchange"]:
        requests_to_make.append(
            {
                "name": f"flows_{segment}",
                "endpoint": NANSEN_ENDPOINTS["flows"],
                "payload": {**base, "holder_segment": segment, "date_range": flow_range},
            }
        )
    return requests_to_make


def fetch_nansen_token_context(
    chain: str,
    token_address: str,
    *,
    api_key: str | None = None,
    date_range: dict[str, str] | None = None,
    max_requests: int = 8,
) -> pd.DataFrame:
    key = api_key or NANSEN_API_KEY or os.getenv("NANSEN_API_KEY")
    if not key:
        return pd.DataFrame(columns=["Endpoint", "Status", "Items", "Summary"])
    request_specs = build_nansen_token_context_requests(chain, token_address, date_range=date_range)[:max_requests]
    rows = []
    session = requests.Session()
    session.headers.update({"X-API-KEY": key, "Content-Type": "application/json"})
    for spec in request_specs:
        try:
            response = session.post(f"{NANSEN_BASE_URL}{spec['endpoint']}", json=spec["payload"], timeout=25)
            response.raise_for_status()
            data = response.json()
            rows.append(_summarise_nansen_response(spec["name"], spec["endpoint"], data, "ok"))
        except Exception as exc:
            rows.append({"Endpoint": spec["endpoint"], "Status": "error", "Items": 0, "Summary": str(exc)[:180]})
    context = pd.DataFrame(rows)
    cache_nansen_context(context, chain, token_address)
    return context


def cache_nansen_context(context: pd.DataFrame, chain: str, token_address: str) -> None:
    if context.empty:
        return
    try:
        enriched = context.copy()
        enriched["Chain"] = chain
        enriched["Token Address"] = token_address
        enriched["Fetched At"] = pd.Timestamp.utcnow()
        current = pd.read_parquet(NANSEN_CONTEXT_PATH) if NANSEN_CONTEXT_PATH.exists() else pd.DataFrame()
        combined = pd.concat([current, enriched], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["Chain", "Token Address", "Endpoint"],
            keep="last",
        )
        combined.to_parquet(NANSEN_CONTEXT_PATH)
    except Exception:
        return


def compute_rolling_30d_return_correlations(
    project_ts: pd.DataFrame,
    selected_ticker: str,
    benchmarks: list[str],
) -> pd.DataFrame:
    columns = ["Benchmark", "Correlation", "Observations", "Selected 30D Return", "Benchmark 30D Return", "Note"]
    if project_ts.empty or not {"date", "ticker", "price"}.issubset(project_ts.columns):
        return pd.DataFrame(columns=columns)

    work = project_ts[["date", "ticker", "price"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["ticker"] = work["ticker"].astype(str).str.upper()
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work = work.dropna(subset=["date", "ticker", "price"])
    if work.empty:
        return pd.DataFrame(columns=columns)

    pivot = work.groupby(["date", "ticker"], observed=True)["price"].last().unstack("ticker").sort_index()
    returns = pivot.pct_change(30)
    selected = str(selected_ticker or "").upper()
    ordered_benchmarks = _dedupe_tickers(benchmarks)
    rows = []
    for benchmark in ordered_benchmarks:
        note = ""
        if selected not in returns:
            corr = None
            observations = 0
            selected_latest = None
            benchmark_latest = None
            note = "Selected asset has no price history."
        elif benchmark not in returns:
            corr = None
            observations = 0
            selected_latest = _latest_pct(returns[selected])
            benchmark_latest = None
            note = "Benchmark has no price history."
        elif selected == benchmark:
            series = returns[selected].dropna()
            corr = 1.0 if not series.empty else None
            observations = int(series.count())
            selected_latest = _latest_pct(returns[selected])
            benchmark_latest = selected_latest
            note = "Selected asset."
        else:
            aligned = returns[[selected, benchmark]].dropna()
            observations = int(len(aligned))
            corr = float(aligned[selected].corr(aligned[benchmark])) if observations >= 2 else None
            selected_latest = _latest_pct(returns[selected])
            benchmark_latest = _latest_pct(returns[benchmark])
            if observations < 2:
                note = "Insufficient aligned 30D return observations."
        rows.append(
            {
                "Benchmark": benchmark,
                "Correlation": corr,
                "Observations": observations,
                "Selected 30D Return": selected_latest,
                "Benchmark 30D Return": benchmark_latest,
                "Note": note,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _first_present(row: pd.Series | dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        try:
            value = row.get(key)
        except AttributeError:
            value = None
        if value is not None and not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}:
            return value
    return None


@lru_cache(maxsize=1)
def _coingecko_platforms_by_id() -> dict[str, dict[str, str]]:
    if not CG_COINS_PATH.exists():
        return {}
    try:
        data = json.loads(CG_COINS_PATH.read_text())
    except Exception:
        return {}
    out: dict[str, dict[str, str]] = {}
    for coin in data:
        coin_id = str(coin.get("id") or "").strip()
        platforms = coin.get("platforms") if isinstance(coin, dict) else None
        if coin_id and isinstance(platforms, dict):
            out[coin_id] = {str(k): str(v) for k, v in platforms.items() if v}
    return out


def _coingecko_nansen_address(coin_id: Any, preferred_chain: Any = None) -> tuple[str, str] | None:
    platforms = _coingecko_platforms_by_id().get(str(coin_id or "").strip(), {})
    if not platforms:
        return None
    preferred = str(preferred_chain or "").lower().strip()
    ordered_platforms = list(NANSEN_PLATFORM_PRIORITY)
    if preferred:
        preferred_platforms = [
            platform
            for platform, nansen_chain in COINGECKO_NANSEN_CHAIN_MAP.items()
            if nansen_chain == preferred and platform in platforms
        ]
        ordered_platforms = [*preferred_platforms, *ordered_platforms]

    seen: set[str] = set()
    for platform in ordered_platforms:
        if platform in seen:
            continue
        seen.add(platform)
        address = platforms.get(platform)
        nansen_chain = COINGECKO_NANSEN_CHAIN_MAP.get(platform)
        if address and nansen_chain in SUPPORTED_NANSEN_CHAINS:
            return nansen_chain, address
    return None


def _infer_nansen_chain(row: pd.Series | dict[str, Any]) -> str:
    category = str(_first_present(row, ["category"]) or "").lower()
    project = str(_first_present(row, ["project", "defillama_slug"]) or "").lower()
    if "solana" in project:
        return "solana"
    if "bnb" in project or "bsc" in project:
        return "bnb"
    if "base" in project:
        return "base"
    if "arbitrum" in project:
        return "arbitrum"
    if "hyperliquid" in project:
        return "hyperevm"
    if "l1" in category or "l2" in category:
        return "ethereum"
    return "ethereum"


def _cache_is_fresh(path: Path, max_age_hours: int) -> bool:
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    return age_seconds <= max_age_hours * 3600


def _parse_unlock_date(value: str) -> datetime | None:
    value = str(value).strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _extract_recipient(chunk: str) -> str | None:
    for label in RECIPIENT_LABELS:
        if re.search(rf"\b{re.escape(label)}\b", chunk, flags=re.I):
            return label
    match = re.search(r"(?:GMT[+-]\d{2}:?\d{2}|AM|PM)\s+([A-Z][A-Za-z0-9 /&+.-]{2,60}?)\s+\$", chunk)
    if match:
        return normalise_whitespace(match.group(1))
    return None


def _extract_usd_value(chunk: str) -> float | None:
    match = re.search(r"Unlock Value\s*\$([0-9][0-9,]*(?:\.[0-9]+)?)\s*([kmbt])?", chunk, flags=re.I)
    if not match:
        match = USD_RE.search(chunk)
    if not match:
        return None
    return parse_number_with_suffix(match.group(1), match.group(2))


def _extract_token_symbol(chunk: str) -> str:
    matches = re.findall(r"\b[0-9][0-9,]*(?:\.[0-9]+)?\s+([A-Z][A-Z0-9]{1,12})\b", chunk)
    filtered = [symbol for symbol in matches if symbol not in {"AM", "PM", "GMT"}]
    return filtered[-1] if filtered else ""


def _extract_token_amount(chunk: str, token_symbol: str) -> float | None:
    if token_symbol:
        match = re.search(rf"\b([0-9][0-9,]*(?:\.[0-9]+)?)\s+{re.escape(token_symbol)}\b", chunk)
        if match:
            return parse_number_with_suffix(match.group(1))
    matches = re.findall(r"\b([0-9][0-9,]*(?:\.[0-9]+)?)\s+[A-Z][A-Z0-9]{1,12}\b", chunk)
    return parse_number_with_suffix(matches[-1]) if matches else None


def _extract_percentages(chunk: str) -> tuple[float | None, float | None]:
    float_match = FLOAT_PERCENT_RE.search(chunk)
    pct_float = float(float_match.group(1)) if float_match else None
    pct_matches = [float(value) for value in PERCENT_RE.findall(chunk)]
    pct_supply = None
    for value in pct_matches:
        if pct_float is None or abs(value - pct_float) > 1e-12:
            pct_supply = value
            break
    return pct_supply, pct_float


def _timestamp_to_date(value: Any) -> pd.Timestamp | pd.NaT:
    try:
        return pd.Timestamp(datetime.fromtimestamp(int(value), tz=timezone.utc).date())
    except Exception:
        return pd.NaT


def _summarise_nansen_response(name: str, endpoint: str, data: Any, status: str) -> dict[str, Any]:
    items = 0
    if isinstance(data, list):
        items = len(data)
    elif isinstance(data, dict):
        for key in ["data", "items", "results", "rows"]:
            value = data.get(key)
            if isinstance(value, list):
                items = len(value)
                break
        else:
            items = len(data)
    summary = name.replace("_", " ").title()
    return {"Endpoint": endpoint, "Status": status, "Items": items, "Summary": summary}


def _dedupe_tickers(tickers: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        clean = str(ticker or "").upper().strip()
        if clean and clean not in seen:
            out.append(clean)
            seen.add(clean)
    return out


def _latest_pct(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1] * 100)
