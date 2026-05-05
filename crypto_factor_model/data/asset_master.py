"""
Canonical asset-master construction.

The screener asset master is DefiLlama-left: rows start from DefiLlama
protocols/chains, then optional external sources attach market data, trading
venues, and source identifiers. The older Blockworks builder remains for
compatibility with historical cache code.

This module does not make API calls. It defines the production data contract and
pure transforms that can be used by the next live-data wiring step.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd


ASSET_MASTER_COLUMNS = [
    "asset_id",
    "entity_key",
    "blockworks_id",
    "blockworks_slug",
    "blockworks_match_slug",
    "ticker",
    "name",
    "asset_type",
    "sector",
    "category",
    "coingecko_id",
    "defillama_slug",
    "defillama_chain",
    "defillama_child_slugs",
    "defillama_group_key",
    "defillama_url",
    "defillama_unlocks_url",
    "blockworks_url",
    "binance_spot_symbol",
    "binance_futures_symbol",
    "match_method",
    "match_confidence",
    "has_price",
    "has_market_cap",
    "has_fdv",
    "has_revenue",
    "has_fees",
    "has_lending_deposits",
    "has_blockworks_tvl",
    "has_defillama_tvl",
    "has_open_interest",
    "primary_tvl_source",
    "universe_source",
]

EXTERNAL_ID_COLUMNS = [
    "coingecko_id",
    "defillama_slug",
    "binance_spot_symbol",
    "binance_futures_symbol",
]

BLOCKWORKS_MARKET_FIELDS = {
    "has_price": ("price", "usd"),
    "has_market_cap": ("market_cap", "usd"),
    "has_fdv": ("fdv", "usd"),
}

DEFILLAMA_EXCLUDED_CATEGORIES = {
    "cex",
    "stablecoins",
    "stablecoin",
    "tokenized treasury",
    "tokenised treasury",
    "treasury manager",
    "reserve currency",
}

DEFILLAMA_EXCLUDED_SYMBOLS = {
    "",
    "-",
    "usdt",
    "usdc",
    "dai",
    "fdusd",
    "tusd",
    "usdd",
    "usdp",
    "usde",
    "usds",
    "busd",
    "lusd",
    "frax",
    "gho",
}

DEFILLAMA_CATEGORY_PRIORITY = {
    "derivatives": 0,
    "perps": 0,
    "dexs": 1,
    "dex": 1,
    "lending": 2,
    "rwa lending": 2,
    "liquid staking": 3,
    "yield": 3,
    "restaking": 3,
    "bridge": 4,
}

METRIC_IDENTIFIER_GROUPS = {
    "has_price": {"token-price-usd"},
    "has_market_cap": {"token-market-cap-usd", "market-cap-usd"},
    "has_fdv": {"token-fdv-usd", "fdv-usd"},
    "has_revenue": {
        "rev-usd",
        "revenue-total-usd",
        "app-revenue-total-usd",
        "dex-revenue-usd",
        "lending-revenue-total-usd",
    },
    "has_fees": {
        "transaction-fee-total-usd",
        "dex-fees-usd",
        "fees-total-usd",
        "app-fees-total-usd",
        "mev-tips-fees-usd",
    },
    "has_lending_deposits": {"lending-deposit-total-usd"},
    "has_blockworks_tvl": {"lending-tvl-total-usd", "tvl-total-usd"},
}


@dataclass(frozen=True)
class EntityMatch:
    """External-source match result for one Blockworks asset."""

    source: str
    external_id: str
    method: str
    confidence: float


def normalise_text(value: Any) -> str:
    """Lowercase and remove punctuation so names/slugs can be compared."""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def normalise_symbol(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _nested_value(row: Mapping[str, Any], path: tuple[str, str]) -> Any:
    current: Any = row
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _asset_type(row: Mapping[str, Any]) -> str:
    raw_type = str(row.get("type") or "").strip().lower()
    category = str(row.get("category") or "").strip().lower()
    sector = str(row.get("sector") or "").strip().lower()
    slug = str(row.get("slug") or "").strip().lower()

    if raw_type in {"infrastructure", "chain"} or category == "infrastructure" or sector in {"l1", "l2", "l1/l2"}:
        return "chain"
    if "etf" in raw_type or "etf" in category or slug in {"ibit", "fbtc", "gbtc", "etha"}:
        return "etf"
    if "treasury" in raw_type or "treasury" in category:
        return "treasury"
    return "project"


def _metric_identifiers(metrics: Sequence[Mapping[str, Any]] | None) -> set[str]:
    if not metrics:
        return set()
    return {
        str(entry.get("identifier", "")).strip()
        for entry in metrics
        if entry.get("identifier")
    }


def summarise_metric_coverage(
    metric_catalog_by_slug: Mapping[str, Sequence[Mapping[str, Any]]] | None,
) -> pd.DataFrame:
    """
    Summarise available Blockworks metric families by project slug.

    The input is expected to be `{blockworks_slug: list_project_metrics(slug)}`.
    Missing slugs simply evaluate to no metric coverage.
    """
    rows = []
    for slug, metrics in (metric_catalog_by_slug or {}).items():
        identifiers = _metric_identifiers(metrics)
        row = {"blockworks_slug": slug}
        for column, known_identifiers in METRIC_IDENTIFIER_GROUPS.items():
            row[column] = bool(identifiers & known_identifiers)
        row["blockworks_metric_count"] = len(identifiers)
        row["blockworks_metric_identifiers"] = sorted(identifiers)
        rows.append(row)
    if not rows:
        columns = ["blockworks_slug", *METRIC_IDENTIFIER_GROUPS, "blockworks_metric_count", "blockworks_metric_identifiers"]
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def build_blockworks_asset_frame(blockworks_assets: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Normalise raw Blockworks asset rows into the canonical left-hand spine."""
    rows = []
    for row in blockworks_assets:
        slug = row.get("slug")
        if not slug:
            continue
        out = {
            "asset_id": f"bw:{slug}",
            "entity_key": slug,
            "blockworks_id": row.get("id"),
            "blockworks_slug": slug,
            "blockworks_match_slug": slug,
            "ticker": str(row.get("code") or "").upper() or pd.NA,
            "name": row.get("title") or row.get("name") or slug,
            "asset_type": _asset_type(row),
            "sector": row.get("type") or row.get("asset_class") or pd.NA,
            "category": row.get("sector") or row.get("type") or row.get("category") or pd.NA,
            "universe_source": "blockworks",
        }
        for column, path in BLOCKWORKS_MARKET_FIELDS.items():
            out[column] = pd.notna(_nested_value(row, path))
        rows.append(out)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=ASSET_MASTER_COLUMNS)

    return df.drop_duplicates("blockworks_slug").sort_values(["asset_type", "ticker", "name"]).reset_index(drop=True)


def _matches_to_frame(
    matches: Mapping[str, Mapping[str, Any]] | pd.DataFrame | None,
    slug_column: str = "blockworks_slug",
) -> pd.DataFrame:
    if matches is None:
        return pd.DataFrame(columns=[slug_column])
    if isinstance(matches, pd.DataFrame):
        return matches.copy()

    rows = []
    for slug, payload in matches.items():
        row = {slug_column: slug}
        row.update(dict(payload))
        rows.append(row)
    return pd.DataFrame(rows)


def score_name_symbol_match(
    blockworks_name: str,
    blockworks_ticker: str,
    external_name: str,
    external_symbol: str,
) -> tuple[str, float]:
    """
    Deterministic fallback scorer for non-contract matches.

    Contract/address and explicit external-id matches should be preferred before
    this scorer. The function returns a method label and confidence in [0, 1].
    """
    name_match = normalise_text(blockworks_name) == normalise_text(external_name)
    symbol_match = normalise_symbol(blockworks_ticker) == normalise_symbol(external_symbol)

    if name_match and symbol_match:
        return "exact_name_symbol", 0.92
    if name_match:
        return "exact_name", 0.82
    if symbol_match:
        return "symbol_only", 0.55
    return "unmatched", 0.0


def slugify_key(value: Any) -> str:
    """Slug-like key suitable for deterministic cache/entity ids."""
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return re.sub(r"-+", "-", text)


def _display_from_slug(slug: str) -> str:
    words = [part for part in str(slug).replace("_", "-").split("-") if part]
    return " ".join(word.upper() if len(word) <= 3 else word.capitalize() for word in words) or str(slug)


def _defillama_symbol(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text in {"", "-"}:
        return ""
    return text.upper()


def _strip_version_suffix(value: Any) -> tuple[str, bool]:
    text = str(value or "").lower()
    text = re.sub(r"[\(\)\[\],.]", " ", text)
    versioned = bool(
        re.search(
            r"\b(v\d+(?:\+\d+)?|version\s*\d+|slipstream|classic|legacy)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    text = re.sub(r"\b(v\d+(?:\+\d+)?|version\s*\d+|slipstream|classic|legacy)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return normalise_text(text), versioned


def _defillama_group_key(row: Mapping[str, Any]) -> str:
    slug = str(row.get("slug") or "").strip()
    parent = str(row.get("parentProtocol") or "").strip()
    if parent.startswith("parent#"):
        return f"parent:{parent.split('#', 1)[1]}"
    if parent:
        return f"parent:{slugify_key(parent)}"

    name_base, name_versioned = _strip_version_suffix(row.get("name"))
    slug_base, slug_versioned = _strip_version_suffix(slug.replace("-", " "))
    base = name_base or slug_base
    token_identity = normalise_text(row.get("gecko_id")) or normalise_symbol(row.get("symbol"))
    if token_identity and base and (name_versioned or slug_versioned):
        return f"heuristic:{token_identity}:{base}"
    return f"slug:{slug}"


def _defillama_is_protocol_candidate(row: Mapping[str, Any]) -> bool:
    slug = str(row.get("slug") or "").strip()
    if not slug:
        return False
    if row.get("latestFetchIsOk") is False or row.get("disabled") is True:
        return False

    symbol = normalise_symbol(row.get("symbol"))
    name = normalise_text(row.get("name"))
    category = normalise_text(row.get("category"))
    if symbol in DEFILLAMA_EXCLUDED_SYMBOLS:
        return False
    if category in {normalise_text(c) for c in DEFILLAMA_EXCLUDED_CATEGORIES}:
        return False
    if "wrapped" in name or name.startswith("staked") or "pegged" in name:
        return False
    if "aggregate" in name and _num_or_none(row.get("tvl")) is None:
        return False
    return True


def _num_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _choose_category(group: pd.DataFrame) -> Any:
    if group.empty:
        return pd.NA
    ranked = group.copy()
    ranked["_priority"] = ranked["category"].map(lambda c: DEFILLAMA_CATEGORY_PRIORITY.get(str(c).strip().lower(), 99))
    ranked["_tvl"] = pd.to_numeric(ranked["tvl"], errors="coerce").fillna(0)
    return ranked.sort_values(["_priority", "_tvl"], ascending=[True, False]).iloc[0]["category"]


def _first_valid(values: Sequence[Any]) -> Any:
    for value in values:
        if value is not None and not pd.isna(value) and str(value).strip() != "":
            return value
    return pd.NA


def _group_child_slugs(group: pd.DataFrame) -> str:
    return "|".join(group["slug"].dropna().astype(str).drop_duplicates().tolist())


def build_defillama_asset_frame(
    protocols: Sequence[Mapping[str, Any]],
    chains: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Build a DefiLlama-left asset master frame.

    Protocol variants are grouped conservatively. `parentProtocol` is preferred;
    otherwise only versioned rows sharing token identity and a normalized base
    name are grouped. Existing Blockworks-named columns are retained as
    compatibility fields for the dashboard cache contract.
    """
    protocol_rows: list[dict[str, Any]] = []
    for row in protocols or []:
        if not _defillama_is_protocol_candidate(row):
            continue
        slug = str(row.get("slug") or "").strip()
        protocol_rows.append(
            {
                "slug": slug,
                "id": row.get("id"),
                "name": row.get("name") or _display_from_slug(slug),
                "symbol": _defillama_symbol(row.get("symbol")),
                "category": row.get("category") or pd.NA,
                "chains": row.get("chains") if isinstance(row.get("chains"), list) else [],
                "tvl": _num_or_none(row.get("tvl")),
                "mcap": _num_or_none(row.get("mcap")),
                "gecko_id": row.get("gecko_id") or pd.NA,
                "parentProtocol": row.get("parentProtocol") or pd.NA,
                "group_key": _defillama_group_key(row),
            }
        )

    rows: list[dict[str, Any]] = []
    protocol_df = pd.DataFrame(protocol_rows)
    if not protocol_df.empty:
        for group_key, group in protocol_df.groupby("group_key", observed=True):
            group = group.copy()
            group["_tvl"] = pd.to_numeric(group["tvl"], errors="coerce").fillna(0)
            group = group.sort_values("_tvl", ascending=False)
            best = group.iloc[0]
            is_grouped = len(group) > 1
            if str(group_key).startswith("parent:") and is_grouped:
                canonical_slug = str(group_key).split(":", 1)[1]
            else:
                canonical_slug = str(best["slug"])

            name = best["name"]
            if is_grouped and str(group_key).startswith("parent:"):
                name = f"{_display_from_slug(canonical_slug)} (Combined)"
            elif is_grouped:
                name = f"{str(best['name']).split(' V')[0]} (Combined)"

            chains_union = sorted({chain for chains_list in group["chains"] for chain in chains_list})
            tvl = group["_tvl"].sum()
            mcap_values = pd.to_numeric(group["mcap"], errors="coerce").dropna()
            gecko_id = _first_valid(group["gecko_id"].tolist())
            ticker = _first_valid(group["symbol"].tolist())
            entity_key = canonical_slug
            rows.append(
                {
                    "asset_id": f"dl:{entity_key}",
                    "entity_key": entity_key,
                    "blockworks_id": pd.NA,
                    "blockworks_slug": entity_key,
                    "blockworks_match_slug": pd.NA,
                    "ticker": ticker,
                    "name": name,
                    "asset_type": "project",
                    "sector": "DeFi",
                    "category": _choose_category(group),
                    "coingecko_id": gecko_id,
                    "defillama_slug": canonical_slug,
                    "defillama_chain": pd.NA,
                    "defillama_child_slugs": _group_child_slugs(group),
                    "defillama_group_key": group_key,
                    "defillama_url": f"https://defillama.com/protocol/{canonical_slug}",
                    "defillama_unlocks_url": f"https://defillama.com/protocol/unlocks/{canonical_slug}",
                    "blockworks_url": pd.NA,
                    "match_method": "defillama_parent_group" if str(group_key).startswith("parent:") and is_grouped else "defillama_exact_slug",
                    "match_confidence": 1.0 if str(group_key).startswith("parent:") or not is_grouped else 0.86,
                    "primary_tvl_source": "defillama" if tvl > 0 else pd.NA,
                    "universe_source": "defillama",
                    "defillama_current_tvl": tvl if tvl > 0 else pd.NA,
                    "defillama_chains": "|".join(chains_union),
                    "defillama_mcap": mcap_values.max() if len(mcap_values) else pd.NA,
                }
            )

    for row in chains or []:
        name = row.get("name")
        symbol = _defillama_symbol(row.get("tokenSymbol"))
        gecko_id = row.get("gecko_id")
        tvl = _num_or_none(row.get("tvl"))
        if not name or not symbol or not gecko_id or symbol.lower() in DEFILLAMA_EXCLUDED_SYMBOLS:
            continue
        entity_key = f"chain:{slugify_key(name)}"
        rows.append(
            {
                "asset_id": f"dl:{entity_key}",
                "entity_key": entity_key,
                "blockworks_id": pd.NA,
                "blockworks_slug": entity_key,
                "blockworks_match_slug": pd.NA,
                "ticker": symbol,
                "name": name,
                "asset_type": "chain",
                "sector": "Infrastructure",
                "category": "L1 / L2",
                "coingecko_id": gecko_id,
                "defillama_slug": name,
                "defillama_chain": name,
                "defillama_child_slugs": pd.NA,
                "defillama_group_key": entity_key,
                "defillama_url": f"https://defillama.com/chain/{str(name).replace(' ', '%20')}",
                "defillama_unlocks_url": pd.NA,
                "blockworks_url": pd.NA,
                "match_method": "defillama_chain_gecko_id",
                "match_confidence": 1.0,
                "primary_tvl_source": "defillama" if tvl and tvl > 0 else pd.NA,
                "universe_source": "defillama",
                "defillama_current_tvl": tvl if tvl and tvl > 0 else pd.NA,
                "defillama_chains": name,
                "defillama_mcap": pd.NA,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=ASSET_MASTER_COLUMNS)

    defaults = {
        "coingecko_id": pd.NA,
        "defillama_slug": pd.NA,
        "binance_spot_symbol": pd.NA,
        "binance_futures_symbol": pd.NA,
        "has_price": False,
        "has_market_cap": False,
        "has_fdv": False,
        "has_revenue": False,
        "has_fees": False,
        "has_lending_deposits": False,
        "has_blockworks_tvl": False,
        "has_defillama_tvl": False,
        "has_open_interest": False,
    }
    for column, default in defaults.items():
        if column not in df:
            df[column] = default
        else:
            df[column] = df[column].fillna(default)
    for column in ASSET_MASTER_COLUMNS:
        if column not in df:
            df[column] = pd.NA
    sort_col = pd.to_numeric(df.get("defillama_current_tvl"), errors="coerce").fillna(0)
    return (
        df.assign(_sort_tvl=sort_col)
        .sort_values(["asset_type", "_sort_tvl", "ticker"], ascending=[True, False, True])
        .drop(columns=["_sort_tvl"])
        .reset_index(drop=True)
    )


def build_asset_master(
    blockworks_assets: Sequence[Mapping[str, Any]],
    metric_catalog_by_slug: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    coingecko_matches: Mapping[str, Mapping[str, Any]] | pd.DataFrame | None = None,
    defillama_matches: Mapping[str, Mapping[str, Any]] | pd.DataFrame | None = None,
    binance_matches: Mapping[str, Mapping[str, Any]] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build the canonical asset master from Blockworks-left inputs.

    External match inputs are optional and keyed by `blockworks_slug`. They are
    designed to be generated automatically by an entity-resolution job in the
    live wiring step, not maintained as a hand-authored asset list.
    """
    master = build_blockworks_asset_frame(blockworks_assets)
    if master.empty:
        return master

    metric_coverage = summarise_metric_coverage(metric_catalog_by_slug)
    if not metric_coverage.empty:
        metric_cols = ["blockworks_slug", *METRIC_IDENTIFIER_GROUPS.keys()]
        master = master.merge(metric_coverage[metric_cols], on="blockworks_slug", how="left", suffixes=("", "_metric"))
        for column in METRIC_IDENTIFIER_GROUPS:
            metric_col = f"{column}_metric"
            if metric_col in master:
                master[column] = master[column].fillna(False) | master[metric_col].fillna(False)
                master = master.drop(columns=[metric_col])

    source_frames = [
        _matches_to_frame(coingecko_matches),
        _matches_to_frame(defillama_matches),
        _matches_to_frame(binance_matches),
    ]
    for source_df in source_frames:
        if source_df.empty:
            continue
        master = master.merge(source_df, on="blockworks_slug", how="left")

    defaults = {
        "entity_key": pd.NA,
        "blockworks_match_slug": pd.NA,
        "coingecko_id": pd.NA,
        "defillama_slug": pd.NA,
        "defillama_chain": pd.NA,
        "defillama_child_slugs": pd.NA,
        "defillama_group_key": pd.NA,
        "defillama_url": pd.NA,
        "defillama_unlocks_url": pd.NA,
        "blockworks_url": pd.NA,
        "binance_spot_symbol": pd.NA,
        "binance_futures_symbol": pd.NA,
        "match_method": "blockworks_only",
        "match_confidence": 1.0,
        "has_price": False,
        "has_market_cap": False,
        "has_fdv": False,
        "has_revenue": False,
        "has_fees": False,
        "has_lending_deposits": False,
        "has_blockworks_tvl": False,
        "has_defillama_tvl": False,
        "has_open_interest": False,
        "primary_tvl_source": pd.NA,
    }
    for column, default in defaults.items():
        if column not in master:
            master[column] = default
        else:
            master[column] = master[column].fillna(default)

    master.loc[master["has_defillama_tvl"].astype(bool), "primary_tvl_source"] = "defillama"
    master.loc[
        ~master["has_defillama_tvl"].astype(bool) & master["has_blockworks_tvl"].astype(bool),
        "primary_tvl_source",
    ] = "blockworks"

    for column in ASSET_MASTER_COLUMNS:
        if column not in master:
            master[column] = pd.NA

    return master[ASSET_MASTER_COLUMNS].sort_values(["asset_type", "ticker", "name"]).reset_index(drop=True)


def asset_master_contract() -> pd.DataFrame:
    """Human-readable contract for Data Dictionary and handoff docs."""
    rows = [
        ("asset_id", "Internal stable key. Defaults to dl:{entity_key}.", "Generated"),
        ("entity_key", "Canonical left-hand row key from DefiLlama protocol slug or chain key.", "DefiLlama"),
        ("blockworks_id", "Optional Blockworks numeric asset identifier when matched.", "Optional Blockworks enrichment"),
        ("blockworks_slug", "Compatibility key retained for dashboard joins; DefiLlama-left rows use entity_key.", "Generated / optional Blockworks"),
        ("blockworks_match_slug", "Optional matched Blockworks slug.", "Optional Blockworks enrichment"),
        ("ticker", "Display/trading symbol.", "DefiLlama symbol, cross-checked externally"),
        ("name", "Project or chain name.", "DefiLlama metadata"),
        ("asset_type", "chain or project.", "DefiLlama protocols/chains"),
        ("sector", "Broad investment sector.", "DefiLlama category mapping"),
        ("category", "Peer benchmark group.", "DefiLlama category mapping"),
        ("coingecko_id", "Market-data join key.", "Automated entity resolution"),
        ("defillama_slug", "TVL/DeFi-data join key.", "DefiLlama"),
        ("defillama_child_slugs", "Pipe-delimited child protocol slugs used for combined rows.", "DefiLlama parent/variant grouping"),
        ("binance_spot_symbol", "Spot-liquidity join key.", "Automated symbol resolution"),
        ("binance_futures_symbol", "OI/funding join key.", "Automated symbol resolution"),
        ("match_method", "Best external match method used.", "Entity resolution"),
        ("match_confidence", "Confidence score for external joins.", "Entity resolution"),
        ("has_* fields", "Coverage booleans for major metric families.", "Source metadata + metric discovery"),
        ("primary_tvl_source", "DefiLlama primary; Blockworks may fill optional gaps.", "TVL source selection"),
        ("universe_source", "DefiLlama for the left-hand universe.", "DefiLlama"),
    ]
    return pd.DataFrame(rows, columns=["Field", "Definition", "Source"])
