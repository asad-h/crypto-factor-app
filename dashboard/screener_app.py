"""
Crypto factor screener mock.

This Streamlit app mirrors the agreed screener hierarchy using mock data only.
The next implementation step can replace load_mock_projects() with the real
pipeline output while keeping the UI, filters, and chart contracts intact.

Usage:
    streamlit run dashboard/screener_app.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import os
import sys
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crypto_factor_model.data.asset_master import asset_master_contract
from dashboard.screener_data import (
    FACTOR_BASKETS_PATH,
    FACTOR_LAG_DAYS,
    FACTOR_SCORES_PATH,
    PROJECT_TS_PATH,
    SNAPSHOT_PATH,
    SUMMARY_TS_PATH,
    load_factor_baskets,
    load_factor_scores,
    load_project_timeseries,
    load_screener_snapshot,
    load_summary_timeseries,
)
from dashboard.macro_monitor import render_macro_monitor
from dashboard.project_events import (
    HYPERLIQUID_STAKING_DOCS_URL,
    HYPURRSCAN_UNSTAKING_URL,
    NANSEN_ENDPOINTS,
    build_nansen_token_context_requests,
    compute_rolling_30d_return_correlations,
    fetch_defillama_unlock_events,
    fetch_nansen_token_context,
    fetch_snapshot_governance_events,
    hyperliquid_unstaking_context,
    source_config_for_project,
)
from crypto_factor_model.config import FAMILY_WEIGHTS, NANSEN_API_KEY


st.set_page_config(
    page_title="Crypto Factor Screener",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


FDV_BUCKETS = [
    ("Micro", 0, 100_000_000, "< $100M"),
    ("Small", 100_000_000, 500_000_000, "$100M-$500M"),
    ("Mid", 500_000_000, 1_000_000_000, "$500M-$1B"),
    ("Large", 1_000_000_000, 5_000_000_000, "$1B-$5B"),
    ("Mega", 5_000_000_000, float("inf"), "$5B+"),
]

FDV_BUCKET_ORDER = [bucket[0] for bucket in FDV_BUCKETS]
MARKET_CAP_BUCKETS = FDV_BUCKETS
MARKET_CAP_BUCKET_ORDER = [bucket[0] for bucket in MARKET_CAP_BUCKETS]

FDV_BUCKET_COLORS = {
    "Micro": "#8eed8a",
    "Small": "#5f9cff",
    "Mid": "#f2c15e",
    "Large": "#f47f6b",
    "Mega": "#b8a5ff",
    "Unknown": "#d8d2c0",
}

CHANGE_METRICS_30D = {
    "revenue_mom_pct": "Revenue",
    "fees_mom_pct": "Fees",
    "deposits_mom_pct": "Deposits",
    "oi_mom_pct": "Open interest",
}

STABLECOIN_TICKERS = {
    "BUSD",
    "CRVUSD",
    "DAI",
    "FDUSD",
    "FRAX",
    "GHO",
    "LUSD",
    "MIM",
    "PYUSD",
    "RLUSD",
    "SUSD",
    "TUSD",
    "USDC",
    "USDC.E",
    "USDD",
    "USDE",
    "USDG",
    "USDL",
    "USDP",
    "USDS",
    "USDT",
    "USDT0",
    "USDY",
}

STABLECOIN_CATEGORIES = {
    "algo-stables",
    "dual-token stablecoin",
    "stablecoin",
    "stablecoins",
}

SUMMARY_RAW_METRICS = {
    "revenue_mom_pct": ("revenue_30d", "Revenue 30D"),
    "fees_mom_pct": ("fees_30d", "Fees 30D"),
    "deposits_mom_pct": ("deposits", "Deposits"),
    "oi_mom_pct": ("open_interest", "Open interest"),
}

DISPERSION_STATS = ["Average", "Median", "99th percentile"]

METRIC_OPTIONS = {
    "revenue_30d": "Revenue 30D",
    "annualised_revenue_30d": "Revenue 30D annualised",
    "fees_30d": "Fees 30D",
    "annualised_fees_30d": "Fees 30D annualised",
    "fdv_annualised_fees": "FDV / annualised fees",
    "revenue_wow_pct": "Revenue WoW change",
    "revenue_mom_pct": "Revenue 30D change",
    "fees_wow_pct": "Fees WoW change",
    "fees_mom_pct": "Fees 30D change",
    "deposits_wow_pct": "Deposits WoW change",
    "deposits_mom_pct": "Deposits 30D change",
    "oi_wow_pct": "Open interest WoW change",
    "oi_mom_pct": "Open interest 30D change",
    "price_7d_pct": "7D price change",
    "factor_score": "Factor score",
    "factor_4w_change": "4W factor score change",
    "revenue_2w_4w_ratio": "Revenue growth 2W / 4W ratio",
}

PROJECT_POSITION_METRICS = {
    "price": "Price",
    "price_btc_corr": "Price correlation to BTC",
    "revenue_30d": "Revenue 30D",
    "annualised_revenue_30d": "Revenue 30D annualised",
    "revenue_2w_4w_ratio": "Revenue ratio 2W / 4W",
    "revenue_btc_corr": "Revenue correlation to BTC",
    "fees_30d": "Fees 30D",
    "annualised_fees_30d": "Fees 30D annualised",
    "fees_2w_4w_ratio": "Fees ratio 2W / 4W",
    "fdv_annualised_fees": "FDV / annualised fees",
    "fdv_annualised_revenue": "FDV / annualised revenue",
    "payback_period_fees": "Payback period",
    "implied_growth_rate": "Implied growth rate",
    "daily_active_users": "Daily active users",
    "revenue_per_active_address": "Revenue per active address",
    "buyback_versus_issuance": "Buyback versus issuance",
    "revenue_mom_pct": "Revenue 30D change",
    "revenue_wow_pct": "Revenue WoW change",
    "fees_mom_pct": "Fees 30D change",
    "deposits_mom_pct": "Deposits 30D change",
    "oi_mom_pct": "Open interest 30D change",
    "oi_wow_pct": "Open interest WoW change",
    "open_interest": "Open interest",
    "volume_24h": "Spot volume",
    "spot_volume_oi_ratio": "Spot volume / OI",
    "daily_pct_over_btc": "Daily % change / % BTC",
    "factor_score": "Factor score",
}

PROJECT_TIMESERIES_METRICS = [
    ("price", "Price"),
    ("price_btc_corr", "Price correlation to BTC"),
    ("annualised_fees_30d", "Fees 30D annualised"),
    ("fees_2w_4w_ratio", "Fees ratio 2W / 4W"),
    ("annualised_revenue_30d", "Revenue 30D annualised"),
    ("revenue_2w_4w_ratio", "Revenue ratio 2W / 4W"),
    ("revenue_btc_corr", "Revenue correlation to BTC"),
    ("fdv_annualised_revenue", "FDV / 30D annualised revenue"),
    ("fdv_annualised_fees", "FDV / 30D annualised fees"),
    ("payback_period_fees", "Payback period"),
    ("implied_growth_rate", "Implied growth rate"),
    ("daily_active_users", "Daily active users"),
    ("revenue_per_active_address", "Revenue per active address"),
    ("buyback_versus_issuance", "Buyback versus issuance"),
    ("daily_pct_over_btc", "Daily % change / % BTC"),
    ("open_interest", "Open interest"),
    ("volume_24h", "Spot volume"),
    ("spot_volume_oi_ratio", "Spot volume / OI"),
]

FACTOR_FAMILIES = {
    "factor_score": "Total",
    "fundamentals_score": "Fundamentals",
    "momentum_score": "Momentum",
    "flows_score": "Flows",
}

FACTOR_FAMILY_CONFIG_KEYS = {
    "fundamentals_score": "fundamentals",
    "momentum_score": "momentum",
    "flows_score": "flows",
}

FACTOR_LAG_COLUMNS = {
    score_col: f"{score_col}_lag_{FACTOR_LAG_DAYS}d"
    for score_col in FACTOR_FAMILIES
}

FACTOR_RESEARCH_NAME = "May24toMay26"
FACTOR_RESEARCH_OUTPUT_DIR = PROJECT_ROOT / "output" / "factor_evaluation"
FACTOR_RESEARCH_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / f"factor_model_walkforward_{FACTOR_RESEARCH_NAME}.ipynb"
FACTOR_RESEARCH_NOTEBOOK_TITLE = f"{FACTOR_RESEARCH_NAME} Factor Research (WIP)"
FACTOR_RESEARCH_NOTEBOOK_URL = os.getenv(
    "FACTOR_RESEARCH_NOTEBOOK_URL",
    f"https://github.com/asad-h/crypto-factor-app/blob/main/notebooks/factor_model_walkforward_{FACTOR_RESEARCH_NAME}.ipynb",
)
FACTOR_RESEARCH_FILES = {
    "plain_english": FACTOR_RESEARCH_OUTPUT_DIR / f"factor_model_walkforward_{FACTOR_RESEARCH_NAME}_plain_english_writeup.md",
    "next_steps": FACTOR_RESEARCH_OUTPUT_DIR / f"next_steps_{FACTOR_RESEARCH_NAME}.md",
    "fundamentals_ic": FACTOR_RESEARCH_OUTPUT_DIR / f"fundamentals_redef_ic_{FACTOR_RESEARCH_NAME}.csv",
    "weights": FACTOR_RESEARCH_OUTPUT_DIR / f"weight_experiment_summary_{FACTOR_RESEARCH_NAME}.csv",
    "regime": FACTOR_RESEARCH_OUTPUT_DIR / f"regime_stratified_folds_{FACTOR_RESEARCH_NAME}.csv",
    "validation": FACTOR_RESEARCH_OUTPUT_DIR / f"validation_model_selection_{FACTOR_RESEARCH_NAME}.csv",
    "test": FACTOR_RESEARCH_OUTPUT_DIR / f"untouched_test_results_{FACTOR_RESEARCH_NAME}.csv",
    "selected": FACTOR_RESEARCH_OUTPUT_DIR / f"selected_model_{FACTOR_RESEARCH_NAME}.json",
}

AXIS_OPTIONS = {
    "category": "Category",
    "fdv_bucket": "FDV bucket",
    "mcap_bucket": "Market cap bucket",
}

USD_COLUMNS = [
    "price",
    "market_cap",
    "fdv",
    "volume_24h",
    "revenue_7d_avg",
    "revenue_30d",
    "annualised_revenue_30d",
    "fees_7d_avg",
    "fees_30d",
    "annualised_fees_30d",
    "deposits",
    "tvl",
    "open_interest",
]

PCT_COLUMNS = [
    "price_7d_pct",
    "revenue_wow_pct",
    "revenue_mom_pct",
    "fees_wow_pct",
    "fees_mom_pct",
    "deposits_wow_pct",
    "deposits_mom_pct",
    "tvl_wow_pct",
    "tvl_mom_pct",
    "oi_wow_pct",
    "oi_mom_pct",
]

FILTERED_UNIVERSE_METRICS = {
    "revenue_30d": ("Revenue 30D", "usd"),
    "fees_30d": ("Fees 30D", "usd"),
    "deposits": ("Deposits", "usd"),
    "tvl": ("TVL", "usd"),
    "open_interest": ("Open Interest", "usd"),
    "volume_24h": ("Volume USD", "usd"),
}

PEER_KPI_METRICS = {
    "revenue_30d": "Revenue",
    "fees_30d": "Fees",
    "tvl": "TVL",
    "deposits": "Deposits",
    "open_interest": "Open interest",
    "volume_24h": "Volume",
    "price": "Price",
}

DEFAULT_CORRELATION_BENCHMARKS = ["BTC", "ETH", "SOL", "HYPE"]

SCREENER_COLUMNS = {
    "ticker": "Ticker",
    "project": "Project",
    "category": "Category",
    "mcap_bucket": "Market Cap Bucket",
    "fdv_bucket": "FDV Bucket",
    "price": "Price",
    "market_cap": "Market Cap",
    "fdv": "FDV",
    "revenue_7d_avg": "Rev 7D Avg",
    "revenue_30d": "Rev 30D",
    "annualised_revenue_30d": "Revenue 30D Annualised",
    "revenue_wow_pct": "Rev WoW",
    "revenue_mom_pct": "Rev 30D Chg",
    "fees_7d_avg": "Fees 7D Avg",
    "fees_30d": "Fees 30D",
    "annualised_fees_30d": "Fees 30D Annualised",
    "fees_wow_pct": "Fees WoW",
    "fees_mom_pct": "Fees 30D Chg",
    "deposits": "Deposits",
    "deposits_wow_pct": "Dep WoW",
    "deposits_mom_pct": "Dep 30D Chg",
    "tvl": "TVL",
    "tvl_wow_pct": "TVL WoW",
    "tvl_mom_pct": "TVL MoM",
    "open_interest": "OI",
    "oi_wow_pct": "OI WoW",
    "oi_mom_pct": "OI 30D Chg",
    "price_7d_pct": "Price 7D",
    "factor_score": "Factor Score",
    "fundamentals_score": "Fundamentals Score",
    "momentum_score": "Momentum Score",
    "flows_score": "Flows Score",
    "factor_4w_change": "Score 4W",
    "factor_score_lag_30d": "Factor Score 30D Lag",
    "fundamentals_score_lag_30d": "Fundamentals Score 30D Lag",
    "momentum_score_lag_30d": "Momentum Score 30D Lag",
    "flows_score_lag_30d": "Flows Score 30D Lag",
    "factor_lag_date": "Factor Lag Date",
}

SIGNAL_DETAIL_COLUMNS = [
    "Fundamentals Signal Detail",
    "Momentum Signal Detail",
    "Flows Signal Detail",
]


@dataclass(frozen=True)
class CoverageRule:
    metric: str
    source: str
    expected_categories: tuple[str, ...] | None
    gap_rule: str


COVERAGE_RULES = [
    CoverageRule(
        "Price, market cap, FDV",
        "CoinGecko enrichment, with Binance/DefiLlama price fallback where needed",
        None,
        "Required for FDV buckets and valuation screens.",
    ),
    CoverageRule(
        "Revenue",
        "DefiLlama fees/revenue summaries",
        None,
        "Required for revenue-growth and revenue-valuation signals.",
    ),
    CoverageRule(
        "Fees",
        "DefiLlama fees summaries",
        None,
        "Used for fee growth and FDV / fees benchmarks. Flag when redundant with revenue.",
    ),
    CoverageRule(
        "Deposits",
        "Blockworks lending deposit total",
        ("Lending", "Yield / Restaking", "L1 / L2"),
        "Expected for lending, collateral, restaking, and chain ecosystems. Not expected elsewhere.",
    ),
    CoverageRule(
        "TVL",
        "DefiLlama protocol/chain TVL preferred, Blockworks lending TVL fallback",
        ("Lending", "Yield / Restaking", "L1 / L2"),
        "Expected where locked capital is a core usage metric.",
    ),
    CoverageRule(
        "Open interest",
        "Binance Futures, with public DefiLlama open-interest fallback",
        None,
        "Expected only where a liquid Binance perpetual exists.",
    ),
    CoverageRule(
        "Bridge flows",
        "DefiLlama bridge flows",
        ("L1 / L2",),
        "Expected for chains and assets with meaningful bridge routing.",
    ),
]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b0c0a;
            --panel: #121410;
            --panel-2: #171a15;
            --line: #2b3128;
            --text: #f2efe4;
            --muted: #a8ad9c;
            --faint: #707866;
            --green: #8eed8a;
            --amber: #f2c15e;
            --red: #ff746d;
            --blue: #73a8ff;
            --cyan: #68d5cb;
            --violet: #b999ff;
        }
        .stApp { background: var(--bg); color: var(--text); }
        .block-container {
            max-width: min(96vw, 1840px) !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        [data-testid="stHeader"] { background: transparent; }
        h1, h2, h3 { letter-spacing: 0; }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 12px;
        }
        div[data-testid="stMetricLabel"] { color: var(--faint); }
        div[data-testid="stMetricValue"] { color: var(--text); }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 8px;
        }
        .hero-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            padding: 16px;
            margin-bottom: 12px;
        }
        .hero-card h3 { margin: 0 0 4px 0; font-size: 18px; }
        .hero-card p { margin: 0; color: var(--muted); font-size: 13px; }
        .bucket-grid {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: var(--panel);
            padding: 8px 10px;
            margin: 8px 0 14px;
            color: var(--muted);
            font-size: 12px;
        }
        .bucket-grid b { color: var(--text); }
        .note {
            color: var(--muted);
            font-size: 12px;
            margin-top: -4px;
            margin-bottom: 12px;
        }
        .footer-credit {
            color: var(--faint);
            font-size: 12px;
            text-align: center;
            padding: 18px 0 6px;
        }
        .footer-credit a { color: var(--muted); text-decoration: none; }
        .footer-credit a:hover { color: var(--text); text-decoration: underline; }
        div[data-testid="stTabs"] div[role="tablist"] button[role="tab"]:last-child,
        div[data-testid="stTabs"] div[role="tablist"] button[role="tab"]:last-child p {
            color: var(--faint) !important;
            opacity: 0.76;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def assign_value_bucket(value: float | None, buckets: list[tuple[str, float, float, str]]) -> str:
    if pd.isna(value):
        return "Unknown"
    for label, lower, upper, _ in buckets:
        if lower <= float(value) < upper:
            return label
    return "Unknown"


def assign_fdv_bucket(fdv: float | None) -> str:
    return assign_value_bucket(fdv, FDV_BUCKETS)


def assign_market_cap_bucket(market_cap: float | None) -> str:
    return assign_value_bucket(market_cap, MARKET_CAP_BUCKETS)


def compact_usd(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        scaled = value / 1_000_000_000
        return f"${scaled:.0f}B" if abs(scaled) >= 10 else f"${scaled:.1f}B"
    if abs_value >= 1_000_000:
        scaled = value / 1_000_000
        return f"${scaled:.0f}M" if abs(scaled) >= 10 else f"${scaled:.1f}M"
    if abs_value >= 1_000:
        scaled = value / 1_000
        return f"${scaled:.0f}K" if abs(scaled) >= 10 else f"${scaled:.1f}K"
    if abs_value >= 1:
        return f"${value:,.2f}"
    return f"${value:,.4f}"


def pct(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):+.1f}%"


def ratio(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.1f}x"


def pp(value: float) -> str:
    return f"{value:+.0f} pp"


def pct_style(value: Any) -> str:
    if pd.isna(value):
        return "color: #707866;"
    value = float(value)
    alpha = min(abs(value) / 30, 1) * 0.34 + 0.05
    if value > 0:
        return f"background-color: rgba(142, 237, 138, {alpha:.2f}); color: #f2efe4;"
    if value < 0:
        return f"background-color: rgba(255, 116, 109, {alpha:.2f}); color: #f2efe4;"
    return "background-color: rgba(242, 193, 94, .12); color: #f2efe4;"


def score_style(value: Any) -> str:
    if pd.isna(value):
        return "color: #707866;"
    value = float(value)
    if value > 0:
        return "background-color: rgba(142, 237, 138, .18); color: #8eed8a;"
    if value < 0:
        return "background-color: rgba(255, 116, 109, .18); color: #ff746d;"
    return "color: #f2c15e;"


def format_pct(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):+.1f}%"


def add_indexed_100_bands(fig: go.Figure, values: pd.Series) -> None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return
    max_deviation = max(abs(float(clean.max()) - 100), abs(100 - float(clean.min())), 3.0)
    pad = max_deviation * 0.18
    y_min = 100 - max_deviation - pad
    y_max = 100 + max_deviation + pad
    fig.add_hrect(
        y0=100,
        y1=y_max,
        fillcolor="rgba(142, 237, 138, 0.09)",
        line_width=0,
        layer="below",
    )
    fig.add_hrect(
        y0=y_min,
        y1=100,
        fillcolor="rgba(255, 116, 109, 0.10)",
        line_width=0,
        layer="below",
    )
    fig.add_hline(y=100, line_color="#d8d2c0", line_width=1.2, opacity=0.72)
    fig.update_yaxes(range=[y_min, y_max])


def format_number(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):+.2f}"


def format_return(value: Any) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:+.1f}%"


def winsorise_frame_values(
    frame: pd.DataFrame,
    value_col: str,
    *,
    by: list[str] | None = None,
    upper_quantile: float = 0.995,
    lower_quantile: float | None = 0.005,
) -> pd.DataFrame:
    if frame.empty or value_col not in frame:
        return frame
    out = frame.copy()
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")

    if by:
        grouped = out.groupby(by, observed=True)[value_col]
        upper = grouped.transform(lambda values: values.dropna().quantile(upper_quantile))
        lower = (
            grouped.transform(lambda values: values.dropna().quantile(lower_quantile))
            if lower_quantile is not None
            else None
        )
        out[value_col] = out[value_col].clip(lower=lower, upper=upper, axis=0)
        return out

    values = out[value_col].dropna()
    if values.empty:
        return out
    upper = values.quantile(upper_quantile)
    lower = values.quantile(lower_quantile) if lower_quantile is not None else None
    out[value_col] = out[value_col].clip(lower=lower, upper=upper)
    return out


def slugify(value: Any) -> str:
    text = str(value or "").lower()
    return "".join(ch if ch.isalnum() else "-" for ch in text).strip("-").replace("--", "-")


def present(value: Any) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() not in {"", "nan", "None"}


def project_source_links(row: pd.Series) -> str:
    defillama_url = row.get("defillama_url", pd.NA)
    if not present(defillama_url) and present(row.get("defillama_slug")):
        if str(row.get("asset_type", "")).lower() == "chain":
            chain_name = str(row.get("project") or row.get("defillama_slug")).replace(" ", "%20")
            defillama_url = f"https://defillama.com/chain/{chain_name}"
        else:
            defillama_url = f"https://defillama.com/protocol/{row['defillama_slug']}"

    links = []
    if present(defillama_url):
        links.append(f'<a href="{defillama_url}" target="_blank">DefiLlama</a>')
    unlocks_url = row.get("defillama_unlocks_url", pd.NA)
    if present(unlocks_url):
        links.append(f'<a href="{unlocks_url}" target="_blank">Unlocks</a>')
    blockworks_url = row.get("blockworks_url", pd.NA)
    if present(blockworks_url):
        links.append(f'<a href="{blockworks_url}" target="_blank">Blockworks Research</a>')
    return " · ".join(links)


def screener_cache_signature() -> tuple[tuple[str, int], ...]:
    paths = [SNAPSHOT_PATH, PROJECT_TS_PATH, SUMMARY_TS_PATH, FACTOR_SCORES_PATH, FACTOR_BASKETS_PATH]
    return tuple((path.name, path.stat().st_mtime_ns if path.exists() else 0) for path in paths)


def factor_research_signature() -> tuple[tuple[str, int], ...]:
    paths = [FACTOR_RESEARCH_NOTEBOOK_PATH, *FACTOR_RESEARCH_FILES.values()]
    return tuple((path.name, path.stat().st_mtime_ns if path.exists() else 0) for path in paths)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text()
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def _load_factor_research_cached(signature: tuple[tuple[str, int], ...]) -> dict[str, Any]:
    return {
        "selected": _read_json(FACTOR_RESEARCH_FILES["selected"]),
        "plain_english": _read_text(FACTOR_RESEARCH_FILES["plain_english"]),
        "next_steps": _read_text(FACTOR_RESEARCH_FILES["next_steps"]),
        "fundamentals_ic": _read_csv(FACTOR_RESEARCH_FILES["fundamentals_ic"]),
        "weights": _read_csv(FACTOR_RESEARCH_FILES["weights"]),
        "regime": _read_csv(FACTOR_RESEARCH_FILES["regime"]),
        "validation": _read_csv(FACTOR_RESEARCH_FILES["validation"]),
        "test": _read_csv(FACTOR_RESEARCH_FILES["test"]),
    }


def load_factor_research() -> dict[str, Any]:
    return _load_factor_research_cached(factor_research_signature())


def _lagged_factor_snapshot(factor_scores: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker"}
    score_cols = [col for col in FACTOR_FAMILIES if col in factor_scores]
    if factor_scores.empty or not required.issubset(factor_scores.columns) or not score_cols:
        return pd.DataFrame()
    scores = factor_scores.copy()
    scores["date"] = pd.to_datetime(scores["date"], errors="coerce")
    scores = scores.dropna(subset=["date", "ticker"])
    if scores.empty:
        return pd.DataFrame()
    lag_date = scores["date"].max() - pd.Timedelta(days=FACTOR_LAG_DAYS)
    lagged = (
        scores[scores["date"] <= lag_date]
        .sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False, observed=True)
        .tail(1)
    )
    if lagged.empty:
        return pd.DataFrame()
    lagged = lagged[["ticker", "date", *score_cols]].copy()
    lagged = lagged.rename(columns={col: FACTOR_LAG_COLUMNS[col] for col in score_cols})
    lagged = lagged.rename(columns={"date": "factor_lag_date"})
    return lagged.reset_index(drop=True)


def _attach_lagged_factor_scores(df: pd.DataFrame, factor_scores: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "ticker" not in df:
        return df
    lagged = _lagged_factor_snapshot(factor_scores)
    if lagged.empty:
        return df
    out = df.copy()
    lag_cols = ["factor_lag_date", *FACTOR_LAG_COLUMNS.values()]
    out = out.drop(columns=[col for col in lag_cols if col in out], errors="ignore")
    out = out.merge(lagged, on="ticker", how="left")
    return out


@st.cache_data(show_spinner=False)
def _load_projects_cached(cache_signature: tuple[tuple[str, int], ...]) -> pd.DataFrame:
    try:
        df = load_screener_snapshot()
        if not df.empty:
            try:
                return _attach_lagged_factor_scores(df, load_factor_scores())
            except Exception:
                return df
    except Exception:
        pass
    return load_mock_projects()


@st.cache_data(show_spinner=False)
def _load_cached_project_timeseries(ticker: str, cache_signature: tuple[tuple[str, int], ...]) -> pd.DataFrame:
    try:
        return load_project_timeseries(ticker)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_cached_summary_timeseries(metric: str, cache_signature: tuple[tuple[str, int], ...]) -> pd.DataFrame:
    try:
        return load_summary_timeseries(metric)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_cached_factor_baskets(cache_signature: tuple[tuple[str, int], ...]) -> pd.DataFrame:
    try:
        return load_factor_baskets()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_cached_factor_scores(cache_signature: tuple[tuple[str, int], ...]) -> pd.DataFrame:
    try:
        return load_factor_scores()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_cached_full_project_timeseries(cache_signature: tuple[tuple[str, int], ...]) -> pd.DataFrame:
    try:
        if PROJECT_TS_PATH.exists():
            ts = pd.read_parquet(PROJECT_TS_PATH)
            if not ts.empty:
                ts["date"] = pd.to_datetime(ts["date"], errors="coerce")
                return ts.sort_values(["ticker", "date"]).reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()


def load_projects() -> pd.DataFrame:
    return _load_projects_cached(screener_cache_signature())


def load_cached_project_timeseries(ticker: str = "") -> pd.DataFrame:
    return _load_cached_project_timeseries(ticker, screener_cache_signature())


def load_cached_summary_timeseries(metric: str) -> pd.DataFrame:
    return _load_cached_summary_timeseries(metric, screener_cache_signature())


def load_cached_factor_baskets() -> pd.DataFrame:
    return _load_cached_factor_baskets(screener_cache_signature())


def load_cached_factor_scores() -> pd.DataFrame:
    return _load_cached_factor_scores(screener_cache_signature())


def load_cached_full_project_timeseries() -> pd.DataFrame:
    return _load_cached_full_project_timeseries(screener_cache_signature())


@st.cache_data(show_spinner=False)
def load_mock_projects() -> pd.DataFrame:
    rows = [
        {
            "ticker": "UNI", "project": "Uniswap", "category": "DEX", "sector": "DeFi",
            "price": 14.22, "market_cap": 8_600_000_000, "fdv": 9_800_000_000,
            "volume_24h": 410_000_000, "price_7d_pct": 12.4,
            "revenue_7d_avg": 2_100_000, "revenue_wow_pct": 11.2, "revenue_mom_pct": 24.0,
            "fees_7d_avg": 4_400_000, "fees_wow_pct": 8.4, "fees_mom_pct": 18.3,
            "deposits": None, "deposits_wow_pct": None, "deposits_mom_pct": None,
            "tvl": 5_800_000_000, "tvl_wow_pct": 3.1, "tvl_mom_pct": 8.2,
            "open_interest": None, "oi_wow_pct": None,
            "factor_score": 1.31, "factor_4w_change": 0.91, "revenue_2w_4w_ratio": 1.4,
            "main_drivers": "Revenue up, fees up, below peer median",
        },
        {
            "ticker": "AAVE", "project": "Aave", "category": "Lending", "sector": "DeFi",
            "price": 128.40, "market_cap": 1_900_000_000, "fdv": 2_100_000_000,
            "volume_24h": 185_000_000, "price_7d_pct": 9.8,
            "revenue_7d_avg": 920_000, "revenue_wow_pct": 9.1, "revenue_mom_pct": 19.0,
            "fees_7d_avg": 1_300_000, "fees_wow_pct": 7.2, "fees_mom_pct": 15.0,
            "deposits": 18_400_000_000, "deposits_wow_pct": 4.2, "deposits_mom_pct": 12.1,
            "tvl": 19_200_000_000, "tvl_wow_pct": 4.8, "tvl_mom_pct": 13.4,
            "open_interest": 410_000_000, "oi_wow_pct": 7.3,
            "factor_score": 1.18, "factor_4w_change": 0.72, "revenue_2w_4w_ratio": 1.2,
            "main_drivers": "Deposits up, revenue up, below peer median",
        },
        {
            "ticker": "JUP", "project": "Jupiter", "category": "DEX Aggregator", "sector": "DeFi",
            "price": 1.48, "market_cap": 2_100_000_000, "fdv": 6_100_000_000,
            "volume_24h": 620_000_000, "price_7d_pct": 18.2,
            "revenue_7d_avg": 1_600_000, "revenue_wow_pct": 18.0, "revenue_mom_pct": 36.0,
            "fees_7d_avg": 2_800_000, "fees_wow_pct": 14.0, "fees_mom_pct": 29.0,
            "deposits": None, "deposits_wow_pct": None, "deposits_mom_pct": None,
            "tvl": 2_400_000_000, "tvl_wow_pct": 4.2, "tvl_mom_pct": 11.2,
            "open_interest": 220_000_000, "oi_wow_pct": 8.1,
            "factor_score": 0.74, "factor_4w_change": 0.46, "revenue_2w_4w_ratio": 1.7,
            "main_drivers": "Revenue up, OI up, above peer median",
        },
        {
            "ticker": "ARB", "project": "Arbitrum", "category": "L1 / L2", "sector": "Infrastructure",
            "price": 1.14, "market_cap": 4_900_000_000, "fdv": 11_400_000_000,
            "volume_24h": 510_000_000, "price_7d_pct": 16.0,
            "revenue_7d_avg": 1_100_000, "revenue_wow_pct": 21.0, "revenue_mom_pct": 47.0,
            "fees_7d_avg": 1_800_000, "fees_wow_pct": 19.0, "fees_mom_pct": 42.0,
            "deposits": 2_100_000_000, "deposits_wow_pct": 5.0, "deposits_mom_pct": 20.0,
            "tvl": 3_200_000_000, "tvl_wow_pct": 6.4, "tvl_mom_pct": 18.2,
            "open_interest": 384_000_000, "oi_wow_pct": 10.0,
            "factor_score": 1.07, "factor_4w_change": 0.84, "revenue_2w_4w_ratio": 1.8,
            "main_drivers": "Revenue up, fees up, OI up",
        },
        {
            "ticker": "PENDLE", "project": "Pendle", "category": "Yield / Restaking", "sector": "DeFi",
            "price": 6.92, "market_cap": 1_050_000_000, "fdv": 1_800_000_000,
            "volume_24h": 170_000_000, "price_7d_pct": 18.0,
            "revenue_7d_avg": 510_000, "revenue_wow_pct": 13.0, "revenue_mom_pct": 28.0,
            "fees_7d_avg": None, "fees_wow_pct": None, "fees_mom_pct": None,
            "deposits": 6_400_000_000, "deposits_wow_pct": 8.0, "deposits_mom_pct": 17.0,
            "tvl": 6_900_000_000, "tvl_wow_pct": 9.4, "tvl_mom_pct": 22.0,
            "open_interest": None, "oi_wow_pct": None,
            "factor_score": 0.92, "factor_4w_change": 0.62, "revenue_2w_4w_ratio": 1.5,
            "main_drivers": "Deposits up, revenue up, below peer median",
        },
        {
            "ticker": "DYDX", "project": "dYdX", "category": "Perps", "sector": "DeFi",
            "price": 3.12, "market_cap": 1_120_000_000, "fdv": 2_200_000_000,
            "volume_24h": 145_000_000, "price_7d_pct": 6.1,
            "revenue_7d_avg": 430_000, "revenue_wow_pct": -2.0, "revenue_mom_pct": -4.0,
            "fees_7d_avg": 540_000, "fees_wow_pct": 1.0, "fees_mom_pct": 2.0,
            "deposits": None, "deposits_wow_pct": None, "deposits_mom_pct": None,
            "tvl": 480_000_000, "tvl_wow_pct": -1.2, "tvl_mom_pct": 1.4,
            "open_interest": 290_000_000, "oi_wow_pct": 12.0,
            "factor_score": 0.36, "factor_4w_change": 0.18, "revenue_2w_4w_ratio": 0.8,
            "main_drivers": "OI up, price up, above peer median",
        },
        {
            "ticker": "GMX", "project": "GMX", "category": "Perps", "sector": "DeFi",
            "price": 48.60, "market_cap": 510_000_000, "fdv": 650_000_000,
            "volume_24h": 80_000_000, "price_7d_pct": 24.0,
            "revenue_7d_avg": 360_000, "revenue_wow_pct": 12.4, "revenue_mom_pct": 21.0,
            "fees_7d_avg": 620_000, "fees_wow_pct": 11.8, "fees_mom_pct": 25.0,
            "deposits": None, "deposits_wow_pct": None, "deposits_mom_pct": None,
            "tvl": 720_000_000, "tvl_wow_pct": 6.1, "tvl_mom_pct": 14.2,
            "open_interest": 180_000_000, "oi_wow_pct": 15.0,
            "factor_score": 0.88, "factor_4w_change": 0.68, "revenue_2w_4w_ratio": 1.6,
            "main_drivers": "Price up, revenue up, OI up",
        },
        {
            "ticker": "AERO", "project": "Aerodrome", "category": "DEX", "sector": "DeFi",
            "price": 1.22, "market_cap": 390_000_000, "fdv": 920_000_000,
            "volume_24h": 95_000_000, "price_7d_pct": 22.0,
            "revenue_7d_avg": 410_000, "revenue_wow_pct": 16.0, "revenue_mom_pct": 30.0,
            "fees_7d_avg": 790_000, "fees_wow_pct": 13.0, "fees_mom_pct": 26.0,
            "deposits": None, "deposits_wow_pct": None, "deposits_mom_pct": None,
            "tvl": 1_100_000_000, "tvl_wow_pct": 7.2, "tvl_mom_pct": 18.1,
            "open_interest": 36_000_000, "oi_wow_pct": 6.0,
            "factor_score": 0.96, "factor_4w_change": 0.82, "revenue_2w_4w_ratio": 1.7,
            "main_drivers": "Fees up, revenue up, price up",
        },
        {
            "ticker": "MORPHO", "project": "Morpho", "category": "Lending", "sector": "DeFi",
            "price": 2.70, "market_cap": 620_000_000, "fdv": 2_700_000_000,
            "volume_24h": 52_000_000, "price_7d_pct": 7.4,
            "revenue_7d_avg": 310_000, "revenue_wow_pct": 10.2, "revenue_mom_pct": 23.0,
            "fees_7d_avg": 430_000, "fees_wow_pct": 8.8, "fees_mom_pct": 20.0,
            "deposits": 4_900_000_000, "deposits_wow_pct": 6.0, "deposits_mom_pct": 16.0,
            "tvl": 5_400_000_000, "tvl_wow_pct": 5.8, "tvl_mom_pct": 16.6,
            "open_interest": 72_000_000, "oi_wow_pct": 5.0,
            "factor_score": 0.82, "factor_4w_change": 0.44, "revenue_2w_4w_ratio": 1.3,
            "main_drivers": "Deposits up, revenue up, fees up",
        },
        {
            "ticker": "ENA", "project": "Ethena", "category": "Yield / Restaking", "sector": "DeFi",
            "price": 0.88, "market_cap": 1_400_000_000, "fdv": 5_900_000_000,
            "volume_24h": 360_000_000, "price_7d_pct": 15.5,
            "revenue_7d_avg": 740_000, "revenue_wow_pct": 17.2, "revenue_mom_pct": 41.0,
            "fees_7d_avg": None, "fees_wow_pct": None, "fees_mom_pct": None,
            "deposits": 3_900_000_000, "deposits_wow_pct": 7.4, "deposits_mom_pct": 19.0,
            "tvl": 4_100_000_000, "tvl_wow_pct": 7.0, "tvl_mom_pct": 21.0,
            "open_interest": 250_000_000, "oi_wow_pct": 11.2,
            "factor_score": 0.84, "factor_4w_change": 0.74, "revenue_2w_4w_ratio": 1.4,
            "main_drivers": "Revenue up, deposits up, OI up",
        },
        {
            "ticker": "SOL", "project": "Solana", "category": "L1 / L2", "sector": "Infrastructure",
            "price": 182.0, "market_cap": 88_000_000_000, "fdv": 108_000_000_000,
            "volume_24h": 4_500_000_000, "price_7d_pct": 14.0,
            "revenue_7d_avg": 4_800_000, "revenue_wow_pct": 8.0, "revenue_mom_pct": 16.0,
            "fees_7d_avg": 5_600_000, "fees_wow_pct": 7.0, "fees_mom_pct": 14.0,
            "deposits": 9_800_000_000, "deposits_wow_pct": 2.8, "deposits_mom_pct": 9.0,
            "tvl": 10_600_000_000, "tvl_wow_pct": 3.2, "tvl_mom_pct": 10.4,
            "open_interest": 1_900_000_000, "oi_wow_pct": 9.2,
            "factor_score": 0.76, "factor_4w_change": 0.51, "revenue_2w_4w_ratio": 1.2,
            "main_drivers": "Price up, OI up, revenue up",
        },
        {
            "ticker": "ETH", "project": "Ethereum", "category": "L1 / L2", "sector": "Infrastructure",
            "price": 3_420.0, "market_cap": 412_000_000_000, "fdv": 412_000_000_000,
            "volume_24h": 13_000_000_000, "price_7d_pct": 5.2,
            "revenue_7d_avg": 10_600_000, "revenue_wow_pct": 5.4, "revenue_mom_pct": 11.0,
            "fees_7d_avg": 12_200_000, "fees_wow_pct": 4.9, "fees_mom_pct": 10.0,
            "deposits": 68_000_000_000, "deposits_wow_pct": 2.0, "deposits_mom_pct": 7.0,
            "tvl": 70_000_000_000, "tvl_wow_pct": 2.2, "tvl_mom_pct": 7.5,
            "open_interest": 6_200_000_000, "oi_wow_pct": 5.0,
            "factor_score": 0.42, "factor_4w_change": 0.25, "revenue_2w_4w_ratio": 1.3,
            "main_drivers": "Revenue stable, deposits up, OI up",
        },
        {
            "ticker": "OP", "project": "Optimism", "category": "L1 / L2", "sector": "Infrastructure",
            "price": 2.38, "market_cap": 2_600_000_000, "fdv": 10_200_000_000,
            "volume_24h": 240_000_000, "price_7d_pct": 7.8,
            "revenue_7d_avg": 720_000, "revenue_wow_pct": 6.4, "revenue_mom_pct": 18.0,
            "fees_7d_avg": 980_000, "fees_wow_pct": 5.2, "fees_mom_pct": 16.0,
            "deposits": 1_300_000_000, "deposits_wow_pct": 3.5, "deposits_mom_pct": 11.0,
            "tvl": 1_900_000_000, "tvl_wow_pct": 4.0, "tvl_mom_pct": 12.0,
            "open_interest": 260_000_000, "oi_wow_pct": 18.4,
            "factor_score": 0.68, "factor_4w_change": 0.40, "revenue_2w_4w_ratio": 1.1,
            "main_drivers": "OI up, revenue up, price up",
        },
        {
            "ticker": "MPLX", "project": "Metaplex", "category": "NFT / Consumer", "sector": "Consumer",
            "price": 0.42, "market_cap": 88_000_000, "fdv": 96_000_000,
            "volume_24h": 9_000_000, "price_7d_pct": 31.0,
            "revenue_7d_avg": 72_000, "revenue_wow_pct": 24.0, "revenue_mom_pct": 52.0,
            "fees_7d_avg": 130_000, "fees_wow_pct": 19.0, "fees_mom_pct": 44.0,
            "deposits": None, "deposits_wow_pct": None, "deposits_mom_pct": None,
            "tvl": None, "tvl_wow_pct": None, "tvl_mom_pct": None,
            "open_interest": 8_000_000, "oi_wow_pct": 9.0,
            "factor_score": 1.02, "factor_4w_change": 0.61, "revenue_2w_4w_ratio": 1.9,
            "main_drivers": "Revenue acceleration, price up, fees up",
        },
        {
            "ticker": "ZORA", "project": "Zora", "category": "L1 / L2", "sector": "Consumer",
            "price": 0.08, "market_cap": 55_000_000, "fdv": 240_000_000,
            "volume_24h": 7_400_000, "price_7d_pct": -4.2,
            "revenue_7d_avg": None, "revenue_wow_pct": None, "revenue_mom_pct": None,
            "fees_7d_avg": None, "fees_wow_pct": None, "fees_mom_pct": None,
            "deposits": None, "deposits_wow_pct": None, "deposits_mom_pct": None,
            "tvl": 62_000_000, "tvl_wow_pct": 1.1, "tvl_mom_pct": 5.0,
            "open_interest": None, "oi_wow_pct": None,
            "factor_score": -0.28, "factor_4w_change": -0.18, "revenue_2w_4w_ratio": None,
            "main_drivers": "Missing revenue and fee coverage",
        },
        {
            "ticker": "RAY", "project": "Raydium", "category": "DEX", "sector": "DeFi",
            "price": 3.94, "market_cap": 1_020_000_000, "fdv": 2_100_000_000,
            "volume_24h": 210_000_000, "price_7d_pct": 11.5,
            "revenue_7d_avg": 820_000, "revenue_wow_pct": 9.4, "revenue_mom_pct": 22.0,
            "fees_7d_avg": 1_700_000, "fees_wow_pct": 7.5, "fees_mom_pct": 19.0,
            "deposits": None, "deposits_wow_pct": None, "deposits_mom_pct": None,
            "tvl": 1_900_000_000, "tvl_wow_pct": 4.0, "tvl_mom_pct": 13.0,
            "open_interest": 120_000_000, "oi_wow_pct": 5.6,
            "factor_score": 0.64, "factor_4w_change": 0.42, "revenue_2w_4w_ratio": 1.2,
            "main_drivers": "Fees up, revenue up, price up",
        },
        {
            "ticker": "LDO", "project": "Lido", "category": "Yield / Restaking", "sector": "DeFi",
            "price": 2.12, "market_cap": 1_900_000_000, "fdv": 2_120_000_000,
            "volume_24h": 110_000_000, "price_7d_pct": 3.0,
            "revenue_7d_avg": 1_400_000, "revenue_wow_pct": 2.1, "revenue_mom_pct": 7.0,
            "fees_7d_avg": None, "fees_wow_pct": None, "fees_mom_pct": None,
            "deposits": 31_000_000_000, "deposits_wow_pct": 1.4, "deposits_mom_pct": 5.0,
            "tvl": 32_400_000_000, "tvl_wow_pct": 1.6, "tvl_mom_pct": 5.3,
            "open_interest": 130_000_000, "oi_wow_pct": 1.0,
            "factor_score": 0.12, "factor_4w_change": 0.06, "revenue_2w_4w_ratio": 1.0,
            "main_drivers": "Large deposit base, slower growth",
        },
        {
            "ticker": "SNX", "project": "Synthetix", "category": "Perps", "sector": "DeFi",
            "price": 2.62, "market_cap": 850_000_000, "fdv": 860_000_000,
            "volume_24h": 66_000_000, "price_7d_pct": 8.5,
            "revenue_7d_avg": 260_000, "revenue_wow_pct": 7.8, "revenue_mom_pct": 16.0,
            "fees_7d_avg": 420_000, "fees_wow_pct": 6.6, "fees_mom_pct": 14.0,
            "deposits": None, "deposits_wow_pct": None, "deposits_mom_pct": None,
            "tvl": 610_000_000, "tvl_wow_pct": 2.1, "tvl_mom_pct": 8.0,
            "open_interest": 92_000_000, "oi_wow_pct": 7.8,
            "factor_score": 0.44, "factor_4w_change": 0.30, "revenue_2w_4w_ratio": 1.1,
            "main_drivers": "Revenue up, OI up",
        },
    ]
    df = pd.DataFrame(rows)
    df["fdv_bucket"] = df["fdv"].map(assign_fdv_bucket)
    df["fdv_bucket"] = pd.Categorical(df["fdv_bucket"], categories=FDV_BUCKET_ORDER, ordered=True)
    df["mcap_bucket"] = df["market_cap"].map(assign_market_cap_bucket)
    df["mcap_bucket"] = pd.Categorical(df["mcap_bucket"], categories=MARKET_CAP_BUCKET_ORDER, ordered=True)
    df["revenue_30d"] = df["revenue_7d_avg"] * 30
    df["fees_30d"] = df["fees_7d_avg"] * 30
    df["annualised_revenue_30d"] = df["revenue_30d"] * (365 / 30)
    df["annualised_fees_30d"] = df["fees_30d"] * (365 / 30)
    df["oi_mom_pct"] = df["oi_wow_pct"] * 2.2
    df["fdv_annualised_fees"] = df["fdv"] / df["annualised_fees_30d"]
    df["fdv_annualised_revenue"] = df["fdv"] / df["annualised_revenue_30d"]
    df["payback_period_fees"] = df["fdv"] / df["annualised_fees_30d"]
    df["fees_2w_4w_ratio"] = (df["fees_wow_pct"] * 2 / df["fees_mom_pct"]).where(df["fees_mom_pct"].abs() > 0)
    df["spot_volume_oi_ratio"] = df["volume_24h"] / df["open_interest"]
    df.loc[df["fees_7d_avg"].isna(), "fdv_annualised_fees"] = pd.NA
    df.loc[df["revenue_7d_avg"].isna(), "fdv_annualised_revenue"] = pd.NA
    df.loc[df["fees_7d_avg"].isna(), ["payback_period_fees", "fees_2w_4w_ratio"]] = pd.NA
    df.loc[df["open_interest"].isna(), "spot_volume_oi_ratio"] = pd.NA
    df["daily_active_users"] = (
        (df["market_cap"].fillna(df["fdv"]) / 180_000)
        * (1 + df["price_7d_pct"].fillna(0) / 100)
    ).round()
    df["revenue_per_active_address"] = df["revenue_7d_avg"] / df["daily_active_users"]
    df["buyback_versus_issuance"] = df["revenue_7d_avg"] * (df["factor_score"].fillna(0) / 7)
    df["price_btc_corr"] = (0.18 + df["market_cap"].rank(pct=True) * 0.58 + df["price_7d_pct"].fillna(0) / 180).clip(-1, 1)
    df["revenue_btc_corr"] = (0.12 + df["revenue_mom_pct"].fillna(0) / 140 + df["factor_score"].fillna(0) / 12).clip(-1, 1)
    df["daily_pct_over_btc"] = df["price_7d_pct"] / 5.8
    df["implied_growth_rate"] = (
        df["fdv_annualised_revenue"]
        / df.groupby("category", observed=True)["fdv_annualised_revenue"].transform("median")
        - 1
    ) * 100
    df["blockworks_slug"] = df["project"].map(slugify)
    df["blockworks_id"] = range(1, len(df) + 1)
    df["entity_key"] = df["blockworks_slug"]
    df["asset_id"] = "dl:" + df["entity_key"]
    df["asset_type"] = df["category"].eq("L1 / L2").map({True: "chain", False: "project"})
    df["universe_source"] = "defillama"
    df["coingecko_id"] = df["blockworks_slug"]
    df["defillama_slug"] = df["blockworks_slug"].where(df["tvl"].notna(), pd.NA)
    df["defillama_url"] = "https://defillama.com/protocol/" + df["defillama_slug"].astype(str)
    df["defillama_unlocks_url"] = "https://defillama.com/protocol/unlocks/" + df["defillama_slug"].astype(str)
    df.loc[df["defillama_slug"].isna(), ["defillama_url", "defillama_unlocks_url"]] = pd.NA
    df["blockworks_match_slug"] = pd.NA
    df["blockworks_url"] = pd.NA
    df["binance_spot_symbol"] = df["ticker"] + "USDT"
    df["binance_futures_symbol"] = (df["ticker"] + "USDT").where(df["open_interest"].notna(), pd.NA)
    df["match_method"] = "mocked_auto_resolution"
    df["match_confidence"] = 0.95
    df["has_price"] = df["price"].notna()
    df["has_market_cap"] = df["market_cap"].notna()
    df["has_fdv"] = df["fdv"].notna()
    df["has_revenue"] = df["revenue_7d_avg"].notna()
    df["has_fees"] = df["fees_7d_avg"].notna()
    df["has_lending_deposits"] = df["deposits"].notna()
    df["has_defillama_tvl"] = df["tvl"].notna()
    df["has_blockworks_tvl"] = df["tvl"].notna() & df["category"].isin(["Lending", "Yield / Restaking", "L1 / L2"])
    df["has_open_interest"] = df["open_interest"].notna()
    df["primary_tvl_source"] = df["has_defillama_tvl"].map({True: "defillama", False: pd.NA})
    df["fundamentals_score"] = (
        df["revenue_mom_pct"].fillna(0) / 35
        + df["fees_mom_pct"].fillna(0) / 35
        - df["fdv_annualised_revenue"].fillna(df["fdv_annualised_revenue"].median()) / 80
    )
    df["momentum_score"] = (
        df["price_7d_pct"].fillna(0) / 18
        + df["factor_4w_change"].fillna(0)
    )
    df["flows_score"] = (
        df["deposits_mom_pct"].fillna(0) / 22
        + df["tvl_mom_pct"].fillna(0) / 22
        + df["oi_mom_pct"].fillna(0) / 28
    ) / 3
    df["factor_lag_date"] = pd.Timestamp(date(2026, 4, 29)) - pd.Timedelta(days=FACTOR_LAG_DAYS)
    for score_col, lag_col in FACTOR_LAG_COLUMNS.items():
        if score_col in df:
            df[lag_col] = df[score_col] - df["factor_4w_change"].fillna(0)
    return df.sort_values(["factor_score", "ticker"], ascending=[False, True]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_mock_timeseries(ticker: str) -> pd.DataFrame:
    end = date(2026, 4, 29)
    rows = []
    base = sum(ord(ch) for ch in ticker)
    for i in range(30):
        day = end - timedelta(days=29 - i)
        rows.append(
            {
                "date": day,
                "factor_score": -0.4 + (i / 29) * 1.6 + ((base + i) % 5 - 2) * 0.035,
                "revenue_30d_change_pct": -6 + (i / 29) * 31 + ((base + i * 3) % 7 - 3) * 0.7,
                "fees_30d_change_pct": -4 + (i / 29) * 24 + ((base + i * 5) % 7 - 3) * 0.6,
                "deposits_30d_change_pct": -2 + (i / 29) * 15 + ((base + i * 4) % 7 - 3) * 0.45,
                "oi_30d_change_pct": -8 + (i / 29) * 27 + ((base + i * 6) % 9 - 4) * 0.6,
                "price_change_pct": -4 + (i / 29) * 18 + ((base + i * 2) % 9 - 4) * 0.4,
            }
        )
    return pd.DataFrame(rows)


def _value_path(final_value: Any, seed: int, drift: float = 0.12, allow_negative: bool = False) -> list[float | None]:
    if pd.isna(final_value):
        return [None] * 30
    final = float(final_value)
    if abs(final) < 1e-9:
        return [0.0] * 30
    start = final * (1 - drift + ((seed % 9) - 4) * 0.012)
    values = []
    for i in range(30):
        progress = i / 29
        wiggle = final * (((seed + i * 7) % 9 - 4) * 0.004)
        value = start + (final - start) * progress + wiggle
        if not allow_negative:
            value = max(value, 0)
        values.append(value)
    values[-1] = final
    return values


def build_project_actual_timeseries(project: pd.Series) -> pd.DataFrame:
    real = load_cached_project_timeseries(str(project["ticker"]))
    if not real.empty:
        real = real.copy()
        real["date"] = pd.to_datetime(real["date"])
        return real

    end = date(2026, 4, 29)
    dates = [end - timedelta(days=29 - i) for i in range(30)]
    seed = sum(ord(ch) for ch in str(project["ticker"]))
    fields = {
        "price": _value_path(project["price"], seed + 1, 0.16),
        "price_btc_corr": _value_path(project["price_btc_corr"], seed + 2, 0.08, allow_negative=True),
        "annualised_fees_30d": _value_path(project["annualised_fees_30d"], seed + 3, 0.22),
        "fees_2w_4w_ratio": _value_path(project["fees_2w_4w_ratio"], seed + 4, 0.10),
        "annualised_revenue_30d": _value_path(project["annualised_revenue_30d"], seed + 5, 0.24),
        "revenue_2w_4w_ratio": _value_path(project["revenue_2w_4w_ratio"], seed + 6, 0.10),
        "revenue_btc_corr": _value_path(project["revenue_btc_corr"], seed + 7, 0.08, allow_negative=True),
        "fdv_annualised_revenue": _value_path(project["fdv_annualised_revenue"], seed + 8, 0.18),
        "fdv_annualised_fees": _value_path(project["fdv_annualised_fees"], seed + 9, 0.18),
        "payback_period_fees": _value_path(project["payback_period_fees"], seed + 10, 0.18),
        "implied_growth_rate": _value_path(project["implied_growth_rate"], seed + 11, 0.22, allow_negative=True),
        "daily_active_users": _value_path(project["daily_active_users"], seed + 12, 0.14),
        "revenue_per_active_address": _value_path(project["revenue_per_active_address"], seed + 13, 0.20),
        "buyback_versus_issuance": _value_path(project["buyback_versus_issuance"], seed + 14, 0.26, allow_negative=True),
        "daily_pct_over_btc": _value_path(project["daily_pct_over_btc"], seed + 15, 0.28, allow_negative=True),
        "open_interest": _value_path(project["open_interest"], seed + 16, 0.24),
        "volume_24h": _value_path(project["volume_24h"], seed + 17, 0.20),
        "spot_volume_oi_ratio": _value_path(project["spot_volume_oi_ratio"], seed + 18, 0.16),
    }
    out = pd.DataFrame({"date": dates})
    for field, values in fields.items():
        out[field] = values
    return out


def render_project_actual_timeseries(project: pd.Series) -> None:
    st.subheader("30D Evolution")
    indexed = st.toggle("Indexed 100", value=False, key=f"project_ts_indexed_{project['ticker']}")
    ts = build_project_actual_timeseries(project)
    cols = st.columns(3)
    for i, (metric, label) in enumerate(PROJECT_TIMESERIES_METRICS):
        with cols[i % 3]:
            values = ts[["date", metric]].dropna()
            if values.empty:
                st.markdown(f"**{label}**")
                st.info("No data")
                continue
            y_col = metric
            y_title = None
            use_indexed_bands = False
            if indexed:
                first = values[metric].dropna().iloc[0]
                if pd.notna(first) and abs(float(first)) > 1e-12:
                    values = values.copy()
                    values["indexed_value"] = values[metric] / float(first) * 100
                    y_col = "indexed_value"
                    y_title = "Indexed 100"
                    use_indexed_bands = True
            fig = px.line(
                values,
                x="date",
                y=y_col,
                color_discrete_sequence=["#8eed8a"],
            )
            fig.update_traces(line=dict(width=2.2))
            fig.update_layout(
                title=label,
                template="plotly_dark",
                height=250,
                xaxis_title=None,
                yaxis_title=y_title,
                margin=dict(l=8, r=8, t=36, b=18),
                paper_bgcolor="#121410",
                plot_bgcolor="#121410",
                hovermode="x unified",
                showlegend=False,
            )
            if use_indexed_bands:
                add_indexed_100_bands(fig, values[y_col])
            st.plotly_chart(fig, width="stretch", key=f"project_actual_ts_{project['ticker']}_{metric}")


def fdv_bucket_markdown() -> None:
    ranges = " | ".join(
        f"<b>{label}</b> {definition}"
        for label, _, _, definition in FDV_BUCKETS
    )
    st.markdown(
        f'<div class="bucket-grid">FDV and market cap buckets: {ranges}</div>',
        unsafe_allow_html=True,
    )


def render_footer_credit() -> None:
    st.markdown(
        '<div class="footer-credit">built by '
        '<a href="https://x.com/0xsloane" target="_blank">0xsloane</a></div>',
        unsafe_allow_html=True,
    )


def page_header() -> None:
    st.title("Crypto Factor Screener")


def metric_value(df: pd.DataFrame, mask: pd.Series, denominator: int) -> tuple[str, str]:
    count = int(mask.sum())
    share = count / denominator if denominator else 0
    return f"{count} / {denominator}", f"{share:.0%}"


def _share(mask: pd.Series, eligible: pd.Series | None = None) -> float:
    if eligible is None:
        eligible = mask.notna()
    denominator = int(eligible.sum())
    if denominator == 0:
        return float("nan")
    return float(mask[eligible].sum() / denominator * 100)


def build_filtered_universe_aggregate_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    real_ts = load_cached_project_timeseries("")
    if not real_ts.empty:
        tickers = set(df["ticker"].astype(str))
        real_ts = real_ts[real_ts["ticker"].astype(str).isin(tickers)].copy()
        rows = []
        for metric, (label, unit) in FILTERED_UNIVERSE_METRICS.items():
            if metric not in real_ts:
                continue
            base = real_ts.dropna(subset=[metric])
            for day, day_df in base.groupby("date", observed=True):
                values = day_df[metric].dropna()
                if values.empty:
                    continue
                rows.append({"date": day, "Metric": label, "metric": metric, "Value": float(values.sum()), "Unit": unit})
        if rows:
            return pd.DataFrame(rows)

    end = date(2026, 4, 29)
    rows = []
    for metric, (label, unit) in FILTERED_UNIVERSE_METRICS.items():
        if metric not in df:
            continue
        final = df[metric].dropna().sum()
        for i in range(30):
            day = end - timedelta(days=29 - i)
            if pd.isna(final):
                continue
            rows.append({"date": day, "Metric": label, "metric": metric, "Value": final, "Unit": unit})
    return pd.DataFrame(rows)


def _is_stablecoin_like(frame: pd.DataFrame) -> pd.Series:
    ticker = frame.get("ticker", pd.Series("", index=frame.index)).astype("object").fillna("").astype(str).str.upper()
    category = (
        frame.get("category", pd.Series("", index=frame.index))
        .astype("object")
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    return ticker.isin(STABLECOIN_TICKERS) | category.isin(STABLECOIN_CATEGORIES)


def build_top100_30d_price_distribution(df: pd.DataFrame) -> pd.DataFrame:
    try:
        real_ts = pd.read_parquet(PROJECT_TS_PATH)
    except Exception:
        real_ts = load_cached_project_timeseries("")
    if real_ts.empty or not {"ticker", "date", "price", "market_cap"}.issubset(real_ts.columns):
        return pd.DataFrame()

    tickers = set(df["ticker"].dropna().astype(str))
    work = real_ts[real_ts["ticker"].astype(str).isin(tickers)].copy()
    work = work[~_is_stablecoin_like(work)]
    work = work.dropna(subset=["date", "ticker", "price"])
    if work.empty:
        return pd.DataFrame()

    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["market_cap"] = pd.to_numeric(work["market_cap"], errors="coerce")
    work = work.dropna(subset=["date", "price"]).sort_values(["ticker", "date"])
    work["price_30d_pct"] = work.groupby("ticker", observed=True)["price"].pct_change(30) * 100.0

    latest = (
        work.dropna(subset=["price_30d_pct", "market_cap"])
        .sort_values(["ticker", "date"])
        .groupby("ticker", as_index=False, observed=True)
        .tail(1)
    )
    if latest.empty:
        return pd.DataFrame()

    top = latest.nlargest(min(100, latest["ticker"].nunique()), "market_cap").copy()
    top["price_30d_pct"] = top["price_30d_pct"].replace([float("inf"), float("-inf")], pd.NA)
    top = top.dropna(subset=["price_30d_pct"]).sort_values("price_30d_pct", ascending=False)
    return top.reset_index(drop=True)


def _render_universe_metric_chart(plot_df: pd.DataFrame, label: str, key: str, indexed: bool = False) -> None:
    if plot_df.empty:
        st.info("No eligible data for this chart.")
        return
    y_title = None
    yaxis = dict(title=None, tickprefix="$", tickformat="~s")
    hover_data = {"Value": ":$,.0f"}
    use_indexed_bands = False
    if indexed:
        plot_df = plot_df.sort_values("date").copy()
        first = plot_df["Value"].dropna().iloc[0] if plot_df["Value"].notna().any() else float("nan")
        if pd.notna(first) and abs(float(first)) > 1e-12:
            plot_df["Value"] = plot_df["Value"] / float(first) * 100
            y_title = "Indexed to 100%"
            yaxis = dict(title=y_title, ticksuffix="%")
            hover_data = {"Value": ":.1f"}
            use_indexed_bands = True
    fig = px.line(
        plot_df,
        x="date",
        y="Value",
        color_discrete_sequence=["#8eed8a"],
        hover_data=hover_data,
    )
    fig.update_traces(line=dict(width=2.4))
    fig.update_layout(
        title=label,
        template="plotly_dark",
        height=330,
        xaxis_title=None,
        yaxis=yaxis,
        margin=dict(l=10, r=20, t=42, b=20),
        paper_bgcolor="#121410",
        plot_bgcolor="#121410",
        hovermode="x unified",
        showlegend=False,
        uirevision=key,
    )
    if use_indexed_bands:
        add_indexed_100_bands(fig, plot_df["Value"])
    st.plotly_chart(fig, width="stretch", key=key)


def render_filtered_universe_evolution(df: pd.DataFrame, title: str, key: str) -> None:
    title_col, toggle_col = st.columns([3.0, 1.0])
    title_col.subheader(title)
    indexed = toggle_col.toggle("Indexed 100%", value=False, key=f"{key}_indexed")
    ts = build_filtered_universe_aggregate_timeseries(df)
    chart_specs = [(metric, label) for metric, (label, _) in FILTERED_UNIVERSE_METRICS.items()]
    for offset in range(0, len(chart_specs), 2):
        cols = st.columns(2)
        for col, (metric, label) in zip(cols, chart_specs[offset : offset + 2]):
            with col:
                _render_universe_metric_chart(
                    ts[ts["metric"] == metric].copy() if not ts.empty else pd.DataFrame(),
                    label,
                    key=f"{key}_{metric}",
                    indexed=indexed,
                )

    distribution = build_top100_30d_price_distribution(df)
    if distribution.empty:
        return
    median = float(distribution["price_30d_pct"].median())
    count = int(len(distribution))
    nbins = min(24, max(8, count // 4))
    values = pd.to_numeric(distribution["price_30d_pct"], errors="coerce").dropna()
    x_range = None
    if not values.empty:
        low = min(0.0, float(values.min()))
        high = max(0.0, float(values.max()))
        pad = max((high - low) * 0.04, 1.0)
        x_range = [low - pad, high + pad]

    left_col, right_col = st.columns(2)
    fig = px.histogram(
        distribution,
        x="price_30d_pct",
        nbins=nbins,
        color_discrete_sequence=["#f2c15e"],
        hover_data={
            "price_30d_pct": ":.1f",
            "ticker": False,
            "project": False,
            "category": False,
            "market_cap": False,
        },
    )
    fig.update_traces(
        marker_line=dict(width=1, color="#121410"),
        hovertemplate="30D price change: %{x:.1f}%<br>Tokens: %{y}<extra></extra>",
    )
    fig.add_vline(x=0, line_color="#d8d2c0", line_width=1, opacity=0.65)
    fig.add_vline(
        x=median,
        line_color="#8eed8a",
        line_width=2,
        annotation_text=f"Median {median:+.1f}%",
        annotation_position="top",
    )
    fig.update_layout(
        title=f"Top-100 30D Price Change Distribution ({count} non-stablecoins)",
        template="plotly_dark",
        height=330,
        xaxis=dict(title="30D price change", ticksuffix="%"),
        yaxis=dict(title="Token count"),
        margin=dict(l=10, r=20, t=42, b=20),
        paper_bgcolor="#121410",
        plot_bgcolor="#121410",
        hovermode="closest",
        uirevision=key,
        showlegend=False,
    )
    if x_range:
        fig.update_xaxes(range=x_range)
    with left_col:
        st.plotly_chart(fig, width="stretch", key=f"{key}_top100_30d_distribution")

    bucket_distribution = distribution.copy()
    bucket_distribution["fdv_bucket"] = bucket_distribution["fdv_bucket"].astype("object").fillna("Unknown")
    bucket_fig = px.histogram(
        bucket_distribution,
        x="price_30d_pct",
        color="fdv_bucket",
        nbins=nbins,
        category_orders={"fdv_bucket": [*FDV_BUCKET_ORDER, "Unknown"]},
        color_discrete_map=FDV_BUCKET_COLORS,
        opacity=0.42,
        hover_data={
            "price_30d_pct": ":.1f",
            "fdv_bucket": True,
            "ticker": False,
            "project": False,
            "category": False,
            "market_cap": False,
        },
    )
    bucket_fig.update_traces(
        marker_line=dict(width=0.6, color="#121410"),
        hovertemplate="%{fullData.name}<br>30D price change: %{x:.1f}%<br>Tokens: %{y}<extra></extra>",
    )
    bucket_fig.add_vline(x=0, line_color="#d8d2c0", line_width=1, opacity=0.65)
    bucket_fig.update_layout(
        title="Top-100 30D Distribution by FDV Bucket",
        template="plotly_dark",
        height=330,
        barmode="overlay",
        xaxis=dict(title="30D price change", ticksuffix="%"),
        yaxis=dict(title="Token count"),
        margin=dict(l=10, r=20, t=42, b=20),
        paper_bgcolor="#121410",
        plot_bgcolor="#121410",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        uirevision=key,
    )
    if x_range:
        bucket_fig.update_xaxes(range=x_range)
    with right_col:
        st.plotly_chart(bucket_fig, width="stretch", key=f"{key}_top100_30d_distribution_fdv")


def build_peer_kpi_growth_frame(df: pd.DataFrame, tickers: list[str], metric: str) -> pd.DataFrame:
    tickers = [str(ticker).upper() for ticker in tickers if str(ticker).strip()]
    if not tickers or metric not in PEER_KPI_METRICS:
        return pd.DataFrame()

    real = load_cached_full_project_timeseries()
    if not real.empty and {"date", "ticker", metric}.issubset(real.columns):
        base = real[real["ticker"].astype(str).str.upper().isin(tickers)][["date", "ticker", "project", metric]].copy()
    else:
        rows = []
        for _, project in df[df["ticker"].astype(str).str.upper().isin(tickers)].iterrows():
            project_ts = build_project_actual_timeseries(project)
            if project_ts.empty or metric not in project_ts:
                continue
            project_rows = project_ts[["date", metric]].copy()
            project_rows["ticker"] = str(project["ticker"]).upper()
            project_rows["project"] = project.get("project", project["ticker"])
            rows.append(project_rows)
        base = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    if base.empty:
        return pd.DataFrame()

    base = base.rename(columns={metric: "raw_value"}).copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    base["ticker"] = base["ticker"].astype(str).str.upper()
    base["raw_value"] = pd.to_numeric(base["raw_value"], errors="coerce")
    base = base.dropna(subset=["date", "ticker", "raw_value"]).sort_values(["ticker", "date"])
    if base.empty:
        return pd.DataFrame()

    first_dates = base.groupby("ticker", observed=True)["date"].min()
    if first_dates.empty:
        return pd.DataFrame()
    common_start = first_dates.max()
    base = base[base["date"] >= common_start].copy()

    frames = []
    for ticker, group in base.groupby("ticker", observed=True):
        group = group.sort_values("date").copy()
        first = group["raw_value"].dropna().iloc[0] if group["raw_value"].notna().any() else float("nan")
        if pd.isna(first) or abs(float(first)) <= 1e-12:
            continue
        group["Indexed"] = group["raw_value"] / float(first) * 100
        frames.append(group)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def render_peer_kpi_growth(df: pd.DataFrame) -> None:
    st.subheader("Peer KPI Growth")
    if df.empty:
        st.info("No projects are available after the current filters.")
        return

    eligible_tickers = df["ticker"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    if not eligible_tickers:
        st.info("No tickers are available after the current filters.")
        return
    ranked = df.copy()
    if "factor_score" in ranked:
        ranked = ranked.sort_values("factor_score", ascending=False, na_position="last")
    default_tickers = ranked["ticker"].dropna().astype(str).str.upper().drop_duplicates().head(min(5, len(eligible_tickers))).tolist()

    c1, c2 = st.columns([1, 2])
    metric = c1.selectbox(
        "KPI",
        options=list(PEER_KPI_METRICS),
        format_func=PEER_KPI_METRICS.get,
        key="peer_kpi_growth_metric",
    )
    selected_tickers = c2.multiselect(
        "Tokens",
        options=eligible_tickers,
        default=default_tickers,
        key="peer_kpi_growth_tickers",
    )
    plot_df = build_peer_kpi_growth_frame(df, selected_tickers, metric)
    if plot_df.empty:
        st.info("No eligible time series for the selected KPI and tokens.")
        return

    fig = px.line(
        plot_df,
        x="date",
        y="Indexed",
        color="ticker",
        hover_name="project",
        hover_data={"raw_value": ":,.2f", "Indexed": ":.1f", "ticker": False},
        color_discrete_sequence=["#8eed8a", "#68d5cb", "#73a8ff", "#b999ff", "#f2c15e", "#ff746d"],
    )
    fig.update_traces(line=dict(width=2.3))
    fig.update_layout(
        title=f"{PEER_KPI_METRICS[metric]} Indexed Peer Growth",
        template="plotly_dark",
        height=420,
        xaxis_title=None,
        yaxis_title="Indexed to 100",
        margin=dict(l=10, r=20, t=42, b=20),
        paper_bgcolor="#121410",
        plot_bgcolor="#121410",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    add_indexed_100_bands(fig, plot_df["Indexed"])
    st.plotly_chart(fig, width="stretch", key=f"peer_kpi_growth_{metric}_{'_'.join(selected_tickers)}")


def render_wow_distribution_by_category(df: pd.DataFrame) -> None:
    st.subheader("WoW Change Distribution by Category")
    metric_options = {
        "revenue_wow_pct": "Revenue WoW change",
        "fees_wow_pct": "Fees WoW change",
        "deposits_wow_pct": "Deposits WoW change",
        "oi_wow_pct": "Open interest WoW change",
        "price_7d_pct": "7D price change",
    }
    metric = st.selectbox(
        "Metric",
        options=list(metric_options),
        format_func=metric_options.get,
        key="summary_wow_distribution_metric",
    )
    plot_df = df.dropna(subset=["category", metric]).copy()
    if plot_df.empty:
        return

    fig = px.box(
        plot_df,
        x="category",
        y=metric,
        points="all",
        color="category",
        hover_name="ticker",
        hover_data={
            "project": True,
            "category": True,
            "fdv_bucket": True,
            "market_cap": ":$,.0f",
            metric: ":.2f",
            "factor_score": ":.2f",
        },
        color_discrete_sequence=["#8eed8a", "#68d5cb", "#73a8ff", "#b999ff", "#f2c15e", "#ff746d"],
    )
    fig.update_traces(
        boxmean=False,
        jitter=0.35,
        pointpos=0,
        marker=dict(size=7, opacity=0.72, line=dict(width=0)),
        line=dict(width=1.4),
    )
    fig.update_layout(
        template="plotly_dark",
        height=460,
        showlegend=False,
        xaxis_title="Category",
        yaxis_title=metric_options[metric],
        margin=dict(l=10, r=20, t=20, b=20),
        paper_bgcolor="#121410",
        plot_bgcolor="#121410",
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch")


def _mock_change_path(final_value: float, group_name: str, metric_col: str) -> list[float]:
    seed = sum(ord(ch) for ch in f"{group_name}-{metric_col}")
    start = final_value * 0.35 - 4 + (seed % 7)
    values = []
    for i in range(30):
        progress = i / 29
        wiggle = ((seed + i * 5) % 9 - 4) * 0.45
        values.append(start + (final_value - start) * progress + wiggle)
    values[-1] = final_value
    return values


def _project_change_timeseries_frame(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    rows = []
    end = date(2026, 4, 29)
    dates = [end - timedelta(days=29 - i) for i in range(30)]
    for _, project in df.dropna(subset=[metric_col]).iterrows():
        path = _mock_change_path(float(project[metric_col]), str(project["ticker"]), metric_col)
        for day, value in zip(dates, path):
            rows.append(
                {
                    "date": day,
                    "ticker": project["ticker"],
                    "project": project["project"],
                    "category": project["category"],
                    "fdv_bucket": project["fdv_bucket"],
                    "mcap_bucket": project["mcap_bucket"],
                    "Value": value,
                }
            )
    return pd.DataFrame(rows)


def _change_timeseries_frame(df: pd.DataFrame, metric_col: str, group_col: str | None = None) -> pd.DataFrame:
    real = load_cached_summary_timeseries(metric_col)
    if not real.empty:
        group_type = "aggregate" if group_col is None else group_col
        out = real[(real["view"] == "change") & (real["group_type"] == group_type)].copy()
        if not out.empty:
            return out[["date", "Group", "Value", "Eligible Projects", "metric_label"]].rename(
                columns={"metric_label": "Metric"}
            )

    project_ts = _project_change_timeseries_frame(df, metric_col)
    if project_ts.empty:
        return project_ts
    if group_col is None:
        project_ts["Group"] = "Aggregate"
    else:
        project_ts["Group"] = project_ts[group_col].astype(str)
    grouped = (
        project_ts.groupby(["date", "Group"], observed=True)["Value"]
        .agg(Value="mean", **{"Eligible Projects": "count"})
        .reset_index()
    )
    grouped["Metric"] = CHANGE_METRICS_30D[metric_col]
    return grouped


def _summary_raw_timeseries_frame(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str | None = None,
    *,
    winsorise_metrics: bool = False,
) -> pd.DataFrame:
    raw_metric, raw_label = SUMMARY_RAW_METRICS[metric_col]
    real = load_cached_project_timeseries("")
    if not real.empty and raw_metric in real:
        tickers = set(df["ticker"].astype(str))
        real = real[real["ticker"].astype(str).isin(tickers)].copy()
        if group_col is None:
            real["Group"] = "Aggregate"
        elif group_col in real:
            real["Group"] = real[group_col].astype(str)
        else:
            mapping = df.set_index("ticker")[group_col].astype(str)
            real["Group"] = real["ticker"].map(mapping)
        base = real.dropna(subset=[raw_metric, "Group"]).copy()
        if not base.empty:
            if winsorise_metrics:
                base = winsorise_frame_values(base, raw_metric, by=["date"])
            grouped = (
                base.groupby(["date", "Group"], observed=True)[raw_metric]
                .agg(Value="sum", **{"Eligible Projects": "count"})
                .reset_index()
                .sort_values(["Group", "date"])
            )
            grouped["Metric"] = raw_label
            return grouped

    end = date(2026, 4, 29)
    dates = [end - timedelta(days=29 - i) for i in range(30)]
    rows = []
    groups = [("Aggregate", df)] if group_col is None else df.groupby(group_col, observed=True)
    for group_name, group in groups:
        value = group[raw_metric].dropna().sum() if raw_metric in group else float("nan")
        count = int(group[raw_metric].notna().sum()) if raw_metric in group else 0
        for day in dates:
            rows.append({"date": day, "Group": str(group_name), "Value": value, "Eligible Projects": count, "Metric": raw_label})
    return pd.DataFrame(rows)


def _summary_index_timeseries_frame(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str | None = None,
    *,
    winsorise_metrics: bool = False,
) -> pd.DataFrame:
    raw = _summary_raw_timeseries_frame(df, metric_col, group_col, winsorise_metrics=winsorise_metrics)
    if raw.empty:
        return raw
    indexed = _rebase_timeseries_to_first(raw, ["Group"])
    if winsorise_metrics:
        indexed = winsorise_frame_values(indexed, "Value")
    return indexed


def _rebase_timeseries_to_first(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    frames = []
    for group_keys, group in frame.groupby(group_cols, observed=True):
        group = group.sort_values("date").copy()
        values = pd.to_numeric(group["Value"], errors="coerce")
        first = values.dropna().iloc[0] if values.notna().any() else float("nan")
        if pd.isna(first) or abs(float(first)) <= 1e-12:
            continue
        group["Value"] = (values / float(first) - 1) * 100
        if len(group_cols) == 1:
            group_key = group_keys[0] if isinstance(group_keys, tuple) else group_keys
            group[group_cols[0]] = str(group_key)
        frames.append(group)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=frame.columns)


def _summary_raw_stat_timeseries_frame(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str | None = None,
    *,
    winsorise_metrics: bool = False,
) -> pd.DataFrame:
    raw_metric, _ = SUMMARY_RAW_METRICS[metric_col]
    real = load_cached_project_timeseries("")
    if real.empty or raw_metric not in real:
        return pd.DataFrame(columns=["date", "Group", "Statistic", "Value", "Eligible Projects"])

    tickers = set(df["ticker"].astype(str))
    real = real[real["ticker"].astype(str).isin(tickers)].copy()
    if group_col is None:
        real["Group"] = "Aggregate"
    elif group_col in real:
        real["Group"] = real[group_col].astype(str)
    else:
        mapping = df.set_index("ticker")[group_col].astype(str)
        real["Group"] = real["ticker"].map(mapping)

    base = real.dropna(subset=[raw_metric, "Group"]).copy()
    if base.empty:
        return pd.DataFrame(columns=["date", "Group", "Statistic", "Value", "Eligible Projects"])
    if winsorise_metrics:
        base = winsorise_frame_values(base, raw_metric, by=["date"])

    rows = []
    for (day, group_name), group in base.groupby(["date", "Group"], observed=True):
        values = pd.to_numeric(group[raw_metric], errors="coerce").dropna()
        if values.empty:
            continue
        for stat, value in [
            ("Average", values.mean()),
            ("Median", values.median()),
            ("75th percentile", values.quantile(0.75)),
            ("99th percentile", values.quantile(0.99)),
        ]:
            rows.append(
                {
                    "date": day,
                    "Group": str(group_name),
                    "Statistic": stat,
                    "Value": value,
                    "Eligible Projects": len(values),
                }
            )
    return pd.DataFrame(rows)


def _dispersion_frame(df: pd.DataFrame, metric_col: str, group_col: str | None = None) -> pd.DataFrame:
    rows = []
    groups = [("Aggregate", df)] if group_col is None else df.groupby(group_col, observed=True)
    for group_name, group in groups:
        values = group[metric_col].dropna()
        if values.empty:
            continue
        rows.extend(
            [
                {"Group": str(group_name), "Statistic": "Median", "Value": values.median(), "Eligible Projects": len(values)},
                {"Group": str(group_name), "Statistic": "Average", "Value": values.mean(), "Eligible Projects": len(values)},
                {"Group": str(group_name), "Statistic": "75th percentile", "Value": values.quantile(0.75), "Eligible Projects": len(values)},
                {"Group": str(group_name), "Statistic": "99th percentile", "Value": values.quantile(0.99), "Eligible Projects": len(values)},
            ]
        )
    return pd.DataFrame(rows)


def _dispersion_timeseries_frame(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str | None = None,
    *,
    winsorise_metrics: bool = False,
) -> pd.DataFrame:
    raw_stats = _summary_raw_stat_timeseries_frame(df, metric_col, group_col, winsorise_metrics=winsorise_metrics)
    if not raw_stats.empty:
        rebased = _rebase_timeseries_to_first(raw_stats, ["Group", "Statistic"])
        if winsorise_metrics:
            rebased = winsorise_frame_values(rebased, "Value", by=["Statistic"])
        return rebased

    project_ts = _project_change_timeseries_frame(df, metric_col)
    if project_ts.empty:
        return project_ts
    if winsorise_metrics:
        project_ts = winsorise_frame_values(project_ts, "Value", by=["date"])
    if group_col is None:
        project_ts["Group"] = "Aggregate"
    else:
        project_ts["Group"] = project_ts[group_col].astype(str)

    rows = []
    for (day, group_name), group in project_ts.groupby(["date", "Group"], observed=True):
        values = group["Value"].dropna()
        if values.empty:
            continue
        rows.extend(
            [
                {"date": day, "Group": group_name, "Statistic": "Average", "Value": values.mean(), "Eligible Projects": len(values)},
                {"date": day, "Group": group_name, "Statistic": "Median", "Value": values.median(), "Eligible Projects": len(values)},
                {"date": day, "Group": group_name, "Statistic": "75th percentile", "Value": values.quantile(0.75), "Eligible Projects": len(values)},
                {"date": day, "Group": group_name, "Statistic": "99th percentile", "Value": values.quantile(0.99), "Eligible Projects": len(values)},
            ]
        )
    fallback = pd.DataFrame(rows)
    if fallback.empty:
        return fallback
    rebased = _rebase_timeseries_to_first(fallback, ["Group", "Statistic"])
    if winsorise_metrics:
        rebased = winsorise_frame_values(rebased, "Value", by=["Statistic"])
    return rebased


def _bar_chart(
    plot_df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
    key: str,
    category_orders: dict[str, list[str]] | None = None,
) -> None:
    if plot_df.empty:
        st.info("No eligible data for this chart.")
        return
    fig = px.bar(
        plot_df,
        x=x,
        y=y,
        color=color,
        barmode="group",
        text=y,
        hover_data={"Eligible Projects": True, y: ":.2f"},
        category_orders=category_orders,
        color_discrete_sequence=["#8eed8a", "#68d5cb", "#73a8ff", "#b999ff", "#f2c15e", "#ff746d"],
    )
    fig.update_traces(texttemplate="%{text:+.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=360,
        legend_title=None,
        xaxis_title=None,
        yaxis_title="Change",
        yaxis_ticksuffix="%",
        margin=dict(l=10, r=10, t=42, b=24),
        paper_bgcolor="#121410",
        plot_bgcolor="#121410",
    )
    st.plotly_chart(fig, width="stretch", key=key)


def _change_timeseries_chart(
    plot_df: pd.DataFrame,
    title: str,
    key: str,
    category_orders: dict[str, list[str]] | None = None,
    raw_values: bool = False,
    winsorise_metrics: bool = False,
) -> None:
    if plot_df.empty:
        st.info("No eligible data for this chart.")
        return
    if winsorise_metrics:
        plot_df = winsorise_frame_values(plot_df, "Value")
    max_abs = max(5.0, float(plot_df["Value"].abs().max())) * 1.18 if not raw_values else None
    hover_value = ":$,.0f" if raw_values else ":.2f"
    fig = px.line(
        plot_df,
        x="date",
        y="Value",
        color="Group",
        category_orders=category_orders,
        hover_data={"Eligible Projects": True, "Value": hover_value},
        color_discrete_sequence=["#8eed8a", "#68d5cb", "#73a8ff", "#b999ff", "#f2c15e", "#ff746d"],
    )
    fig.update_traces(line=dict(width=2.3))
    if raw_values:
        yaxis = dict(title="Aggregate value", tickprefix="$", tickformat="~s")
    else:
        fig.add_hrect(y0=0, y1=max_abs, fillcolor="rgba(142,237,138,0.07)", line_width=0, layer="below")
        fig.add_hrect(y0=-max_abs, y1=0, fillcolor="rgba(255,116,109,0.08)", line_width=0, layer="below")
        fig.add_hline(y=0, line_color="#8a8f84", line_width=1.2)
        yaxis = dict(title="Change from first date", ticksuffix="%", range=[-max_abs, max_abs])
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=360,
        legend_title=None,
        xaxis_title=None,
        yaxis=yaxis,
        margin=dict(l=10, r=10, t=42, b=24),
        paper_bgcolor="#121410",
        plot_bgcolor="#121410",
        hovermode="x unified",
        uirevision=key,
    )
    st.plotly_chart(fig, width="stretch", key=key)


def render_change_summary_charts(
    df: pd.DataFrame,
    metric_col: str,
    raw_values: bool = False,
    winsorise_metrics: bool = False,
) -> None:
    st.subheader("30D Change")
    mode_label = "Raw aggregate" if raw_values else "% change"
    specs = [
        (
            _summary_raw_timeseries_frame(df, metric_col, winsorise_metrics=winsorise_metrics)
            if raw_values
            else _summary_index_timeseries_frame(df, metric_col, winsorise_metrics=winsorise_metrics),
            f"Aggregate: {CHANGE_METRICS_30D[metric_col]} ({mode_label})",
            "summary_30d_change_aggregate",
            None,
        ),
        (
            _summary_raw_timeseries_frame(df, metric_col, "category", winsorise_metrics=winsorise_metrics)
            if raw_values
            else _summary_index_timeseries_frame(df, metric_col, "category", winsorise_metrics=winsorise_metrics),
            f"By Category: {CHANGE_METRICS_30D[metric_col]} ({mode_label})",
            "summary_30d_change_category",
            None,
        ),
    ]
    for offset in range(0, len(specs), 2):
        cols = st.columns(2)
        for col, (plot_df, title, key, category_orders) in zip(cols, specs[offset : offset + 2]):
            with col:
                _change_timeseries_chart(
                    plot_df,
                    title=title,
                    key=key,
                    category_orders=category_orders,
                    raw_values=raw_values,
                    winsorise_metrics=winsorise_metrics,
                )


def _dispersion_timeseries_chart(
    plot_df: pd.DataFrame,
    title: str,
    key: str,
    category_orders: dict[str, list[str]] | None = None,
    stat_filter: str | None = None,
    winsorise_metrics: bool = False,
) -> None:
    if stat_filter is not None and not plot_df.empty:
        plot_df = plot_df[plot_df["Statistic"] == stat_filter].copy()
    if plot_df.empty:
        st.info("No eligible data for this chart.")
        return
    if winsorise_metrics:
        plot_df = winsorise_frame_values(plot_df, "Value")
    max_abs = max(5.0, float(plot_df["Value"].abs().max())) * 1.18
    color_col = "Statistic" if plot_df["Group"].nunique() == 1 and stat_filter is None else "Group"
    fig = px.line(
        plot_df,
        x="date",
        y="Value",
        color=color_col,
        category_orders=category_orders,
        hover_data={"Eligible Projects": True, "Value": ":.2f"},
        color_discrete_sequence=["#8eed8a", "#68d5cb", "#73a8ff", "#b999ff", "#f2c15e", "#ff746d"],
    )
    fig.update_traces(line=dict(width=2.1))
    fig.add_hrect(y0=0, y1=max_abs, fillcolor="rgba(142,237,138,0.07)", line_width=0, layer="below")
    fig.add_hrect(y0=-max_abs, y1=0, fillcolor="rgba(255,116,109,0.08)", line_width=0, layer="below")
    fig.add_hline(y=0, line_color="#8a8f84", line_width=1.2)
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=360,
        legend_title=None,
        xaxis_title=None,
        yaxis=dict(title="Change from first date", ticksuffix="%", range=[-max_abs, max_abs]),
        margin=dict(l=10, r=10, t=42, b=24),
        paper_bgcolor="#121410",
        plot_bgcolor="#121410",
        hovermode="x unified",
        uirevision=key,
    )
    st.plotly_chart(fig, width="stretch", key=key)


def render_dispersion_charts(df: pd.DataFrame, metric_col: str, winsorise_metrics: bool = False) -> None:
    st.subheader("Dispersion")
    st.caption("Each statistic is indexed to the first visible date, so all series begin at 0%.")
    aggregate = _dispersion_timeseries_frame(df, metric_col, winsorise_metrics=winsorise_metrics)
    by_category = _dispersion_timeseries_frame(df, metric_col, "category", winsorise_metrics=winsorise_metrics)
    for stat in DISPERSION_STATS:
        st.markdown(f"**{stat}**")
        stat_key = stat.lower().replace(" ", "_").replace("th_", "th_")
        specs = [
            (aggregate, f"Aggregate: {CHANGE_METRICS_30D[metric_col]}", f"summary_dispersion_aggregate_{stat_key}", None),
            (by_category, f"By Category: {CHANGE_METRICS_30D[metric_col]}", f"summary_dispersion_category_{stat_key}", None),
        ]
        for offset in range(0, len(specs), 2):
            cols = st.columns(2)
            for col, (plot_df, title, key, category_orders) in zip(cols, specs[offset : offset + 2]):
                with col:
                    _dispersion_timeseries_chart(
                        plot_df,
                        title=title,
                        key=key,
                        category_orders=category_orders,
                        stat_filter=stat,
                        winsorise_metrics=winsorise_metrics,
                    )


def render_executive_summary(df: pd.DataFrame) -> None:
    control_metric, control_axis, control_winsor = st.columns([2.0, 1.0, 1.2])
    metric_col = control_metric.selectbox(
        "Metric",
        options=list(CHANGE_METRICS_30D),
        index=0,
        format_func=CHANGE_METRICS_30D.get,
        key="summary_metric_v2",
    )
    raw_values = control_axis.toggle("Raw aggregate values", value=False, key="summary_raw_aggregate_values")
    winsorise_metrics = control_winsor.toggle(
        "Winsorise at 99.5%",
        value=True,
        key="summary_winsorise_metrics_995",
        help="Caps charted metric values at the 99.5th percentile to keep extreme one-off observations from dominating the executive-summary charts.",
    )

    kpi_cols = st.columns(4)
    for col, (kpi_metric_col, metric_label) in zip(kpi_cols, CHANGE_METRICS_30D.items()):
        aggregate_change = _summary_index_timeseries_frame(df, kpi_metric_col, winsorise_metrics=winsorise_metrics)
        latest = aggregate_change[aggregate_change["Group"] == "Aggregate"].sort_values("date").tail(1)
        value = latest["Value"].iloc[0] if not latest.empty else float("nan")
        eligible = int(latest["Eligible Projects"].iloc[0]) if not latest.empty else 0
        col.metric(
            f"{metric_label} 30D Change",
            "n/a" if pd.isna(value) else f"{value:+.1f}%",
            delta=f"{eligible} / {len(df)} projects",
        )

    fdv_bucket_markdown()

    render_change_summary_charts(df, metric_col, raw_values=raw_values, winsorise_metrics=winsorise_metrics)
    render_dispersion_charts(df, metric_col, winsorise_metrics=winsorise_metrics)


def filter_projects(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns([1.2, 1.2, 1.2])
        categories = f1.multiselect(
            "Category",
            sorted(df["category"].dropna().unique()),
            default=sorted(df["category"].dropna().unique()),
        )
        fdv_buckets = f2.multiselect(
            "FDV bucket",
            FDV_BUCKET_ORDER,
            default=FDV_BUCKET_ORDER,
        )
        mcap_buckets = f3.multiselect(
            "Market cap bucket",
            MARKET_CAP_BUCKET_ORDER,
            default=MARKET_CAP_BUCKET_ORDER,
        )
        f4, f5, f6 = st.columns([1.2, 1.2, 1])
        min_revenue_ann_m = f4.number_input(
            "Min revenue 30D annualised ($M)",
            min_value=0.0,
            value=0.0,
            step=10.0,
        )
        min_fees_ann_m = f5.number_input(
            "Min fees 30D annualised ($M)",
            min_value=0.0,
            value=0.0,
            step=10.0,
        )
        show_missing = f6.toggle("Show rows with data gaps", value=True)

    filtered = df[
        df["category"].isin(categories)
        & df["fdv_bucket"].astype(str).isin(fdv_buckets)
        & df["mcap_bucket"].astype(str).isin(mcap_buckets)
    ]
    if min_revenue_ann_m > 0:
        filtered = filtered[filtered["annualised_revenue_30d"] >= min_revenue_ann_m * 1_000_000]
    if min_fees_ann_m > 0:
        filtered = filtered[filtered["annualised_fees_30d"] >= min_fees_ann_m * 1_000_000]
    if not show_missing:
        filtered = filtered[filtered[["revenue_7d_avg", "fees_7d_avg", "fdv"]].notna().all(axis=1)]
    return filtered.copy()


def render_screener(df: pd.DataFrame) -> pd.DataFrame:
    st.markdown(
        '<div class="hero-card"><h3>Tokens Screener</h3>'
        '<p>One row per project. Change filters, inspect project-level metrics, then use the distribution explorer below.</p></div>',
        unsafe_allow_html=True,
    )
    filtered = filter_projects(df)
    render_filtered_universe_evolution(filtered, "30D Filtered Universe Evolution", "screener_filtered_universe")
    render_peer_kpi_growth(filtered)
    return filtered


def render_screener_table(filtered: pd.DataFrame) -> None:
    st.subheader("Ticker Table")
    table_cols = [
        "ticker", "project", "category", "mcap_bucket", "fdv_bucket", "price", "market_cap", "fdv",
        "revenue_7d_avg", "revenue_30d", "annualised_revenue_30d", "revenue_wow_pct", "revenue_mom_pct",
        "fees_7d_avg", "fees_30d", "annualised_fees_30d", "fees_wow_pct", "fees_mom_pct",
        "deposits", "deposits_wow_pct", "deposits_mom_pct",
        "tvl", "tvl_wow_pct", "tvl_mom_pct",
        "open_interest", "oi_wow_pct", "oi_mom_pct", "price_7d_pct",
        "factor_score", "fundamentals_score", "momentum_score", "flows_score", "factor_4w_change",
        "factor_lag_date", "factor_score_lag_30d", "fundamentals_score_lag_30d",
        "momentum_score_lag_30d", "flows_score_lag_30d",
    ]
    table_cols = [col for col in table_cols if col in filtered]
    display = filtered[table_cols].rename(columns=SCREENER_COLUMNS).copy()
    signal_details = (
        filtered.apply(signal_detail_cells, axis=1, result_type="expand")
        if not filtered.empty
        else pd.DataFrame(columns=SIGNAL_DETAIL_COLUMNS, index=filtered.index)
    )
    display = pd.concat([display, signal_details], axis=1)
    usd_labels = [SCREENER_COLUMNS[col] for col in USD_COLUMNS if col in table_cols]
    pct_labels = [SCREENER_COLUMNS[col] for col in PCT_COLUMNS if col in table_cols]
    score_labels = [
        label
        for label in [
            "Factor Score",
            "Fundamentals Score",
            "Momentum Score",
            "Flows Score",
            "Score 4W",
            "Factor Score 30D Lag",
            "Fundamentals Score 30D Lag",
            "Momentum Score 30D Lag",
            "Flows Score 30D Lag",
        ]
        if label in display
    ]
    formatters = {label: compact_usd for label in usd_labels}
    formatters.update({label: format_number for label in score_labels})
    formatters.update({label: format_pct for label in pct_labels})
    column_config = {
        col: st.column_config.TextColumn(col, width="large")
        for col in SIGNAL_DETAIL_COLUMNS
        if col in display
    }
    column_config.update(
        {
            col: st.column_config.NumberColumn(col, width="small")
            for col in usd_labels
            if col in display
        }
    )
    styled = (
        display.style
        .format(formatters, na_rep="n/a")
        .bar(
            subset=pct_labels,
            align="mid",
            color=["rgba(255, 116, 109, 0.30)", "rgba(142, 237, 138, 0.28)"],
            vmin=-50,
            vmax=50,
        )
        .map(score_style, subset=score_labels)
    )
    st.dataframe(
        styled,
        hide_index=True,
        width="stretch",
        height=min(620, 90 + len(filtered) * 36),
        column_config=column_config,
    )


def render_distribution_explorer(df: pd.DataFrame) -> None:
    st.subheader("Distribution Explorer")
    st.caption(
        "Box = middle 50% of projects, whisker = observed range, dots = individual projects. "
        "Hover any dot to see the project name and selected metric."
    )
    c1, c2 = st.columns([1, 1])
    x_axis = c1.selectbox("X-axis", options=list(AXIS_OPTIONS), format_func=AXIS_OPTIONS.get)
    y_axis = c2.selectbox("Y-axis", options=list(METRIC_OPTIONS), format_func=METRIC_OPTIONS.get)

    plot_df = df.dropna(subset=[x_axis, y_axis]).copy()
    if plot_df.empty:
        st.info("No projects have this metric after the current filters.")
        return

    plot_df["hover_value"] = plot_df[y_axis]
    fig = px.box(
        plot_df,
        x=y_axis,
        y=x_axis,
        points="all",
        color=x_axis,
        orientation="h",
        hover_name="ticker",
        hover_data={
            "project": True,
            "category": True,
            "mcap_bucket": True,
            "fdv_bucket": True,
            "fdv": ":$,.0f",
            y_axis: ":.2f",
            "factor_score": ":.2f",
            x_axis: False,
        },
        category_orders={"fdv_bucket": FDV_BUCKET_ORDER, "mcap_bucket": MARKET_CAP_BUCKET_ORDER},
        color_discrete_sequence=["#8eed8a", "#68d5cb", "#73a8ff", "#b999ff", "#f2c15e", "#ff746d"],
    )
    fig.update_traces(
        boxmean=False,
        jitter=0.35,
        pointpos=0,
        marker=dict(size=7, opacity=0.72, line=dict(width=0)),
        line=dict(width=1.4),
    )

    fig.update_layout(
        template="plotly_dark",
        height=520,
        showlegend=False,
        xaxis_title=METRIC_OPTIONS[y_axis],
        yaxis_title=AXIS_OPTIONS[x_axis],
        margin=dict(l=10, r=20, t=20, b=20),
        paper_bgcolor="#121410",
        plot_bgcolor="#121410",
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch")


def _selection_points(event: Any) -> list[dict]:
    """Return selected Plotly points from Streamlit's selection event shape."""
    if event is None:
        return []
    selection = event.get("selection", {}) if isinstance(event, dict) else getattr(event, "selection", {})
    if selection is None:
        return []
    if isinstance(selection, dict):
        return selection.get("points", []) or []
    return getattr(selection, "points", []) or []


def _peer_normalise(values: pd.Series, selected_value: Any | None = None) -> tuple[pd.Series, float | None]:
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    iqr = numeric.quantile(0.75) - numeric.quantile(0.25)
    scale = iqr if pd.notna(iqr) and abs(iqr) > 1e-9 else numeric.std()
    if pd.isna(scale) or abs(scale) <= 1e-9:
        scale = 1.0
    selected_norm = None
    if selected_value is not None and pd.notna(selected_value):
        selected_norm = (float(selected_value) - median) / scale
    return (numeric - median) / scale, selected_norm


def render_category_position(df: pd.DataFrame, selected_ticker: str) -> None:
    selected = df[df["ticker"] == selected_ticker].iloc[0]
    st.subheader("Category Position")
    category_options = sorted(df["category"].dropna().unique())
    default_category_index = category_options.index(selected["category"]) if selected["category"] in category_options else 0
    benchmark_category = st.selectbox(
        "Benchmark category",
        options=category_options,
        index=default_category_index,
        key=f"project_benchmark_category_{selected_ticker}",
    )

    available_metrics = [
        (metric, label)
        for metric, label in PROJECT_POSITION_METRICS.items()
        if df[(df["category"] == benchmark_category) & df[metric].notna()].shape[0] > 0
    ]
    if not available_metrics:
        st.info("No benchmark metrics available for this category.")
        return

    fig = go.Figure()
    for metric, label in available_metrics:
        peers = df[(df["category"] == benchmark_category) & df[metric].notna()].copy()
        peers = peers.sort_values("ticker").reset_index(drop=True)
        normalised, selected_norm = _peer_normalise(peers[metric], selected[metric])
        custom = peers[["ticker", "project", "category", "fdv_bucket", metric]].to_numpy()
        fig.add_trace(
            go.Box(
                x=normalised,
                y=[label] * len(peers),
                orientation="h",
                name=label,
                boxpoints="all",
                jitter=0.34,
                pointpos=0,
                fillcolor="rgba(142,237,138,0.13)",
                line=dict(color="#8eed8a", width=1.4),
                marker=dict(size=7, color="rgba(185,190,180,0.68)", line=dict(width=0)),
                customdata=custom,
                hovertemplate=(
                    "<b>%{customdata[0]} - %{customdata[1]}</b><br>"
                    "Category: %{customdata[2]}<br>"
                    "FDV bucket: %{customdata[3]}<br>"
                    f"{label}: %{{customdata[4]:.2f}}<br>"
                    "Peer-normalised: %{x:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
        if selected_norm is not None:
            fig.add_trace(
                go.Scatter(
                    x=[selected_norm],
                    y=[label],
                    mode="markers",
                    name=selected_ticker,
                    marker=dict(size=14, color="#73a8ff", line=dict(width=2, color="#f2efe4")),
                    customdata=[[
                        selected["ticker"],
                        selected["project"],
                        selected["category"],
                        selected["fdv_bucket"],
                        selected[metric],
                    ]],
                    hovertemplate=(
                        "<b>%{customdata[0]} - %{customdata[1]}</b><br>"
                        "Category: %{customdata[2]}<br>"
                        "FDV bucket: %{customdata[3]}<br>"
                        f"{label}: %{{customdata[4]:.2f}}<br>"
                        "Peer-normalised: %{x:.2f}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    fig.add_vline(x=0, line_color="#8a8f84", line_width=1.1)
    fig.update_layout(
        template="plotly_dark",
        height=max(520, 64 + len(available_metrics) * 46),
        xaxis_title="Peer-normalised value (0 = benchmark median)",
        yaxis_title=None,
        margin=dict(l=10, r=20, t=20, b=20),
        paper_bgcolor="#121410",
        plot_bgcolor="#121410",
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch", key=f"category_position_all_metrics_{selected_ticker}_{benchmark_category}")


def render_fdv_bucket_position(df: pd.DataFrame, selected_ticker: str) -> None:
    selected = df[df["ticker"] == selected_ticker].iloc[0]
    st.subheader("FDV Bucket Position")
    metric = st.radio(
        "Metric",
        options=list(PROJECT_POSITION_METRICS),
        format_func=PROJECT_POSITION_METRICS.get,
        horizontal=True,
        key=f"fdv_bucket_metric_{selected_ticker}",
    )
    plot_df = df.dropna(subset=["fdv_bucket", metric]).copy()
    if plot_df.empty:
        st.info("No projects have this metric.")
        return
    fig = px.box(
        plot_df,
        x=metric,
        y="fdv_bucket",
        orientation="h",
        points="all",
        color="fdv_bucket",
        hover_name="ticker",
        hover_data={
            "project": True,
            "category": True,
            "fdv": ":$,.0f",
            metric: ":.2f",
            "fdv_bucket": False,
        },
        category_orders={"fdv_bucket": FDV_BUCKET_ORDER},
        color_discrete_sequence=["#8eed8a", "#68d5cb", "#73a8ff", "#b999ff", "#f2c15e"],
    )
    fig.update_traces(
        boxmean=False,
        jitter=0.35,
        pointpos=0,
        marker=dict(size=7, opacity=0.7, line=dict(width=0)),
        line=dict(width=1.4),
    )
    if pd.notna(selected[metric]):
        fig.add_trace(
            go.Scatter(
                x=[selected[metric]],
                y=[selected["fdv_bucket"]],
                mode="markers+text",
                text=[selected_ticker],
                textposition="middle right",
                marker=dict(size=16, color="#73a8ff", line=dict(width=2, color="#f2efe4")),
                showlegend=False,
                hovertemplate=(
                    f"<b>{selected_ticker} - {selected['project']}</b><br>"
                    f"{PROJECT_POSITION_METRICS[metric]}: %{{x:.2f}}<br>"
                    "FDV bucket: %{y}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=420,
        showlegend=False,
        xaxis_title=PROJECT_POSITION_METRICS[metric],
        yaxis_title="FDV bucket",
        margin=dict(l=10, r=20, t=20, b=20),
        paper_bgcolor="#121410",
        plot_bgcolor="#121410",
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch", key=f"fdv_bucket_position_{selected_ticker}_{metric}")


def render_project_peer_position(df: pd.DataFrame, selected_ticker: str) -> None:
    selected = df[df["ticker"] == selected_ticker].iloc[0]
    st.subheader("Peer Positioning")
    metric = st.selectbox(
        "Metric",
        options=list(PROJECT_POSITION_METRICS),
        format_func=PROJECT_POSITION_METRICS.get,
        key=f"project_peer_metric_{selected_ticker}",
    )
    category_peers = df[(df["category"] == selected["category"]) & df[metric].notna()].copy()
    fdv_peers = df[(df["category"] == selected["category"]) & df[metric].notna()].copy()
    c1, c2 = st.columns(2)

    with c1:
        if category_peers.empty:
            st.info("No category benchmark data.")
        else:
            fig = px.box(
                category_peers,
                x=metric,
                y="category",
                orientation="h",
                points="all",
                hover_name="ticker",
                hover_data={
                    "project": True,
                    "fdv_bucket": True,
                    metric: ":.2f",
                    "category": False,
                },
                color_discrete_sequence=["#8eed8a"],
            )
            fig.update_traces(
                boxmean=False,
                jitter=0.34,
                pointpos=0,
                fillcolor="rgba(142,237,138,0.13)",
                line=dict(color="#8eed8a", width=1.5),
                marker=dict(size=8, opacity=0.68, line=dict(width=0)),
            )
            if pd.notna(selected[metric]):
                fig.add_trace(
                    go.Scatter(
                        x=[selected[metric]],
                        y=[selected["category"]],
                        mode="markers+text",
                        text=[selected_ticker],
                        textposition="middle right",
                        marker=dict(size=16, color="#73a8ff", line=dict(width=2, color="#f2efe4")),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{selected_ticker} - {selected['project']}</b><br>"
                            f"{PROJECT_POSITION_METRICS[metric]}: %{{x:.2f}}<br>"
                            "Category: %{y}<extra></extra>"
                        ),
                    )
                )
            fig.update_layout(
                title="Peer Position within Category",
                template="plotly_dark",
                height=360,
                showlegend=False,
                xaxis_title=PROJECT_POSITION_METRICS[metric],
                yaxis_title=None,
                margin=dict(l=10, r=20, t=42, b=20),
                paper_bgcolor="#121410",
                plot_bgcolor="#121410",
                hovermode="closest",
            )
            st.plotly_chart(fig, width="stretch", key=f"category_raw_position_{selected_ticker}_{metric}")

    with c2:
        if fdv_peers.empty:
            st.info("No FDV bucket data.")
        else:
            fig = px.box(
                fdv_peers,
                x=metric,
                y="fdv_bucket",
                orientation="h",
                points="all",
                color="fdv_bucket",
                hover_name="ticker",
                hover_data={
                    "project": True,
                    "category": True,
                    metric: ":.2f",
                    "fdv_bucket": False,
                },
                category_orders={"fdv_bucket": FDV_BUCKET_ORDER},
                color_discrete_sequence=["#8eed8a", "#68d5cb", "#73a8ff", "#b999ff", "#f2c15e"],
            )
            fig.update_traces(
                boxmean=False,
                jitter=0.34,
                pointpos=0,
                marker=dict(size=8, opacity=0.68, line=dict(width=0)),
                line=dict(width=1.4),
            )
            if pd.notna(selected[metric]):
                fig.add_trace(
                    go.Scatter(
                        x=[selected[metric]],
                        y=[selected["fdv_bucket"]],
                        mode="markers+text",
                        text=[selected_ticker],
                        textposition="middle right",
                        marker=dict(size=16, color="#73a8ff", line=dict(width=2, color="#f2efe4")),
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{selected_ticker} - {selected['project']}</b><br>"
                            f"{PROJECT_POSITION_METRICS[metric]}: %{{x:.2f}}<br>"
                            "FDV bucket: %{y}<extra></extra>"
                        ),
                    )
                )
            fig.update_layout(
                title="FDV bucket distribution within Category",
                template="plotly_dark",
                height=360,
                showlegend=False,
                xaxis_title=PROJECT_POSITION_METRICS[metric],
                yaxis_title=None,
                margin=dict(l=10, r=20, t=42, b=20),
                paper_bgcolor="#121410",
                plot_bgcolor="#121410",
                hovermode="closest",
            )
            st.plotly_chart(fig, width="stretch", key=f"fdv_raw_position_{selected_ticker}_{metric}")


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_project_unlock_events_cached(slug: str, project: str, ticker: str, source_url: str) -> pd.DataFrame:
    return fetch_defillama_unlock_events(slug, project=project, ticker=ticker, source_url=source_url)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_snapshot_governance_cached(
    space_id: str,
    project: str,
    ticker: str,
    defillama_slug: str,
    governance_url: str,
) -> pd.DataFrame:
    return fetch_snapshot_governance_events(
        space_id,
        project=project,
        ticker=ticker,
        defillama_slug=defillama_slug,
        governance_url=governance_url,
    )


def render_project_unlocks(row: pd.Series) -> None:
    st.subheader("Emissions and Unlocks")
    config = source_config_for_project(row)
    slug = str(config.get("defillama_slug") or row.get("defillama_slug") or "").strip()
    source_url = str(config.get("unlock_url") or row.get("defillama_unlocks_url") or "")
    if not slug or not source_url:
        st.info("No DefiLlama unlock source is configured for this project.")
        return

    events = _load_project_unlock_events_cached(slug, str(row["project"]), str(row["ticker"]), source_url)
    if events.empty:
        st.info("No parsable unlock events are available from the public DefiLlama page. The source link is still available for manual review.")
        st.markdown(f"[Open DefiLlama unlocks]({source_url})")
        return

    today = pd.Timestamp.today().normalize()
    upcoming = events[pd.to_datetime(events["date"], errors="coerce") >= today].sort_values("date")
    potential = upcoming[upcoming["bucket"].eq("Potential Selling")]
    c1, c2, c3 = st.columns(3)
    c1.metric("Upcoming Unlocks", str(len(upcoming)))
    c2.metric("Potential Selling Value", compact_usd(potential["value_usd"].dropna().sum() if not potential.empty else pd.NA))
    c3.metric("Next Unlock", upcoming["date"].dt.strftime("%Y-%m-%d").iloc[0] if not upcoming.empty else "n/a")

    display = events.copy()
    display["Date"] = pd.to_datetime(display["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    display = display.rename(
        columns={
            "recipient": "Recipient",
            "bucket": "Bucket",
            "token_amount": "Token Amount",
            "token_symbol": "Token",
            "value_usd": "USD Value",
            "pct_supply": "% Supply",
            "pct_float": "% Float",
            "notes": "Notes",
            "source_url": "Source URL",
        }
    )
    display_cols = ["Date", "Recipient", "Bucket", "Token Amount", "Token", "USD Value", "% Supply", "% Float", "Notes", "Source URL"]
    st.dataframe(
        display[[col for col in display_cols if col in display]],
        hide_index=True,
        width="stretch",
        column_config={
            "Token Amount": st.column_config.NumberColumn("Token Amount", format="%.0f"),
            "USD Value": st.column_config.NumberColumn("USD Value", format="$%.2f"),
            "% Supply": st.column_config.NumberColumn("% Supply", format="%.3f%%"),
            "% Float": st.column_config.NumberColumn("% Float", format="%.3f%%"),
            "Source URL": st.column_config.LinkColumn("Source URL"),
        },
    )


def render_project_unstaking_flows(row: pd.Series) -> None:
    config = source_config_for_project(row)
    if str(config.get("flow_adapter", "")).lower() != "hyperliquid" and str(row.get("ticker", "")).upper() != "HYPE":
        return

    st.subheader("Flows: Staking and Unstaking")
    events = hyperliquid_unstaking_context(str(row.get("project", "Hyperliquid")), str(row.get("ticker", "HYPE")))
    c1, c2, c3 = st.columns(3)
    c1.metric("Delegation Lock", "1 day")
    c2.metric("Unstaking Queue", "7 days")
    c3.metric("Pending Withdrawals", "Max 5/address")
    st.caption(
        "Live Hypurrscan queue parsing is source-dependent; mechanics are shown from the official Hyperliquid staking docs."
    )
    display = events.rename(columns={"title": "Item", "notes": "Detail", "source_url": "Source URL"})
    st.dataframe(
        display[["Item", "Detail", "Source URL"]],
        hide_index=True,
        width="stretch",
        column_config={"Source URL": st.column_config.LinkColumn("Source URL")},
    )
    st.markdown(f"[Hypurrscan unstaking]({HYPURRSCAN_UNSTAKING_URL}) · [Hyperliquid staking docs]({HYPERLIQUID_STAKING_DOCS_URL})")


def render_project_governance(row: pd.Series) -> None:
    st.subheader("Governance")
    config = source_config_for_project(row)
    slug = str(config.get("defillama_slug") or row.get("defillama_slug") or "").strip()
    governance_url = str(config.get("governance_url") or (f"https://defillama.com/governance/{slug}" if slug else ""))
    space_id = str(config.get("snapshot_space") or "").strip()
    if not governance_url:
        st.info("No governance source is configured for this project.")
        return

    proposals = _load_snapshot_governance_cached(
        space_id,
        str(row["project"]),
        str(row["ticker"]),
        slug,
        governance_url,
    ) if space_id else pd.DataFrame()

    if proposals.empty:
        detail = "No configured Snapshot space or no recent Snapshot proposals were available."
        st.info(f"{detail} Use the DefiLlama governance source for manual review.")
        st.markdown(f"[Open DefiLlama governance]({governance_url})")
        return

    active = proposals[proposals["state"].astype(str).str.lower().isin(["active", "pending"])]
    closed = proposals[proposals["state"].astype(str).str.lower().eq("closed")]
    c1, c2, c3 = st.columns(3)
    c1.metric("Active / Pending", str(len(active)))
    c2.metric("Recent Closed", str(len(closed)))
    c3.metric("Snapshot Space", space_id)
    display = proposals.copy()
    display["Start"] = pd.to_datetime(display["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    display["End"] = pd.to_datetime(display["end_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    display = display.rename(columns={"title": "Proposal", "state": "State", "value_usd": "Votes", "source_url": "Source URL"})
    st.dataframe(
        display[["Start", "End", "State", "Proposal", "Votes", "Source URL"]],
        hide_index=True,
        width="stretch",
        column_config={
            "Votes": st.column_config.NumberColumn("Votes", format="%.0f"),
            "Source URL": st.column_config.LinkColumn("Source URL"),
        },
    )
    st.markdown(f"[Open DefiLlama governance]({governance_url})")


def render_project_nansen_context(row: pd.Series) -> None:
    with st.expander("Nansen Flow Context", expanded=False):
        config = source_config_for_project(row)
        chain = str(config.get("nansen_chain") or "").lower()
        token_address = str(config.get("nansen_token_address") or "").strip()
        endpoint_rows = [
            {"Use": name.replace("_", " ").title(), "Endpoint": endpoint}
            for name, endpoint in NANSEN_ENDPOINTS.items()
            if name in {
                "token_information",
                "indicators",
                "token_ohlcv",
                "holders",
                "flow_intelligence",
                "flows",
                "who_bought_sold",
                "dex_trades",
                "transfers",
                "pnl_leaderboard",
            }
        ]
        st.dataframe(pd.DataFrame(endpoint_rows), hide_index=True, width="stretch")
        if not NANSEN_API_KEY:
            st.info("Set `NANSEN_API_KEY` in Streamlit secrets to enable live Nansen enrichment. The dashboard will continue without it.")
            return
        if not chain or not token_address:
            st.info(
                f"No supported Nansen token address is mapped yet for {str(row['ticker']).upper()}."
            )
            return
        request_count = len(build_nansen_token_context_requests(chain, token_address))
        masked_address = f"{token_address[:8]}...{token_address[-6:]}" if len(token_address) > 16 else token_address
        st.caption(f"Mapped Nansen asset: {chain} / {masked_address}. Full enrichment queries {request_count} endpoint slices.")
        if st.button("Fetch full Nansen enrichment", key=f"fetch_nansen_context_{row['ticker']}"):
            with st.spinner("Fetching Nansen context..."):
                context = fetch_nansen_token_context(
                    chain,
                    token_address,
                    api_key=NANSEN_API_KEY,
                    max_requests=max(request_count, 1),
                )
            if context.empty:
                st.info("Nansen returned no context for this token.")
            else:
                st.dataframe(context, hide_index=True, width="stretch")


def render_project_correlation_table(df: pd.DataFrame, selected_ticker: str) -> None:
    st.subheader("30D Price-Change Correlation")
    full_ts = load_cached_full_project_timeseries()
    if full_ts.empty:
        st.info("Full project price history is unavailable for correlation analysis.")
        return

    options = sorted(
        ticker
        for ticker in df["ticker"].dropna().astype(str).str.upper().unique().tolist()
        if ticker not in set(DEFAULT_CORRELATION_BENCHMARKS + [str(selected_ticker).upper()])
    )
    extras = st.multiselect(
        "Additional benchmarks",
        options=options,
        default=[],
        key=f"correlation_extra_benchmarks_{selected_ticker}",
    )
    benchmarks = [*DEFAULT_CORRELATION_BENCHMARKS, *extras]
    correlations = compute_rolling_30d_return_correlations(full_ts, selected_ticker, benchmarks)
    if correlations.empty:
        st.info("No eligible price history for the selected asset and benchmarks.")
        return
    st.dataframe(
        correlations,
        hide_index=True,
        width="stretch",
        column_config={
            "Correlation": st.column_config.NumberColumn("Correlation", format="%.2f"),
            "Selected 30D Return": st.column_config.NumberColumn("Selected 30D Return", format="%.1f%%"),
            "Benchmark 30D Return": st.column_config.NumberColumn("Benchmark 30D Return", format="%.1f%%"),
        },
    )


def render_project_detail(df: pd.DataFrame) -> str:
    default_index = int(df.index[df["ticker"] == st.session_state.get("project_detail_selected", "UNI")][0]) if st.session_state.get("project_detail_selected", "UNI") in df["ticker"].values else 0
    selected_ticker = st.selectbox(
        "Project",
        options=df["ticker"].tolist(),
        index=default_index,
        key="project_detail_selected",
    )
    row = df[df["ticker"] == selected_ticker].iloc[0]
    source_links = project_source_links(row)
    source_html = f" {source_links}" if source_links else ""
    st.markdown(
        f'<div class="hero-card"><h3>{row["ticker"]} Project Detail</h3>'
        f'<p>{row["project"]} · {row["category"]} · {row["mcap_bucket"]} market cap bucket · {row["fdv_bucket"]} FDV bucket. '
        f'{source_html}</p></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue 30D Change", pct(row["revenue_mom_pct"]), delta=f"WoW {pct(row['revenue_wow_pct'])}")
    c2.metric("Fees 30D Change", pct(row["fees_mom_pct"]), delta=f"WoW {pct(row['fees_wow_pct'])}")
    c3.metric("Deposits 30D Change", pct(row["deposits_mom_pct"]), delta=f"WoW {pct(row['deposits_wow_pct'])}")
    c4.metric("OI 30D Change", pct(row["oi_mom_pct"]), delta=f"WoW {pct(row['oi_wow_pct'])}")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Price", compact_usd(row["price"]), delta=pct(row["price_7d_pct"]))
    r2.metric("Revenue 30D Annualised", compact_usd(row["annualised_revenue_30d"]), delta=compact_usd(row["revenue_30d"]))
    r3.metric("Fees 30D Annualised", compact_usd(row["annualised_fees_30d"]), delta=compact_usd(row["fees_30d"]))
    r4.metric("Factor Score", f"{row['factor_score']:+.2f}", delta=f"4W {row['factor_4w_change']:+.2f}")

    render_project_actual_timeseries(row)
    render_project_correlation_table(df, selected_ticker)
    render_project_unlocks(row)
    render_project_unstaking_flows(row)
    render_project_governance(row)
    render_project_nansen_context(row)
    render_project_peer_position(df, selected_ticker)
    return selected_ticker


def signal_rows(project: pd.Series) -> pd.DataFrame:
    raw = [
        ("Fundamentals", "Revenue", compact_usd(project["revenue_7d_avg"]), 1.1, "Current 7d average revenue"),
        ("Fundamentals", "Revenue growth 2W", pct(project["revenue_wow_pct"]), 1.4, "Short-term revenue acceleration"),
        ("Fundamentals", "Revenue growth 4W", pct(project["revenue_mom_pct"]), 1.8, "Monthly revenue acceleration"),
        ("Fundamentals", "Revenue stability", "0.62", 0.4, "Lower volatility of revenue growth"),
        ("Fundamentals", "Revenue correlation to BTC", "-0.18", 0.6, "Revenue less tied to BTC beta"),
        ("Fundamentals", "Active revenue share", "-0.21", 0.1, "Higher direct protocol revenue is better"),
        ("Fundamentals", "Protocol margin", compact_usd(project["revenue_7d_avg"] * 0.8 if pd.notna(project["revenue_7d_avg"]) else pd.NA), 0.8, "Revenue after token issuance"),
        ("Fundamentals", "Trading fees", compact_usd(project["fees_7d_avg"]), 1.2, "Current 7d average fees"),
        ("Fundamentals", "Market cap / fees mean reversion", "0.74x", 1.0, "Below own fee-history valuation"),
        ("Fundamentals", "FDV / revenue", ratio(project["fdv_annualised_revenue"]), 1.3, "Lower FDV to revenue is better"),
        ("Fundamentals", "Payback period", "5.4y", 0.9, "Lower market cap payback is better"),
        ("Fundamentals", "Implied growth rate", "0.82x", 0.7, "Market expectations versus peer median"),
        ("Fundamentals", "Daily active user growth 2W", "+8.0%", 0.2, "Short-term active user growth"),
        ("Fundamentals", "Daily active user growth 4W", "+13.0%", 0.5, "Monthly active user growth"),
        ("Fundamentals", "Growth composite", "+1.5", 1.5, "Revenue growth plus activity growth"),
        ("Fundamentals", "Revenue per active address", "$0.42", 0.1, "Unit economics per active address"),
        ("Fundamentals", "Buyback versus issuance", "n/a", None, "Token burn versus issuance"),
        ("Fundamentals", "Staked supply ratio", "n/a", None, "Supply locked in staking"),
        ("Momentum", "Volatility-adjusted momentum 3W", "0.58", 0.7, "Return adjusted for realised volatility"),
        ("Momentum", "Relative strength versus BTC", "+9.0%", 0.9, "Outperformance versus BTC"),
        ("Momentum", "Drawdown from all-time high", "-34.0%", 0.2, "Distance from all-time high"),
        ("Momentum", "Momentum breadth", "62%", 0.5, "Share of positive trailing weeks"),
        ("Momentum", "Short-term reversal", "-3.0%", -0.2, "One-week reversal signal"),
        ("Momentum", "Realised volatility rank", "68%", 0.1, "Lower realised volatility ranks better"),
        ("Momentum", "Size signal", compact_usd(project["market_cap"]), -0.4, "Smaller market cap scores higher"),
        ("Flows", "Stablecoin supply growth", "+3.0%", 0.3, "Liquidity growth on relevant chains"),
        ("Flows", "DEX volume growth", "+18.0%", 1.1, "Trading activity growth"),
        ("Flows", "Bridge net flow", "+$42M", 0.2, "Net bridge inflows"),
        ("Flows", "Open interest change", pct(project["oi_wow_pct"]), None if pd.isna(project["oi_wow_pct"]) else 0.6, "Futures open interest change where available"),
    ]
    return pd.DataFrame(raw, columns=["Family", "Metric", "Raw Value", "Peer Score", "Meaning"])


def _signal_score_text(value: Any) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):+.2f}"


def signal_detail_cells(project: pd.Series) -> pd.Series:
    signals = signal_rows(project)
    cells: dict[str, str] = {}
    for family in ["Fundamentals", "Momentum", "Flows"]:
        parts = []
        family_rows = signals[signals["Family"] == family]
        for _, signal in family_rows.iterrows():
            parts.append(
                f"{signal['Metric']}: {signal['Raw Value']} "
                f"(peer score {_signal_score_text(signal['Peer Score'])}) - {signal['Meaning']}"
            )
        cells[f"{family} Signal Detail"] = "; ".join(parts)
    return pd.Series(cells)


def render_signal_detail(df: pd.DataFrame) -> None:
    default = st.session_state.get("signal_detail_selected", st.session_state.get("project_detail_selected", "UNI"))
    default_index = int(df.index[df["ticker"] == default][0]) if default in df["ticker"].values else 0
    selected_ticker = st.selectbox(
        "Project",
        options=df["ticker"].tolist(),
        index=default_index,
        key="signal_detail_selected",
    )
    project = df[df["ticker"] == selected_ticker].iloc[0]
    st.markdown(
        f'<div class="hero-card"><h3>{selected_ticker} Signal Detail</h3>'
        '<p>Audit layer for all 29 signals behind the project score.</p></div>',
        unsafe_allow_html=True,
    )
    signals = signal_rows(project)
    st.dataframe(
        signals,
        hide_index=True,
        width="stretch",
        height=760,
        column_config={
            "Peer Score": st.column_config.NumberColumn("Peer Score", format="%.2f")
        },
    )


def _factor_table(df: pd.DataFrame, score_col: str, ascending: bool, n: int = 5) -> pd.DataFrame:
    rows = (
        df.dropna(subset=[score_col])
        .sort_values(score_col, ascending=ascending)
        .head(n)
        [["ticker", "project", "category", "fdv_bucket", "fdv", "price_7d_pct", score_col]]
        .rename(
            columns={
                "ticker": "Ticker",
                "project": "Project",
                "category": "Category",
                "fdv_bucket": "FDV Bucket",
                "fdv": "FDV",
                "price_7d_pct": "7D Price",
                score_col: "Score",
            }
        )
    )
    rows["FDV"] = rows["FDV"].map(compact_usd)
    rows["7D Price"] = rows["7D Price"].map(pct)
    return rows


def _render_factor_tables(df: pd.DataFrame, title: str, ascending: bool) -> None:
    st.subheader(title)
    cols = st.columns(len(FACTOR_FAMILIES))
    for col, (score_col, label) in zip(cols, FACTOR_FAMILIES.items()):
        with col:
            st.markdown(f"**{label}**")
            table = _factor_table(df, score_col, ascending=ascending)
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                height=252,
                column_config={"Score": st.column_config.NumberColumn("Score", format="%.2f")},
            )


def _weighted_factor_index(df: pd.DataFrame, tickers: list[str], label: str) -> pd.DataFrame:
    basket = df[df["ticker"].isin(tickers)].copy()
    if basket.empty:
        return pd.DataFrame(columns=["date", "Basket", "Index"])
    frames = []
    real_ts = load_cached_project_timeseries("")
    weight_values: dict[str, float] = {}
    basket_weights = basket.set_index("ticker")
    for ticker in basket["ticker"].astype(str):
        ts = real_ts[real_ts["ticker"].astype(str).eq(str(ticker))].sort_values("date") if not real_ts.empty else pd.DataFrame()
        if not ts.empty and "price" in ts and ts["price"].notna().sum() >= 2:
            ts = ts[["date", "price"]].dropna().copy()
            ts["date"] = pd.to_datetime(ts["date"])
            ts[str(ticker)] = ts["price"] / ts["price"].iloc[0] * 100
            frames.append(ts[["date", str(ticker)]].set_index("date"))
            source_rows = real_ts[real_ts["ticker"].astype(str).eq(str(ticker))].sort_values("date")
            source_rows = source_rows.dropna(subset=["price"]) if "price" in source_rows else pd.DataFrame()
            start_row = source_rows.iloc[0] if not source_rows.empty else pd.Series(dtype=float)
            weight = pd.NA
            for weight_col in ["fdv", "market_cap"]:
                if weight_col in start_row and pd.notna(start_row[weight_col]):
                    weight = start_row[weight_col]
                    break
        else:
            ts = load_mock_timeseries(str(ticker))[["date", "price_change_pct"]].copy()
            ts["date"] = pd.to_datetime(ts["date"])
            ts[str(ticker)] = (1 + ts["price_change_pct"] / 100) * 100
            frames.append(ts[["date", str(ticker)]].set_index("date"))
            weight = pd.NA
        if (pd.isna(weight) or float(weight) <= 0) and ticker in basket_weights.index:
            for weight_col in ["fdv", "market_cap"]:
                if weight_col in basket_weights and pd.notna(basket_weights.at[ticker, weight_col]):
                    weight = basket_weights.at[ticker, weight_col]
                    break
        weight_values[ticker] = float(weight) if pd.notna(weight) and float(weight) > 0 else 1.0
    if not frames:
        return pd.DataFrame(columns=["date", "Basket", "Index"])
    indexed_prices = pd.concat(frames, axis=1, sort=False).sort_index()
    weights = pd.Series(weight_values, dtype=float)
    weights = weights / weights.sum() if weights.sum() > 0 else pd.Series(1.0 / len(tickers), index=tickers)
    aligned_weights = pd.Series(weights.to_numpy(), index=basket["ticker"].astype(str)).reindex(indexed_prices.columns).fillna(0)
    available_weights = indexed_prices.notna().mul(aligned_weights, axis=1)
    weight_sum = available_weights.sum(axis=1).replace(0, float("nan"))
    basket_index = indexed_prices.mul(aligned_weights, axis=1).sum(axis=1, min_count=1) / weight_sum
    basket_index = basket_index.dropna()
    if basket_index.empty:
        return pd.DataFrame(columns=["date", "Basket", "Index"])
    basket_index = basket_index / basket_index.iloc[0] * 100
    combined = basket_index.reset_index()
    combined.columns = ["date", "Index"]
    combined["Basket"] = label
    return combined[["date", "Basket", "Index"]]


def render_factor_index(df: pd.DataFrame) -> None:
    st.subheader("Top vs Bottom Factor Basket")
    score_col = st.selectbox(
        "Basket factor",
        options=list(FACTOR_FAMILIES),
        format_func=FACTOR_FAMILIES.get,
        key="factor_basket_score_select",
    )
    lag_col = FACTOR_LAG_COLUMNS[score_col]
    factor_label = FACTOR_FAMILIES[score_col]
    if lag_col not in df:
        st.info("30D lagged factor scores are unavailable. Refresh the screener cache to populate basket rankings.")
        return
    ranked = df.dropna(subset=[lag_col]).sort_values(lag_col, ascending=False)
    if ranked.empty:
        st.info("No projects have 30D lagged factor scores after the current filters.")
        return
    top = ranked.head(5)
    bottom = ranked.tail(5).sort_values(lag_col, ascending=True)
    st.caption(
        f"Baskets are selected from the {factor_label} score observed about {FACTOR_LAG_DAYS} days ago, "
        "then indexed forward over the subsequent 30D price window. Basket weights use FDV/market cap from the start of that window where available."
    )
    index_df = pd.concat(
        [
            _weighted_factor_index(df, top["ticker"].tolist(), "Top factor basket"),
            _weighted_factor_index(df, bottom["ticker"].tolist(), "Bottom factor basket"),
        ],
        ignore_index=True,
    )
    if index_df.empty:
        index_df = load_cached_factor_baskets()
        if "Factor" in index_df:
            index_df = index_df[index_df["Factor"].eq(factor_label)]
    if index_df.empty:
        return
    fig = px.line(
        index_df,
        x="date",
        y="Index",
        color="Basket",
        color_discrete_map={
            "Top factor basket": "#8eed8a",
            "Bottom factor basket": "#ff746d",
        },
    )
    fig.update_traces(line=dict(width=2.6))
    fig.update_layout(
        template="plotly_dark",
        height=430,
        xaxis_title=None,
        yaxis_title="Indexed performance",
        legend_title=None,
        margin=dict(l=10, r=20, t=20, b=20),
        paper_bgcolor="#121410",
        plot_bgcolor="#121410",
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")


def _stable_fundamentals_table(research: dict[str, Any]) -> pd.DataFrame:
    ic = research.get("fundamentals_ic", pd.DataFrame())
    if ic.empty or not {"signal", "period", "mean_ic"}.issubset(ic.columns):
        return pd.DataFrame()
    flags = ic.groupby("signal", observed=True).agg(
        Stable=("positive_all_windows_and_fold2", "max"),
        Min_Decision_IC=("min_train_validation_fold2_ic", "max"),
    )
    pivot = ic.pivot_table(index="signal", columns="period", values="mean_ic", aggfunc="mean")
    cols = [col for col in ["Train", "Validation", "Test", "fold_2"] if col in pivot]
    out = flags.join(pivot[cols], how="left").reset_index()
    out = out[out["Stable"].astype(bool)].sort_values("Min_Decision_IC", ascending=False)
    return out.rename(
        columns={
            "signal": "Signal",
            "Min_Decision_IC": "Min Train/Validation/fold_2 IC",
            "fold_2": "Mixed fold IC",
        }
    )


def _weight_experiment_validation_table(research: dict[str, Any]) -> pd.DataFrame:
    weights = research.get("weights", pd.DataFrame())
    if weights.empty or "period" not in weights:
        return pd.DataFrame()
    validation = weights[weights["period"].eq("Validation")].copy()
    if validation.empty:
        return pd.DataFrame()
    cols = [
        "scheme",
        "total_return",
        "sharpe",
        "fold_retrained_rolling_positive_folds",
        "fold_retrained_rolling_avg_total_return",
        "would_clear_strict_gate_no_leakage",
    ]
    validation = validation[[col for col in cols if col in validation]]
    return validation.rename(
        columns={
            "scheme": "Scheme",
            "total_return": "Validation return",
            "sharpe": "Validation Sharpe",
            "fold_retrained_rolling_positive_folds": "Positive fold-retrained folds",
            "fold_retrained_rolling_avg_total_return": "Rolling avg return",
            "would_clear_strict_gate_no_leakage": "Clears no-leakage gate",
        }
    )


def _selected_validation_row(research: dict[str, Any]) -> pd.Series:
    validation = research.get("validation", pd.DataFrame())
    selected = research.get("selected", {})
    if validation.empty or not selected:
        return pd.Series(dtype=object)
    mask = (
        validation.get("model", pd.Series(index=validation.index, dtype=object)).eq(selected.get("selected_model"))
        & validation.get("variant", pd.Series(index=validation.index, dtype=object)).eq(selected.get("selected_variant"))
        & validation.get("eligible_universe", pd.Series(index=validation.index, dtype=object)).eq(selected.get("selected_eligible_universe"))
        & validation.get("horizon", pd.Series(index=validation.index, dtype=float)).eq(selected.get("primary_horizon", 14))
    )
    if not mask.any():
        return pd.Series(dtype=object)
    return validation[mask].iloc[0]


def _selected_test_row(research: dict[str, Any]) -> pd.Series:
    test = research.get("test", pd.DataFrame())
    selected = research.get("selected", {})
    if test.empty or not selected:
        return pd.Series(dtype=object)
    mask = test.get("horizon", pd.Series(index=test.index, dtype=float)).eq(selected.get("primary_horizon", 14))
    if "model" in test:
        mask &= test["model"].eq(selected.get("selected_model"))
    if "variant" in test:
        mask &= test["variant"].eq(selected.get("selected_variant"))
    if not mask.any():
        return pd.Series(dtype=object)
    return test[mask].iloc[0]


def render_factor_research_wip() -> None:
    research = load_factor_research()
    selected = research.get("selected", {})
    selected_val = _selected_validation_row(research)
    selected_test = _selected_test_row(research)
    stable = _stable_fundamentals_table(research)
    weight_validation = _weight_experiment_validation_table(research)

    st.markdown(
        '<div class="hero-card"><h3>Factor Research (WIP)</h3>'
        '<p>Research notebook and audit-facing diagnostics. Production factor rankings are unchanged.</p></div>',
        unsafe_allow_html=True,
    )
    if FACTOR_RESEARCH_NOTEBOOK_PATH.exists():
        st.download_button(
            "Download notebook",
            data=FACTOR_RESEARCH_NOTEBOOK_PATH.read_bytes(),
            file_name=f"factor_model_walkforward_{FACTOR_RESEARCH_NAME}_WIP.ipynb",
            mime="application/x-ipynb+json",
        )

    st.info(
        "The latest research improved the audit trail and found better candidate fundamentals signals, "
        "but it did not approve a new production factor model. The dashboard factors shown elsewhere still use the existing production inputs."
    )

    status = selected.get("selection_status", "Unknown")
    strict_pass = bool(selected.get("strict_gate_passed", False))
    stable_count = int(len(stable)) if not stable.empty else 0
    clears = 0
    if not weight_validation.empty and "Clears no-leakage gate" in weight_validation:
        clears = int(weight_validation["Clears no-leakage gate"].fillna(False).astype(bool).sum())
    regime_pass = selected_val.get("passes_regime_stratified_gate", pd.NA) if not selected_val.empty else pd.NA
    beta_share = pd.NA
    if not selected_test.empty and pd.notna(selected_test.get("total_return")) and float(selected_test.get("total_return")) != 0:
        beta_share = float(selected_test.get("beta_attributed_return", 0.0)) / float(selected_test.get("total_return"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Selection status", "Strict pass" if strict_pass else status.replace("_", " ").title())
    c2.metric("Stable new signals", str(stable_count), "not active yet")
    c3.metric("Weight schemes cleared", str(clears))
    c4.metric("Regime sub-gate", "Pass" if bool(regime_pass) else "Fail")
    c5.metric("Test beta share", "n/a" if pd.isna(beta_share) else f"{beta_share:.1%}")

    with st.expander("Research update", expanded=True):
        text = research.get("plain_english") or research.get("next_steps") or "Research update is unavailable in this deployment."
        st.markdown(text)

    with st.expander("Stable fundamentals candidates", expanded=True):
        if stable.empty:
            st.info("No stable fundamentals redefinition table is available.")
        else:
            st.caption("These are research candidates only. They are not yet blended into the production screener scores.")
            st.dataframe(
                stable,
                hide_index=True,
                width="stretch",
                column_config={
                    "Train": st.column_config.NumberColumn("Train IC", format="%.4f"),
                    "Validation": st.column_config.NumberColumn("Validation IC", format="%.4f"),
                    "Test": st.column_config.NumberColumn("Test IC", format="%.4f"),
                    "Mixed fold IC": st.column_config.NumberColumn("Mixed fold IC", format="%.4f"),
                    "Min Train/Validation/fold_2 IC": st.column_config.NumberColumn("Min decision IC", format="%.4f"),
                },
            )

    with st.expander("Weight robustness and validation gates", expanded=True):
        if weight_validation.empty:
            st.info("Weight experiment summary is unavailable.")
        else:
            st.caption("Gate checks use validation and fold-retrained rolling evidence only, not test labels.")
            st.dataframe(
                weight_validation,
                hide_index=True,
                width="stretch",
                column_config={
                    "Validation return": st.column_config.NumberColumn("Validation return", format="%.1%"),
                    "Validation Sharpe": st.column_config.NumberColumn("Validation Sharpe", format="%.2f"),
                    "Rolling avg return": st.column_config.NumberColumn("Rolling avg return", format="%.1%"),
                },
            )

    with st.expander("Selected diagnostic basket: beta and regime checks", expanded=True):
        if selected_val.empty or selected_test.empty:
            st.info("Selected basket diagnostics are unavailable.")
        else:
            diagnostic = pd.DataFrame(
                [
                    {
                        "Model": selected.get("selected_model"),
                        "Variant": selected.get("selected_variant"),
                        "Universe": selected.get("selected_eligible_universe"),
                        "Validation return": selected_val.get("total_return"),
                        "Validation Sharpe": selected_val.get("sharpe"),
                        "Positive rolling folds": selected_val.get("rolling_positive_folds"),
                        "Regime sub-gate": bool(selected_val.get("passes_regime_stratified_gate", False)),
                        "Test return": selected_test.get("total_return"),
                        "Beta-attributed return": selected_test.get("beta_attributed_return"),
                        "Dispersion alpha": selected_test.get("dispersion_alpha"),
                    }
                ]
            )
            st.dataframe(
                diagnostic,
                hide_index=True,
                width="stretch",
                column_config={
                    "Validation return": st.column_config.NumberColumn("Validation return", format="%.1%"),
                    "Validation Sharpe": st.column_config.NumberColumn("Validation Sharpe", format="%.2f"),
                    "Test return": st.column_config.NumberColumn("Test return", format="%.1%"),
                    "Beta-attributed return": st.column_config.NumberColumn("Beta-attributed return", format="%.1%"),
                    "Dispersion alpha": st.column_config.NumberColumn("Dispersion alpha", format="%.1%"),
                },
            )


def render_factor_aggregation_info() -> None:
    active_total = sum(FAMILY_WEIGHTS.get(key, 0.0) for key in FACTOR_FAMILY_CONFIG_KEYS.values())
    rows = []
    for score_col, label in FACTOR_FAMILIES.items():
        if score_col == "factor_score":
            continue
        config_key = FACTOR_FAMILY_CONFIG_KEYS[score_col]
        configured = float(FAMILY_WEIGHTS.get(config_key, 0.0))
        rows.append(
            {
                "Family": label,
                "Configured Weight": configured,
                "Active Total Weight": configured / active_total if active_total else pd.NA,
                "Aggregation": "Equal-weight average of available z-scored signals",
            }
        )
    with st.expander("How total factor score is aggregated", expanded=False):
        st.caption(
            "Signal values are winsorised and cross-sectionally z-scored, averaged inside each family, "
            "then combined into Total. The configured model also reserves weight for Team, but that family is not yet populated in this dashboard, so active weights are normalised across the available families."
        )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Configured Weight": st.column_config.NumberColumn("Configured Weight", format="%.1%"),
                "Active Total Weight": st.column_config.NumberColumn("Active Total Weight", format="%.1%"),
            },
        )


def render_factors(df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="hero-card"><h3>Factor Screener</h3>'
        '<p>Factor leaders and laggards by total score and by family.</p></div>',
        unsafe_allow_html=True,
    )
    render_factor_screener_section(df)


def render_factor_screener_section(df: pd.DataFrame) -> None:
    st.divider()
    st.subheader("Factor Screener")
    if df.empty or "factor_score" not in df:
        st.info("No tokens match the current screener filters.")
        return
    render_factor_aggregation_info()
    with st.expander("Factor Filters", expanded=True):
        c1, c2 = st.columns(2)
        use_score_filter = c1.toggle("Filter by factor score", value=False, key="merged_factor_score_filter_toggle")
        use_fdv_filter = c2.toggle("Filter by FDV bucket", value=False, key="merged_factor_fdv_filter_toggle")
        filtered = df.copy()
        if use_score_filter:
            min_score = float(df["factor_score"].min())
            max_score = float(df["factor_score"].max())
            if min_score == max_score:
                score_range = (min_score, max_score)
            else:
                score_range = st.slider(
                    "Factor score range",
                    min_value=min_score,
                    max_value=max_score,
                    value=(min_score, max_score),
                    step=0.05,
                    key="merged_factor_score_range",
                )
            filtered = filtered[filtered["factor_score"].between(score_range[0], score_range[1])]
        if use_fdv_filter:
            selected_buckets = st.multiselect(
                "FDV buckets",
                FDV_BUCKET_ORDER,
                default=FDV_BUCKET_ORDER,
                key="merged_factor_fdv_bucket_filter",
            )
            filtered = filtered[filtered["fdv_bucket"].astype(str).isin(selected_buckets)]

    _render_factor_tables(filtered, "Top 5 Tokens", ascending=False)
    _render_factor_tables(filtered, "Bottom 5 Tokens", ascending=True)
    render_factor_index(filtered)
    render_screener_table(filtered)


def render_data_quality(df: pd.DataFrame) -> None:
    st.markdown(
        '<div class="hero-card"><h3>Data Quality</h3>'
        '<p>Coverage and gaps only. Investment rankings are deliberately not shown here.</p></div>',
        unsafe_allow_html=True,
    )

    coverage_rows = []
    stale_defillama = df.get(
        "defillama_stale_metric",
        df.get("blockworks_stale_metric", pd.Series(False, index=df.index)),
    ).fillna(False).astype(bool)
    blockworks_match = df.get("blockworks_match_slug", pd.Series(pd.NA, index=df.index)).notna()
    coverage_map = {
        "DefiLlama universe": df["universe_source"].eq("defillama"),
        "CoinGecko market-data match": df["coingecko_id"].notna(),
        "DefiLlama TVL match": df["defillama_slug"].notna(),
        "Binance spot symbol": df["binance_spot_symbol"].notna(),
        "Binance futures symbol": df["binance_futures_symbol"].notna(),
        "Price, market cap, FDV": df[["price", "market_cap", "fdv"]].notna().all(axis=1),
        "Revenue": df["revenue_7d_avg"].notna(),
        "Fees": df["fees_7d_avg"].notna(),
        "Deposits": df["deposits"].notna(),
        "TVL": df["tvl"].notna(),
        "Open interest": df["open_interest"].notna(),
        "Bridge flows": df["category"].eq("L1 / L2"),
        "Fresh DefiLlama metric data": ~stale_defillama,
        "Optional Blockworks match": blockworks_match,
        "Revenue/fees not redundant": ~df.get("revenue_fees_redundant", pd.Series(False, index=df.index)).fillna(False).astype(bool),
    }
    rules_by_metric = {rule.metric: rule for rule in COVERAGE_RULES}
    for metric, mask in coverage_map.items():
        rule = rules_by_metric.get(metric)
        coverage_rows.append(
            {
                "Metric": metric,
                "Coverage": f"{mask.mean():.0%}",
                "Projects": f"{int(mask.sum())} / {len(mask)}",
                "Source": rule.source if rule else "Automated asset master / entity resolution",
                "Gap Rule": rule.gap_rule if rule else "Required to explain source coverage and unresolved joins.",
            }
        )
    st.dataframe(pd.DataFrame(coverage_rows), hide_index=True, width="stretch")

    gap_rows = []
    for _, row in df.iterrows():
        gap_rows.append(
            {
                "Ticker": row["ticker"],
                "Project": row["project"],
                "Category": row["category"],
                "FDV bucket": row["fdv_bucket"],
                "CG": "OK" if pd.notna(row["coingecko_id"]) else "Missing",
                "DL TVL": "OK" if pd.notna(row["defillama_slug"]) else "n/a or missing",
                "Binance Spot": "OK" if pd.notna(row["binance_spot_symbol"]) else "Missing",
                "Binance Futures": "OK" if pd.notna(row["binance_futures_symbol"]) else "n/a or missing",
                "Revenue": "OK" if pd.notna(row["revenue_7d_avg"]) else "Missing",
                "Fees": "OK" if pd.notna(row["fees_7d_avg"]) else "Missing",
                "DL Fees/Rev": "Missing" if bool(row.get("missing_defillama_fees_revenue", False)) else "OK",
                "Deposits": "OK" if pd.notna(row["deposits"]) else "n/a or missing",
                "OI": "OK" if pd.notna(row["open_interest"]) else "n/a or missing",
                "Stale DL": "Yes" if bool(row.get("defillama_stale_metric", row.get("blockworks_stale_metric", False))) else "No",
                "BW Match": "OK" if pd.notna(row.get("blockworks_match_slug", pd.NA)) else "Optional",
                "Rev/Fee Corr": "n/a" if pd.isna(row.get("revenue_fees_corr_90d", pd.NA)) else f"{float(row.get('revenue_fees_corr_90d')):.2f}",
                "Deduped": "Yes" if bool(row.get("fees_signal_deduped", False)) else "No",
            }
        )
    st.subheader("Per-Project Gaps")
    st.dataframe(pd.DataFrame(gap_rows), hide_index=True, width="stretch")

    redundant = df[df.get("revenue_fees_redundant", pd.Series(False, index=df.index)).fillna(False).astype(bool)].copy()
    if not redundant.empty:
        st.subheader("Redundant Metric Warnings")
        warning_cols = ["ticker", "project", "category", "revenue_fees_corr_90d", "fees_signal_deduped"]
        warnings = redundant[warning_cols].rename(
            columns={
                "ticker": "Ticker",
                "project": "Project",
                "category": "Category",
                "revenue_fees_corr_90d": "Revenue / Fees Corr",
                "fees_signal_deduped": "Fees Signal Deduped",
            }
        )
        st.dataframe(
            warnings,
            hide_index=True,
            width="stretch",
            column_config={"Revenue / Fees Corr": st.column_config.NumberColumn("Revenue / Fees Corr", format="%.2f")},
        )


def metric_dictionary() -> pd.DataFrame:
    rows = [
        ("Universe source", "Canonical screener universe.", "Rows start from DefiLlama protocols and chains, then market/trading sources enrich the row.", "DefiLlama /protocols + /v2/chains"),
        ("Ticker", "Liquid token symbol used for trading and display.", "Mapped from DefiLlama symbol/native token and cross-checked against external market data.", "DefiLlama asset master"),
        ("Project", "Protocol or chain name.", "Mapped from DefiLlama protocol and chain metadata.", "DefiLlama metadata"),
        ("Sector", "Broad investable area such as DeFi, Infrastructure, or Consumer.", "Mapped from DefiLlama categories.", "DefiLlama metadata"),
        ("Category", "Peer group used for benchmarking.", "Mapped from DefiLlama category and retained as the primary benchmark grouping.", "DefiLlama metadata"),
        ("FDV bucket", "Size class based on fully diluted valuation.", "Micro < $100M; Small $100M-$500M; Mid $500M-$1B; Large $1B-$5B; Mega $5B+.", "CoinGecko FDV"),
        ("Market cap bucket", "Size class based on circulating market capitalisation.", "Micro < $100M; Small $100M-$500M; Mid $500M-$1B; Large $1B-$5B; Mega $5B+.", "CoinGecko market cap"),
        ("Price", "Current token price.", "Latest available USD token price.", "Binance/CoinGecko primary; DefiLlama prices fallback"),
        ("Market cap", "Circulating market capitalisation.", "Price multiplied by circulating supply where available.", "CoinGecko primary"),
        ("FDV", "Fully diluted valuation.", "Price multiplied by fully diluted supply.", "CoinGecko primary"),
        ("24h volume", "Latest exchange trading volume.", "Token spot and futures volume where mapped.", "Binance + CoinGecko"),
        ("7D price change", "Token price change over the last seven days.", "(latest price / price 7 days ago - 1) * 100.", "CoinGecko / Binance"),
        ("Revenue 7D avg", "Current revenue run-rate proxy.", "Average daily revenue over the latest seven observations.", "DefiLlama fees/revenue summaries"),
        ("Revenue 30D", "Trailing monthly protocol revenue.", "Sum of the latest 30 daily revenue observations.", "DefiLlama fees/revenue summaries"),
        ("Revenue 30D annualised", "Current annualised revenue run-rate.", "Trailing 30D revenue multiplied by 365 / 30.", "DefiLlama fees/revenue summaries"),
        ("Revenue WoW change", "One-week change in current revenue.", "Latest 7D average versus prior 7D average.", "DefiLlama fees/revenue summaries"),
        ("Revenue MoM change", "One-month change in current revenue.", "Latest 7D average versus 7D average from roughly 30 days earlier.", "DefiLlama fees/revenue summaries"),
        ("Fees 7D avg", "Current fee generation proxy.", "Average daily fees over the latest seven observations.", "DefiLlama fees summaries"),
        ("Fees 30D", "Trailing monthly fees.", "Sum of the latest 30 daily fee observations.", "DefiLlama fees summaries"),
        ("Fees 30D annualised", "Current annualised fee run-rate.", "Trailing 30D fees multiplied by 365 / 30.", "DefiLlama fees summaries"),
        ("Fees WoW change", "One-week change in fees.", "Latest 7D fee average versus prior 7D average.", "DefiLlama fees summaries"),
        ("Fees MoM change", "One-month change in fees.", "Latest 7D average versus 7D average from roughly 30 days earlier.", "DefiLlama fees summaries"),
        ("Deposits", "Current lending or collateral deposits.", "Latest TVL used as deposit proxy where applicable.", "DefiLlama TVL"),
        ("Deposits WoW change", "One-week change in deposits.", "Latest deposits versus deposits seven days earlier.", "DefiLlama TVL"),
        ("Deposits MoM change", "One-month change in deposits.", "Latest deposits versus deposits roughly 30 days earlier.", "DefiLlama TVL"),
        ("TVL", "Current capital locked in the protocol or chain ecosystem.", "Latest total value locked.", "DefiLlama protocol/chain TVL preferred; Blockworks lending TVL fallback"),
        ("TVL WoW change", "One-week change in TVL.", "Latest TVL versus TVL seven days earlier.", "DefiLlama / Blockworks"),
        ("TVL MoM change", "One-month change in TVL.", "Latest TVL versus TVL roughly 30 days earlier.", "DefiLlama / Blockworks"),
        ("Open interest", "Current futures open interest.", "Latest Binance perpetual open interest in USD.", "Binance Futures"),
        ("Open interest WoW change", "One-week change in futures positioning.", "Latest OI versus OI seven days earlier.", "Binance Futures"),
        ("Open interest 30D change", "One-month change in futures positioning.", "Latest OI versus OI roughly 30 days earlier.", "Binance Futures"),
        ("FDV / annualised fees", "Valuation versus current fee base.", "FDV divided by fees 30D annualised.", "CoinGecko + DefiLlama fees"),
        ("FDV / annualised revenue", "Valuation versus current revenue base.", "FDV divided by revenue 30D annualised.", "CoinGecko + DefiLlama revenue"),
        ("Revenue growth 2W / 4W ratio", "Revenue acceleration ratio.", "Two-week revenue growth divided by four-week revenue growth.", "DefiLlama revenue metrics"),
        ("Factor score", "Composite model score.", "Weighted, winsorised, peer-normalised average of active signals.", "Internal factor model"),
        ("30D lagged factor score", "Prior factor signal used for forward performance checks.", "Latest score from the most recent observation at or before 30 days ago.", "Internal factor model"),
        ("4W factor score change", "Change in composite score over four weeks.", "Latest factor score minus factor score 28 days earlier.", "Internal factor model"),
        ("Top-100 30D price distribution", "Market dispersion metric.", "Latest cached non-stablecoin top 100 by market cap, with 30D price change calculated as latest price / price 30 daily rows earlier - 1.", "CoinGecko / Binance"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Definition", "Calculation", "Source"])


def signal_dictionary() -> pd.DataFrame:
    rows = [
        ("Fundamentals", "Revenue", "Current protocol revenue level.", "Latest 7D average revenue.", "DefiLlama revenue metrics"),
        ("Fundamentals", "Revenue growth 2W", "Short-term revenue acceleration.", "Latest 7D average versus prior 7D average.", "DefiLlama revenue metrics"),
        ("Fundamentals", "Revenue growth 4W", "Monthly revenue acceleration.", "Latest 7D average versus 7D average from roughly 30 days earlier.", "DefiLlama revenue metrics"),
        ("Fundamentals", "Revenue stability", "Persistence and smoothness of revenue.", "Inverse volatility of recent revenue growth.", "DefiLlama revenue metrics"),
        ("Fundamentals", "Revenue correlation to BTC", "How much revenue moves with BTC beta.", "Rolling correlation of revenue growth versus BTC returns.", "DefiLlama + Binance BTC"),
        ("Fundamentals", "Active revenue share", "Direct active revenue quality.", "Lower passive or non-operating revenue share scores better.", "DefiLlama revenue metrics where available"),
        ("Fundamentals", "Protocol margin", "Revenue retained after token incentives.", "Revenue minus token issuance or incentive cost.", "DefiLlama revenue + token issuance sources"),
        ("Fundamentals", "Trading fees", "Current fee generation.", "Latest 7D average fees.", "DefiLlama fee metrics"),
        ("Fundamentals", "Market cap / fees mean reversion", "Valuation versus own fee history.", "Current market cap / fees versus trailing historical average.", "CoinGecko + DefiLlama fees"),
        ("Fundamentals", "FDV / revenue", "Valuation versus current revenue.", "FDV divided by revenue 30D annualised; lower scores better.", "CoinGecko + DefiLlama revenue"),
        ("Fundamentals", "Payback period", "Years of current revenue needed to cover market cap.", "Market cap divided by annualised protocol revenue.", "CoinGecko + DefiLlama revenue"),
        ("Fundamentals", "Implied growth rate", "Market-implied growth versus peers.", "Current valuation multiple versus category median.", "CoinGecko + DefiLlama"),
        ("Fundamentals", "Daily active user growth 2W", "Short-term active user growth.", "Latest active users versus two weeks ago.", "Activity metrics where available"),
        ("Fundamentals", "Daily active user growth 4W", "Monthly active user growth.", "Latest active users versus roughly 30 days ago.", "Activity metrics where available"),
        ("Fundamentals", "Growth composite", "Combined usage and revenue growth.", "Average of revenue growth and activity growth scores.", "Internal factor model"),
        ("Fundamentals", "Revenue per active address", "Unit economics per user.", "Revenue divided by active addresses.", "DefiLlama revenue + activity where available"),
        ("Fundamentals", "Buyback versus issuance", "Tokenholder value return versus dilution.", "Token buybacks or burns minus issuance.", "Protocol/token supply sources"),
        ("Fundamentals", "Staked supply ratio", "Supply locked in staking.", "Staked supply divided by circulating supply.", "Token supply / staking sources"),
        ("Momentum", "Volatility-adjusted momentum 3W", "Recent return adjusted for risk.", "Three-week return divided by realised volatility.", "Binance / CoinGecko prices"),
        ("Momentum", "Relative strength versus BTC", "Token outperformance versus BTC.", "Token return minus BTC return over the same window.", "Binance / CoinGecko prices"),
        ("Momentum", "Drawdown from all-time high", "Distance from prior peak.", "Current price divided by all-time high minus one.", "CoinGecko"),
        ("Momentum", "Momentum breadth", "Consistency of positive returns.", "Share of positive trailing return windows.", "Binance / CoinGecko prices"),
        ("Momentum", "Short-term reversal", "One-week reversal signal.", "Negative of most recent one-week return after strong move.", "Binance / CoinGecko prices"),
        ("Momentum", "Realised volatility rank", "Riskiness versus peers.", "Recent realised volatility percentile; lower scores better.", "Binance / CoinGecko prices"),
        ("Momentum", "Size signal", "Smaller-cap tilt.", "Cross-sectional inverse rank of market cap.", "CoinGecko"),
        ("Flows", "Stablecoin supply growth", "Liquidity growth on relevant chains.", "Stablecoin supply change over the selected window.", "DefiLlama stablecoin metrics"),
        ("Flows", "DEX volume growth", "Trading activity growth.", "DEX volume change over the selected window.", "DefiLlama DEX volume metrics"),
        ("Flows", "Bridge net flow", "Net capital moving into ecosystem.", "Bridge inflows minus outflows.", "DefiLlama bridge flows"),
        ("Flows", "Open interest change", "Futures positioning change.", "Latest futures OI versus prior period.", "Binance Futures"),
    ]
    return pd.DataFrame(rows, columns=["Family", "Signal", "Definition", "Calculation", "Source"])


def render_data_dictionary() -> None:
    st.markdown(
        '<div class="hero-card"><h3>Data Dictionary</h3>'
        '<p>Definitions, formulas, and sources for the screener metrics and the 29 model signals.</p></div>',
        unsafe_allow_html=True,
    )
    st.subheader("Data Dictionary")
    st.dataframe(asset_master_contract(), hide_index=True, width="stretch", height=420)
    st.subheader("Screener Metrics")
    st.dataframe(metric_dictionary(), hide_index=True, width="stretch", height=560)
    st.subheader("All 29 Signals")
    st.dataframe(signal_dictionary(), hide_index=True, width="stretch", height=720)


def main() -> None:
    inject_css()
    df = load_projects()

    page_header()
    summary_tab, screener_tab, project_tab, signal_tab, market_tab, quality_tab, dictionary_tab, research_tab = st.tabs(
        [
            "Executive Summary",
            "Tokens Screener",
            "Project Detail",
            "Signal Detail",
            "Macro Signals",
            "Data Quality",
            "Data Dictionary",
            "Factor Research (WIP)",
        ]
    )

    with summary_tab:
        render_executive_summary(df)

    with screener_tab:
        filtered = render_screener(df)
        render_distribution_explorer(filtered)
        render_factor_screener_section(filtered)

    with project_tab:
        render_project_detail(df)

    with signal_tab:
        render_signal_detail(df)

    with market_tab:
        render_macro_monitor()

    with quality_tab:
        render_data_quality(df)

    with dictionary_tab:
        render_data_dictionary()

    with research_tab:
        render_factor_research_wip()

    render_footer_credit()


if __name__ == "__main__":
    main()
