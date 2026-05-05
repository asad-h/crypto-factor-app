"""
Crypto Regime Audit Dashboard.

Streamlit app that displays the weekly regime classification with full
indicator audit, family scoring, DAT mNAV tracking, macro KPIs,
and token/equity watchlist.

Consumes real data via the data_adapter layer.
Mock fallbacks are clearly marked when a source is unavailable.

Usage:
    cd "Crypto Factor Model"
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sys
from io import StringIO
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data_adapter import (
    compute_current_regime,
    compute_weekly_regime_df,
    build_market_kpis_df,
    load_dat_mnav,
    fetch_btc_weekly,
)

st.set_page_config(
    page_title="Crypto Regime Audit",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Constants (display only, not data)
# ---------------------------------------------------------------------------

REGIME_COLORS = {
    "Risk-on": "#a8ff8f",
    "Choppy": "#9ea596",
    "Risk-off": "#ff6f61",
    "Local top": "#ffd166",
    "Local bottom": "#78a6ff",
}

WATCHLIST_PATH = os.getenv(
    "WATCHLIST_CSV",
    str(Path(__file__).resolve().parent / "data" / "watchlist.csv"),
)

REGIME_PLAYBOOK = [
    {"Regime": "Risk-on", "Factor emphasis": "More weight to momentum, growth, flows",
     "Exposure response": "Fully invested, higher conviction sizing"},
    {"Regime": "Choppy", "Factor emphasis": "Balanced across families; require confirmation",
     "Exposure response": "Normal sizing, watchlist active, avoid chasing"},
    {"Regime": "Risk-off", "Factor emphasis": "Value, quality, defensiveness dominant",
     "Exposure response": "40-60% cash / BTC / stablecoin yield"},
    {"Regime": "Local top", "Factor emphasis": "Trim momentum, raise value",
     "Exposure response": "Reduce gross, trim extended names"},
    {"Regime": "Local bottom", "Factor emphasis": "Lean into strong fundamentals",
     "Exposure response": "Selective adds where scoring supports"},
]

FAMILY_EXPLAINERS = [
    {"Family": "BTC trend / momentum", "Weight": 20,
     "Why it matters": "BTC is the primary beta anchor. If BTC trend is weak, alt signals and flow signals deserve less confidence."},
    {"Family": "Market breadth", "Weight": 15,
     "Why it matters": "Breadth tells us whether the bid is broad or concentrated in a few names."},
    {"Family": "ETF + DAT flows", "Weight": 30,
     "Why it matters": "This is the largest marginal liquidity engine. ETFs measure spot institutional demand; DAT mNAV measures balance-sheet buying capacity."},
    {"Family": "Stablecoin liquidity", "Weight": 10,
     "Why it matters": "Stablecoin supply is crypto-native dry powder."},
    {"Family": "Leverage / volatility", "Weight": 10,
     "Why it matters": "Leverage and volatility show whether moves are healthy or fragile."},
    {"Family": "Macro / AI", "Weight": 10,
     "Why it matters": "Crypto is still exposed to TradFi risk shocks. AI is a sector tilt."},
    {"Family": "Valuation / sentiment", "Weight": 5,
     "Why it matters": "Most useful at extremes. Should confirm tops/bottoms, not dominate."},
]


# ---------------------------------------------------------------------------
# CSS injection (preserved from original app)
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #080907;
            --panel: #10120f;
            --line: #2b3027;
            --text: #f3f0e7;
            --muted: #a1a895;
            --faint: #687160;
            --green: #a8ff8f;
            --amber: #ffd166;
            --red: #ff6f61;
            --blue: #78a6ff;
        }
        .stApp { background: #080907; color: var(--text); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMetric"] {
            background: #10120f; border: 1px solid #2b3027;
            border-radius: 8px; padding: 14px;
        }
        [data-testid="stMetricLabel"] { color: var(--faint); }
        h1, h2, h3 { letter-spacing: 0; }
        .hero {
            border: 1px solid #2b3027; border-radius: 8px; padding: 22px;
            background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.01)), #10120f;
        }
        .eyebrow {
            color: var(--faint); font-size: 11px; font-weight: 800;
            letter-spacing: .12em; text-transform: uppercase; margin-bottom: 6px;
        }
        .regime {
            font-size: clamp(56px, 8vw, 104px); line-height: .9;
            font-weight: 750; color: var(--text);
        }
        .copy {
            margin-top: 14px; color: var(--muted); font-size: 15px;
            line-height: 1.55; max-width: 980px;
        }
        .chip {
            display: inline-flex; min-height: 26px; align-items: center;
            padding: 4px 9px; border: 1px solid #2b3027; border-radius: 999px;
            color: var(--muted); background: rgba(255,255,255,.018);
            font-size: 11px; font-weight: 780; letter-spacing: .04em;
            text-transform: uppercase; margin-right: 6px; margin-bottom: 8px;
        }
        .chip.warn { color: var(--amber); border-color: rgba(255,209,102,.34); background: rgba(255,209,102,.12); }
        .chip.good { color: var(--green); border-color: rgba(168,255,143,.32); background: rgba(168,255,143,.11); }
        .chip.blue { color: var(--blue); border-color: rgba(120,166,255,.32); background: rgba(120,166,255,.11); }
        .kpi-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
            gap: 14px; margin-top: 14px;
        }
        .kpi-card {
            border: 1px solid #2b3027; border-radius: 8px;
            background: #10120f; padding: 14px; min-height: 112px;
        }
        .kpi-label { color: var(--faint); font-size: 11px; font-weight: 780; letter-spacing: .08em; text-transform: uppercase; }
        .kpi-value { margin-top: 16px; color: var(--text); font-size: 30px; line-height: 1; font-weight: 720; overflow-wrap: anywhere; }
        .kpi-help { margin-top: 10px; color: var(--muted); font-size: 12px; line-height: 1.35; }
        .section-spacer { height: 10px; }
        div[data-testid="stDataFrame"] { border: 1px solid #2b3027; border-radius: 8px; }
        .simple-table {
            width: 100%; border-collapse: collapse; border: 1px solid #2b3027;
            border-radius: 8px; overflow: hidden; background: #10120f;
        }
        .simple-table th, .simple-table td {
            padding: 12px 14px; border-bottom: 1px solid #1d211b;
            text-align: left; vertical-align: top;
        }
        .simple-table th {
            color: var(--faint); font-size: 11px; font-weight: 800;
            letter-spacing: .08em; text-transform: uppercase;
        }
        .simple-table td { color: var(--muted); font-size: 13px; line-height: 1.4; }
        .simple-table td:first-child { color: var(--text); font-weight: 700; }
        .bar-cell { min-width: 110px; }
        .mini-bar {
            display: inline-block; width: 92px; height: 7px; border-radius: 99px;
            background: #292f26; overflow: hidden; vertical-align: middle; margin-right: 8px;
        }
        .mini-bar > i { display: block; height: 100%; border-radius: inherit; }
        .mini-bar.family > i { background: var(--amber); }
        .mini-bar.risk-on > i { background: var(--green); }
        .mini-bar.choppy > i { background: #9ea596; }
        .mini-bar.risk-off > i { background: var(--red); }
        .mini-bar.local-top > i { background: var(--amber); }
        .mini-bar.local-bottom > i { background: var(--blue); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Display components (preserved from original, consuming adapter data)
# ---------------------------------------------------------------------------

def metric_row(regime_result: dict, btc_latest: float | None) -> None:
    health = 0
    if "indicator_audit_df" in regime_result and not regime_result["indicator_audit_df"].empty:
        total = regime_result["indicator_audit_df"]["score"].sum()
        max_total = len(regime_result["indicator_audit_df"]) * 2
        health = round(total / max_total * 100) if max_total > 0 else 0

    # ETF+DAT family score
    etf_score = "n/a"
    etf_read = ""
    if "family_scores_df" in regime_result:
        fdf = regime_result["family_scores_df"]
        etf_row = fdf[fdf["family"] == "ETF + DAT flows"]
        if not etf_row.empty:
            etf_score = f"{etf_row.iloc[0]['family_score']:.0f}%"
            etf_read = etf_row.iloc[0]["read"]

    btc_str = f"${btc_latest / 1000:.1f}k" if btc_latest else "n/a"

    tilt_map = {
        "Risk-on": "Momentum",
        "Risk-off": "Defensive",
        "Choppy": "Quality",
        "Local top": "Trim",
        "Local bottom": "Selective",
    }
    tilt = tilt_map.get(regime_result.get("regime", "Choppy"), "Quality")

    items = [
        ("Current regime", regime_result.get("regime", "Choppy"),
         f"{regime_result.get('risk_on_score', 0):.0f}% risk-on evidence"),
        ("BTC close", btc_str, "Latest weekly close"),
        ("Weighted health", f"{health}%", "Criteria met across families"),
        ("ETF + DAT", etf_score, etf_read[:50] if etf_read else ""),
        ("Tilt", tilt, f"Factor emphasis for {regime_result.get('regime', 'Choppy')} regime"),
    ]
    cards = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-help">{help_text}</div></div>'
        for label, value, help_text in items
    )
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)


def regime_chart(btc_df: pd.DataFrame) -> alt.LayerChart:
    if btc_df.empty:
        return alt.Chart(pd.DataFrame()).mark_text().encode()

    y_min = int(btc_df["btc_close"].min() * 0.92)
    y_max = int(btc_df["btc_close"].max() * 1.08)

    bands = (
        alt.Chart(btc_df)
        .mark_rect(opacity=0.18)
        .encode(
            x=alt.X("date:T", title=None), x2="end_date:T",
            y=alt.value(0), y2=alt.value(420),
            color=alt.Color("regime:N", title="Regime",
                scale=alt.Scale(domain=list(REGIME_COLORS.keys()),
                                range=[REGIME_COLORS[k] for k in REGIME_COLORS]),
                legend=alt.Legend(orient="top", labelColor="#a1a895", titleColor="#687160")),
            tooltip=[alt.Tooltip("date:T", title="Week", format="%b %d, %Y"),
                     alt.Tooltip("regime:N", title="Regime")],
        )
    )
    line = (
        alt.Chart(btc_df)
        .mark_line(color="#f3f0e7", strokeWidth=2.4)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("btc_close:Q", title="BTCUSDT weekly close",
                     scale=alt.Scale(domain=[y_min, y_max]),
                     axis=alt.Axis(format="$,.0f", labelColor="#a1a895", titleColor="#687160")),
            tooltip=[alt.Tooltip("date:T", title="Week", format="%b %d, %Y"),
                     alt.Tooltip("btc_close:Q", title="BTC close", format="$,.0f"),
                     alt.Tooltip("regime:N", title="Regime")],
        )
    )
    points = (
        alt.Chart(btc_df.tail(1))
        .mark_point(color="#a8ff8f", filled=True, size=70)
        .encode(x="date:T", y="btc_close:Q")
    )
    return (
        (bands + line + points)
        .properties(height=420)
        .configure(background="#10120f")
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor="rgba(255,255,255,.08)", domainColor="#2b3027")
        .configure_legend(labelColor="#a1a895", titleColor="#687160")
    )


def show_family_scores(families_df: pd.DataFrame) -> None:
    def bar(value: float, css_class: str) -> str:
        return (
            f'<span class="mini-bar {css_class}"><i style="width:{value}%"></i></span>'
            f"<b>{value:.0f}%</b>"
        )

    rows = []
    for _, row in families_df.iterrows():
        rows.append(
            "<tr>"
            f"<td>{row['family']}</td>"
            f"<td>{int(row['weight'])}%</td>"
            f"<td>{int(row.get('indicators_met', 0))}/{int(row.get('indicators_total', 0))}</td>"
            f'<td class="bar-cell">{bar(row["family_score"], "family")}</td>'
            f'<td class="bar-cell">{bar(row["risk_on"], "risk-on")}</td>'
            f'<td class="bar-cell">{bar(row["choppy"], "choppy")}</td>'
            f'<td class="bar-cell">{bar(row["risk_off"], "risk-off")}</td>'
            f'<td class="bar-cell">{bar(row["local_top"], "local-top")}</td>'
            f'<td class="bar-cell">{bar(row["local_bottom"], "local-bottom")}</td>'
            f"<td>{row['read']}</td>"
            "</tr>"
        )
    html = (
        '<table class="simple-table"><thead><tr>'
        "<th>Family</th><th>Weight</th><th>Indicators met</th><th>Family score</th>"
        "<th>Risk-on</th><th>Choppy</th><th>Risk-off</th><th>Local top</th><th>Local bottom</th><th>Read</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    st.markdown(html, unsafe_allow_html=True)


def show_simple_table(rows: list[dict]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{row[h]}</td>" for h in headers) + "</tr>"
    st.markdown(
        f'<table class="simple-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>',
        unsafe_allow_html=True,
    )


def show_indicator_audit(indicators_df: pd.DataFrame) -> None:
    if indicators_df.empty:
        st.info("No indicator data available.")
        return

    for family in indicators_df["family"].drop_duplicates():
        group = indicators_df[indicators_df["family"] == family].copy()
        n = len(group)
        score_pct = round(group["score"].sum() / (n * 2) * 100) if n > 0 else 0
        default_expand = family in ["BTC trend / momentum", "ETF + DAT flows"]
        with st.expander(f"{family} -- {score_pct}% criteria met", expanded=default_expand):
            display_cols = ["indicator", "value", "criteria", "status", "score", "meaning", "regime_fit"]
            available_cols = [c for c in display_cols if c in group.columns]
            table = group[available_cols].rename(columns={
                "indicator": "Indicator", "value": "Current value",
                "criteria": "Criteria", "status": "Status", "score": "Score",
                "meaning": "What it means", "regime_fit": "Regime fit",
            })
            # Add source and asof if available
            if "source" in group.columns:
                table["Source"] = group["source"].values
            if "asof" in group.columns:
                table["As of"] = group["asof"].values

            st.dataframe(
                table, hide_index=True, use_container_width=True,
                height=min(520, 96 + len(table) * 54),
                column_config={
                    "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=2, format="%d / 2")
                },
            )


def show_dat_kpis(dat_df: pd.DataFrame) -> None:
    cols = st.columns(min(3, len(dat_df)))
    for col, (_, row) in zip(cols, dat_df.iterrows()):
        col.metric(
            f"{row['DAT']} {row['Metric']}",
            f"{row['mNAV']:.2f}x",
            row.get("Source status", ""),
        )
    with st.expander("DAT scrape status and sources", expanded=False):
        st.dataframe(dat_df, hide_index=True, use_container_width=True)


def show_market_kpis(kpi_df: pd.DataFrame) -> None:
    rows = [kpi_df.iloc[i: i + 3] for i in range(0, len(kpi_df), 3)]
    for metric_group in rows:
        cols = st.columns(len(metric_group))
        for col, (_, row) in zip(cols, metric_group.iterrows()):
            col.metric(row["KPI"], row["Value"], row["Status"])
    with st.expander("Macro and market KPI details", expanded=False):
        st.dataframe(kpi_df, hide_index=True, use_container_width=True)


@st.cache_data(ttl=15 * 60)
def _fetch_stooq_quote(symbol: str) -> dict:
    try:
        resp = requests.get(
            f"https://stooq.com/q/l/?s={symbol}&f=sd2t2ohlcv&h&e=csv", timeout=10
        )
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        close = df.iloc[0]["Close"]
        if pd.isna(close) or str(close) == "N/D":
            raise ValueError("No quote")
        return {"close": float(close), "status": "Live via Stooq"}
    except Exception:
        return {"close": float("nan"), "status": "Unavailable"}


@st.cache_data(ttl=15 * 60)
def _fetch_yf_history(symbol: str, period: str = "7d"):
    if yf is None:
        return {"close": float("nan"), "history": [], "status": "yfinance not installed"}
    try:
        hist = yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)
        closes = hist["Close"].dropna().astype(float).tolist()
        if not closes:
            raise ValueError
        return {"close": closes[-1], "history": closes, "status": "Live via yfinance"}
    except Exception:
        return {"close": float("nan"), "history": [], "status": "Unavailable"}


# ---------------------------------------------------------------------------
# CoinGecko helpers
# ---------------------------------------------------------------------------

# Map watchlist token names/symbols to CoinGecko IDs.
# CoinGecko IDs are lowercase slugs; this covers the user's watchlist.
_TOKEN_TO_CG_ID = {
    "WET": "wet",
    "JITO": "jito-governance-token",
    "Kamino": "kamino",
    "Hyperlend": "hyperlend",
    "Huma": "huma-finance",
    "CC": "canton-coin",
    "Bonk": "bonk",
    "Plasma": "plasma-2",
    "2Z": "2z",
    "Raydium": "raydium",
    "Fluid": "fluid",
    "DeBridge": "debridge",
    "MagicEden": "magic-eden",
    "Jupiter": "jupiter-exchange-solana",
    "MetaDAO": "meta-dao",
    "Layer3": "layer3",
    "Pendle": "pendle",
    "Grass": "grass",
    "Geodnet": "geodnet",
    "Zora": "zora",
    "TON": "the-open-network",
    "Helium": "helium",
    "Metaplex": "metaplex",
    "Aerodrome": "aerodrome-finance",
    "Morpho": "morpho",
    "Sanctum": "sanctum-2",
    "EtherFi": "ether-fi",
    "Spark": "spark",
    "Umbra": "umbra-exchange",
    "Lighter": "lighter",
    "NEAR": "near",
    "Derive": "derive-2",
    "Turtle": "turtle",
    "Meteora": "meteora",
}


@st.cache_data(ttl=5 * 60)
def _fetch_coingecko_batch(cg_ids: tuple[str, ...], api_key: str) -> dict:
    """
    Batch-fetch price, 7d sparkline, and 24h/7d change from CoinGecko /coins/markets.
    Returns {cg_id: {price, sparkline_7d, price_change_7d_pct}}.
    """
    if not cg_ids or not api_key:
        return {}

    results = {}
    # Detect Pro vs Demo key. Pro keys start with "CG-".
    is_pro = api_key.startswith("CG-")
    base_url = "https://pro-api.coingecko.com/api/v3" if is_pro else "https://api.coingecko.com/api/v3"
    header_key = "x-cg-pro-api-key" if is_pro else "x-cg-demo-api-key"

    batch_size = 100
    for i in range(0, len(cg_ids), batch_size):
        chunk = cg_ids[i: i + batch_size]
        ids_str = ",".join(chunk)
        try:
            resp = requests.get(
                f"{base_url}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "ids": ids_str,
                    "order": "market_cap_desc",
                    "sparkline": "true",
                    "price_change_percentage": "7d",
                },
                headers={header_key: api_key},
                timeout=20,
            )
            resp.raise_for_status()
            for coin in resp.json():
                cid = coin["id"]
                spark = coin.get("sparkline_in_7d", {}).get("price", [])
                results[cid] = {
                    "price": coin.get("current_price"),
                    "sparkline_7d": spark,
                    "price_change_7d_pct": coin.get("price_change_percentage_7d_in_currency"),
                }
        except Exception as e:
            logger.warning(f"CoinGecko batch fetch failed: {e}")
    return results


@st.cache_data(ttl=5 * 60)
def load_watchlist_prices(
    api_key: str = "",
    enable_coingecko: bool = False,
    watchlist_path: str = WATCHLIST_PATH,
) -> pd.DataFrame:
    """Load and price the watchlist. Tokens via CoinGecko, equities via yfinance."""
    watchlist = _load_watchlist_csv(watchlist_path)

    # Build CoinGecko ID list for all tokens
    cg_id_map = {}  # row_idx -> cg_id
    cg_ids_needed = []
    for idx, row in watchlist.iterrows():
        is_token = str(row.get("asset_type", "")).lower() == "token"
        if is_token:
            name = row["name"]
            cg_id = _TOKEN_TO_CG_ID.get(name)
            if cg_id:
                cg_id_map[idx] = cg_id
                cg_ids_needed.append(cg_id)

    # Batch fetch from CoinGecko (always, if we have a key)
    cg_data = {}
    if api_key and cg_ids_needed:
        cg_data = _fetch_coingecko_batch(tuple(cg_ids_needed), api_key)

    rows = []
    for idx, row in watchlist.iterrows():
        symbol = row["symbol"]
        is_token = str(row.get("asset_type", "")).lower() == "token"

        price = float("nan")
        sparkline = []
        wow_pct = float("nan")
        source = "Unavailable"

        if is_token:
            cg_id = cg_id_map.get(idx)
            if cg_id and cg_id in cg_data:
                d = cg_data[cg_id]
                if d["price"] is not None:
                    price = d["price"]
                    source = "CoinGecko"
                if d["sparkline_7d"]:
                    sparkline = d["sparkline_7d"]
                if d["price_change_7d_pct"] is not None:
                    wow_pct = d["price_change_7d_pct"]
        else:
            # Equity: yfinance / Stooq
            if yf is not None:
                result = _fetch_yf_history(symbol)
                if pd.notna(result["close"]):
                    price = result["close"]
                    sparkline = result["history"]
                    source = result["status"]
                    if len(sparkline) >= 2 and sparkline[0] > 0:
                        wow_pct = (sparkline[-1] / sparkline[0] - 1) * 100
                else:
                    sq = _fetch_stooq_quote(f"{symbol.lower()}.us")
                    if pd.notna(sq["close"]):
                        price, source = sq["close"], sq["status"]

        output = {
            "Type": row.get("asset_type", "token"),
            "Name": row["name"],
            "Symbol": symbol,
            "Price": price,
            "WoW %": wow_pct,
            "7d sparkline": sparkline if sparkline else [],
            "Source": source,
        }
        for extra in ["chain", "portco", "relationship", "note"]:
            if extra in row.index:
                output[extra.replace("_", " ").title()] = row.get(extra, "")
        rows.append(output)

    return pd.DataFrame(rows)


def _load_watchlist_csv(path: str) -> pd.DataFrame:
    fallback = pd.DataFrame([
        {"asset_type": "token", "name": "Bitcoin", "symbol": "BTC", "coingecko_id": "bitcoin"},
        {"asset_type": "token", "name": "Ethereum", "symbol": "ETH", "coingecko_id": "ethereum"},
        {"asset_type": "equity", "name": "Strategy", "symbol": "MSTR", "coingecko_id": ""},
        {"asset_type": "equity", "name": "BitMine", "symbol": "BMNR", "coingecko_id": ""},
        {"asset_type": "equity", "name": "Strategy STRC", "symbol": "STRC", "coingecko_id": ""},
    ])
    if not os.path.exists(path):
        return fallback
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        if {"asset_type", "name", "symbol"}.issubset(df.columns):
            return df
        if "Token" in df.columns:
            tdf = df.dropna(subset=["Token"]).copy()
            tdf["Token"] = tdf["Token"].astype(str).str.strip()
            tdf = tdf[tdf["Token"] != ""]
            tdf["asset_type"] = "token"
            tdf["name"] = tdf["Token"]
            tdf["symbol"] = tdf["Token"]
            tdf["coingecko_id"] = ""
            rename_map = {"Chain": "chain", "Portco": "portco",
                          "Existing relationship": "relationship", "Note": "note"}
            tdf = tdf.rename(columns=rename_map)
            extra = [c for c in ["chain", "portco", "relationship", "note"] if c in tdf.columns]
            return tdf[["asset_type", "name", "symbol", "coingecko_id"] + extra]
    except Exception:
        pass
    return fallback


def show_watchlist(api_key: str, enable_coingecko: bool) -> None:
    wdf = load_watchlist_prices(api_key, enable_coingecko)
    st.dataframe(
        wdf, hide_index=True, use_container_width=True,
        column_config={
            "Price": st.column_config.NumberColumn("Price", format="$%.4f"),
            "WoW %": st.column_config.NumberColumn("WoW %", format="%.1f%%"),
            "7d sparkline": st.column_config.LineChartColumn("7d sparkline", y_min=0, y_max=None),
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    inject_css()

    with st.sidebar:
        st.header("Classifier settings")
        default_cg_key = os.getenv("COINGECKO_API_KEY", "")
        api_key = st.text_input(
            "CoinGecko API key", type="password",
            value=default_cg_key,
            help="Set COINGECKO_API_KEY in env or paste here. Pro keys (CG-...) use the Pro API.",
        )
        enable_coingecko = st.checkbox(
            "Enable CoinGecko pricing for watchlist",
            value=bool(default_cg_key),
            help="When enabled, token identifiers are sent to CoinGecko for live prices.",
        )
        risk_on_thresh = st.slider("Risk-on threshold", 50, 90, 65)
        risk_off_thresh = st.slider("Risk-off threshold", 50, 90, 65)
        overlay_thresh = st.slider("Local overlay threshold", 20, 60, 40)
        st.checkbox("Use Choppy as the display label", value=True)
        st.checkbox("Treat local top/bottom as overlays", value=True)
        st.caption("The safest architecture is a small core regime model plus transparent overlays.")

    # ── Compute regime ────────────────────────────────────────────
    with st.spinner("Computing regime classification..."):
        regime_result = compute_current_regime()

    regime = regime_result.get("regime", "Choppy")
    summary = regime_result.get("summary_text", "")
    families_df = regime_result.get("family_scores_df", pd.DataFrame())
    indicators_df = regime_result.get("indicator_audit_df", pd.DataFrame())
    btc_weekly = regime_result.get("btc_weekly", pd.Series())

    btc_latest = btc_weekly.iloc[-1] if not btc_weekly.empty else None

    # Determine data source status
    live_sources = 0
    if not indicators_df.empty and "source" in indicators_df.columns:
        live_sources = (indicators_df["source"] != "Fallback").sum()
    total_indicators = len(indicators_df) if not indicators_df.empty else 0

    source_chip = "good" if live_sources > total_indicators * 0.7 else "warn" if live_sources > 0 else "blue"
    source_label = f"{live_sources}/{total_indicators} live sources" if total_indicators > 0 else "Mock pipeline data"

    st.markdown(
        f'<span class="chip">Weekly classifier</span>'
        f'<span class="chip warn">Display label: Choppy</span>'
        f'<span class="chip {source_chip}">{source_label}</span>',
        unsafe_allow_html=True,
    )

    regime_color_map = {
        "Risk-on": "var(--green)", "Risk-off": "var(--red)",
        "Choppy": "var(--muted)", "Local top": "var(--amber)", "Local bottom": "var(--blue)",
    }

    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">Current regime call</div>
          <div class="regime" style="color: {regime_color_map.get(regime, 'var(--text)')}">{regime}</div>
          <div class="copy">{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    metric_row(regime_result, btc_latest)

    st.markdown("## Regime playbook")
    st.caption("What each regime means for factor emphasis and exposure.")
    show_simple_table(REGIME_PLAYBOOK)

    st.markdown("## Family explainer and weights")
    st.caption("Families group indicators by economic role.")
    show_simple_table([{**r, "Weight": f"{r['Weight']}%"} for r in FAMILY_EXPLAINERS])

    st.markdown("## DAT mNAV and credit KPIs")
    st.caption("Weekly scrape targets for digital asset treasury companies.")
    dat_df = load_dat_mnav()
    show_dat_kpis(dat_df)

    st.markdown("## Macro and market structure KPIs")
    st.caption("DXY, gold, M2, and STRC preferred-stack monitoring.")
    with st.spinner("Fetching macro KPIs..."):
        market_kpis = build_market_kpis_df()
    show_market_kpis(market_kpis)

    st.markdown("## BTC price and weekly regime shading")
    with st.spinner("Computing historical regime classifications..."):
        weekly_df = compute_weekly_regime_df(btc_weekly)
    if not weekly_df.empty:
        st.altair_chart(regime_chart(weekly_df), use_container_width=True)
        st.caption("Regime shading is computed from BTC trend indicators. Full multi-family classification applies to the current week only.")
    else:
        st.info("Insufficient BTC data for historical regime chart.")

    st.markdown("## Regime evidence by family")
    st.caption("Family contribution shows which regime each family currently supports.")
    if not families_df.empty:
        show_family_scores(families_df)

    st.markdown("## Indicator audit")
    st.caption("Every row: current value, threshold, status, score, meaning, regime fit, source, as-of date.")
    show_indicator_audit(indicators_df)

    st.markdown("## Token and equity watchlist")
    st.caption("Token prices via CoinGecko. Equity prices via yfinance/Stooq.")
    show_watchlist(api_key, enable_coingecko)


if __name__ == "__main__":
    main()
