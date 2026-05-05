#!/usr/bin/env python3
"""Pull and chart reported hyperscaler capex from SEC XBRL companyconcept JSON.

The metric is reported cash capex where available. Amazon uses the broader
`PaymentsToAcquireProductiveAssets` tag, which matches its cash-flow statement
presentation after it stopped using the narrower PPE tag.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


@dataclass(frozen=True)
class CompanySpec:
    symbol: str
    cik: str
    tag: str

    @property
    def cache_name(self) -> str:
        return f"{self.symbol}_{self.tag}.json"

    @property
    def url(self) -> str:
        return (
            "https://data.sec.gov/api/xbrl/companyconcept/"
            f"CIK{self.cik}/us-gaap/{self.tag}.json"
        )


COMPANIES = [
    CompanySpec("AMZN", "0001018724", "PaymentsToAcquireProductiveAssets"),
    CompanySpec("GOOGL", "0001652044", "PaymentsToAcquirePropertyPlantAndEquipment"),
    CompanySpec("META", "0001326801", "PaymentsToAcquirePropertyPlantAndEquipment"),
    CompanySpec("MSFT", "0000789019", "PaymentsToAcquirePropertyPlantAndEquipment"),
    CompanySpec("ORCL", "0001341439", "PaymentsToAcquirePropertyPlantAndEquipment"),
]

PALETTE = {
    "AMZN": "#4E9CDF",
    "GOOGL": "#7A2E3B",
    "META": "#C7B56B",
    "MSFT": "#625BD6",
    "ORCL": "#2F5B4C",
}


def download(specs: list[CompanySpec], cache_dir: Path, user_agent: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener()
    for spec in specs:
        request = urllib.request.Request(spec.url, headers={"User-Agent": user_agent})
        with opener.open(request, timeout=30) as response:
            payload = response.read()
        (cache_dir / spec.cache_name).write_bytes(payload)
        time.sleep(0.15)


def load_raw_facts(specs: list[CompanySpec], cache_dir: Path) -> pd.DataFrame:
    rows = []
    for spec in specs:
        path = cache_dir / spec.cache_name
        payload = json.loads(path.read_text())
        for fact in payload.get("units", {}).get("USD", []):
            if fact.get("form") not in {"10-K", "10-Q"}:
                continue
            rows.append(
                {
                    "symbol": spec.symbol,
                    "cik": spec.cik,
                    "tag": spec.tag,
                    "entity": payload.get("entityName"),
                    "start": fact.get("start"),
                    "end": fact.get("end"),
                    "value": fact.get("val"),
                    "accession": fact.get("accn"),
                    "fy": fact.get("fy"),
                    "fp": fact.get("fp"),
                    "form": fact.get("form"),
                    "filed": fact.get("filed"),
                    "frame": fact.get("frame"),
                    "source": "reported",
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    for column in ["start", "end", "filed"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    df["duration_days"] = (df["end"] - df["start"]).dt.days + 1
    df["value_bn"] = df["value"] / 1e9
    df = df.dropna(subset=["start", "end", "filed", "value_bn"])
    return df


def latest_by_period(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["symbol", "start", "end", "filed", "accession"]
    return (
        df.sort_values(sort_cols)
        .drop_duplicates(["symbol", "start", "end"], keep="last")
        .reset_index(drop=True)
    )


def derive_quarters_from_ytd(df: pd.DataFrame) -> pd.DataFrame:
    cumulative_mask = (
        df["duration_days"].between(70, 110)
        | df["duration_days"].between(140, 220)
        | df["duration_days"].between(230, 310)
        | ((df["fp"] == "FY") & df["duration_days"].between(340, 390))
    )
    cumulative = latest_by_period(
        df[cumulative_mask].copy()
    ).sort_values(["symbol", "start", "end"])
    derived_rows = []

    for (_symbol, _year_start), group in cumulative.groupby(["symbol", "start"], sort=False):
        group = group.sort_values("end")
        for _, current in group.iterrows():
            if current["duration_days"] <= 110:
                continue
            previous = group[group["end"] < current["end"]].sort_values("end")
            if previous.empty:
                continue
            previous_row = previous.iloc[-1]
            value = current["value"] - previous_row["value"]
            if value < 0:
                continue
            row = current.to_dict()
            row["start"] = previous_row["end"] + pd.Timedelta(days=1)
            row["end"] = current["end"]
            row["value"] = value
            row["value_bn"] = value / 1e9
            row["form"] = "derived"
            row["frame"] = None
            row["source"] = "derived_from_ytd_delta"
            row["duration_days"] = (row["end"] - row["start"]).days + 1
            derived_rows.append(row)

    return pd.DataFrame(derived_rows)


def build_quarterly_series(df: pd.DataFrame) -> pd.DataFrame:
    standalone = df[df["duration_days"].between(70, 110)].copy()
    derived = derive_quarters_from_ytd(df)
    series = pd.concat([standalone, derived], ignore_index=True)
    if series.empty:
        return series
    series["source_rank"] = series["source"].map({"reported": 1}).fillna(0)
    series = (
        series.sort_values(["symbol", "start", "end", "source_rank", "filed", "accession"])
        .drop_duplicates(["symbol", "start", "end"], keep="last")
        .drop(columns=["source_rank"])
        .sort_values(["symbol", "end"])
        .reset_index(drop=True)
    )
    return series


def write_plot(df: pd.DataFrame, path: Path, title: str, x_col: str, x_title: str) -> None:
    fig = go.Figure()
    for symbol in [spec.symbol for spec in COMPANIES]:
        chart = df[df["symbol"] == symbol].sort_values(x_col)
        fig.add_trace(
            go.Scatter(
                x=chart[x_col],
                y=chart["value_bn"],
                mode="lines+markers",
                name=symbol,
                line={"color": PALETTE[symbol], "width": 3},
                marker={"size": 8},
                customdata=chart[["end", "filed", "tag", "source"]].astype(str),
                hovertemplate=(
                    "%{fullData.name}<br>"
                    f"{x_title}: %{{x|%Y-%m-%d}}<br>"
                    "Capex: $%{y:.1f}bn<br>"
                    "Period end: %{customdata[0]}<br>"
                    "Filed: %{customdata[1]}<br>"
                    "Tag: %{customdata[2]}<br>"
                    "Source: %{customdata[3]}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=620,
        width=1050,
        hovermode="x unified",
        legend_title_text="Company",
        margin={"l": 72, "r": 28, "t": 86, "b": 68},
        xaxis_title=x_title,
        yaxis_title="Reported capex / productive asset spend ($bn)",
        font={"family": "Arial, sans-serif", "size": 13},
    )
    fig.update_yaxes(rangemode="tozero", gridcolor="#E6E8EF")
    fig.update_xaxes(gridcolor="#F1F2F6")
    fig.write_html(path, include_plotlyjs=True)


def write_readme(
    output_dir: Path,
    as_of: date,
    quarterly_30d_period_end: pd.DataFrame,
    updates_30d: pd.DataFrame,
    quarterly_1y: pd.DataFrame,
) -> None:
    lines = [
        "# Hyperscaler Capex SEC Test Run",
        "",
        f"As of: {as_of.isoformat()}",
        "",
        "Metric: reported cash capex from SEC XBRL companyconcept data. "
        "Amazon uses `PaymentsToAcquireProductiveAssets`; the other four use "
        "`PaymentsToAcquirePropertyPlantAndEquipment`.",
        "",
        f"Last 30 days by period end: {len(quarterly_30d_period_end)} observations.",
        f"Last 30 days by SEC filing date: {len(updates_30d)} observations.",
        f"Last 1 year by period end: {len(quarterly_1y)} observations.",
        "",
        "Quarterly capex is sparse by design; the 30-day chart is keyed to SEC filing "
        "date because quarterly period-end data usually does not exist inside a "
        "calendar 30-day window.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/hyperscaler_capex")
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--user-agent",
        default="hyperscaler-capex-research contact@example.com",
        help="SEC requests an identifying User-Agent for automated access.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "sec_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    as_of = date.fromisoformat(args.as_of)
    if args.download:
        download(COMPANIES, cache_dir, args.user_agent)

    raw = load_raw_facts(COMPANIES, cache_dir)
    quarterly = build_quarterly_series(raw)

    cutoff_30d = pd.Timestamp(as_of - timedelta(days=30))
    cutoff_1y = pd.Timestamp(as_of - timedelta(days=365))
    as_of_ts = pd.Timestamp(as_of)

    quarterly_1y = quarterly[(quarterly["end"] >= cutoff_1y) & (quarterly["end"] <= as_of_ts)]
    quarterly_30d_period_end = quarterly[
        (quarterly["end"] >= cutoff_30d) & (quarterly["end"] <= as_of_ts)
    ]
    updates_30d = quarterly[
        (quarterly["filed"] >= cutoff_30d)
        & (quarterly["filed"] <= as_of_ts)
        & (quarterly["end"] >= pd.Timestamp(as_of - timedelta(days=180)))
    ]

    raw.to_csv(output_dir / "sec_capex_raw_facts.csv", index=False)
    quarterly.to_csv(output_dir / "sec_capex_quarterly_series.csv", index=False)
    quarterly_1y.to_csv(output_dir / "sec_capex_quarterly_last_1y.csv", index=False)
    quarterly_30d_period_end.to_csv(
        output_dir / "sec_capex_quarterly_last_30d_period_end.csv", index=False
    )
    updates_30d.to_csv(output_dir / "sec_capex_filing_updates_last_30d.csv", index=False)

    latest = quarterly.sort_values(["symbol", "end"]).groupby("symbol").tail(1)
    latest.to_csv(output_dir / "sec_capex_latest_quarter_by_company.csv", index=False)

    write_plot(
        quarterly_1y,
        output_dir / "sec_capex_quarterly_last_1y.html",
        f"Hyperscaler Reported Capex: Last 1 Year by Period End (as of {as_of})",
        "end",
        "Fiscal/quarter period end",
    )
    write_plot(
        updates_30d,
        output_dir / "sec_capex_filing_updates_last_30d.html",
        f"Hyperscaler Reported Capex: SEC Filing Updates in Last 30 Days (as of {as_of})",
        "filed",
        "SEC filing date",
    )
    write_readme(output_dir, as_of, quarterly_30d_period_end, updates_30d, quarterly_1y)

    print(f"Wrote outputs to {output_dir}")
    print(f"last_30d_period_end_rows={len(quarterly_30d_period_end)}")
    print(f"last_30d_filing_update_rows={len(updates_30d)}")
    print(f"last_1y_period_end_rows={len(quarterly_1y)}")


if __name__ == "__main__":
    main()
