"""
Regime classifier.

Takes indicator-level outputs and produces:
1. Family-level scores and regime contributions
2. Weighted overall regime scores
3. Final regime classification: Risk-on, Choppy, Risk-off
4. Local top / Local bottom overlays

Classification logic (from spec):
- Risk-on if weighted risk-on evidence >= 65 and leads choppy/risk-off by >= 10 pts
- Risk-off if weighted risk-off evidence >= 65 or portfolio drawdown defensive override
- Choppy otherwise
- Local top overlay if local-top evidence crosses threshold and risk-on is extended
- Local bottom overlay if local-bottom evidence crosses threshold after stress
"""
from __future__ import annotations

import pandas as pd
import numpy as np


FAMILIES_ORDERED = [
    "BTC trend / momentum",
    "Market breadth",
    "ETF + DAT flows",
    "Stablecoin liquidity",
    "Leverage / volatility",
    "Macro / AI",
    "Valuation / sentiment",
]

FAMILY_WEIGHTS = {
    "BTC trend / momentum": 20,
    "Market breadth": 15,
    "ETF + DAT flows": 30,
    "Stablecoin liquidity": 10,
    "Leverage / volatility": 10,
    "Macro / AI": 10,
    "Valuation / sentiment": 5,
}


def classify_regime(
    indicator_results: list[dict],
    risk_on_threshold: float = 65.0,
    risk_off_threshold: float = 65.0,
    lead_margin: float = 10.0,
    local_overlay_threshold: float = 40.0,
    portfolio_dd_override: float = -15.0,
) -> dict:
    """
    Run full classification from indicator results.

    Returns dict with:
        regime: str  (Risk-on, Choppy, Risk-off, Local top, Local bottom)
        risk_on_score, choppy_score, risk_off_score: float 0-100
        local_top_score, local_bottom_score: float 0-100
        family_scores_df: pd.DataFrame
        indicator_audit_df: pd.DataFrame
        summary_text: str
    """
    # ── 1. Build indicator audit df ───────────────────────────────
    audit_rows = []
    for ind in indicator_results:
        row = {k: v for k, v in ind.items() if k != "regime_votes"}
        audit_rows.append(row)

    indicator_audit_df = pd.DataFrame(audit_rows)

    # ── 2. Compute family-level aggregation ───────────────────────
    family_rows = []
    for fam in FAMILIES_ORDERED:
        fam_indicators = [i for i in indicator_results if i["family"] == fam]
        n = len(fam_indicators)
        if n == 0:
            family_rows.append(_empty_family(fam))
            continue

        total_score = sum(i["score"] for i in fam_indicators)
        met_count = sum(1 for i in fam_indicators if i["score"] == 2)
        family_score = 100 * total_score / (2 * n)

        # Aggregate regime votes (average across indicators in family)
        vote_keys = ["risk_on", "choppy", "risk_off", "local_top", "local_bottom"]
        avg_votes = {}
        for key in vote_keys:
            vals = [i["regime_votes"][key] for i in fam_indicators if "regime_votes" in i]
            avg_votes[key] = np.mean(vals) if vals else 0.0

        # Generate family read
        dominant = max(vote_keys[:3], key=lambda k: avg_votes[k])
        read = _generate_family_read(fam, family_score, avg_votes, met_count, n)

        family_rows.append({
            "family": fam,
            "weight": FAMILY_WEIGHTS.get(fam, 0),
            "indicators_met": met_count,
            "indicators_total": n,
            "family_score": round(family_score, 1),
            "risk_on": round(avg_votes["risk_on"], 1),
            "choppy": round(avg_votes["choppy"], 1),
            "risk_off": round(avg_votes["risk_off"], 1),
            "local_top": round(avg_votes["local_top"], 1),
            "local_bottom": round(avg_votes["local_bottom"], 1),
            "read": read,
        })

    family_scores_df = pd.DataFrame(family_rows)

    # ── 3. Weighted regime scores ─────────────────────────────────
    total_weight = sum(FAMILY_WEIGHTS.get(f, 0) for f in FAMILIES_ORDERED)
    weighted_scores = {
        "risk_on": 0.0, "choppy": 0.0, "risk_off": 0.0,
        "local_top": 0.0, "local_bottom": 0.0,
    }
    for _, row in family_scores_df.iterrows():
        w = row["weight"] / total_weight
        for key in weighted_scores:
            weighted_scores[key] += row[key] * w

    risk_on = weighted_scores["risk_on"]
    choppy = weighted_scores["choppy"]
    risk_off = weighted_scores["risk_off"]
    local_top = weighted_scores["local_top"]
    local_bottom = weighted_scores["local_bottom"]

    # ── 4. Classification ─────────────────────────────────────────
    # Check portfolio drawdown override
    dd_indicators = [i for i in indicator_results if i["indicator"] == "Portfolio drawdown"]
    dd_override = False
    if dd_indicators:
        dd_val = dd_indicators[0].get("value", "0%")
        try:
            dd_pct = float(dd_val.replace("%", ""))
            dd_override = dd_pct <= portfolio_dd_override
        except (ValueError, AttributeError):
            pass

    if dd_override:
        regime = "Risk-off"
        summary = "Portfolio drawdown defensive override is active."
    elif risk_off >= risk_off_threshold:
        regime = "Risk-off"
        summary = f"Risk-off evidence ({risk_off:.0f}%) exceeds threshold ({risk_off_threshold:.0f}%)."
    elif risk_on >= risk_on_threshold and (risk_on - choppy) >= lead_margin and (risk_on - risk_off) >= lead_margin:
        regime = "Risk-on"
        summary = f"Risk-on evidence ({risk_on:.0f}%) leads with sufficient margin."
    else:
        regime = "Choppy"
        summary = f"Evidence is mixed (risk-on {risk_on:.0f}%, choppy {choppy:.0f}%, risk-off {risk_off:.0f}%). No clear signal dominates."

    # Local overlays
    if local_top >= local_overlay_threshold and risk_on >= 50:
        regime = "Local top"
        summary = f"Local top overlay active ({local_top:.0f}%) during extended risk-on ({risk_on:.0f}%)."
    elif local_bottom >= local_overlay_threshold and risk_off >= 40:
        regime = "Local bottom"
        summary = f"Local bottom overlay active ({local_bottom:.0f}%) after stress period."

    return {
        "regime": regime,
        "risk_on_score": round(risk_on, 1),
        "choppy_score": round(choppy, 1),
        "risk_off_score": round(risk_off, 1),
        "local_top_score": round(local_top, 1),
        "local_bottom_score": round(local_bottom, 1),
        "family_scores_df": family_scores_df,
        "indicator_audit_df": indicator_audit_df,
        "summary_text": summary,
    }


def classify_weekly_history(
    btc_weekly: pd.Series,
    indicator_func,
    start_date: str = "2025-01-06",
) -> pd.DataFrame:
    """
    Compute historical weekly regime classifications.

    Args:
        btc_weekly: Full BTC weekly close series.
        indicator_func: Callable that takes (btc_weekly_up_to_date) and returns
                        list[dict] of indicator results. Must handle partial data.
        start_date: First week to classify.

    Returns:
        DataFrame matching weekly_regime_df contract.
    """
    start = pd.Timestamp(start_date)
    records = []

    for i in range(len(btc_weekly)):
        dt = btc_weekly.index[i]
        if dt < start:
            continue

        # Slice data up to this week
        btc_slice = btc_weekly.iloc[: i + 1]
        if len(btc_slice) < 21:
            continue

        # Compute indicators with data available up to this point
        try:
            indicators = indicator_func(btc_slice)
            result = classify_regime(indicators)
        except Exception:
            result = {
                "regime": "Choppy",
                "risk_on_score": 50, "choppy_score": 50, "risk_off_score": 30,
                "local_top_score": 10, "local_bottom_score": 10,
            }

        records.append({
            "date": dt,
            "btc_close": btc_weekly.iloc[i],
            "regime": result["regime"],
            "risk_on_score": result["risk_on_score"],
            "choppy_score": result["choppy_score"],
            "risk_off_score": result["risk_off_score"],
            "local_top_score": result["local_top_score"],
            "local_bottom_score": result["local_bottom_score"],
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df["end_date"] = df["date"].shift(-1)
        df.loc[df.index[-1], "end_date"] = df.loc[df.index[-1], "date"] + pd.Timedelta(days=7)
    return df


def _empty_family(fam):
    return {
        "family": fam, "weight": FAMILY_WEIGHTS.get(fam, 0),
        "indicators_met": 0, "indicators_total": 0, "family_score": 0.0,
        "risk_on": 0.0, "choppy": 100.0, "risk_off": 0.0,
        "local_top": 0.0, "local_bottom": 0.0,
        "read": "No indicator data available for this family.",
    }


def _generate_family_read(
    family: str,
    score: float,
    votes: dict,
    met: int,
    total: int,
) -> str:
    """Generate a one-sentence read for a family."""
    dominant = max(["risk_on", "choppy", "risk_off"], key=lambda k: votes[k])

    if score >= 75:
        strength = "strongly supportive"
    elif score >= 50:
        strength = "moderately supportive"
    elif score >= 25:
        strength = "mixed"
    else:
        strength = "weak"

    regime_label = {"risk_on": "risk-on", "choppy": "choppy", "risk_off": "risk-off"}[dominant]

    return f"{family} is {strength} ({met}/{total} met); primary lean is {regime_label}."
