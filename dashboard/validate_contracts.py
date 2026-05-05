"""
Validate that all data contracts are met.

Run standalone:
    python3 -m dashboard.validate_contracts

Checks:
1. indicator_audit_df has all required columns
2. family_scores_df has all required columns and valid ranges
3. Every indicator has score in {0, 1, 2}
4. Every indicator has regime_votes keys
5. No "Connector pending" rows
6. All 7 families are represented
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def validate():
    from dashboard.regime_indicators import (
        btc_vs_20w_sma, sma_20w_slope, btc_4w_return, btc_12w_return,
        stablecoin_supply_growth, realised_vol_8w,
        funding_rates_indicator, oi_to_mcap_indicator, basis_indicator,
        spy_qqq_trend, dxy_trend_indicator, real_yields_indicator, ai_macro_bias,
        mvrv_nupl_indicator, retail_froth_indicator, portfolio_drawdown_indicator,
    )
    from dashboard.regime_etf_dat import (
        btc_etf_flows, eth_etf_flows, mstr_mnav, bmnr_mnav, purr_mnav,
    )
    from dashboard.regime_classifier import classify_regime, FAMILIES_ORDERED
    import pandas as pd
    import numpy as np

    print("=" * 60)
    print("CONTRACT VALIDATION")
    print("=" * 60)

    errors = []

    # ── 1. Test indicator output shape ────────────────────────────
    print("\n1. Testing indicator output contracts...")

    # Synthetic BTC data for testing
    dates = pd.date_range("2024-01-01", periods=200, freq="W-MON")
    btc = pd.Series(
        np.random.lognormal(11.2, 0.05, 200).cumsum() / 200 * 70000,
        index=dates,
    )

    required_keys = {
        "family", "indicator", "value", "criteria", "status",
        "score", "meaning", "regime_fit", "source", "asof", "regime_votes",
    }
    vote_keys = {"risk_on", "choppy", "risk_off", "local_top", "local_bottom"}

    test_indicators = [
        btc_vs_20w_sma(btc),
        sma_20w_slope(btc),
        btc_4w_return(btc),
        btc_12w_return(btc),
        stablecoin_supply_growth(None),
        realised_vol_8w(pd.Series(np.random.lognormal(11, 0.02, 400), index=pd.date_range("2024-01-01", periods=400))),
        funding_rates_indicator(6.5),
        oi_to_mcap_indicator(3.8),
        basis_indicator(7.0),
        spy_qqq_trend(None, None),
        dxy_trend_indicator(105.0, 1.5),
        real_yields_indicator(15.0),
        ai_macro_bias(5.0),
        mvrv_nupl_indicator(2.1, 0.47),
        retail_froth_indicator(145),
        portfolio_drawdown_indicator(-7.0),
    ]

    # DAT indicators
    dat_df = pd.DataFrame([
        {"DAT": "MSTR", "Metric": "mNAV", "mNAV": 1.34, "Source": "test", "Source status": "test"},
        {"DAT": "BMNR", "Metric": "mNAV", "mNAV": 1.02, "Source": "test", "Source status": "test"},
        {"DAT": "PURR", "Metric": "mNAV", "mNAV": 0.94, "Source": "test", "Source status": "test"},
    ])
    test_indicators.extend([mstr_mnav(dat_df), bmnr_mnav(dat_df), purr_mnav(dat_df)])

    for ind in test_indicators:
        name = f"{ind['family']}/{ind['indicator']}"
        missing = required_keys - set(ind.keys())
        if missing:
            errors.append(f"  FAIL: {name} missing keys: {missing}")
        if ind["score"] not in {0, 1, 2}:
            errors.append(f"  FAIL: {name} score={ind['score']} not in {{0,1,2}}")
        if ind["status"] not in {"Met", "Partial", "Not met"}:
            errors.append(f"  FAIL: {name} status='{ind['status']}' invalid")
        if "regime_votes" in ind:
            missing_votes = vote_keys - set(ind["regime_votes"].keys())
            if missing_votes:
                errors.append(f"  FAIL: {name} missing vote keys: {missing_votes}")
        if "Connector pending" in str(ind.get("source", "")):
            errors.append(f"  FAIL: {name} has 'Connector pending' source")

    print(f"  Tested {len(test_indicators)} indicators")

    # ── 2. Test classifier output ─────────���───────────────────────
    print("\n2. Testing classifier output...")

    result = classify_regime(test_indicators)

    required_result_keys = {
        "regime", "risk_on_score", "choppy_score", "risk_off_score",
        "local_top_score", "local_bottom_score",
        "family_scores_df", "indicator_audit_df", "summary_text",
    }
    missing = required_result_keys - set(result.keys())
    if missing:
        errors.append(f"  FAIL: classifier result missing: {missing}")

    if result["regime"] not in {"Risk-on", "Choppy", "Risk-off", "Local top", "Local bottom"}:
        errors.append(f"  FAIL: regime='{result['regime']}' invalid")

    for key in ["risk_on_score", "choppy_score", "risk_off_score", "local_top_score", "local_bottom_score"]:
        val = result.get(key, -1)
        if not (0 <= val <= 100):
            errors.append(f"  FAIL: {key}={val} out of range [0, 100]")

    # ── 3. Test family_scores_df ───────��──────────────────────────
    print("\n3. Testing family_scores_df contract...")

    fdf = result["family_scores_df"]
    required_fdf_cols = {
        "family", "weight", "indicators_met", "indicators_total",
        "family_score", "risk_on", "choppy", "risk_off",
        "local_top", "local_bottom", "read",
    }
    missing_cols = required_fdf_cols - set(fdf.columns)
    if missing_cols:
        errors.append(f"  FAIL: family_scores_df missing columns: {missing_cols}")

    families_present = set(fdf["family"].tolist())
    families_expected = set(FAMILIES_ORDERED)
    missing_families = families_expected - families_present
    if missing_families:
        errors.append(f"  FAIL: missing families: {missing_families}")

    # ── 4. Test indicator_audit_df ─────────��──────────────────────
    print("\n4. Testing indicator_audit_df contract...")

    idf = result["indicator_audit_df"]
    required_idf_cols = {"family", "indicator", "value", "criteria", "status", "score", "meaning", "regime_fit"}
    missing_cols = required_idf_cols - set(idf.columns)
    if missing_cols:
        errors.append(f"  FAIL: indicator_audit_df missing columns: {missing_cols}")

    connector_pending = idf[idf.apply(lambda r: "Connector pending" in str(r.values), axis=1)]
    if len(connector_pending) > 0:
        errors.append(f"  FAIL: {len(connector_pending)} rows have 'Connector pending'")

    # ── Summary ─────────��───────────────────────────���─────────────
    print(f"\n{'=' * 60}")
    if errors:
        print(f"FAILED: {len(errors)} errors")
        for e in errors:
            print(e)
        return False
    else:
        print("ALL CONTRACTS PASSED")
        print(f"  Indicators: {len(test_indicators)}")
        print(f"  Families: {len(fdf)}")
        print(f"  Regime: {result['regime']}")
        print(f"  Scores: risk-on={result['risk_on_score']:.0f}%, "
              f"choppy={result['choppy_score']:.0f}%, "
              f"risk-off={result['risk_off_score']:.0f}%")
        return True


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
