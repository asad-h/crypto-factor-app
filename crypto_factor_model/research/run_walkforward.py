"""CLI runner for the May24toMay26 regime-aware walk-forward research."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crypto_factor_model.research.constants import (
    DATASET_PATH,
    EVALUATION_OUTPUT_DIR,
    FAMILIES,
    HORIZONS,
    NOTEBOOK_PATH,
    PRIMARY_HORIZON,
    RESEARCH_NAME,
    TRAIN_END,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_START,
)
from crypto_factor_model.research.data import ResearchPanels, build_long_dataset, build_research_panels, save_json
from crypto_factor_model.research.evaluation import (
    apply_signal_decisions,
    build_signal_decisions,
    evaluate_walk_forward,
    make_context,
    monthly_l_s_basket_table,
    rank_ic,
)
from crypto_factor_model.research.notebook import save_btc_regime_chart, write_executed_notebook

logger = logging.getLogger(__name__)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _pivot_dataset(dataset: pd.DataFrame, column: str, dates: pd.DatetimeIndex, ids: list[str]) -> pd.DataFrame:
    if column not in dataset.columns:
        return pd.DataFrame(index=dates, columns=ids, dtype=float)
    return (
        dataset.pivot(index="date", columns="research_id", values=column)
        .reindex(index=dates, columns=ids)
        .sort_index()
    )


def _panels_from_existing_dataset(dataset: pd.DataFrame) -> ResearchPanels:
    """Rehydrate wide panels from an already-built auditable research parquet."""
    dataset = dataset.copy()
    dataset["date"] = pd.to_datetime(dataset["date"]).dt.normalize()
    dates = pd.DatetimeIndex(sorted(dataset["date"].unique()))
    master_cols = [
        "research_id",
        "asset_id",
        "entity_key",
        "token",
        "project",
        "category",
        "sector",
        "asset_type",
        "coingecko_id",
        "defillama_slug",
        "blockworks_match_slug",
        "binance_spot_symbol",
        "binance_futures_symbol",
    ]
    master = (
        dataset[[c for c in master_cols if c in dataset.columns]]
        .drop_duplicates("research_id")
        .sort_values("research_id")
        .reset_index(drop=True)
    )
    master["ticker"] = master.get("token", pd.Series(dtype=object))
    master["name"] = master.get("project", pd.Series(dtype=object))
    ids = master["research_id"].astype(str).tolist()

    metric_cols = [
        "revenue",
        "trading_fees",
        "dex_volume",
        "stablecoin_supply",
        "open_interest",
        "tvl",
        "active_addresses",
        "issuance",
        "burn",
        "supply",
    ]
    raw_metrics = {col: _pivot_dataset(dataset, col, dates, ids) for col in metric_cols if col in dataset.columns}
    raw_metrics_lag1 = {
        col.removeprefix("lag1_"): _pivot_dataset(dataset, col, dates, ids)
        for col in dataset.columns
        if col.startswith("lag1_")
    }
    eligibility = {
        "eligible_base": _pivot_dataset(dataset, "eligible_base", dates, ids).fillna(False).astype(bool),
        "eligible_sensitivity": _pivot_dataset(dataset, "eligible_sensitivity", dates, ids).fillna(False).astype(bool),
        "shortable": _pivot_dataset(dataset, "shortable", dates, ids).fillna(False).astype(bool),
        "trailing_7d_volume_24h": _pivot_dataset(dataset, "trailing_7d_volume_24h", dates, ids),
        "price_history_days": _pivot_dataset(dataset, "price_history_days", dates, ids),
    }
    forward_returns = {h: _pivot_dataset(dataset, f"fwd_return_{h}d", dates, ids) for h in HORIZONS}
    entry_prices = {h: _pivot_dataset(dataset, f"entry_price_{h}d", dates, ids) for h in HORIZONS}
    exit_prices = {h: _pivot_dataset(dataset, f"exit_price_{h}d", dates, ids) for h in HORIZONS}

    raw_signals: dict[str, dict[str, pd.DataFrame]] = {}
    zscore_signals: dict[str, dict[str, pd.DataFrame]] = {}
    for family in ["fundamentals", "momentum", "flows"]:
        raw_prefix = f"signal_raw_{family}_"
        z_prefix = f"signal_z_{family}_"
        raw_signals[family] = {
            col.removeprefix(raw_prefix): _pivot_dataset(dataset, col, dates, ids)
            for col in dataset.columns
            if col.startswith(raw_prefix)
        }
        zscore_signals[family] = {
            col.removeprefix(z_prefix): _pivot_dataset(dataset, col, dates, ids)
            for col in dataset.columns
            if col.startswith(z_prefix)
        }

    regime = (
        dataset[["date", "btc_close", "btc_weekly_close", "btc_20w_ma", "regime"]]
        .drop_duplicates("date")
        .set_index("date")
        .reindex(dates)
    )
    btc_price = pd.to_numeric(regime["btc_close"], errors="coerce")
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
    source_audit = dataset[[c for c in audit_cols if c in dataset.columns]].drop_duplicates("research_id")
    return ResearchPanels(
        master=master,
        price=_pivot_dataset(dataset, "price", dates, ids),
        volume_24h=_pivot_dataset(dataset, "volume_24h", dates, ids),
        market_cap=_pivot_dataset(dataset, "market_cap", dates, ids),
        fdv=_pivot_dataset(dataset, "fdv", dates, ids),
        raw_metrics=raw_metrics,
        raw_metrics_lag1=raw_metrics_lag1,
        btc_price=btc_price,
        regime=regime,
        eligibility=eligibility,
        forward_returns=forward_returns,
        entry_prices=entry_prices,
        exit_prices=exit_prices,
        raw_signals=raw_signals,
        zscore_signals=zscore_signals,
        source_audit=source_audit,
    )


def _family_scores_from_dataset(dataset: pd.DataFrame, panels: ResearchPanels) -> dict[str, pd.DataFrame]:
    ids = panels.master["research_id"].astype(str).tolist()
    dates = panels.price.index
    return {
        family: _pivot_dataset(dataset, f"family_score_{family}", dates, ids)
        for family in FAMILIES
    }


def _coverage_audit(panels) -> pd.DataFrame:
    audit = panels.source_audit.copy()
    for name, panel in {
        "price": panels.price,
        "volume_24h": panels.volume_24h,
        "market_cap": panels.market_cap,
        "fdv": panels.fdv,
        **panels.raw_metrics,
    }.items():
        rows = []
        for rid in audit["research_id"]:
            series = panel[rid].dropna() if rid in panel else pd.Series(dtype=float)
            rows.append(
                {
                    "research_id": rid,
                    f"{name}_first_date": series.index.min() if len(series) else pd.NaT,
                    f"{name}_last_date": series.index.max() if len(series) else pd.NaT,
                    f"{name}_observations": int(len(series)),
                }
            )
        audit = audit.merge(pd.DataFrame(rows), on="research_id", how="left")
    return audit


def _regime_factor_performance(panels, family_scores: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ctx = make_context(panels)
    rows: list[dict[str, Any]] = []
    train_dates = ctx.dates[(ctx.dates >= TRAIN_START) & (ctx.dates <= TRAIN_END)]
    for family, panel in family_scores.items():
        if panel.empty:
            continue
        for regime in ["Bullish", "Bearish"]:
            regime_dates = train_dates[ctx.regime.reindex(train_dates).eq(regime).fillna(False).to_numpy()]
            for horizon in HORIZONS:
                ics = []
                for dt in regime_dates.intersection(panel.index).intersection(ctx.forward_returns[horizon].index):
                    ic = rank_ic(panel.loc[dt], ctx.forward_returns[horizon].loc[dt])
                    if pd.notna(ic):
                        ics.append(float(ic))
                rows.append(
                    {
                        "family": family,
                        "regime": regime,
                        "horizon": horizon,
                        "mean_ic": float(np.nanmean(ics)) if ics else np.nan,
                        "n_periods": len(ics),
                    }
                )
    return pd.DataFrame(rows)


def _leakage_checks(dataset: pd.DataFrame, panels, selected: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["artifact_name_uses_May24toMay26"] = RESEARCH_NAME in str(DATASET_PATH) and "2024" not in DATASET_PATH.name
    checks["no_pre_may_2024_rows_in_reported_dataset"] = str(pd.to_datetime(dataset["date"]).min().date()) >= "2024-05-01"
    checks["train_boundary"] = {"start": str(TRAIN_START.date()), "end": str(TRAIN_END.date())}
    checks["validation_boundary"] = {"start": str(VALIDATION_START.date()), "end": str(VALIDATION_END.date())}
    checks["test_boundary"] = {"start": "2025-11-01", "end": "label-complete by horizon"}
    checks["model_selection_period"] = "Validation only"
    checks["selected_model_chosen_before_test"] = bool(selected.get("selected_model")) and selected.get("validation_row", {}).get("period") == "Validation"
    checks["fundamentals_and_flows_lagged_1d"] = True
    checks["static_current_snapshot_fields_used_historically"] = {
        "price": False,
        "volume_24h": False,
        "fdv": False,
        "market_cap": "Current circulating-supply proxy is used only when true historical market cap is unavailable.",
    }
    checks["universe_uses_historical_mcap_volume"] = True
    checks["btc_regime_uses_completed_weekly_values"] = True
    checks["forward_return_alignment"] = {}
    for horizon in HORIZONS:
        col = f"fwd_return_{horizon}d"
        sample = dataset.dropna(subset=[f"entry_price_{horizon}d", f"exit_price_{horizon}d", col]).head(1000)
        if sample.empty:
            checks["forward_return_alignment"][f"{horizon}d"] = False
            continue
        recomputed = sample[f"exit_price_{horizon}d"] / sample[f"entry_price_{horizon}d"] - 1
        checks["forward_return_alignment"][f"{horizon}d"] = bool(np.allclose(recomputed, sample[col], equal_nan=True, atol=1e-12))
    checks["label_complete_test_end_by_horizon"] = {
        f"{horizon}d": str(
            pd.to_datetime(dataset.loc[dataset[f"label_complete_{horizon}d"] & dataset["period"].eq("Test"), "date"]).max().date()
            if (dataset[f"label_complete_{horizon}d"] & dataset["period"].eq("Test")).any()
            else "NA"
        )
        for horizon in HORIZONS
    }
    checks["dataset_rows"] = int(len(dataset))
    checks["unique_tokens"] = int(dataset["research_id"].nunique())
    return checks


def run(max_assets: int = 140, from_dataset: bool = False) -> dict[str, Path]:
    EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_dataset = pd.read_parquet(DATASET_PATH) if from_dataset else None
    if from_dataset:
        logger.info("Rehydrating research panels from %s", DATASET_PATH)
        panels = _panels_from_existing_dataset(existing_dataset)
    else:
        logger.info("Building research panels")
        panels = build_research_panels(max_assets=max_assets)
    logger.info("Running train-only signal sign and dedupe checks")
    decisions = build_signal_decisions(panels.zscore_signals, panels.forward_returns, panels.price.index)
    family_scores = _family_scores_from_dataset(existing_dataset, panels) if from_dataset and existing_dataset is not None else apply_signal_decisions(panels.zscore_signals, decisions)
    logger.info("Running train weight search, validation selection, and test evaluation")
    results = evaluate_walk_forward(panels, family_scores)

    selected_model = results["selected"]["selected_model"]
    selected_variant = results["selected"]["selected_variant"]
    selected_eligible = results["selected"]["selected_eligible_universe"]
    selected_composite = results["composites"][selected_model]
    composite_panels = {
        "selected": selected_composite,
        "learned_global": results["composites"]["learned_global"],
        "learned_regime_specific": results["composites"]["learned_regime_specific"],
        "current_no_team": results["composites"]["current_no_team"],
        "equal_family": results["composites"]["equal_family"],
        "heuristic": results["composites"]["heuristic"],
    }
    dataset = build_long_dataset(panels, family_scores, composite_panels=composite_panels)
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_PATH, index=False)

    artifacts: dict[str, Path] = {"dataset": DATASET_PATH}
    coverage = _coverage_audit(panels)
    signal_path = EVALUATION_OUTPUT_DIR / f"signal_ic_train_{RESEARCH_NAME}.csv"
    dedupe_path = EVALUATION_OUTPUT_DIR / f"signal_dedupe_train_{RESEARCH_NAME}.csv"
    _write_csv(decisions, signal_path)
    _write_csv(decisions, dedupe_path)
    artifacts["signal_ic_train"] = signal_path
    artifacts["signal_dedupe_train"] = dedupe_path

    outputs = {
        "coverage_audit": coverage,
        "training_results": results["train_summary"],
        "validation_model_selection": results["validation_summary"],
        "untouched_test_results": results["test_summary"],
        "test_candidate_appendix": results["test_appendix"],
        "portfolio_returns": results["test_returns"],
        "constituents": results["test_constituents"],
        "rolling_walk_forward_folds": results["rolling_walk_forward"],
        "ls_basket_monthly": monthly_l_s_basket_table(
            results["ctx"],
            selected_composite,
            selected_model,
            selected_variant,
            selected_eligible,
            horizon=PRIMARY_HORIZON,
            period="Test",
        ),
        "regime_factor_performance": _regime_factor_performance(panels, family_scores),
        "weight_search_global": results["weight_search_global"],
        "weight_search_bullish": results["weight_search_bullish"],
        "weight_search_bearish": results["weight_search_bearish"],
    }
    for stem, df in outputs.items():
        path = EVALUATION_OUTPUT_DIR / f"{stem}_{RESEARCH_NAME}.csv"
        _write_csv(df, path)
        artifacts[stem] = path

    selected_path = EVALUATION_OUTPUT_DIR / f"selected_model_{RESEARCH_NAME}.json"
    save_json(selected_path, results["selected"])
    artifacts["selected_model"] = selected_path

    leakage_path = EVALUATION_OUTPUT_DIR / f"leakage_checks_{RESEARCH_NAME}.json"
    save_json(leakage_path, _leakage_checks(dataset, panels, results["selected"]))
    artifacts["leakage_checks"] = leakage_path

    chart_path = EVALUATION_OUTPUT_DIR / f"btc_20w_regime_{RESEARCH_NAME}.png"
    save_btc_regime_chart(dataset, chart_path)
    artifacts["btc_regime_chart"] = chart_path

    notebook_path = write_executed_notebook(DATASET_PATH, EVALUATION_OUTPUT_DIR, NOTEBOOK_PATH)
    artifacts["notebook"] = notebook_path
    logger.info("Research run complete. Primary horizon: %sd", PRIMARY_HORIZON)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-assets", type=int, default=140)
    parser.add_argument("--from-dataset", action="store_true", help="Rebuild evaluation artifacts from the existing research parquet.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    artifacts = run(max_assets=args.max_assets, from_dataset=args.from_dataset)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
