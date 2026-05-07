"""Executed notebook writer for the May24toMay26 research artifacts."""
from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from crypto_factor_model.research.constants import DATASET_PATH, EVALUATION_OUTPUT_DIR, NOTEBOOK_PATH, RESEARCH_NAME


def save_btc_regime_chart(dataset: pd.DataFrame, path: Path) -> Path:
    """Render a BTC close vs 20W MA regime chart."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    regime = (
        dataset[["date", "btc_close", "btc_20w_ma", "regime"]]
        .drop_duplicates("date")
        .sort_values("date")
        .dropna(subset=["btc_close"])
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    bull = regime["regime"].eq("Bullish")
    ax.fill_between(regime["date"], regime["btc_close"].min(), regime["btc_close"].max(), where=bull, color="#d6f5df", alpha=0.45)
    ax.fill_between(regime["date"], regime["btc_close"].min(), regime["btc_close"].max(), where=~bull, color="#fde2dd", alpha=0.45)
    ax.plot(regime["date"], regime["btc_close"], label="BTC close", color="#111827", linewidth=1.5)
    ax.plot(regime["date"], regime["btc_20w_ma"], label="20W MA", color="#2563eb", linewidth=1.4)
    ax.set_title(f"BTC 20-Week Regime ({RESEARCH_NAME})")
    ax.set_ylabel("BTC USD")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _table_text(df: pd.DataFrame, max_rows: int = 18, float_format: str = "{:.4f}") -> str:
    if df.empty:
        return "<empty>"
    show = df.head(max_rows).copy()
    for col in show.select_dtypes(include="number").columns:
        show[col] = show[col].map(lambda x: "" if pd.isna(x) else float_format.format(x))
    return show.to_string(index=False)


def _json_text(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _code_cell(source: str, text_output: str | None = None, image_path: Path | None = None, execution_count: int = 1) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    if text_output is not None:
        outputs.append({"output_type": "execute_result", "execution_count": execution_count, "data": {"text/plain": text_output}, "metadata": {}})
    if image_path is not None and image_path.exists():
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        outputs.append({"output_type": "display_data", "data": {"image/png": encoded, "text/plain": "<Figure>"}, "metadata": {}})
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs,
        "source": source,
    }


def _markdown_cell(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text}


def write_executed_notebook(
    dataset_path: Path = DATASET_PATH,
    output_dir: Path = EVALUATION_OUTPUT_DIR,
    notebook_path: Path = NOTEBOOK_PATH,
) -> Path:
    """Create an executed notebook from persisted research outputs."""
    dataset = pd.read_parquet(dataset_path)
    coverage = pd.read_csv(output_dir / f"coverage_audit_{RESEARCH_NAME}.csv")
    leakage = json.loads((output_dir / f"leakage_checks_{RESEARCH_NAME}.json").read_text())
    signal_ic = pd.read_csv(output_dir / f"signal_ic_train_{RESEARCH_NAME}.csv")
    dedupe = pd.read_csv(output_dir / f"signal_dedupe_train_{RESEARCH_NAME}.csv")
    train = pd.read_csv(output_dir / f"training_results_{RESEARCH_NAME}.csv")
    validation = pd.read_csv(output_dir / f"validation_model_selection_{RESEARCH_NAME}.csv")
    test = pd.read_csv(output_dir / f"untouched_test_results_{RESEARCH_NAME}.csv")
    appendix = pd.read_csv(output_dir / f"test_candidate_appendix_{RESEARCH_NAME}.csv")
    monthly_path = output_dir / f"ls_basket_monthly_{RESEARCH_NAME}.csv"
    monthly_basket = pd.read_csv(monthly_path) if monthly_path.exists() else pd.DataFrame()
    rolling_folds_path = output_dir / f"rolling_walk_forward_folds_{RESEARCH_NAME}.csv"
    rolling_folds = pd.read_csv(rolling_folds_path) if rolling_folds_path.exists() else pd.DataFrame()
    selected = json.loads((output_dir / f"selected_model_{RESEARCH_NAME}.json").read_text())
    constituents = pd.read_csv(output_dir / f"constituents_{RESEARCH_NAME}.csv")
    chart_path = output_dir / f"btc_20w_regime_{RESEARCH_NAME}.png"
    if not chart_path.exists():
        save_btc_regime_chart(dataset, chart_path)

    date_coverage = dataset.groupby("period")["date"].agg(["min", "max", "nunique"]).reset_index()
    regime_counts = dataset.drop_duplicates("date")["regime"].value_counts(dropna=False).rename_axis("regime").reset_index(name="days")
    best_signals = signal_ic.sort_values("mean_ic_avg", ascending=False).head(12)
    rejected = dedupe[~dedupe["selected"].astype(bool)].head(18)
    validation_primary = validation[validation["horizon"].eq(selected.get("primary_horizon", 14))].sort_values(
        ["period", "total_return"], ascending=[True, False]
    )
    test_primary = test.sort_values(["horizon", "eligible_universe"])
    long_short_compare = validation[validation["model"].eq("learned_global")].sort_values(["horizon", "variant"])
    btc_relative_cols = [
        "model",
        "variant",
        "horizon",
        "period",
        "eligible_universe",
        "excess_return",
        "beta",
        "alpha",
        "information_ratio",
        "correlation",
        "up_capture",
        "down_capture",
        "drawdown_vs_btc",
    ]
    btc_relative = test[[c for c in btc_relative_cols if c in test]]

    weights = selected.get("weights", {})
    recommendation = {
        "selected_model": selected.get("selected_model"),
        "selected_variant": selected.get("selected_variant"),
        "selected_eligible_universe": selected.get("selected_eligible_universe"),
        "primary_horizon": selected.get("primary_horizon"),
        "selection_status": selected.get("selection_status"),
        "strict_gate_passed": selected.get("strict_gate_passed"),
        "final_recommendation": selected.get("final_recommendation"),
        "selection_reason": selected.get("selection_reason"),
        "learned_global_weights": weights.get("learned_global"),
        "learned_bullish_weights": weights.get("learned_bullish"),
        "learned_bearish_weights": weights.get("learned_bearish"),
    }

    cells: list[dict[str, Any]] = [
        _markdown_cell(f"# Crypto Factor Walk-Forward Research: {RESEARCH_NAME}"),
        _markdown_cell("## TLDR\nSelected model, recommended weights, signal highlights, and whether regime awareness helped."),
        _code_cell("selected_model", _json_text(recommendation), execution_count=1),
        _code_cell("best_train_signals", _table_text(best_signals), execution_count=2),
        _code_cell("rejected_or_deduped_signals", _table_text(rejected), execution_count=3),
        _markdown_cell("## Data Coverage And Source Audit"),
        _code_cell("date_coverage_by_period", _table_text(date_coverage), execution_count=4),
        _code_cell("source_coverage_audit", _table_text(coverage), execution_count=5),
        _markdown_cell("## Leakage And As-Of Checks"),
        _code_cell("leakage_checks", _json_text(leakage), execution_count=6),
        _markdown_cell("## BTC 20W MA Regime"),
        _code_cell("btc_20w_regime_chart", image_path=chart_path, execution_count=7),
        _code_cell("regime_counts", _table_text(regime_counts), execution_count=8),
        _markdown_cell("## Signal Definitions, Sign Audit, Coverage, And Train-Only Dedupe"),
        _code_cell("signal_ic_train", _table_text(signal_ic.sort_values('mean_ic_avg', ascending=False), max_rows=30), execution_count=9),
        _code_cell("signal_dedupe_decisions", _table_text(dedupe, max_rows=35), execution_count=10),
        _markdown_cell("## Training Results"),
        _code_cell("training_results", _table_text(train, max_rows=36), execution_count=11),
        _markdown_cell("## Validation Model Selection"),
        _code_cell("validation_model_selection", _table_text(validation_primary, max_rows=36), execution_count=12),
        _markdown_cell("## Purged Rolling Walk-Forward Folds"),
        _code_cell("rolling_walk_forward_folds", _table_text(rolling_folds, max_rows=50), execution_count=21),
        _markdown_cell("## Current Vs Equal-Family Vs Heuristic Vs Learned Composites"),
        _code_cell("composite_candidates", _table_text(validation[validation['variant'].isin(['always_l_s','regime_aware_l_s'])], max_rows=36), execution_count=13),
        _markdown_cell("## Bullish Vs Bearish Factor Performance"),
        _code_cell("regime_weight_search_top", _table_text(pd.read_csv(output_dir / f"weight_search_bullish_{RESEARCH_NAME}.csv").head(10)), execution_count=14),
        _code_cell("bearish_weight_search_top", _table_text(pd.read_csv(output_dir / f"weight_search_bearish_{RESEARCH_NAME}.csv").head(10)), execution_count=15),
        _markdown_cell("## Always L/S Vs Long-Only Vs Short-Only Vs Regime-Switch"),
        _code_cell("exposure_variant_comparison", _table_text(long_short_compare, max_rows=36), execution_count=16),
        _markdown_cell("## Untouched Test Results"),
        _code_cell("selected_model_test_results", _table_text(test_primary, max_rows=24), execution_count=17),
        _markdown_cell("## BTC-Relative Performance"),
        _code_cell("btc_relative_test", _table_text(btc_relative, max_rows=24), execution_count=18),
        _markdown_cell("## Constituents By Rebalance Date"),
        _code_cell("constituents_sample", _table_text(constituents.sort_values(['horizon','rebalance_date','side']).head(50), max_rows=50), execution_count=19),
        _markdown_cell("## Monthly L/S Basket And MoM Returns"),
        _code_cell("monthly_l_s_basket", _table_text(monthly_basket, max_rows=24), execution_count=22),
        _markdown_cell("## Appendix: Non-Selected Test Candidates"),
        _code_cell("test_candidate_appendix", _table_text(appendix, max_rows=50), execution_count=20),
        _markdown_cell(
            "## Limitations\n"
            "- Survivorship bias remains because the asset master is seeded from the current mapped universe.\n"
            "- Binance USD-M shortability is treated as a current shortability proxy, not a full historical listing calendar.\n"
            "- Public CoinGecko and DefiLlama coverage is sparse for some assets and can lag exchange prices.\n"
            "- Binance open-interest history is recent-history limited, so OI signals rely mostly on DefiLlama coverage.\n"
            "- Regime-specific samples are smaller than global samples, especially once label completeness and universe filters are applied."
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with open(notebook_path, "w") as f:
        json.dump(notebook, f)
    return notebook_path
