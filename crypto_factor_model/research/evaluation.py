"""Walk-forward signal, weight, portfolio, and model-selection helpers."""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from crypto_factor_model.research.constants import (
    FAMILIES,
    HORIZONS,
    MAX_WEIGHT_PER_FAMILY,
    PRIMARY_HORIZON,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    TX_COST_BPS,
    VALIDATION_END,
    VALIDATION_START,
    WEIGHT_GRID_STEP,
)
from crypto_factor_model.research.data import ResearchPanels
from crypto_factor_model.signals.utils import rank_ic, rank_zscore_panel

logger = logging.getLogger(__name__)

L_S_VARIANTS = {"always_l_s", "sticky_l_s", "regime_aware_l_s"}
DIRECTIONAL_VARIANTS = {"long_only", "short_only", "regime_switch"}


@dataclass
class EvaluationContext:
    """Wide panels needed to evaluate models without repeatedly pivoting."""

    dates: pd.DatetimeIndex
    metadata: pd.DataFrame
    forward_returns: dict[int, pd.DataFrame]
    entry_prices: dict[int, pd.DataFrame]
    exit_prices: dict[int, pd.DataFrame]
    eligible_base: pd.DataFrame
    eligible_sensitivity: pd.DataFrame
    shortable: pd.DataFrame
    regime: pd.Series
    btc_forward_returns: dict[int, pd.Series]


def make_context(panels: ResearchPanels) -> EvaluationContext:
    btc_ret = {}
    btc_entry = panels.btc_price.shift(-1)
    for horizon in HORIZONS:
        btc_exit = panels.btc_price.shift(-(horizon + 1))
        btc_ret[horizon] = (btc_exit / btc_entry - 1).replace([np.inf, -np.inf], np.nan)
    return EvaluationContext(
        dates=panels.price.index,
        metadata=panels.master.set_index("research_id"),
        forward_returns=panels.forward_returns,
        entry_prices=panels.entry_prices,
        exit_prices=panels.exit_prices,
        eligible_base=panels.eligibility["eligible_base"],
        eligible_sensitivity=panels.eligibility["eligible_sensitivity"],
        shortable=panels.eligibility["shortable"],
        regime=panels.regime["regime"],
        btc_forward_returns=btc_ret,
    )


def _period_bounds(period: str, horizon: int, dates: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    latest = dates.max() - pd.Timedelta(days=horizon + 1)
    if period == "Train":
        return TRAIN_START, min(TRAIN_END - pd.Timedelta(days=horizon + 1), latest)
    if period == "Validation":
        return VALIDATION_START, min(VALIDATION_END - pd.Timedelta(days=horizon + 1), latest)
    if period == "Test":
        return TEST_START, latest
    raise ValueError(f"Unknown period: {period}")


def _period_dates(dates: pd.DatetimeIndex, period: str, horizon: int) -> pd.DatetimeIndex:
    start, end = _period_bounds(period, horizon, dates)
    if pd.isna(end) or end < start:
        return pd.DatetimeIndex([])
    return dates[(dates >= start) & (dates <= end)]


def _sample_rebalance_dates(dates: pd.DatetimeIndex, period: str, horizon: int) -> pd.DatetimeIndex:
    available = _period_dates(dates, period, horizon)
    if len(available) == 0:
        return available
    return available[::horizon]


def _eligible_panel(ctx: EvaluationContext, eligible_name: str) -> pd.DataFrame:
    if eligible_name == "eligible_base":
        return ctx.eligible_base
    if eligible_name == "eligible_sensitivity":
        return ctx.eligible_sensitivity
    raise ValueError(f"Unknown eligible universe: {eligible_name}")


def _ic_for_panel(signal: pd.DataFrame, returns: pd.DataFrame, dates: pd.DatetimeIndex) -> list[float]:
    common = dates.intersection(signal.index).intersection(returns.index)
    ics: list[float] = []
    for dt in common:
        ic = rank_ic(signal.loc[dt], returns.loc[dt])
        if pd.notna(ic):
            ics.append(float(ic))
    return ics


def build_signal_decisions(
    zscore_signals: dict[str, dict[str, pd.DataFrame]],
    forward_returns: dict[int, pd.DataFrame],
    dates: pd.DatetimeIndex,
    train_dates_by_horizon: dict[int, pd.DatetimeIndex] | None = None,
    corr_threshold: float = 0.90,
) -> pd.DataFrame:
    """Train-only sign checks and within-family redundancy decisions."""
    rows: list[dict[str, Any]] = []
    for family, signals in zscore_signals.items():
        for name, panel in signals.items():
            row: dict[str, Any] = {"family": family, "signal": name}
            all_ics: list[float] = []
            for horizon in HORIZONS:
                train_dates = (
                    train_dates_by_horizon[horizon]
                    if train_dates_by_horizon is not None and horizon in train_dates_by_horizon
                    else _period_dates(dates, "Train", horizon)
                )
                ics = _ic_for_panel(panel, forward_returns[horizon], train_dates)
                row[f"mean_ic_{horizon}d"] = float(np.nanmean(ics)) if ics else np.nan
                row[f"n_ic_{horizon}d"] = len(ics)
                all_ics.extend(ics)
            row["mean_ic_avg"] = float(np.nanmean(all_ics)) if all_ics else np.nan
            row["sign"] = -1 if pd.notna(row["mean_ic_avg"]) and row["mean_ic_avg"] < 0 else 1
            row["selected"] = bool(pd.notna(row["mean_ic_avg"]))
            row["dedupe_reason"] = ""
            rows.append(row)
    decisions = pd.DataFrame(rows)
    if decisions.empty:
        return decisions

    for family, group in decisions.groupby("family"):
        names = group.loc[group["selected"], "signal"].tolist()
        if len(names) < 2:
            continue
        dedupe_dates = (
            train_dates_by_horizon[PRIMARY_HORIZON]
            if train_dates_by_horizon is not None and PRIMARY_HORIZON in train_dates_by_horizon
            else _period_dates(dates, "Train", PRIMARY_HORIZON)
        )
        flattened = {}
        for name in names:
            panel = zscore_signals[family][name].reindex(dedupe_dates)
            sign = int(decisions.loc[(decisions["family"] == family) & (decisions["signal"] == name), "sign"].iloc[0])
            flattened[name] = (panel * sign).stack().dropna()
        matrix = pd.DataFrame(flattened).dropna(thresh=2)
        if matrix.empty:
            continue
        corr = matrix.corr(method="spearman").abs()
        dropped: set[str] = set()
        for left, right in itertools.combinations(names, 2):
            if left in dropped or right in dropped:
                continue
            value = corr.loc[left, right] if left in corr.index and right in corr.columns else np.nan
            if pd.isna(value) or value < corr_threshold:
                continue
            left_ic = abs(float(decisions.loc[(decisions["family"] == family) & (decisions["signal"] == left), "mean_ic_avg"].iloc[0]))
            right_ic = abs(float(decisions.loc[(decisions["family"] == family) & (decisions["signal"] == right), "mean_ic_avg"].iloc[0]))
            loser = right if left_ic >= right_ic else left
            dropped.add(loser)
            mask = (decisions["family"] == family) & (decisions["signal"] == loser)
            decisions.loc[mask, "selected"] = False
            decisions.loc[mask, "dedupe_reason"] = f"train corr >= {corr_threshold:.2f} with {left if loser == right else right}"
    return decisions.sort_values(["family", "selected", "mean_ic_avg"], ascending=[True, False, False]).reset_index(drop=True)


def apply_signal_decisions(
    zscore_signals: dict[str, dict[str, pd.DataFrame]],
    decisions: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Create signed, deduped family score panels from train-only decisions."""
    family_scores: dict[str, pd.DataFrame] = {}
    for family in ["fundamentals", "momentum", "flows"]:
        selected = decisions[(decisions["family"] == family) & (decisions["selected"])].copy()
        panels = []
        for _, row in selected.iterrows():
            panel = zscore_signals.get(family, {}).get(row["signal"])
            if panel is not None and not panel.empty:
                panels.append(panel * int(row["sign"]))
        if panels:
            avg = pd.concat(panels, keys=range(len(panels))).groupby(level=1).mean()
            family_scores[family] = rank_zscore_panel(avg, winsorise_first=False)
        else:
            family_scores[family] = pd.DataFrame()

    base = composite_from_weights(
        {key: panel for key, panel in family_scores.items() if key in {"fundamentals", "momentum", "flows"}},
        {"fundamentals": 1 / 3, "momentum": 1 / 3, "flows": 1 / 3},
        min_token_families=1,
    )
    improvement = rank_zscore_panel(base - base.shift(28), winsorise_first=True) if not base.empty else pd.DataFrame()
    family_scores["factor_improvement"] = improvement
    return family_scores


def composite_from_weights(
    family_scores: dict[str, pd.DataFrame],
    weights: dict[str, float],
    min_token_families: int = 1,
) -> pd.DataFrame:
    panels = {family: family_scores.get(family, pd.DataFrame()) for family in FAMILIES if weights.get(family, 0) > 0}
    panels = {family: panel for family, panel in panels.items() if panel is not None and not panel.empty}
    if not panels:
        return pd.DataFrame()
    all_dates = sorted(set().union(*(panel.index for panel in panels.values())))
    all_cols = sorted(set().union(*(panel.columns for panel in panels.values())))
    numerator = pd.DataFrame(0.0, index=all_dates, columns=all_cols)
    denominator = pd.DataFrame(0.0, index=all_dates, columns=all_cols)
    coverage = pd.DataFrame(0, index=all_dates, columns=all_cols)
    for family, panel in panels.items():
        weight = float(weights.get(family, 0))
        aligned = panel.reindex(index=all_dates, columns=all_cols)
        valid = aligned.notna()
        numerator += aligned.fillna(0) * weight
        denominator += valid.astype(float) * weight
        coverage += valid.astype(int)
    composite = numerator / denominator.replace(0, np.nan)
    composite[coverage < min_token_families] = np.nan
    return composite


def regime_specific_composite(
    family_scores: dict[str, pd.DataFrame],
    bullish_weights: dict[str, float],
    bearish_weights: dict[str, float],
    regime: pd.Series,
) -> pd.DataFrame:
    bullish = composite_from_weights(family_scores, bullish_weights)
    bearish = composite_from_weights(family_scores, bearish_weights)
    all_dates = bullish.index.union(bearish.index)
    all_cols = bullish.columns.union(bearish.columns)
    bullish = bullish.reindex(index=all_dates, columns=all_cols)
    bearish = bearish.reindex(index=all_dates, columns=all_cols)
    out = bullish.copy()
    aligned_regime = regime.reindex(all_dates).ffill()
    out.loc[aligned_regime.eq("Bearish")] = bearish.loc[aligned_regime.eq("Bearish")]
    return out


def current_no_team_weights() -> dict[str, float]:
    raw = {"fundamentals": 0.55, "momentum": 0.175, "flows": 0.125, "factor_improvement": 0.0}
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


def equal_family_weights() -> dict[str, float]:
    return {family: 1.0 / len(FAMILIES) for family in FAMILIES}


def heuristic_weights() -> dict[str, float]:
    return {"fundamentals": 0.25, "momentum": 0.40, "flows": 0.25, "factor_improvement": 0.10}


def weight_grid() -> list[dict[str, float]]:
    steps = int(round(1 / WEIGHT_GRID_STEP))
    max_step = int(round(MAX_WEIGHT_PER_FAMILY / WEIGHT_GRID_STEP))
    combos: list[dict[str, float]] = []
    for values in itertools.product(range(max_step + 1), repeat=len(FAMILIES)):
        if sum(values) != steps:
            continue
        if sum(v > 0 for v in values) < 2:
            continue
        combos.append({family: value * WEIGHT_GRID_STEP for family, value in zip(FAMILIES, values)})
    return combos


def _top_bottom_spread(composite: pd.DataFrame, returns: pd.DataFrame, eligible: pd.DataFrame, dates: pd.DatetimeIndex, n: int = 5) -> float:
    spreads: list[float] = []
    for dt in dates.intersection(composite.index).intersection(returns.index):
        scores = composite.loc[dt].where(eligible.reindex(composite.index).loc[dt]).dropna()
        if len(scores) < n * 2:
            continue
        fwd = returns.loc[dt]
        top = fwd.reindex(scores.sort_values(ascending=False).head(n).index).dropna()
        bottom = fwd.reindex(scores.sort_values(ascending=True).head(n).index).dropna()
        if len(top) >= n and len(bottom) >= n:
            spreads.append(float(top.mean() - bottom.mean()))
    return float(np.nanmean(spreads)) if spreads else np.nan


def _mean_ic_for_composite(composite: pd.DataFrame, ctx: EvaluationContext, period: str, regime_filter: str | None = None) -> float:
    all_ics: list[float] = []
    for horizon in HORIZONS:
        dates = _period_dates(ctx.dates, period, horizon)
        if regime_filter:
            dates = dates[ctx.regime.reindex(dates).eq(regime_filter).fillna(False).to_numpy()]
        all_ics.extend(_ic_for_panel(composite, ctx.forward_returns[horizon], dates))
    return float(np.nanmean(all_ics)) if all_ics else np.nan


def construct_positions(
    score_row: pd.Series,
    return_row: pd.Series,
    eligible_row: pd.Series,
    shortable_row: pd.Series,
    variant: str,
    regime: str | None = None,
    prev_weights: pd.Series | None = None,
    short_rank_buffer: int = 10,
    n: int = 5,
) -> pd.Series:
    eligible_scores = score_row.where(eligible_row).dropna()
    eligible_scores = eligible_scores[return_row.reindex(eligible_scores.index).notna()]
    if variant == "regime_switch":
        variant = "long_only" if regime == "Bullish" else "short_only"

    weights: dict[str, float] = {}
    if variant in {"always_l_s", "sticky_l_s", "regime_aware_l_s"}:
        long_names = eligible_scores.sort_values(ascending=False).head(n).index.tolist()
        short_pool = eligible_scores[shortable_row.reindex(eligible_scores.index).fillna(False)]
        short_pool = short_pool.sort_values(ascending=True)
        if variant == "sticky_l_s" and prev_weights is not None and not prev_weights.empty:
            candidate_names = set(short_pool.head(max(short_rank_buffer, n)).index)
            previous_shorts = prev_weights[prev_weights < 0].index.tolist()
            short_names = [
                name
                for name in previous_shorts
                if name in candidate_names and name not in long_names
            ][:n]
            for name in short_pool.index:
                if len(short_names) >= n:
                    break
                if name not in short_names and name not in long_names:
                    short_names.append(name)
        else:
            short_names = [name for name in short_pool.head(n).index.tolist() if name not in long_names]
        if len(long_names) < n or len(short_names) < n:
            return pd.Series(dtype=float)
        weights.update({name: 0.5 / n for name in long_names})
        weights.update({name: -0.5 / n for name in short_names})
    elif variant == "long_only":
        long_names = eligible_scores.sort_values(ascending=False).head(n).index.tolist()
        if len(long_names) < n:
            return pd.Series(dtype=float)
        weights.update({name: 1.0 / n for name in long_names})
    elif variant == "short_only":
        short_pool = eligible_scores[shortable_row.reindex(eligible_scores.index).fillna(False)]
        short_names = short_pool.sort_values(ascending=True).head(n).index.tolist()
        if len(short_names) < n:
            return pd.Series(dtype=float)
        weights.update({name: -1.0 / n for name in short_names})
    else:
        raise ValueError(f"Unknown portfolio variant: {variant}")
    return pd.Series(weights, dtype=float)


def run_portfolio_backtest(
    ctx: EvaluationContext,
    composite: pd.DataFrame,
    horizon: int,
    period: str,
    variant: str,
    eligible_name: str = "eligible_base",
    cost_bps: float = TX_COST_BPS,
    rebalance_dates: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = _eligible_panel(ctx, eligible_name)
    dates = rebalance_dates if rebalance_dates is not None else _sample_rebalance_dates(ctx.dates, period, horizon)
    returns_rows: list[dict[str, Any]] = []
    constituent_rows: list[dict[str, Any]] = []
    prev_weights = pd.Series(dtype=float)
    cost_rate = cost_bps / 10000.0

    for dt in dates:
        if dt not in composite.index or dt not in ctx.forward_returns[horizon].index:
            continue
        regime = ctx.regime.reindex([dt]).iloc[0] if dt in ctx.regime.index else None
        weights = construct_positions(
            composite.loc[dt],
            ctx.forward_returns[horizon].loc[dt],
            eligible.reindex(composite.index).loc[dt],
            ctx.shortable.reindex(composite.index).loc[dt],
            variant,
            regime=regime,
            prev_weights=prev_weights,
        )
        if weights.empty:
            continue
        all_names = weights.index.union(prev_weights.index)
        traded_notional = float((weights.reindex(all_names).fillna(0) - prev_weights.reindex(all_names).fillna(0)).abs().sum())
        cost = traded_notional * cost_rate
        ret = ctx.forward_returns[horizon].loc[dt].reindex(weights.index)
        gross_return = float((weights * ret).sum())
        net_return = gross_return - cost
        returns_rows.append(
            {
                "date": dt,
                "horizon": horizon,
                "period": period,
                "variant": variant,
                "eligible_universe": eligible_name,
                "gross_return": gross_return,
                "cost": cost,
                "net_return": net_return,
                "turnover": traded_notional,
                "regime": regime,
                "n_positions": len(weights),
            }
        )
        gross = float(weights.abs().sum())
        for rid, weight in weights.items():
            entry = ctx.entry_prices[horizon].loc[dt, rid] if rid in ctx.entry_prices[horizon] else np.nan
            exit_ = ctx.exit_prices[horizon].loc[dt, rid] if rid in ctx.exit_prices[horizon] else np.nan
            asset_ret = ctx.forward_returns[horizon].loc[dt, rid]
            cost_alloc = cost * (abs(weight) / gross) if gross > 0 else 0.0
            meta = ctx.metadata.loc[rid] if rid in ctx.metadata.index else pd.Series(dtype=object)
            constituent_rows.append(
                {
                    "rebalance_date": dt,
                    "horizon": horizon,
                    "period": period,
                    "variant": variant,
                    "eligible_universe": eligible_name,
                    "research_id": rid,
                    "token": meta.get("token", rid),
                    "project": meta.get("name", pd.NA),
                    "side": "Long" if weight > 0 else "Short",
                    "weight": weight,
                    "entry_price": entry,
                    "exit_price": exit_,
                    "asset_forward_return": asset_ret,
                    "return_contribution": weight * asset_ret - cost_alloc,
                    "turnover": traded_notional,
                    "cost_allocated": cost_alloc,
                    "regime": regime,
                }
            )
        prev_weights = weights

    return pd.DataFrame(returns_rows), pd.DataFrame(constituent_rows)


def _nav_from_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0)).cumprod()


def _max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return np.nan
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def performance_metrics(returns: pd.DataFrame, btc_returns: pd.Series | None, horizon: int) -> dict[str, float]:
    if returns.empty:
        keys = [
            "total_return",
            "annualized_return",
            "annualized_vol",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "hit_rate",
            "avg_turnover",
            "total_cost",
            "btc_total_return",
            "excess_return",
            "beta",
            "alpha",
            "information_ratio",
            "correlation",
            "up_capture",
            "down_capture",
            "drawdown_vs_btc",
        ]
        return {key: np.nan for key in keys}
    r = returns.sort_values("date")["net_return"].astype(float)
    nav = _nav_from_returns(r)
    years = max((pd.Timestamp(returns["date"].max()) - pd.Timestamp(returns["date"].min())).days + horizon, horizon) / 365.0
    total = float(nav.iloc[-1] - 1.0)
    ann = float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1 and years > 0 else np.nan
    periods_per_year = 365.0 / horizon
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else np.nan
    sharpe = float((r.mean() / r.std(ddof=1)) * np.sqrt(periods_per_year)) if len(r) > 1 and r.std(ddof=1) > 0 else np.nan
    downside = r[r < 0].std(ddof=1)
    sortino = float((r.mean() / downside) * np.sqrt(periods_per_year)) if pd.notna(downside) and downside > 0 else np.nan
    mdd = _max_drawdown(nav)
    calmar = float(ann / abs(mdd)) if pd.notna(mdd) and mdd < 0 else np.nan

    out = {
        "total_return": total,
        "annualized_return": ann,
        "annualized_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "hit_rate": float((r > 0).mean()),
        "avg_turnover": float(returns["turnover"].mean()),
        "total_cost": float(returns["cost"].sum()),
    }

    if btc_returns is None or btc_returns.empty:
        out.update(
            {
                "btc_total_return": np.nan,
                "excess_return": np.nan,
                "beta": np.nan,
                "alpha": np.nan,
                "information_ratio": np.nan,
                "correlation": np.nan,
                "up_capture": np.nan,
                "down_capture": np.nan,
                "drawdown_vs_btc": np.nan,
            }
        )
        return out

    aligned = pd.DataFrame({"portfolio": r.to_numpy()}, index=pd.to_datetime(returns.sort_values("date")["date"]))
    aligned["btc"] = btc_returns.reindex(aligned.index)
    aligned = aligned.dropna()
    if aligned.empty:
        out.update({key: np.nan for key in ["btc_total_return", "excess_return", "beta", "alpha", "information_ratio", "correlation", "up_capture", "down_capture", "drawdown_vs_btc"]})
        return out
    btc_nav = _nav_from_returns(aligned["btc"])
    btc_total = float(btc_nav.iloc[-1] - 1.0)
    excess = aligned["portfolio"] - aligned["btc"]
    beta = float(np.cov(aligned["portfolio"], aligned["btc"])[0, 1] / np.var(aligned["btc"])) if np.var(aligned["btc"]) > 0 else np.nan
    btc_ann = float((1.0 + btc_total) ** (1.0 / years) - 1.0) if btc_total > -1 and years > 0 else np.nan
    alpha = float(ann - beta * btc_ann) if pd.notna(beta) and pd.notna(ann) and pd.notna(btc_ann) else np.nan
    ir = float((excess.mean() / excess.std(ddof=1)) * np.sqrt(periods_per_year)) if len(excess) > 1 and excess.std(ddof=1) > 0 else np.nan
    up = aligned[aligned["btc"] > 0]
    down = aligned[aligned["btc"] < 0]
    out.update(
        {
            "btc_total_return": btc_total,
            "excess_return": total - btc_total,
            "beta": beta,
            "alpha": alpha,
            "information_ratio": ir,
            "correlation": float(aligned["portfolio"].corr(aligned["btc"])) if len(aligned) > 1 else np.nan,
            "up_capture": float(up["portfolio"].mean() / up["btc"].mean()) if len(up) and up["btc"].mean() != 0 else np.nan,
            "down_capture": float(down["portfolio"].mean() / down["btc"].mean()) if len(down) and down["btc"].mean() != 0 else np.nan,
            "drawdown_vs_btc": mdd - _max_drawdown(btc_nav),
        }
    )
    return out


def _candidate_row(
    model_name: str,
    variant: str,
    horizon: int,
    period: str,
    returns: pd.DataFrame,
    ctx: EvaluationContext,
    eligible_name: str,
) -> dict[str, Any]:
    metrics = performance_metrics(returns, ctx.btc_forward_returns[horizon], horizon)
    return {
        "model": model_name,
        "variant": variant,
        "horizon": horizon,
        "period": period,
        "eligible_universe": eligible_name,
        "n_rebalances": int(len(returns)),
        **metrics,
    }


def evaluate_candidate(
    ctx: EvaluationContext,
    composite: pd.DataFrame,
    model_name: str,
    periods: list[str],
    variants: list[str],
    eligible_name: str = "eligible_base",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    return_frames: list[pd.DataFrame] = []
    constituent_frames: list[pd.DataFrame] = []
    for period in periods:
        for horizon in HORIZONS:
            for variant in variants:
                returns, constituents = run_portfolio_backtest(ctx, composite, horizon, period, variant, eligible_name=eligible_name)
                if not returns.empty:
                    returns["model"] = model_name
                    constituents["model"] = model_name
                    return_frames.append(returns)
                    constituent_frames.append(constituents)
                summary_rows.append(_candidate_row(model_name, variant, horizon, period, returns, ctx, eligible_name))
    return (
        pd.DataFrame(summary_rows),
        pd.concat(return_frames, ignore_index=True) if return_frames else pd.DataFrame(),
        pd.concat(constituent_frames, ignore_index=True) if constituent_frames else pd.DataFrame(),
    )


def search_weights(
    ctx: EvaluationContext,
    family_scores: dict[str, pd.DataFrame],
    regime_filter: str | None = None,
    eligible_name: str = "eligible_base",
    train_dates: pd.DatetimeIndex | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    train_dates = train_dates if train_dates is not None else _period_dates(ctx.dates, "Train", PRIMARY_HORIZON)
    if regime_filter:
        train_dates = train_dates[ctx.regime.reindex(train_dates).eq(regime_filter).fillna(False).to_numpy()]
    search_dates = train_dates[::30] if len(train_dates) > 30 else train_dates
    eligible = _eligible_panel(ctx, eligible_name)
    for weights in weight_grid():
        comp = composite_from_weights(family_scores, weights)
        ics = _ic_for_panel(comp, ctx.forward_returns[PRIMARY_HORIZON], search_dates)
        ic = float(np.nanmean(ics)) if ics else np.nan
        rows.append(
            {
                **{f"weight_{family}": weights[family] for family in FAMILIES},
                "mean_ic": ic,
                "spread": np.nan,
                "total_return": np.nan,
                "max_drawdown": np.nan,
                "sharpe": np.nan,
                "turnover": np.nan,
            }
        )
    grid = pd.DataFrame(rows)
    if grid.empty:
        return equal_family_weights(), grid
    for col in ["mean_ic"]:
        values = pd.to_numeric(grid[col], errors="coerce")
        grid[f"pre_z_{col}"] = 0.0 if values.notna().sum() < 2 or values.std(ddof=0) == 0 else ((values - values.mean()) / values.std(ddof=0)).fillna(-5)
    grid["pre_score"] = grid["pre_z_mean_ic"]
    shortlist = grid.sort_values("pre_score", ascending=False).head(60).index
    for idx in shortlist:
        weights = {family: float(grid.loc[idx, f"weight_{family}"]) for family in FAMILIES}
        comp = composite_from_weights(family_scores, weights)
        grid.loc[idx, "spread"] = _top_bottom_spread(comp, ctx.forward_returns[PRIMARY_HORIZON], eligible, search_dates)
        returns, _ = run_portfolio_backtest(
            ctx,
            comp,
            PRIMARY_HORIZON,
            "Train",
            "always_l_s",
            eligible_name=eligible_name,
            rebalance_dates=train_dates[::PRIMARY_HORIZON],
        )
        if regime_filter and not returns.empty:
            returns = returns[returns["regime"] == regime_filter].copy()
        metrics = performance_metrics(returns, ctx.btc_forward_returns[PRIMARY_HORIZON], PRIMARY_HORIZON)
        grid.loc[idx, "total_return"] = metrics["total_return"]
        grid.loc[idx, "max_drawdown"] = metrics["max_drawdown"]
        grid.loc[idx, "sharpe"] = metrics["sharpe"]
        grid.loc[idx, "turnover"] = metrics["avg_turnover"]
    score_cols = {
        "mean_ic": 0.35,
        "spread": 0.20,
        "total_return": 0.25,
        "sharpe": 0.15,
        "max_drawdown": 0.10,
        "turnover": -0.05,
    }
    objective = pd.Series(0.0, index=grid.index)
    for col, weight in score_cols.items():
        values = pd.to_numeric(grid[col], errors="coerce")
        if values.notna().sum() < 2 or values.std(ddof=0) == 0:
            z = values.fillna(values.median()).fillna(-5 if weight > 0 else 5)
        else:
            z = (values - values.mean()) / values.std(ddof=0)
            z = z.fillna(z.min() if pd.notna(z.min()) else 0)
        objective += z * weight
    grid["objective"] = objective
    grid = grid.sort_values("objective", ascending=False).reset_index(drop=True)
    best = grid.iloc[0]
    weights = {family: float(best[f"weight_{family}"]) for family in FAMILIES}
    return weights, grid


def search_weights_ic_only(
    ctx: EvaluationContext,
    family_scores: dict[str, pd.DataFrame],
    regime_filter: str | None = None,
    train_dates: pd.DatetimeIndex | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Fast fold-local weight search using only train-period rank IC."""
    rows: list[dict[str, Any]] = []
    train_dates = train_dates if train_dates is not None else _period_dates(ctx.dates, "Train", PRIMARY_HORIZON)
    if regime_filter:
        train_dates = train_dates[ctx.regime.reindex(train_dates).eq(regime_filter).fillna(False).to_numpy()]
    search_dates = train_dates[::30] if len(train_dates) > 30 else train_dates
    for weights in weight_grid():
        comp = composite_from_weights(family_scores, weights)
        ics = _ic_for_panel(comp, ctx.forward_returns[PRIMARY_HORIZON], search_dates)
        ic = float(np.nanmean(ics)) if ics else np.nan
        rows.append(
            {
                **{f"weight_{family}": weights[family] for family in FAMILIES},
                "mean_ic": ic,
                "objective": ic,
            }
        )
    grid = pd.DataFrame(rows)
    if grid.empty:
        return equal_family_weights(), grid
    grid = grid.sort_values("objective", ascending=False, na_position="last").reset_index(drop=True)
    best = grid.iloc[0]
    weights = {family: float(best[f"weight_{family}"]) for family in FAMILIES}
    return weights, grid


def purged_walk_forward_folds(
    dates: pd.DatetimeIndex,
    horizon: int = PRIMARY_HORIZON,
    train_months: int = 6,
    validation_months: int = 3,
) -> list[dict[str, Any]]:
    """Expanding-train, rolling-validation folds before the untouched test window."""
    folds: list[dict[str, Any]] = []
    validation_start = TRAIN_START + pd.DateOffset(months=train_months)
    fold_id = 1
    latest_label_complete = dates.max() - pd.Timedelta(days=horizon + 1)
    while validation_start <= VALIDATION_END:
        validation_end = min(validation_start + pd.DateOffset(months=validation_months) - pd.Timedelta(days=1), VALIDATION_END)
        validation_end = min(pd.Timestamp(validation_end), latest_label_complete)
        train_end = pd.Timestamp(validation_start) - pd.Timedelta(days=horizon + 1)
        train_dates = dates[(dates >= TRAIN_START) & (dates <= train_end)]
        validation_dates = dates[(dates >= validation_start) & (dates <= validation_end)]
        if len(train_dates) and len(validation_dates):
            folds.append(
                {
                    "fold": f"fold_{fold_id}",
                    "train_start": TRAIN_START,
                    "train_end": train_end,
                    "validation_start": pd.Timestamp(validation_start),
                    "validation_end": validation_end,
                    "purge_days": horizon + 1,
                    "train_days": int(len(train_dates)),
                    "validation_days": int(len(validation_dates)),
                }
            )
            fold_id += 1
        validation_start = validation_start + pd.DateOffset(months=validation_months)
    return folds


def rolling_walk_forward_diagnostics(panels: ResearchPanels) -> pd.DataFrame:
    """Retrain signal signs and learned global weights on each purged fold, then validate forward."""
    ctx = make_context(panels)
    rows: list[dict[str, Any]] = []
    for fold in purged_walk_forward_folds(ctx.dates):
        validation_dates = ctx.dates[
            (ctx.dates >= fold["validation_start"])
            & (ctx.dates <= fold["validation_end"])
        ]
        train_dates_by_horizon = {
            horizon: ctx.dates[
                (ctx.dates >= TRAIN_START)
                & (ctx.dates <= (pd.Timestamp(fold["validation_start"]) - pd.Timedelta(days=horizon + 1)))
            ]
            for horizon in HORIZONS
        }
        decisions = build_signal_decisions(
            panels.zscore_signals,
            panels.forward_returns,
            panels.price.index,
            train_dates_by_horizon=train_dates_by_horizon,
        )
        fold_family_scores = apply_signal_decisions(panels.zscore_signals, decisions)
        fold_weights, _ = search_weights_ic_only(
            ctx,
            fold_family_scores,
            train_dates=train_dates_by_horizon[PRIMARY_HORIZON],
        )
        composites = {
            "current_no_team": composite_from_weights(fold_family_scores, current_no_team_weights()),
            "equal_family": composite_from_weights(fold_family_scores, equal_family_weights()),
            "heuristic": composite_from_weights(fold_family_scores, heuristic_weights()),
            "learned_global": composite_from_weights(fold_family_scores, fold_weights),
        }
        for model, composite in composites.items():
            variants = ["always_l_s", "sticky_l_s"]
            if model == "learned_global":
                variants.extend(["long_only", "short_only", "regime_switch"])
            for eligible_name in ["eligible_base", "eligible_sensitivity"]:
                for variant in variants:
                    returns, _ = run_portfolio_backtest(
                        ctx,
                        composite,
                        PRIMARY_HORIZON,
                        fold["fold"],
                        variant,
                        eligible_name=eligible_name,
                        rebalance_dates=validation_dates[::PRIMARY_HORIZON],
                    )
                    row = _candidate_row(model, variant, PRIMARY_HORIZON, fold["fold"], returns, ctx, eligible_name)
                    row.update(
                        {
                            "train_start": fold["train_start"],
                            "train_end": fold["train_end"],
                            "validation_start": fold["validation_start"],
                            "validation_end": fold["validation_end"],
                            "purge_days": fold["purge_days"],
                            **{f"weight_{family}": fold_weights.get(family, np.nan) for family in FAMILIES},
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def rolling_selection_stats(rolling: pd.DataFrame) -> pd.DataFrame:
    """Summarize isolated rolling folds for model-selection gates."""
    if rolling.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "variant",
                "eligible_universe",
                "rolling_folds",
                "rolling_positive_folds",
                "rolling_avg_total_return",
                "rolling_min_total_return",
                "rolling_avg_sharpe",
            ]
        )
    primary = rolling[rolling["horizon"].eq(PRIMARY_HORIZON)].copy()
    if primary.empty:
        return pd.DataFrame()
    grouped = primary.groupby(["model", "variant", "eligible_universe"], dropna=False)
    stats_df = grouped.agg(
        rolling_folds=("total_return", "count"),
        rolling_positive_folds=("total_return", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        rolling_avg_total_return=("total_return", "mean"),
        rolling_min_total_return=("total_return", "min"),
        rolling_avg_sharpe=("sharpe", "mean"),
    ).reset_index()
    return stats_df


def annotate_validation_selection(validation: pd.DataFrame) -> pd.DataFrame:
    """Add strict L/S-first validation-selection gates to the summary table."""
    annotated = validation.copy()
    default_bool_cols = [
        "is_l_s_candidate",
        "is_directional_candidate",
        "passes_validation_gate",
        "passes_rolling_gate",
        "passes_strict_gate",
    ]
    for col in default_bool_cols:
        annotated[col] = False
    annotated["selection_score"] = np.nan
    for col in [
        "rolling_folds",
        "rolling_positive_folds",
        "rolling_avg_total_return",
        "rolling_min_total_return",
        "rolling_avg_sharpe",
    ]:
        if col not in annotated.columns:
            annotated[col] = np.nan

    primary_mask = annotated["horizon"].eq(PRIMARY_HORIZON) & annotated["period"].eq("Validation")
    if not primary_mask.any():
        return annotated

    primary = annotated.loc[primary_mask].copy()
    primary["selection_score"] = (
        primary["sharpe"].fillna(-5)
        + primary["calmar"].fillna(-5).clip(lower=-5, upper=5) * 0.25
        + primary["total_return"].fillna(-1) * 2.0
        - primary["avg_turnover"].fillna(2) * 0.05
    )
    primary["is_l_s_candidate"] = primary["variant"].isin(L_S_VARIANTS)
    primary["is_directional_candidate"] = primary["variant"].isin(DIRECTIONAL_VARIANTS)
    primary["passes_validation_gate"] = (
        primary["is_l_s_candidate"]
        & (pd.to_numeric(primary["total_return"], errors="coerce") > 0)
        & (pd.to_numeric(primary["sharpe"], errors="coerce") > 0)
        & (pd.to_numeric(primary["n_rebalances"], errors="coerce") >= 3)
        & (pd.to_numeric(primary["max_drawdown"], errors="coerce") >= -0.50)
    )
    primary["passes_rolling_gate"] = (
        (pd.to_numeric(primary["rolling_folds"], errors="coerce") >= 3)
        & (pd.to_numeric(primary["rolling_positive_folds"], errors="coerce") >= 3)
        & (pd.to_numeric(primary["rolling_avg_total_return"], errors="coerce") > 0)
    )
    primary["passes_strict_gate"] = (
        primary["is_l_s_candidate"]
        & primary["passes_validation_gate"]
        & primary["passes_rolling_gate"]
    )
    annotated.loc[primary.index, primary.columns] = primary
    return annotated


def choose_strict_l_s_candidate(validation: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, str, str, bool]:
    """Choose an L/S model or mark the best L/S row as reporting-only."""
    annotated = annotate_validation_selection(validation)
    primary = annotated[annotated["horizon"].eq(PRIMARY_HORIZON) & annotated["period"].eq("Validation")].copy()
    if primary.empty:
        raise ValueError("No primary-horizon validation rows are available for selection.")

    strict_pool = primary[primary["passes_strict_gate"].astype(bool)].copy()
    status = "STRICT_PASS"
    passed = True
    if not strict_pool.empty:
        sensitivity_pool = strict_pool[strict_pool["eligible_universe"].eq("eligible_sensitivity")]
        pool = sensitivity_pool if not sensitivity_pool.empty else strict_pool
        selected = pool.sort_values("selection_score", ascending=False).iloc[0]
        baseline = primary[
            primary["model"].eq("learned_global")
            & primary["variant"].eq("always_l_s")
            & primary["eligible_universe"].eq(selected["eligible_universe"])
        ].head(1)
        if not baseline.empty:
            base = baseline.iloc[0]
            selected_is_baseline = (
                selected["model"] == base["model"]
                and selected["variant"] == base["variant"]
                and selected["eligible_universe"] == base["eligible_universe"]
            )
            selected_beats_base = (
                pd.to_numeric(pd.Series([selected["total_return"]]), errors="coerce").iloc[0]
                > pd.to_numeric(pd.Series([base["total_return"]]), errors="coerce").iloc[0] + 0.01
                and pd.to_numeric(pd.Series([selected["sharpe"]]), errors="coerce").iloc[0]
                > pd.to_numeric(pd.Series([base["sharpe"]]), errors="coerce").iloc[0] + 0.05
                and pd.to_numeric(pd.Series([selected["max_drawdown"]]), errors="coerce").iloc[0]
                >= pd.to_numeric(pd.Series([base["max_drawdown"]]), errors="coerce").iloc[0] - 0.03
            )
            if (
                not selected_is_baseline
                and bool(base.get("passes_strict_gate", False))
                and not selected_beats_base
            ):
                selected = base
                reason = "Strict gates passed, but the alternate L/S candidate did not clearly beat learned-global always L/S; selected the simpler L/S baseline."
            else:
                reason = "Selected the best strict-gated L/S validation candidate, with the $10M universe preferred when available."
        else:
            reason = "Selected the best strict-gated L/S validation candidate; no same-universe learned-global L/S baseline was available."
        return selected, annotated, reason, status, passed

    status = "NO_STRICT_PASS"
    passed = False
    l_s_pool = primary[primary["is_l_s_candidate"].astype(bool)].copy()
    if l_s_pool.empty:
        selected = primary.sort_values("selection_score", ascending=False).iloc[0]
        reason = "No L/S validation candidate was available; selected the best validation row for reporting only, not as a strategy recommendation."
    else:
        sensitivity_pool = l_s_pool[l_s_pool["eligible_universe"].eq("eligible_sensitivity")]
        pool = sensitivity_pool if not sensitivity_pool.empty else l_s_pool
        selected = pool.sort_values("selection_score", ascending=False).iloc[0]
        reason = (
            "No L/S candidate cleared the positive validation and rolling-fold gates; "
            "selected the best L/S row for basket reporting only, not as a strategy recommendation."
        )
    return selected, annotated, reason, status, passed


def monthly_l_s_basket_table(
    ctx: EvaluationContext,
    composite: pd.DataFrame,
    model_name: str,
    variant: str,
    eligible_name: str,
    horizon: int = PRIMARY_HORIZON,
    period: str = "Test",
) -> pd.DataFrame:
    """Monthly selected L/S constituents and realized forward basket returns."""
    dates = _period_dates(ctx.dates, period, horizon)
    if len(dates) == 0:
        return pd.DataFrame()
    monthly_dates = pd.DatetimeIndex(dates.to_series().groupby(dates.to_period("M")).max().tolist())
    returns, constituents = run_portfolio_backtest(
        ctx,
        composite,
        horizon,
        period,
        variant,
        eligible_name=eligible_name,
        rebalance_dates=monthly_dates,
    )
    if returns.empty:
        return pd.DataFrame()

    returns = returns.sort_values("date").copy()
    returns["model"] = model_name
    returns["nav"] = _nav_from_returns(returns["net_return"].astype(float)).to_numpy()
    btc = ctx.btc_forward_returns[horizon].reindex(pd.to_datetime(returns["date"]))
    returns["btc_return"] = btc.to_numpy()
    returns["excess_return"] = returns["net_return"].astype(float) - returns["btc_return"].astype(float)

    rows: list[dict[str, Any]] = []
    prev_longs: set[str] | None = None
    prev_shorts: set[str] | None = None
    for _, ret_row in returns.iterrows():
        dt = pd.Timestamp(ret_row["date"])
        group = constituents[constituents["rebalance_date"].eq(dt)]
        long_tokens = group[group["side"].eq("Long")]["token"].astype(str).tolist()
        short_tokens = group[group["side"].eq("Short")]["token"].astype(str).tolist()
        long_set = set(long_tokens)
        short_set = set(short_tokens)
        rows.append(
            {
                "month": dt.to_period("M").strftime("%Y-%m"),
                "signal_date": dt.date().isoformat(),
                "entry_date": (dt + pd.Timedelta(days=1)).date().isoformat(),
                "exit_date": (dt + pd.Timedelta(days=horizon + 1)).date().isoformat(),
                "model": model_name,
                "variant": variant,
                "eligible_universe": eligible_name,
                "horizon": horizon,
                "regime": ret_row.get("regime"),
                "long_tokens": ", ".join(long_tokens),
                "short_tokens": ", ".join(short_tokens),
                "long_overlap_prev": np.nan if prev_longs is None else len(long_set & prev_longs),
                "short_overlap_prev": np.nan if prev_shorts is None else len(short_set & prev_shorts),
                "gross_return": ret_row["gross_return"],
                "cost": ret_row["cost"],
                "net_return": ret_row["net_return"],
                "nav": ret_row["nav"],
                "btc_return": ret_row["btc_return"],
                "excess_return": ret_row["excess_return"],
                "turnover": ret_row["turnover"],
            }
        )
        prev_longs = long_set
        prev_shorts = short_set
    return pd.DataFrame(rows)


def evaluate_walk_forward(panels: ResearchPanels, family_scores: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Run train weight search, validation model selection, and test evaluation."""
    ctx = make_context(panels)
    global_weights, global_grid = search_weights(ctx, family_scores)
    bullish_weights, bullish_grid = search_weights(ctx, family_scores, regime_filter="Bullish")
    bearish_weights, bearish_grid = search_weights(ctx, family_scores, regime_filter="Bearish")

    composites = {
        "current_no_team": composite_from_weights(family_scores, current_no_team_weights()),
        "equal_family": composite_from_weights(family_scores, equal_family_weights()),
        "heuristic": composite_from_weights(family_scores, heuristic_weights()),
        "learned_global": composite_from_weights(family_scores, global_weights),
        "learned_regime_specific": regime_specific_composite(family_scores, bullish_weights, bearish_weights, ctx.regime),
    }
    weights = {
        "current_no_team": current_no_team_weights(),
        "equal_family": equal_family_weights(),
        "heuristic": heuristic_weights(),
        "learned_global": global_weights,
        "learned_bullish": bullish_weights,
        "learned_bearish": bearish_weights,
    }

    validation_summaries = []
    validation_returns = []
    validation_constituents = []
    eligible_universes = ["eligible_base", "eligible_sensitivity"]
    for model, composite in composites.items():
        variants = ["always_l_s", "sticky_l_s"]
        if model == "learned_global":
            variants.extend(["long_only", "short_only", "regime_switch"])
        if model == "learned_regime_specific":
            variants = ["regime_aware_l_s"]
        for eligible_name in eligible_universes:
            summary, rets, const = evaluate_candidate(ctx, composite, model, ["Validation"], variants, eligible_name=eligible_name)
            validation_summaries.append(summary)
            validation_returns.append(rets)
            validation_constituents.append(const)
    validation = pd.concat(validation_summaries, ignore_index=True)
    rolling = rolling_walk_forward_diagnostics(panels)
    rolling_stats = rolling_selection_stats(rolling)
    if not rolling_stats.empty:
        validation = validation.merge(
            rolling_stats,
            on=["model", "variant", "eligible_universe"],
            how="left",
        )
    selected, validation, selection_reason, selection_status, strict_gate_passed = choose_strict_l_s_candidate(validation)

    selected_model = str(selected["model"])
    selected_variant = str(selected["variant"])
    selected_eligible = str(selected["eligible_universe"])
    selected_composite = composites[selected_model]
    test_summary, test_returns, test_constituents = evaluate_candidate(
        ctx,
        selected_composite,
        selected_model,
        ["Test"],
        [selected_variant],
        eligible_name=selected_eligible,
    )

    appendix_summaries = []
    for model, composite in composites.items():
        variants = ["always_l_s", "sticky_l_s"]
        if model == "learned_global":
            variants.extend(["long_only", "short_only", "regime_switch"])
        if model == "learned_regime_specific":
            variants = ["regime_aware_l_s"]
        for eligible_name in eligible_universes:
            summary, _, _ = evaluate_candidate(ctx, composite, model, ["Test"], variants, eligible_name=eligible_name)
            appendix_summaries.append(summary)

    train_summary_rows = []
    for model in ["current_no_team", "equal_family", "heuristic", "learned_global", "learned_regime_specific"]:
        variants = ["always_l_s", "sticky_l_s"] if model != "learned_regime_specific" else ["regime_aware_l_s"]
        for eligible_name in eligible_universes:
            summary, _, _ = evaluate_candidate(ctx, composites[model], model, ["Train"], variants, eligible_name=eligible_name)
            train_summary_rows.append(summary)

    selected_payload = {
        "selected_model": selected_model,
        "selected_variant": selected_variant,
        "selected_eligible_universe": selected_eligible,
        "primary_horizon": PRIMARY_HORIZON,
        "selection_status": selection_status,
        "strict_gate_passed": strict_gate_passed,
        "final_recommendation": "Use selected L/S model" if strict_gate_passed else "No model selected; reporting basket is diagnostic only",
        "selection_reason": selection_reason,
        "weights": weights,
        "validation_row": selected.to_dict(),
    }

    return {
        "ctx": ctx,
        "weights": weights,
        "weight_search_global": global_grid,
        "weight_search_bullish": bullish_grid,
        "weight_search_bearish": bearish_grid,
        "composites": composites,
        "train_summary": pd.concat(train_summary_rows, ignore_index=True),
        "validation_summary": validation,
        "validation_returns": pd.concat([df for df in validation_returns if df is not None and not df.empty], ignore_index=True)
        if any(not df.empty for df in validation_returns)
        else pd.DataFrame(),
        "validation_constituents": pd.concat([df for df in validation_constituents if df is not None and not df.empty], ignore_index=True)
        if any(not df.empty for df in validation_constituents)
        else pd.DataFrame(),
        "selected": selected_payload,
        "rolling_walk_forward": rolling,
        "test_summary": test_summary,
        "test_returns": test_returns,
        "test_constituents": test_constituents,
        "test_appendix": pd.concat(appendix_summaries, ignore_index=True),
    }
