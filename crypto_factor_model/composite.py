"""
Composite score computation.

Takes z-scored signal panels from each family, applies family weights,
and produces a single composite ranking per token at each rebalance date.
"""
import logging

import numpy as np
import pandas as pd

from crypto_factor_model.config import FAMILY_WEIGHTS, MIN_POSITIONS, MAX_POSITIONS
from crypto_factor_model.signals.utils import rank_zscore_panel

logger = logging.getLogger(__name__)


def compute_family_score(
    signals: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Equal-weight average of z-scored signals within a family.

    Args:
        signals: dict of signal_name -> panel (dates x tokens)

    Returns:
        Single DataFrame: family z-score per token per date.
    """
    if not signals:
        return pd.DataFrame()

    # Z-score each signal cross-sectionally
    z_scored = {}
    for name, panel in signals.items():
        z_scored[name] = rank_zscore_panel(panel)

    # Stack and average
    stacked = pd.concat(z_scored.values(), keys=z_scored.keys())
    # Group by date (level 1) and token (columns), take mean
    family_score = stacked.groupby(level=1).mean()
    return family_score


def compute_composite(
    family_scores: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
    min_families: int = 2,
) -> pd.DataFrame:
    """
    Weighted combination of family scores into a single composite.

    Args:
        family_scores: dict mapping family name -> score panel
        weights: family weights (defaults to config FAMILY_WEIGHTS)
        min_families: minimum number of families with valid scores
                      for a token to receive a composite

    Returns:
        DataFrame: composite score per token per date.
    """
    if weights is None:
        weights = FAMILY_WEIGHTS

    # Normalise weights to sum to 1 for available families
    available = [f for f in family_scores if not family_scores[f].empty]
    if not available:
        return pd.DataFrame()

    total_weight = sum(weights.get(f, 0) for f in available)
    if total_weight == 0:
        return pd.DataFrame()

    norm_weights = {f: weights.get(f, 0) / total_weight for f in available}

    # Align all panels to common dates and tokens
    all_dates = sorted(set().union(*(f.index for f in family_scores.values() if not f.empty)))
    all_tokens = sorted(set().union(*(f.columns for f in family_scores.values() if not f.empty)))

    composite = pd.DataFrame(0.0, index=all_dates, columns=all_tokens)
    coverage = pd.DataFrame(0, index=all_dates, columns=all_tokens)

    for family, panel in family_scores.items():
        if panel.empty or family not in norm_weights:
            continue
        w = norm_weights[family]
        aligned = panel.reindex(index=all_dates, columns=all_tokens)
        valid_mask = aligned.notna()
        composite += aligned.fillna(0) * w
        coverage += valid_mask.astype(int)

    # Mask tokens with insufficient family coverage
    n_families = len(available)
    composite[coverage < min_families] = np.nan

    return composite


def rank_and_select(
    composite: pd.DataFrame,
    n_positions: int = MAX_POSITIONS,
    eligible_tokens: list[str] | None = None,
) -> pd.DataFrame:
    """
    Rank tokens by composite score at each date and select top N.

    Args:
        composite: composite score panel
        n_positions: max positions to select
        eligible_tokens: restrict to these tokens (from universe screen)

    Returns:
        DataFrame with columns: date, token, rank, composite_score, selected
    """
    records = []

    for date in composite.index:
        row = composite.loc[date].dropna()
        if eligible_tokens:
            row = row[row.index.isin(eligible_tokens)]

        if row.empty:
            continue

        ranked = row.sort_values(ascending=False)
        for i, (token, score) in enumerate(ranked.items()):
            records.append({
                "date": date,
                "token": token,
                "rank": i + 1,
                "composite_score": score,
                "selected": i < n_positions,
            })

    return pd.DataFrame(records)
