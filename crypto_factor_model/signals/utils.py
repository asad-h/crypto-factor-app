"""
Signal processing utilities.

Core operations:
  - Winsorisation (clip outliers)
  - Cross-sectional rank z-scores
  - Rank IC computation
  - Signal diagnostics
"""
import numpy as np
import pandas as pd
from scipy import stats

from crypto_factor_model.config import (
    WINSORISE_LOWER,
    WINSORISE_UPPER,
    MIN_NONZERO_OBS,
)


def winsorise(
    series: pd.Series,
    lower: float = WINSORISE_LOWER,
    upper: float = WINSORISE_UPPER,
) -> pd.Series:
    """Clip values at the lower/upper percentiles."""
    series = series.replace([np.inf, -np.inf], np.nan)
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def cross_sectional_rank_zscore(row: pd.Series) -> pd.Series:
    """
    Convert a cross-section (one date, many tokens) to rank z-scores.
    Rank 1..N, then convert to z-score (mean 0, std 1).
    NaN values stay NaN.
    """
    ranked = row.rank(method="average", na_option="keep")
    n = ranked.count()
    if n < 3:
        return pd.Series(np.nan, index=row.index)
    # Convert ranks to z-scores: (rank - mean_rank) / std_rank
    mean_rank = (n + 1) / 2
    std_rank = np.sqrt((n ** 2 - 1) / 12)
    return (ranked - mean_rank) / std_rank


def rank_zscore_panel(
    df: pd.DataFrame,
    winsorise_first: bool = True,
) -> pd.DataFrame:
    """
    Apply winsorisation then cross-sectional rank z-scoring to a panel.

    Args:
        df: DataFrame where rows = dates, columns = tokens, values = raw signal
        winsorise_first: clip outliers before ranking

    Returns:
        Same-shaped DataFrame with z-scored values.
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    if winsorise_first:
        df = df.apply(winsorise, axis=1)
    return df.apply(cross_sectional_rank_zscore, axis=1)


def rank_ic(
    signal: pd.Series,
    forward_return: pd.Series,
) -> float:
    """
    Spearman rank correlation between signal scores and forward returns.
    Both inputs should have the same index (tokens at a single date).
    """
    valid = pd.DataFrame({"signal": signal, "ret": forward_return}).dropna()
    if len(valid) < 5:
        return np.nan
    corr, _ = stats.spearmanr(valid["signal"], valid["ret"])
    return corr


def rolling_rank_ic(
    signal_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
    window: int = 13,
) -> pd.Series:
    """
    Compute rolling rank IC over time.

    Args:
        signal_panel: rows = rebalance dates, columns = tokens, values = signal z-scores
        return_panel: rows = dates, columns = tokens, values = forward returns
        window: rolling window in periods

    Returns:
        Series of rank IC values indexed by date.
    """
    # Align dates
    common_dates = signal_panel.index.intersection(return_panel.index)
    ics = []
    for dt in common_dates:
        ic = rank_ic(signal_panel.loc[dt], return_panel.loc[dt])
        ics.append({"date": dt, "ic": ic})

    ic_series = pd.DataFrame(ics).set_index("date")["ic"]
    return ic_series.rolling(window, min_periods=max(3, window // 2)).mean()


def signal_diagnostics(
    signal_panel: pd.DataFrame,
    return_panel: pd.DataFrame,
) -> dict:
    """
    Quick diagnostic for a signal.
    Returns dict with mean IC, IC std, IC IR, % positive IC periods, coverage.
    """
    common_dates = signal_panel.index.intersection(return_panel.index)
    ics = []
    for dt in common_dates:
        ic = rank_ic(signal_panel.loc[dt], return_panel.loc[dt])
        if not np.isnan(ic):
            ics.append(ic)

    if not ics:
        return {"mean_ic": np.nan, "ic_std": np.nan, "ic_ir": np.nan,
                "pct_positive": np.nan, "n_periods": 0}

    ics = np.array(ics)
    mean_ic = ics.mean()
    ic_std = ics.std()
    ic_ir = mean_ic / ic_std if ic_std > 0 else np.nan

    return {
        "mean_ic": round(mean_ic, 4),
        "ic_std": round(ic_std, 4),
        "ic_ir": round(ic_ir, 4),
        "pct_positive": round((ics > 0).mean(), 4),
        "n_periods": len(ics),
    }


def check_min_observations(
    series: pd.Series,
    min_nonzero: int = MIN_NONZERO_OBS,
) -> bool:
    """Check if a signal has enough non-zero, non-NaN observations."""
    valid = series.dropna()
    nonzero = (valid != 0).sum()
    return nonzero >= min_nonzero


def pairwise_signal_correlation(
    signals: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute pairwise Spearman correlation between signal panels.
    Each signal panel: rows = dates, columns = tokens.

    Returns correlation matrix of signals.
    """
    # Flatten each signal panel to a single vector (stacked cross-sections)
    flat = {}
    for name, panel in signals.items():
        flat[name] = panel.stack().dropna()

    # Align on common (date, token) pairs
    combined = pd.DataFrame(flat).dropna()
    return combined.corr(method="spearman")
