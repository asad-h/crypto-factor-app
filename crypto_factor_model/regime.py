"""
Regime overlay module.

Adjusts gross portfolio exposure based on three market-regime signals:
    1. BTC trend (price vs 20-week SMA)
    2. Volatility targeting (realised vol vs annual target)
    3. Drawdown-based scaling (peak-to-trough exposure reduction)

The final exposure scalar multiplies the raw composite scores before
position sizing, so the factor model's relative rankings stay intact
but absolute sizing flexes with market conditions.

Usage:
    from crypto_factor_model.regime import RegimeOverlay
    overlay = RegimeOverlay(btc_price)
    exposure = overlay.compute_exposure()          # Series: date -> scalar [0, MAX_LEVERAGE]
    adjusted = overlay.apply_to_composite(composite)
"""
import logging

import numpy as np
import pandas as pd

from crypto_factor_model.config import (
    BTC_SMA_WEEKS,
    VOL_TARGET_ANNUAL,
    VOL_LOOKBACK_WEEKS,
    MAX_LEVERAGE,
    DRAWDOWN_START,
    DRAWDOWN_MIN_EXPOSURE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weekly_to_daily(weekly_param: int) -> int:
    """Convert a weekly lookback to approximate trading days."""
    return weekly_param * 7


def _annualise_daily_vol(daily_vol: float) -> float:
    """Scale daily vol to annualised (sqrt-365 for crypto, 24/7 markets)."""
    return daily_vol * np.sqrt(365)


# ---------------------------------------------------------------------------
# Individual regime signals
# ---------------------------------------------------------------------------

def btc_trend_signal(
    btc_price: pd.Series,
    sma_weeks: int = BTC_SMA_WEEKS,
) -> pd.Series:
    """
    BTC trend signal: price relative to N-week SMA.

    Returns a Series of scalars:
        > 1.0 when price is above SMA (bullish)
        < 1.0 when price is below SMA (bearish)

    The raw ratio is clipped to [0.5, 1.5] to avoid extreme swings.
    """
    sma_days = _weekly_to_daily(sma_weeks)
    sma = btc_price.rolling(window=sma_days, min_periods=sma_days // 2).mean()
    ratio = btc_price / sma

    # Clip to prevent extreme values
    ratio = ratio.clip(lower=0.5, upper=1.5)
    return ratio.rename("btc_trend")


def btc_trend_binary(
    btc_price: pd.Series,
    sma_weeks: int = BTC_SMA_WEEKS,
) -> pd.Series:
    """
    Binary BTC trend: 1.0 if above SMA, 0.5 if below.

    Simpler variant. Use as a hard risk-off toggle when BTC breaks
    below its 20-week moving average.
    """
    sma_days = _weekly_to_daily(sma_weeks)
    sma = btc_price.rolling(window=sma_days, min_periods=sma_days // 2).mean()
    signal = (btc_price >= sma).astype(float)
    signal = signal.replace(0.0, 0.5)
    return signal.rename("btc_trend_binary")


def volatility_target_scalar(
    portfolio_returns: pd.Series,
    vol_target: float = VOL_TARGET_ANNUAL,
    lookback_weeks: int = VOL_LOOKBACK_WEEKS,
) -> pd.Series:
    """
    Volatility targeting scalar.

    Computes trailing realised vol and returns a scalar that scales
    exposure inversely: when realised vol > target, reduce; when < target,
    increase (capped at MAX_LEVERAGE).

    scalar = vol_target / realised_vol, clipped to [0.25, MAX_LEVERAGE]
    """
    lookback_days = _weekly_to_daily(lookback_weeks)
    daily_vol = portfolio_returns.rolling(
        window=lookback_days,
        min_periods=lookback_days // 2,
    ).std()

    annual_vol = daily_vol.apply(_annualise_daily_vol)

    # Avoid division by zero
    annual_vol = annual_vol.replace(0.0, np.nan)
    scalar = vol_target / annual_vol

    # Floor at 0.25 (never go below 25% exposure), cap at MAX_LEVERAGE
    scalar = scalar.clip(lower=0.25, upper=MAX_LEVERAGE)
    return scalar.rename("vol_target_scalar")


def drawdown_exposure_scalar(
    portfolio_nav: pd.Series,
    dd_start: float = DRAWDOWN_START,
    dd_min_exposure: float = DRAWDOWN_MIN_EXPOSURE,
) -> pd.Series:
    """
    Drawdown-based exposure reduction.

    As the portfolio draws down from its peak, exposure is linearly
    reduced between dd_start (-10%) and dd_min_exposure (-20%).

    Returns:
        Series of scalars in [dd_min_exposure_scalar, 1.0]
        where dd_min_exposure_scalar maps to the minimum exposure at max drawdown.

    At 0% drawdown:  scalar = 1.0
    At -10% drawdown: scalar = 1.0 (just entering reduction zone)
    At -20% drawdown: scalar = 0.25 (minimum)
    Below -20%:       scalar = 0.25 (floor)
    """
    # Running peak
    peak = portfolio_nav.expanding().max()
    drawdown = (portfolio_nav - peak) / peak  # negative values

    # Linear interpolation between dd_start and dd_min_exposure
    # dd_start = -0.10, dd_min_exposure = -0.20
    # At dd_start: scalar = 1.0
    # At dd_min_exposure: scalar = 0.25
    min_scalar = 0.25
    max_scalar = 1.0

    # Normalise drawdown to [0, 1] within the reduction zone
    dd_range = dd_min_exposure - dd_start  # e.g. -0.20 - (-0.10) = -0.10
    pct_through = (drawdown - dd_start) / dd_range  # 0 at start, 1 at max
    pct_through = pct_through.clip(lower=0.0, upper=1.0)

    scalar = max_scalar - pct_through * (max_scalar - min_scalar)

    # Above dd_start (no drawdown concern): scalar = 1.0
    scalar[drawdown > dd_start] = max_scalar

    return scalar.rename("drawdown_scalar")


# ---------------------------------------------------------------------------
# Combined overlay
# ---------------------------------------------------------------------------

class RegimeOverlay:
    """
    Combines BTC trend, vol targeting, and drawdown signals into a single
    exposure multiplier that scales the composite score panel.

    The three signals are combined multiplicatively:
        exposure = btc_trend * vol_scalar * drawdown_scalar

    This means all three must agree for full exposure. A single bearish
    signal pulls exposure down, which is the conservative behavior we want.
    """

    def __init__(
        self,
        btc_price: pd.Series,
        portfolio_returns: pd.Series | None = None,
        portfolio_nav: pd.Series | None = None,
        use_binary_trend: bool = False,
    ):
        """
        Args:
            btc_price: Daily BTC close prices (DatetimeIndex).
            portfolio_returns: Daily portfolio returns for vol targeting.
                               If None, uses BTC returns as proxy.
            portfolio_nav: Portfolio NAV series for drawdown calc.
                           If None, uses BTC price as proxy (reasonable
                           for a crypto-only portfolio).
            use_binary_trend: If True, use hard 1.0/0.5 BTC trend instead
                              of smooth ratio.
        """
        self.btc_price = btc_price.sort_index().dropna()
        self.use_binary_trend = use_binary_trend

        # Default to BTC returns if no portfolio returns provided
        if portfolio_returns is not None:
            self.portfolio_returns = portfolio_returns.sort_index().dropna()
        else:
            self.portfolio_returns = self.btc_price.pct_change().dropna()
            logger.info("No portfolio returns provided; using BTC returns as vol proxy")

        # Default to BTC price as NAV proxy
        if portfolio_nav is not None:
            self.portfolio_nav = portfolio_nav.sort_index().dropna()
        else:
            # Normalise BTC price to start at 1.0 for drawdown calc
            self.portfolio_nav = (self.btc_price / self.btc_price.iloc[0])
            logger.info("No portfolio NAV provided; using normalised BTC price as drawdown proxy")

        # Cache computed signals
        self._btc_trend: pd.Series | None = None
        self._vol_scalar: pd.Series | None = None
        self._dd_scalar: pd.Series | None = None
        self._exposure: pd.Series | None = None

    @property
    def btc_trend(self) -> pd.Series:
        if self._btc_trend is None:
            if self.use_binary_trend:
                self._btc_trend = btc_trend_binary(self.btc_price)
            else:
                self._btc_trend = btc_trend_signal(self.btc_price)
        return self._btc_trend

    @property
    def vol_scalar(self) -> pd.Series:
        if self._vol_scalar is None:
            self._vol_scalar = volatility_target_scalar(self.portfolio_returns)
        return self._vol_scalar

    @property
    def dd_scalar(self) -> pd.Series:
        if self._dd_scalar is None:
            self._dd_scalar = drawdown_exposure_scalar(self.portfolio_nav)
        return self._dd_scalar

    def compute_exposure(self) -> pd.Series:
        """
        Compute the combined exposure scalar.

        Returns:
            Series: date -> exposure multiplier, clipped to [0.25, MAX_LEVERAGE].
        """
        if self._exposure is not None:
            return self._exposure

        # Align all three signals to a common index
        combined = pd.DataFrame({
            "btc_trend": self.btc_trend,
            "vol_scalar": self.vol_scalar,
            "dd_scalar": self.dd_scalar,
        }).dropna()

        if combined.empty:
            logger.warning("No overlapping dates across regime signals")
            return pd.Series(dtype=float, name="exposure")

        # Multiplicative combination
        exposure = combined["btc_trend"] * combined["vol_scalar"] * combined["dd_scalar"]

        # Final clip
        exposure = exposure.clip(lower=0.25, upper=MAX_LEVERAGE)
        exposure.name = "exposure"

        self._exposure = exposure
        return exposure

    def apply_to_composite(
        self,
        composite: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Scale the composite score panel by the regime exposure scalar.

        The relative ranking of tokens is preserved (scores are multiplied
        by the same scalar on each date). The absolute magnitude changes,
        which flows through to position sizing.

        Args:
            composite: DataFrame (dates x tokens) of composite scores.

        Returns:
            Adjusted composite panel.
        """
        exposure = self.compute_exposure()

        # Align exposure to composite dates
        aligned_exposure = exposure.reindex(composite.index).ffill()

        # Multiply each row by that day's exposure scalar
        adjusted = composite.multiply(aligned_exposure, axis=0)

        n_reduced = (aligned_exposure < 1.0).sum()
        n_elevated = (aligned_exposure > 1.0).sum()
        logger.info(
            f"Regime overlay applied: {n_reduced} days reduced exposure, "
            f"{n_elevated} days elevated exposure, "
            f"median scalar = {aligned_exposure.median():.3f}"
        )

        return adjusted

    def diagnostics(self) -> pd.DataFrame:
        """
        Return a diagnostics DataFrame with all regime signals and the
        combined exposure for analysis and charting.
        """
        exposure = self.compute_exposure()

        diag = pd.DataFrame({
            "btc_price": self.btc_price,
            "btc_trend": self.btc_trend,
            "vol_scalar": self.vol_scalar,
            "dd_scalar": self.dd_scalar,
            "exposure": exposure,
        })
        return diag

    def summary(self) -> dict:
        """
        Quick summary stats for logging.
        """
        exposure = self.compute_exposure()
        return {
            "n_dates": len(exposure),
            "mean_exposure": exposure.mean(),
            "median_exposure": exposure.median(),
            "min_exposure": exposure.min(),
            "max_exposure": exposure.max(),
            "pct_below_1": (exposure < 1.0).mean(),
            "pct_above_1": (exposure > 1.0).mean(),
            "current_exposure": exposure.iloc[-1] if len(exposure) > 0 else None,
            "current_btc_trend": self.btc_trend.iloc[-1] if len(self.btc_trend) > 0 else None,
            "current_vol_scalar": self.vol_scalar.iloc[-1] if len(self.vol_scalar) > 0 else None,
            "current_dd_scalar": self.dd_scalar.iloc[-1] if len(self.dd_scalar) > 0 else None,
        }
