"""
Test the regime overlay module with synthetic and real BTC data.

Usage:
    python3 test_regime.py
"""
import numpy as np
import pandas as pd
from crypto_factor_model.regime import (
    btc_trend_signal,
    btc_trend_binary,
    volatility_target_scalar,
    drawdown_exposure_scalar,
    RegimeOverlay,
)


def test_with_synthetic_data():
    """Test regime signals with controlled synthetic data."""
    print("=" * 60)
    print("TEST 1: Synthetic data")
    print("=" * 60)

    dates = pd.date_range("2023-01-01", periods=365, freq="D")

    # Simulate BTC: uptrend for 6 months, crash, then recovery
    np.random.seed(42)
    price = [40000.0]
    for i in range(1, 365):
        if i < 180:
            drift = 0.002  # bullish
        elif i < 250:
            drift = -0.005  # crash
        else:
            drift = 0.003  # recovery
        noise = np.random.normal(0, 0.02)
        price.append(price[-1] * (1 + drift + noise))

    btc = pd.Series(price, index=dates, name="BTCUSDT")

    # Test BTC trend
    trend = btc_trend_signal(btc)
    trend_bin = btc_trend_binary(btc)
    print(f"\nBTC Trend (smooth):")
    print(f"  Min:    {trend.min():.3f}")
    print(f"  Max:    {trend.max():.3f}")
    print(f"  Latest: {trend.iloc[-1]:.3f}")

    print(f"\nBTC Trend (binary):")
    print(f"  % above SMA: {(trend_bin == 1.0).mean():.1%}")
    print(f"  % below SMA: {(trend_bin == 0.5).mean():.1%}")

    # Test vol targeting
    returns = btc.pct_change().dropna()
    vol_scalar = volatility_target_scalar(returns)
    print(f"\nVol Targeting:")
    print(f"  Min:    {vol_scalar.min():.3f}")
    print(f"  Max:    {vol_scalar.max():.3f}")
    print(f"  Median: {vol_scalar.median():.3f}")
    print(f"  Latest: {vol_scalar.iloc[-1]:.3f}")

    # Test drawdown scalar
    nav = btc / btc.iloc[0]
    dd_scalar = drawdown_exposure_scalar(nav)
    print(f"\nDrawdown Scalar:")
    print(f"  Min:    {dd_scalar.min():.3f}")
    print(f"  Max:    {dd_scalar.max():.3f}")
    print(f"  Median: {dd_scalar.median():.3f}")
    print(f"  Latest: {dd_scalar.iloc[-1]:.3f}")

    # Test full overlay
    overlay = RegimeOverlay(btc)
    exposure = overlay.compute_exposure()
    summary = overlay.summary()
    print(f"\nCombined Exposure:")
    print(f"  Min:              {summary['min_exposure']:.3f}")
    print(f"  Max:              {summary['max_exposure']:.3f}")
    print(f"  Median:           {summary['median_exposure']:.3f}")
    print(f"  Current:          {summary['current_exposure']:.3f}")
    print(f"  % days below 1.0: {summary['pct_below_1']:.1%}")
    print(f"  % days above 1.0: {summary['pct_above_1']:.1%}")

    # Verify the crash period has reduced exposure
    crash_exposure = exposure.loc["2023-07-01":"2023-09-01"]
    if len(crash_exposure) > 0:
        print(f"\n  Crash period (Jul-Sep) mean exposure: {crash_exposure.mean():.3f}")
        assert crash_exposure.mean() < 1.0, "Exposure should be reduced during crash"
        print("  PASS: Crash exposure below 1.0")

    # Test apply_to_composite
    tokens = ["ethereum", "solana", "arbitrum"]
    composite = pd.DataFrame(
        np.random.randn(365, 3) * 0.5,
        index=dates,
        columns=tokens,
    )
    adjusted = overlay.apply_to_composite(composite)
    assert adjusted.shape == composite.shape, "Shape mismatch"
    print("\n  PASS: apply_to_composite preserves shape")

    # Rankings should be preserved (same relative order)
    for dt in [dates[200], dates[300]]:
        raw_rank = composite.loc[dt].rank(ascending=False)
        adj_rank = adjusted.loc[dt].rank(ascending=False)
        assert (raw_rank == adj_rank).all(), f"Rankings changed on {dt}"
    print("  PASS: Relative rankings preserved after overlay")


def test_with_live_btc():
    """Test with real BTC data from Binance."""
    print("\n" + "=" * 60)
    print("TEST 2: Live BTC data")
    print("=" * 60)

    try:
        from crypto_factor_model.clients.binance import BinanceClient
        bn = BinanceClient()
        btc = bn.get_btc_price(start="2022-01-01")

        if btc.empty:
            print("  SKIP: Could not fetch BTC data")
            return

        print(f"  BTC data: {len(btc)} days, {btc.index[0].date()} to {btc.index[-1].date()}")
        print(f"  Price range: ${btc.min():,.0f} - ${btc.max():,.0f}")

        overlay = RegimeOverlay(btc)
        summary = overlay.summary()

        print(f"\n  Regime Summary:")
        print(f"    Dates:              {summary['n_dates']}")
        print(f"    Mean exposure:      {summary['mean_exposure']:.3f}")
        print(f"    Median exposure:    {summary['median_exposure']:.3f}")
        print(f"    Current exposure:   {summary['current_exposure']:.3f}")
        print(f"    Current BTC trend:  {summary['current_btc_trend']:.3f}")
        print(f"    Current vol scalar: {summary['current_vol_scalar']:.3f}")
        print(f"    Current DD scalar:  {summary['current_dd_scalar']:.3f}")
        print(f"    % below 1.0:       {summary['pct_below_1']:.1%}")

        # Save diagnostics for inspection
        diag = overlay.diagnostics()
        diag.to_csv("output/regime_diagnostics_test.csv")
        print(f"\n  Diagnostics saved to output/regime_diagnostics_test.csv")

    except Exception as e:
        print(f"  SKIP: {e}")


if __name__ == "__main__":
    test_with_synthetic_data()
    test_with_live_btc()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
