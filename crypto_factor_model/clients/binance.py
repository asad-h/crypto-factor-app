"""
Binance public API client for price data.
No API key required for public market data endpoints.
"""
import time
import logging
from pathlib import Path

import pandas as pd
import requests

from crypto_factor_model.config import CACHE_DIR

logger = logging.getLogger(__name__)

BASE_URL = "https://api.binance.com"
FAPI_URL = "https://fapi.binance.com"


class BinanceClient:
    """Fetches OHLCV data from Binance public API."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir / "binance"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def get_klines(
        self,
        symbol: str,
        interval: str = "1d",
        start: str = "2021-01-01",
        end: str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV klines for a symbol.

        Args:
            symbol: trading pair (e.g. "BTCUSDT", "ETHUSDT")
            interval: candle interval ("1h", "4h", "1d", "1w")
            start: ISO date string
            end: ISO date string (defaults to now)

        Returns:
            DataFrame with columns: open, high, low, close, volume, quote_volume
        """
        cache_path = self.cache_dir / f"{symbol}_{interval}_{start}_{end}.parquet"
        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = int(pd.Timestamp(end or pd.Timestamp.now()).timestamp() * 1000)

        all_rows = []
        current = start_ms

        while current < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current,
                "endTime": end_ms,
                "limit": 1000,
            }

            resp = self.session.get(f"{BASE_URL}/api/v3/klines", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            all_rows.extend(data)
            current = data[-1][6] + 1  # close_time + 1ms
            time.sleep(0.1)  # rate limit

        if not all_rows:
            logger.warning(f"No kline data for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_vol",
            "taker_buy_quote_vol", "ignore",
        ])

        df["date"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)

        df = df.set_index("date")[["open", "high", "low", "close", "volume", "quote_volume"]]
        df = df.sort_index()

        df.to_parquet(cache_path)
        return df

    def get_futures_klines(
        self,
        symbol: str,
        interval: str = "1d",
        start: str = "2021-01-01",
        end: str | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch USD-M futures OHLCV klines for a symbol."""
        cache_path = self.cache_dir / f"futures_{symbol}_{interval}_{start}_{end}.parquet"
        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = int(pd.Timestamp(end or pd.Timestamp.now()).timestamp() * 1000)

        all_rows = []
        current = start_ms

        while current < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current,
                "endTime": end_ms,
                "limit": 1000,
            }

            resp = self.session.get(f"{FAPI_URL}/fapi/v1/klines", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            all_rows.extend(data)
            current = data[-1][6] + 1
            time.sleep(0.1)

        if not all_rows:
            logger.warning(f"No futures kline data for {symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_vol",
            "taker_buy_quote_vol", "ignore",
        ])

        df["date"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)

        df = df.set_index("date")[["open", "high", "low", "close", "volume", "quote_volume"]]
        df = df.sort_index()

        df.to_parquet(cache_path)
        return df

    def get_daily_close(
        self,
        symbol: str,
        start: str = "2021-01-01",
        end: str | None = None,
    ) -> pd.Series:
        """Return daily close prices as a Series."""
        df = self.get_klines(symbol, interval="1d", start=start, end=end)
        if df.empty:
            return pd.Series(dtype=float, name=symbol)
        return df["close"].rename(symbol)

    def get_btc_price(self, start: str = "2021-01-01") -> pd.Series:
        """Convenience: BTC daily close."""
        return self.get_daily_close("BTCUSDT", start=start)

    def get_multiple_daily(
        self,
        symbols: list[str],
        start: str = "2021-01-01",
    ) -> pd.DataFrame:
        """
        Fetch daily close for multiple symbols.
        Returns DataFrame: DatetimeIndex, columns = symbols.
        """
        frames = {}
        for sym in symbols:
            try:
                s = self.get_daily_close(sym, start=start)
                if len(s) > 0:
                    frames[sym] = s
            except Exception as e:
                logger.warning(f"Failed to fetch {sym}: {e}")
        return pd.DataFrame(frames).sort_index()

    def get_multiple_daily_ohlcv(
        self,
        symbols: list[str],
        start: str = "2021-01-01",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch daily OHLCV frames for multiple symbols.

        Returns a dict keyed by Binance symbol so callers can use close prices
        and actual quote volume instead of reconstructing volume placeholders.
        """
        frames: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = self.get_klines(sym, interval="1d", start=start)
                if not df.empty:
                    frames[sym] = df
            except Exception as e:
                logger.warning(f"Failed to fetch OHLCV for {sym}: {e}")
        return frames

    def get_multiple_futures_daily_ohlcv(
        self,
        symbols: list[str],
        start: str = "2021-01-01",
    ) -> dict[str, pd.DataFrame]:
        """Fetch daily USD-M futures OHLCV frames for multiple symbols."""
        frames: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = self.get_futures_klines(sym, interval="1d", start=start)
                if not df.empty:
                    frames[sym] = df
            except Exception as e:
                logger.warning(f"Failed to fetch futures OHLCV for {sym}: {e}")
        return frames

    def get_exchange_symbols(self, quote_asset: str = "USDT") -> set[str]:
        """Return actively traded spot symbols for a quote asset."""
        quote_asset = quote_asset.upper()
        cache_path = self.cache_dir / f"spot_symbols_{quote_asset}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            return set(df["symbol"].astype(str))

        resp = self.session.get(f"{BASE_URL}/api/v3/exchangeInfo", timeout=20)
        resp.raise_for_status()
        rows = []
        for entry in resp.json().get("symbols", []):
            if entry.get("status") != "TRADING":
                continue
            if entry.get("quoteAsset") != quote_asset:
                continue
            rows.append({"symbol": entry.get("symbol")})

        df = pd.DataFrame(rows).dropna().drop_duplicates()
        df.to_parquet(cache_path)
        return set(df["symbol"].astype(str))

    def get_futures_symbols(self, quote_asset: str = "USDT") -> set[str]:
        """Return actively traded USD-M perpetual futures symbols."""
        quote_asset = quote_asset.upper()
        cache_path = self.cache_dir / f"futures_symbols_{quote_asset}.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            return set(df["symbol"].astype(str))

        resp = self.session.get(f"{FAPI_URL}/fapi/v1/exchangeInfo", timeout=20)
        resp.raise_for_status()
        rows = []
        for entry in resp.json().get("symbols", []):
            if entry.get("status") != "TRADING":
                continue
            if entry.get("quoteAsset") != quote_asset:
                continue
            if entry.get("contractType") not in {"PERPETUAL", None}:
                continue
            rows.append({"symbol": entry.get("symbol")})

        df = pd.DataFrame(rows).dropna().drop_duplicates()
        df.to_parquet(cache_path)
        return set(df["symbol"].astype(str))

    def get_hourly_ohlcv(
        self,
        symbol: str,
        start: str = "2021-01-01",
    ) -> pd.DataFrame:
        """Convenience: 1h OHLCV for momentum signal computation."""
        return self.get_klines(symbol, interval="1h", start=start)

    def get_funding_rate(
        self,
        symbol: str = "BTCUSDT",
        start: str = "2024-01-01",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates from Binance Futures.

        Endpoint: GET /fapi/v1/fundingRate (public, no auth needed)
        Returns every 8h funding rate. Columns: fundingTime, fundingRate, symbol.
        """
        cache_path = self.cache_dir / f"funding_{symbol}_{start}.parquet"
        if use_cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        fapi_url = "https://fapi.binance.com"
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        all_rows = []
        current = start_ms

        while True:
            params = {
                "symbol": symbol,
                "startTime": current,
                "limit": 1000,
            }
            resp = self.session.get(
                f"{fapi_url}/fapi/v1/fundingRate",
                params=params, timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_rows.extend(data)
            current = data[-1]["fundingTime"] + 1
            time.sleep(0.1)
            if len(data) < 1000:
                break

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["fundingRate"] = df["fundingRate"].astype(float)
        df = df.set_index("date")[["symbol", "fundingRate"]].sort_index()

        df.to_parquet(cache_path)
        return df

    def get_current_funding_annualized(
        self,
        symbol: str = "BTCUSDT",
        lookback_days: int = 7,
    ) -> float | None:
        """
        Get annualized funding rate for a symbol.

        Funding is paid 3x/day on Binance. Annualized = avg_8h_rate * 3 * 365 * 100.
        Uses the last `lookback_days` of funding data for the average.
        """
        try:
            start = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days + 5)).strftime("%Y-%m-%d")
            df = self.get_funding_rate(symbol, start=start)
            if df.empty:
                return None
            recent = df.tail(lookback_days * 3)  # 3 per day
            avg_rate = recent["fundingRate"].mean()
            return avg_rate * 3 * 365 * 100  # annualized percentage
        except Exception as e:
            logger.warning(f"Funding rate fetch failed for {symbol}: {e}")
            return None

    def get_open_interest(
        self,
        symbol: str = "BTCUSDT",
    ) -> float | None:
        """
        Get current open interest in USDT from Binance Futures.

        Endpoint: GET /fapi/v1/openInterest (public)
        """
        try:
            resp = self.session.get(
                f"{FAPI_URL}/fapi/v1/openInterest",
                params={"symbol": symbol}, timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("openInterest", 0))
        except Exception as e:
            logger.warning(f"OI fetch failed for {symbol}: {e}")
            return None

    def get_open_interest_history(
        self,
        symbol: str = "BTCUSDT",
        start: str | None = None,
        end: str | None = None,
        period: str = "1d",
        use_cache: bool = True,
    ) -> pd.Series:
        """
        Fetch Binance USD-M futures open-interest history in quote currency.

        Binance's openInterestHist endpoint returns `sumOpenInterestValue`,
        which is the useful USD/USDT notional for the screener. The endpoint is
        public but only exposes recent history, so callers should request the
        latest 30-45 days for WoW/MoM comparisons.
        """
        if start is None:
            start = (pd.Timestamp.now() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        cache_path = self.cache_dir / f"open_interest_{symbol}_{period}_{start}_{end}.parquet"
        if use_cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            return df.iloc[:, 0].rename(symbol)

        if end is None:
            params = {
                "symbol": symbol,
                "period": period,
                "limit": 60,
            }
            resp = self.session.get(f"{FAPI_URL}/futures/data/openInterestHist", params=params, timeout=15)
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                return pd.Series(dtype=float, name=symbol)
            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
            value_col = "sumOpenInterestValue" if "sumOpenInterestValue" in df else "sumOpenInterest"
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
            series = (
                df.set_index("date")[value_col]
                .sort_index()
                .groupby(level=0)
                .last()
                .rename(symbol)
            )
            series.to_frame().to_parquet(cache_path)
            return series

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.now()
        # The endpoint is recent-history only. Clamp old starts to keep the
        # request valid while preserving the 30D changes the dashboard needs.
        start_ts = max(start_ts, end_ts - pd.Timedelta(days=45))
        current = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)

        rows = []
        while current < end_ms:
            params = {
                "symbol": symbol,
                "period": period,
                "startTime": current,
                "endTime": end_ms,
                "limit": 500,
            }
            resp = self.session.get(f"{FAPI_URL}/futures/data/openInterestHist", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            rows.extend(data)
            current = int(data[-1]["timestamp"]) + 1
            time.sleep(0.1)
            if len(data) < 500:
                break

        if not rows:
            return pd.Series(dtype=float, name=symbol)

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        value_col = "sumOpenInterestValue" if "sumOpenInterestValue" in df else "sumOpenInterest"
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        series = (
            df.set_index("date")[value_col]
            .sort_index()
            .groupby(level=0)
            .last()
            .rename(symbol)
        )
        series.to_frame().to_parquet(cache_path)
        return series

    def get_multiple_open_interest_history(
        self,
        symbols: list[str],
        start: str | None = None,
    ) -> pd.DataFrame:
        """Fetch open-interest history for multiple futures symbols."""
        frames = {}
        for sym in symbols:
            try:
                s = self.get_open_interest_history(sym, start=start)
                if len(s) > 0:
                    frames[sym] = s
            except Exception as e:
                logger.warning(f"Failed to fetch OI history for {sym}: {e}")
        return pd.DataFrame(frames).sort_index()
