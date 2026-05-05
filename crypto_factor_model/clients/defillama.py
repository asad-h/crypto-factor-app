"""
DefiLlama public API client.

The screener intentionally uses public DefiLlama endpoints only. Pro-only
endpoints from the OpenAPI spec are not called here because this project should
refresh without a DefiLlama Pro key.
"""
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from crypto_factor_model.config import CACHE_DIR

logger = logging.getLogger(__name__)

BASE_URL = "https://bridges.llama.fi"
API_BASE_URL = "https://api.llama.fi"
STABLECOINS_BASE_URL = "https://stablecoins.llama.fi"
COINS_BASE_URL = "https://coins.llama.fi"


class DefiLlamaClient:
    """Fetch cached public DefiLlama data."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir / "defillama"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

    def _json_cache_path(self, name: str) -> Path:
        safe_name = str(name).replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_name}.json"

    def _get_json(
        self,
        base_url: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        cache_name: str | None = None,
        use_cache: bool = True,
        timeout: int = 30,
    ) -> Any:
        cache_path = self._json_cache_path(cache_name) if cache_name else None
        if use_cache and cache_path and cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

        resp = self.session.get(f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}", params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if cache_path:
            with open(cache_path, "w") as f:
                json.dump(data, f)
        return data

    def list_protocols(self, use_cache: bool = True) -> list[dict]:
        """
        Return DefiLlama protocol metadata.

        The protocol list is used for deterministic Blockworks-left matching.
        It includes slug, name, symbol, category, chains, and current TVL.
        """
        return self._get_json(API_BASE_URL, "/protocols", cache_name="protocols", use_cache=use_cache)

    def list_chains(self, use_cache: bool = True) -> list[dict]:
        """Return DefiLlama chain TVL metadata from /v2/chains."""
        return self._get_json(API_BASE_URL, "/v2/chains", cache_name="chains", use_cache=use_cache)

    def get_protocol(self, slug: str, use_cache: bool = True) -> dict:
        """Return full protocol metadata and historical TVL for one slug."""
        encoded = quote(str(slug), safe="")
        return self._get_json(
            API_BASE_URL,
            f"/protocol/{encoded}",
            cache_name=f"protocol_{slug}",
            use_cache=use_cache,
        )

    def get_protocol_tvl(
        self,
        slug: str,
        use_cache: bool = True,
    ) -> pd.Series:
        """Fetch daily total protocol TVL for a DefiLlama slug."""
        cache_path = self.cache_dir / f"protocol_tvl_{slug}.parquet"
        if use_cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            return df.iloc[:, 0].rename(slug)

        data = self.get_protocol(slug, use_cache=use_cache)
        records = data.get("tvl", [])
        if not records:
            return pd.Series(dtype=float, name=slug)

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], unit="s")
        value_col = "totalLiquidityUSD" if "totalLiquidityUSD" in df else "tvl"
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        series = df.set_index("date")[value_col].sort_index().rename(slug)
        series.to_frame().to_parquet(cache_path)
        return series

    def get_chain_tvl(
        self,
        chain: str,
        use_cache: bool = True,
    ) -> pd.Series:
        """Fetch daily chain TVL for a DefiLlama chain name."""
        safe_chain = str(chain).replace("/", "-")
        cache_path = self.cache_dir / f"chain_tvl_{safe_chain}.parquet"
        if use_cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            return df.iloc[:, 0].rename(chain)

        encoded = quote(str(chain), safe="")
        data = self._get_json(
            API_BASE_URL,
            f"/v2/historicalChainTvl/{encoded}",
            cache_name=f"chain_tvl_json_{safe_chain}",
            use_cache=use_cache,
        )
        records = data if isinstance(data, list) else data.get("tvl", [])
        if not records:
            return pd.Series(dtype=float, name=chain)

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], unit="s")
        value_col = "tvl" if "tvl" in df else "totalLiquidityUSD"
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        series = df.set_index("date")[value_col].sort_index().rename(chain)
        series.to_frame().to_parquet(cache_path)
        return series

    def get_fees_summary(
        self,
        slug: str,
        data_type: str = "dailyFees",
        use_cache: bool = True,
    ) -> dict:
        """Return /summary/fees/{protocol} for dailyFees or dailyRevenue."""
        encoded = quote(str(slug), safe="")
        return self._get_json(
            API_BASE_URL,
            f"/summary/fees/{encoded}",
            params={"dataType": data_type},
            cache_name=f"fees_summary_{slug}_{data_type}",
            use_cache=use_cache,
        )

    def get_fees_overview(self, data_type: str = "dailyFees", use_cache: bool = True) -> dict:
        """Return /overview/fees with public protocol breakdown history."""
        return self._get_json(
            API_BASE_URL,
            "/overview/fees",
            params={
                "excludeTotalDataChart": False,
                "excludeTotalDataChartBreakdown": False,
                "dataType": data_type,
            },
            cache_name=f"fees_overview_{data_type}",
            use_cache=use_cache,
        )

    def get_fees_series(
        self,
        slug: str,
        data_type: str = "dailyFees",
        use_cache: bool = True,
    ) -> pd.Series:
        """Return a daily fees/revenue series from /summary/fees/{protocol}."""
        cache_path = self.cache_dir / f"fees_{data_type}_{slug}.parquet"
        if use_cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            return df.iloc[:, 0].rename(slug)

        data = self.get_fees_summary(slug, data_type=data_type, use_cache=use_cache)
        series = _chart_pairs_to_series(data.get("totalDataChart", []), name=slug)
        if not series.empty:
            series.to_frame().to_parquet(cache_path)
        return series

    def get_dex_summary(self, slug: str, use_cache: bool = True) -> dict:
        """Return /summary/dexs/{protocol} for public DEX volume history."""
        encoded = quote(str(slug), safe="")
        return self._get_json(
            API_BASE_URL,
            f"/summary/dexs/{encoded}",
            params={"excludeTotalDataChart": False, "excludeTotalDataChartBreakdown": True},
            cache_name=f"dex_summary_{slug}",
            use_cache=use_cache,
        )

    def get_dex_overview(self, use_cache: bool = True) -> dict:
        """Return /overview/dexs with public protocol breakdown history."""
        return self._get_json(
            API_BASE_URL,
            "/overview/dexs",
            params={"excludeTotalDataChart": False, "excludeTotalDataChartBreakdown": False},
            cache_name="dex_overview",
            use_cache=use_cache,
        )

    def get_dex_volume_series(self, slug: str, use_cache: bool = True) -> pd.Series:
        """Return daily DEX volume from /summary/dexs/{protocol}."""
        cache_path = self.cache_dir / f"dex_volume_{slug}.parquet"
        if use_cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            return df.iloc[:, 0].rename(slug)

        data = self.get_dex_summary(slug, use_cache=use_cache)
        series = _chart_pairs_to_series(data.get("totalDataChart", []), name=slug)
        if not series.empty:
            series.to_frame().to_parquet(cache_path)
        return series

    def get_open_interest_overview(self, use_cache: bool = True) -> dict:
        """Return public perps/open-interest overview with breakdown history."""
        return self._get_json(
            API_BASE_URL,
            "/overview/open-interest",
            params={"excludeTotalDataChart": False, "excludeTotalDataChartBreakdown": False},
            cache_name="open_interest_overview",
            use_cache=use_cache,
        )

    def get_stablecoins(self, include_prices: bool = False, use_cache: bool = True) -> dict:
        """Return public stablecoin metadata."""
        return self._get_json(
            STABLECOINS_BASE_URL,
            "/stablecoins",
            params={"includePrices": include_prices},
            cache_name=f"stablecoins_{include_prices}",
            use_cache=use_cache,
        )

    def get_stablecoin_chain_supply(self, chain: str, use_cache: bool = True) -> pd.Series:
        """Return total stablecoin supply history for a DefiLlama chain."""
        safe_chain = str(chain).replace("/", "-")
        cache_path = self.cache_dir / f"stablecoin_supply_{safe_chain}.parquet"
        if use_cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            return df.iloc[:, 0].rename(chain)

        encoded = quote(str(chain), safe="")
        data = self._get_json(
            STABLECOINS_BASE_URL,
            f"/stablecoincharts/{encoded}",
            cache_name=f"stablecoincharts_{safe_chain}",
            use_cache=use_cache,
        )
        rows = []
        for entry in data if isinstance(data, list) else []:
            total = entry.get("totalCirculating", {}) if isinstance(entry, dict) else {}
            rows.append(
                {
                    "date": pd.to_datetime(entry.get("date"), unit="s"),
                    "value": total.get("peggedUSD"),
                }
            )
        if not rows:
            return pd.Series(dtype=float, name=chain)
        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        series = df.set_index("date")["value"].sort_index().rename(chain)
        series.to_frame().to_parquet(cache_path)
        return series

    def get_current_prices(
        self,
        coins: list[str],
        use_cache: bool = True,
        search_width: str = "6h",
    ) -> dict:
        """Return current prices for coins like coingecko:ethereum."""
        coins = [str(c) for c in coins if c]
        if not coins:
            return {}
        coin_key = ",".join(coins)
        encoded = quote(coin_key, safe=",:")
        return self._get_json(
            COINS_BASE_URL,
            f"/prices/current/{encoded}",
            params={"searchWidth": search_width},
            cache_name=f"prices_current_{coin_key}",
            use_cache=use_cache,
        )

    def get_price_chart(
        self,
        coin: str,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp | None = None,
        span: int = 120,
        period: str = "1d",
        use_cache: bool = True,
    ) -> pd.Series:
        """Return public DefiLlama coin price chart for one coin id."""
        start_ts = int(pd.Timestamp(start).timestamp())
        end_ts = int(pd.Timestamp(end or pd.Timestamp.now()).timestamp())
        cache_path = self.cache_dir / f"price_chart_{str(coin).replace(':', '_')}_{start_ts}_{end_ts}_{period}.parquet"
        if use_cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            return df.iloc[:, 0].rename(coin)

        encoded = quote(str(coin), safe=",:")
        data = self._get_json(
            COINS_BASE_URL,
            f"/chart/{encoded}",
            params={"start": start_ts, "end": end_ts, "span": span, "period": period},
            cache_name=f"price_chart_json_{coin}_{start_ts}_{end_ts}_{period}",
            use_cache=use_cache,
        )
        payload = (data.get("coins") or {}).get(coin, {}) if isinstance(data, dict) else {}
        rows = payload.get("prices", [])
        if not rows:
            return pd.Series(dtype=float, name=coin)
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["timestamp"], unit="s")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        series = df.set_index("date")["price"].sort_index().rename(coin)
        series.to_frame().to_parquet(cache_path)
        return series

    def get_bridge_volume(self, chain: str) -> pd.DataFrame:
        """
        Get bridge volume for a chain.
        Returns DataFrame with date, deposits, withdrawals, net_flow.
        """
        cache_path = self.cache_dir / f"bridge_{chain}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        resp = self.session.get(
            f"{BASE_URL}/transactions/{chain}",
            params={"id": chain},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        records = []
        for entry in data.get("data", data if isinstance(data, list) else []):
            records.append({
                "date": pd.to_datetime(entry.get("date", entry.get("timestamp", 0)), unit="s"),
                "deposits": float(entry.get("depositUSD", 0)),
                "withdrawals": float(entry.get("withdrawUSD", 0)),
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("date").sort_index()
        df["net_flow"] = df["deposits"] - df["withdrawals"]

        df.to_parquet(cache_path)
        return df

    def get_chain_bridge_flows(self, chains: list[str]) -> pd.DataFrame:
        """
        Net bridge flows for multiple chains.
        Returns DataFrame: DatetimeIndex, columns = chain names (net flow values).
        """
        frames = {}
        for chain in chains:
            try:
                df = self.get_bridge_volume(chain)
                if not df.empty:
                    frames[chain] = df["net_flow"]
            except Exception as e:
                logger.warning(f"Bridge data failed for {chain}: {e}")

        if not frames:
            return pd.DataFrame()

        return pd.DataFrame(frames).sort_index()


def _chart_pairs_to_series(records: list, name: str) -> pd.Series:
    """Convert DefiLlama [timestamp, value] chart pairs to a daily Series."""
    if not records:
        return pd.Series(dtype=float, name=name)
    rows = []
    for entry in records:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        rows.append({"date": pd.to_datetime(entry[0], unit="s"), "value": entry[1]})
    if not rows:
        return pd.Series(dtype=float, name=name)
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")["value"].sort_index().rename(name)
