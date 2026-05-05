"""
Live API test script. Run from project root:
    python test_api.py

Tests:
    1. Metric discovery for chains and projects
    2. Chain timeseries fetch (Ethereum revenue)
    3. Batch chain fetch (revenue across multiple chains)
    4. Project metric discovery and fetch (Uniswap)
    5. Asset market cap snapshot
    6. Binance price fetch
"""
import json
import logging
from crypto_factor_model.clients.blockworks import BlockworksClient
from crypto_factor_model.clients.binance import BinanceClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    bw = BlockworksClient()
    bn = BinanceClient()

    print("=" * 60)
    print("BLOCKWORKS API TESTS")
    print("=" * 60)

    # ── Test 1: Metric discovery ──────────────────────────────
    print("\n--- Test 1: Discover Ethereum metrics ---")
    eth_metrics = bw.list_project_metrics("ethereum")
    print(f"Found {len(eth_metrics)} metrics for Ethereum:")
    for m in eth_metrics:
        print(f"  {m['identifier']:40s} {m['name']:30s} {m['category']}")

    print("\n--- Test 1b: Discover Uniswap metrics ---")
    uni_metrics = bw.list_project_metrics("uniswap")
    print(f"Found {len(uni_metrics)} metrics for Uniswap:")
    for m in uni_metrics:
        print(f"  {m['identifier']:40s} {m['name']:30s} {m['category']}")

    # ── Test 2: Chain timeseries ──────────────────────────────
    print("\n--- Test 2: Ethereum revenue timeseries ---")
    eth_rev = bw.get_timeseries("ethereum", "rev-usd", start="2024-01-01", use_cache=False)
    print(f"Got {len(eth_rev)} data points")
    print(f"Date range: {eth_rev.index.min().date()} to {eth_rev.index.max().date()}")
    print("Last 5 values:")
    print(eth_rev.tail())

    # ── Test 3: Batch chain fetch ─────────────────────────────
    print("\n--- Test 3: Revenue across chains (batch) ---")
    chains = ["ethereum", "solana", "arbitrum", "base", "polygon"]
    rev_panel = bw.get_bulk_metric("rev-usd", chains, start="2024-06-01")
    print(f"Panel shape: {rev_panel.shape}")
    print(f"Chains with data: {list(rev_panel.columns)}")
    print("Latest row:")
    print(rev_panel.tail(1))

    # ── Test 4: Project data (auto-discover) ──────────────────
    print("\n--- Test 4: Uniswap full data (auto-discover) ---")
    uni_data = bw.get_project_data("uniswap", start="2024-06-01")
    print(f"Columns: {list(uni_data.columns)}")
    print(f"Shape: {uni_data.shape}")
    print("Latest row:")
    print(uni_data.tail(1))

    # ── Test 5: Asset market cap ──────────────────────────────
    print("\n--- Test 5: ETH asset market cap ---")
    try:
        eth_mcap = bw.get_asset_market_cap("ethereum")
        print(json.dumps(eth_mcap, indent=2))
    except Exception as e:
        print(f"Error: {e}")

    # ── Test 6: Binance prices ────────────────────────────────
    print("\n--- Test 6: Binance ETH price ---")
    eth_price = bn.get_daily_close("ETHUSDT", start="2024-06-01")
    print(f"Got {len(eth_price)} daily closes")
    print(f"Latest: ${eth_price.iloc[-1]:.2f}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
