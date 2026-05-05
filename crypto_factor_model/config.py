"""
Configuration for the crypto factor model.
All tuneable parameters in one place.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _get_secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return default


# ── API keys ───────────────────────────────────────────────────────────
BLOCKWORKS_API_KEY = _get_secret("BLOCKWORKS_API_KEY")
BLOCKWORKS_BASE_URL = _get_secret("BLOCKWORKS_BASE_URL", "https://api.blockworks.com")
COINGECKO_API_KEY = _get_secret("COINGECKO_API_KEY") or _get_secret("Coingecko_API_KEY")
COINGECKO_BASE_URL = _get_secret(
    "COINGECKO_BASE_URL",
    "https://pro-api.coingecko.com/api/v3" if COINGECKO_API_KEY else "https://api.coingecko.com/api/v3",
)

# ── Universe filters ───────────────────────────────────────────────────
MIN_MCAP_USD = 100_000_000          # $100M
MIN_DAILY_VOLUME_USD = 10_000_000   # $10M
MIN_LISTING_AGE_DAYS = 30
MIN_HISTORY_WEEKS = 13              # 13 weeks before eligible
EXCLUDE_CATEGORIES = [
    "stablecoin", "wrapped", "pegged", "tokenised-treasury",
    "basket", "infrastructure-aggregate",
]

# ── Rebalance ──────────────────────────────────────────────────────────
REBALANCE_FREQ = "M"                # monthly
TURNOVER_BUFFER_PCT = 0.02          # skip changes below 2%

# ── Signal processing ──────────────────────────────────────────────────
WINSORISE_LOWER = 0.01              # 1st percentile
WINSORISE_UPPER = 0.99              # 99th percentile
MIN_OBS_WINDOW = 13                 # weeks
MIN_NONZERO_OBS = 8                 # within the window
MAX_PAIRWISE_CORR = 0.12            # signal redundancy threshold

# ── Factor family weights (heuristic, regime-neutral) ──────────────────
FAMILY_WEIGHTS = {
    "fundamentals": 0.55,
    "momentum": 0.175,
    "flows": 0.125,
    "team": 0.15,
}

# ── Portfolio constraints ──────────────────────────────────────────────
MAX_POSITIONS = 8
MIN_POSITIONS = 3
MAX_SINGLE_NAME_PCT = 0.15          # 15%
MAX_CHAIN_NARRATIVE_PCT = 0.40      # 40%
MAX_SECTOR_PCT = 0.40               # 40%

# ── Regime thresholds ──────────────────────────────────────────────────
BTC_SMA_WEEKS = 20
VOL_TARGET_ANNUAL = 0.15            # 15% annualised
VOL_LOOKBACK_WEEKS = 8
MAX_LEVERAGE = 1.5
DRAWDOWN_START = -0.10              # start reducing at -10%
DRAWDOWN_MIN_EXPOSURE = -0.20       # minimum exposure at -20%

# ── Backtest ───────────────────────────────────────────────────────────
BACKTEST_START = "2022-01-01"
TX_COST_BPS = 50                    # base case
TX_COST_SENSITIVITY = [25, 50, 75, 100, 138]
DATA_LAG_DAYS = 1                   # T+1 for fundamentals

# ── Blockworks metric identifiers ─────────────────────────────────────
# Chain-level and project-level metrics use DIFFERENT identifiers.
# Use GET /v1/metrics?project={slug} to discover available metrics.
# Verified against docs.blockworksresearch.com/api-reference + live API.

# Chain-level metrics (ethereum, solana, arbitrum, base, etc.)
# These 29 chains have rich fundamentals data (11-19+ metrics each).
BW_CHAIN_METRICS = {
    # Price & supply
    "price": "token-price-usd",
    "supply": "token-supply",
    "fdv": "token-fdv-usd",
    # Revenue & fees (USD)
    "revenue": "rev-usd",
    "trading_fees": "transaction-fee-total-usd",
    "base_fees": "transaction-base-fee-total-usd",
    "priority_fees": "transaction-priority-fee-total-usd",
    "other_fees": "transaction-other-fee-total-usd",
    "mev_tips": "mev-tips-fees-usd",
    "burn": "burn-usd",
    "app_revenue": "app-revenue-total-usd",
    # Activity
    "txn_total": "transaction-total",
    "txn_succeeded": "transaction-succeed-total",
    "txn_failed": "transaction-fail-total",
    "active_addresses": "active-address-total",
    "fee_median_usd": "transaction-fee-med-usd",
    # DEX
    "dex_volume": "dex-spot-volume-total-usd",
    # Stablecoins
    "stablecoin_supply": "stablecoin-supply-total-usd",
    "stablecoin_circulating": "stablecoin-circulating-supply-total-usd",
    # Supply dynamics
    "issuance": "issuance-usd",
    "staked_supply": "token-supply-staked",
    "staking_rate": "staking-rate",
    "liquid_staking_rate": "staking-liquid-rate",
    "net_staking_flow": "staking-net-flow-native",
    # Staking APRs
    "staking_apr_total": "staking-apr-total",
    "staking_apr_issuance": "staking-apr-issuance",
    "staking_apr_mev": "staking-apr-mev",
    "staking_apr_priority": "staking-apr-priority-fees",
    # Lending
    "lending_tvl": "lending-tvl-total-usd",
    "lending_borrows": "lending-borrow-total-usd",
    "lending_deposits": "lending-deposit-total-usd",
    "lending_revenue": "lending-revenue-total-usd",
}

# Project-level metrics (uniswap, aave, lido, etc.)
# 136 projects with sparser data (1-5 metrics, mostly financials + trading).
BW_PROJECT_METRICS = {
    "revenue": "revenue-total-usd",
    "dex_revenue": "dex-revenue-usd",
    "dex_volume": "dex-spot-volume-usd",
    "dex_fees": "dex-fees-usd",
    "memecoin_share": "dex-memecoin-volume-share",
}

# Backward compat alias: default to chain metrics
BW_METRICS = BW_CHAIN_METRICS

# Known chain slugs (from Blockworks API docs)
BW_CHAIN_SLUGS = [
    "aptos", "arbitrum", "avalanche", "base", "berachain", "bitcoin",
    "bnb", "bob", "boba", "celestia", "ethereum", "fogo", "hyperevm",
    "ink", "katana", "megaeth", "mode", "monad", "optimism", "plasma",
    "polygon", "shape", "solana", "superseed", "tron", "unichain",
    "worldchain", "zksync", "zora",
]

# Blockworks slug -> Binance USDT pair mapping
# Chains without a liquid native token on Binance are excluded.
SLUG_TO_BINANCE = {
    "aptos": "APTUSDT",
    "arbitrum": "ARBUSDT",
    "avalanche": "AVAXUSDT",
    "base": None,               # no native token on Binance
    "berachain": "BERAUSDT",
    "bitcoin": "BTCUSDT",
    "bnb": "BNBUSDT",
    "bob": None,
    "boba": "BOBAUSDT",
    "celestia": "TIAUSDT",
    "ethereum": "ETHUSDT",
    "fogo": None,
    "hyperevm": None,
    "ink": None,
    "katana": None,
    "megaeth": None,
    "mode": "MODEUSDT",
    "monad": None,               # not yet launched
    "optimism": "OPUSDT",
    "plasma": None,
    "polygon": "POLUSDT",
    "shape": None,
    "solana": "SOLUSDT",
    "superseed": None,
    "tron": "TRXUSDT",
    "unichain": None,
    "worldchain": "WLDUSDT",
    "zksync": "ZKUSDT",
    "zora": None,
}
