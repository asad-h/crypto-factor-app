# Crypto Factor Screener

Streamlit dashboard for crypto factor rankings, project-level metrics, market updates, and data quality checks.

## Local Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud

Use `streamlit_app.py` as the app entrypoint. The repo includes the compact `cache/screener` parquet outputs so the dashboard can render immediately on Streamlit Cloud without private API keys.

Add these app-level secrets in Streamlit Cloud under App settings -> Secrets so the live Macro Signals rebuild button can call the same data sources as the scheduled refresh:

```toml
BLOCKWORKS_API_KEY = "..."
BLOCKWORKS_BASE_URL = "https://api.blockworks.com"
COINGECKO_API_KEY = "..."
FRED_API_KEY = "..."
SEC_USER_AGENT = "CryptoFactorModel/1.0 your-email@example.com"
NANSEN_API_KEY = "..."
```

## Daily Data Refresh

GitHub Actions runs `.github/workflows/daily-data-refresh.yml` daily at 03:15 UTC. It refreshes the screener parquet files and Macro Signals snapshot, commits any changed cache files, and pushes to `main` so Streamlit Cloud redeploys automatically.

Required GitHub repo secret:

- `BLOCKWORKS_API_KEY`: used for Blockworks Research factor and metric history.

Recommended GitHub repo secret:

- `COINGECKO_API_KEY`: used by the screener refresh for CoinGecko market caps, FDV, prices, volumes, 7D/30D performance, and fallback chart history. Also used by Macro Signals token movers through the shared screener client. Without it, the workflow falls back to public CoinGecko endpoints and may hit rate limits.
- `NANSEN_API_KEY`: optional, used for Project Detail Nansen holder and flow enrichment when token address overrides are configured. Rotate any key pasted into chat before using it here.

Optional GitHub repo secrets:

- `FRED_API_KEY`: macro data from FRED.
- `SEC_USER_AGENT`: SEC filing fetches in Macro Signals.
