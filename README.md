# Crypto Factor Screener

Streamlit dashboard for crypto factor rankings, project-level metrics, market updates, and data quality checks.

## Local Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud

Use `streamlit_app.py` as the app entrypoint. The repo includes the compact `cache/screener` parquet outputs so the dashboard can render immediately on Streamlit Cloud without private API keys.

## Daily Data Refresh

GitHub Actions runs `.github/workflows/daily-data-refresh.yml` daily at 03:15 UTC. It refreshes the screener parquet files and Market Update snapshot, commits any changed cache files, and pushes to `main` so Streamlit Cloud redeploys automatically.

Required GitHub repo secret:

- `BLOCKWORKS_API_KEY`: used for Blockworks Research factor and metric history.

Recommended GitHub repo secret:

- `COINGECKO_API_KEY`: used by the screener refresh for CoinGecko market caps, FDV, prices, volumes, 7D/30D performance, and fallback chart history. Also used by Market Update token movers through the shared screener client. Without it, the workflow falls back to public CoinGecko endpoints and may hit rate limits.

Optional GitHub repo secrets:

- `FRED_API_KEY`: macro data; the app has a default fallback key.
- `SEC_USER_AGENT`: SEC filing fetches in Market Update.
