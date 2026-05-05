# Crypto Factor Screener

Streamlit dashboard for crypto factor rankings, project-level metrics, market updates, and data quality checks.

## Local Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud

Use `streamlit_app.py` as the app entrypoint. The repo includes the compact `cache/screener` parquet outputs so the dashboard can render immediately on Streamlit Cloud without private API keys.
