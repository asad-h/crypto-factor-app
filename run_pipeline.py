"""
Run the full factor model pipeline.

Usage:
    python3 run_pipeline.py              # full universe (29 chains)
    python3 run_pipeline.py --fast       # quick test (5 chains)
"""
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# Patch chain list for fast mode before importing main
if "--fast" in sys.argv:
    import crypto_factor_model.config as cfg
    cfg.BW_CHAIN_SLUGS = ["ethereum", "solana", "arbitrum", "optimism", "polygon"]
    cfg.BACKTEST_START = "2024-01-01"  # shorter history
    print("FAST MODE: 5 chains, start 2024-01-01\n")

from crypto_factor_model.main import run_pipeline

if __name__ == "__main__":
    composite, rankings = run_pipeline()

    # Print summary
    selected = rankings[rankings["selected"]]
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total dates ranked: {rankings['date'].nunique()}")
    print(f"Total chains scored: {rankings['token'].nunique()}")
    print(f"Selected positions (most recent):")

    latest_date = selected["date"].max()
    latest = selected[selected["date"] == latest_date].sort_values("rank")
    for _, r in latest.iterrows():
        print(f"  #{int(r['rank']):2d}  {r['token']:15s}  score={r['composite_score']:.4f}")
