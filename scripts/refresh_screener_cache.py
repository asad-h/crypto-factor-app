#!/usr/bin/env python3
"""Refresh the real-data screener cache for the Streamlit dashboard."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.screener_data import refresh_screener_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh cache/screener parquet outputs.")
    parser.add_argument("--start", default="2024-06-01", help="Historical start date for source pulls.")
    parser.add_argument("--no-force", action="store_true", help="Skip refresh when all output files already exist.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    outputs = refresh_screener_cache(force=not args.no_force, start=args.start)
    print(f"refreshed_at={datetime.now(timezone.utc).isoformat()}")
    for name, path in outputs.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
