#!/usr/bin/env python3
"""Refresh the daily Macro Signals snapshot used by the Streamlit dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.macro_monitor import MARKET_UPDATE_SNAPSHOT_PATH, build_market_update_snapshot


def main() -> int:
    snapshot = build_market_update_snapshot()
    print(f"snapshot={MARKET_UPDATE_SNAPSHOT_PATH}")
    print(f"created_at={snapshot.get('created_at')}")
    errors = snapshot.get("errors") or []
    print(f"errors={len(errors)}")
    for error in errors[:20]:
        print(f"error={error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
