"""Shared constants for the May24toMay26 factor research run."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_NAME = "May24toMay26"

WARMUP_START = pd.Timestamp("2023-12-01")
EVALUATION_START = pd.Timestamp("2024-05-01")
TRAIN_START = pd.Timestamp("2024-05-01")
TRAIN_END = pd.Timestamp("2025-04-30")
VALIDATION_START = pd.Timestamp("2025-05-01")
VALIDATION_END = pd.Timestamp("2025-10-31")
TEST_START = pd.Timestamp("2025-11-01")

# Current date in this Codex session is 2026-05-06. Passing this as the
# exclusive API end keeps daily candles label-complete through 2026-05-05.
DATA_END_EXCLUSIVE = pd.Timestamp("2026-05-06")

HORIZONS = (7, 14, 30)
PRIMARY_HORIZON = 14

RESEARCH_CACHE_DIR = PROJECT_ROOT / "cache" / "research"
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "factor_evaluation"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / f"factor_model_walkforward_{RESEARCH_NAME}.ipynb"
DATASET_PATH = RESEARCH_CACHE_DIR / f"factor_research_{RESEARCH_NAME}.parquet"

FAMILIES = ("fundamentals", "momentum", "flows", "factor_improvement")
TX_COST_BPS = 50.0

BASE_VOLUME_USD = 1_000_000.0
SENSITIVITY_VOLUME_USD = 10_000_000.0
MIN_MARKET_CAP_USD = 100_000_000.0
MIN_PRICE_HISTORY_DAYS = 90
MAX_WEIGHT_PER_FAMILY = 0.60
WEIGHT_GRID_STEP = 0.05

