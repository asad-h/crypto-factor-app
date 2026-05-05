"""
Blockworks Research API client.

Fetches chain-level and project-level metrics and caches locally as parquet.

API docs: https://docs.blockworksresearch.com/api-reference
Base URL: https://api.blockworks.com
Auth: x-api-key header

Key endpoints:
    GET /v1/metrics?project={slug}       -> list available metrics for a project
    GET /v1/metrics/{id}?project={slug}  -> timeseries data
    GET /v1/assets                       -> list assets (tokens)
    GET /v1/assets/{slug}/market_cap     -> current market cap snapshot

IMPORTANT: Chain-level and project-level metrics use DIFFERENT identifiers.
    Chain (ethereum):  rev-usd, burn-usd, active-address-total
    Project (uniswap): revenue-total-usd, dex-fees-usd, dex-spot-volume-usd
Use list_project_metrics() to discover available metrics per project.
"""
import time
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from crypto_factor_model.config import (
    BLOCKWORKS_API_KEY,
    BLOCKWORKS_BASE_URL,
    CACHE_DIR,
    BW_CHAIN_METRICS,
    BW_CHAIN_SLUGS,
)

logger = logging.getLogger(__name__)


class BlockworksClient:
    """Thin wrapper around the Blockworks Research API."""

    def __init__(
        self,
        api_key: str = BLOCKWORKS_API_KEY,
        base_url: str = BLOCKWORKS_BASE_URL,
        cache_dir: Path = CACHE_DIR,
        rate_limit_pause: float = 0.25,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir / "blockworks"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_pause = rate_limit_pause
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": api_key,
            "Accept": "application/json",
        })
        # Cache of project -> available metrics (populated lazily)
        self._metric_catalog: dict[str, list[dict]] = {}

    def _cache_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{h}.parquet"

    def _get(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make an authenticated GET request with rate limiting."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        logger.debug(f"GET {url} params={params}")
        time.sleep(self.rate_limit_pause)
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # ── Discovery ─────────────────────────────────────────────────────

    def list_project_metrics(self, project: str) -> list[dict]:
        """
        Discover all available metrics for a project/chain.

        Endpoint: GET /v1/metrics?project={slug}
        Returns list of dicts with keys: name, identifier, category, etc.
        Cached in memory after first call.
        """
        if project in self._metric_catalog:
            return self._metric_catalog[project]

        # Check disk cache
        cache_file = self.cache_dir / f"catalog_{project}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                metrics = json.load(f)
                self._metric_catalog[project] = metrics
                return metrics

        data = self._get("/v1/metrics", params={"project": project})
        metrics = data.get("data", [])
        self._metric_catalog[project] = metrics

        # Persist to disk
        with open(cache_file, "w") as f:
            json.dump(metrics, f)

        logger.info(f"Discovered {len(metrics)} metrics for {project}")
        return metrics

    def get_metric_identifiers(self, project: str) -> dict[str, str]:
        """
        Return dict of metric_name -> identifier for a project.
        e.g. {"Revenue": "rev-usd", "Burn": "burn-usd"} for chains
        or   {"Revenue": "revenue-total-usd", ...} for projects
        """
        metrics = self.list_project_metrics(project)
        return {m["name"]: m["identifier"] for m in metrics}

    def is_chain(self, slug: str) -> bool:
        """Check if a slug is a known chain vs a project."""
        return slug.lower() in BW_CHAIN_SLUGS

    def list_assets(
        self,
        page: int = 1,
        limit: int = 50,
        expand: Optional[list[str]] = None,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        List assets with pagination.
        Endpoint: GET /v1/assets?page={page}&limit={limit}
        Returns list of asset dicts with id, code, slug, sector, category.
        """
        params = {"page": page, "limit": limit}
        if expand:
            params["expand"] = ",".join(expand)
        if filters:
            params.update(filters)
        data = self._get("/v1/assets", params=params)
        return data.get("data", [])

    def list_all_assets(
        self,
        limit: int = 100,
        expand: Optional[list[str]] = None,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Fetch the full paginated Blockworks asset directory.

        Blockworks uses page-based pagination. The live API currently accepts a
        max page size of 100, so clamp larger requests to keep callers from
        tripping a 400 before pagination starts.
        This method is intentionally a discovery primitive; callers still decide
        which assets are eligible for a screener universe based on metric
        coverage, market data, and product type.
        """
        limit = min(max(int(limit), 1), 100)
        page = 1
        assets: list[dict] = []

        while True:
            params = {"page": page, "limit": limit}
            if expand:
                params["expand"] = ",".join(expand)
            if filters:
                params.update(filters)

            data = self._get("/v1/assets", params=params)
            rows = data.get("data", []) or []
            assets.extend(rows)

            total = data.get("total")
            if total is None and isinstance(data.get("meta"), dict):
                total = data["meta"].get("total")
            if not rows:
                break
            if total is not None and page * limit >= int(total):
                break
            if len(rows) < limit:
                break
            page += 1

        logger.info("Fetched %s Blockworks assets across %s pages", len(assets), page)
        return assets

    def get_asset_market_cap(self, slug: str) -> dict:
        """
        Get current market cap snapshot for an asset.
        Tries multiple endpoint patterns since Blockworks docs
        show /market_cap but actual API may differ.
        """
        for path in [f"/v1/assets/{slug}/market_cap",
                     f"/v1/assets/{slug}/market-cap"]:
            try:
                return self._get(path)
            except requests.exceptions.HTTPError:
                continue
        # Fallback: use the expand=market_cap approach
        data = self._get(f"/v1/assets/{slug}", params={"expand": "market_cap"})
        return data.get("market_cap", data)

    # ── Timeseries ────────────────────────────────────────────────────

    def get_timeseries(
        self,
        project: str,
        metric: str,
        start: str = "2021-01-01",
        end: Optional[str] = None,
        use_cache: bool = True,
    ) -> pd.Series:
        """
        Fetch a single metric timeseries for a project/chain.

        Args:
            project: project slug (e.g. "ethereum", "uniswap")
            metric: metric identifier (e.g. "rev-usd", "revenue-total-usd")
            start: ISO date string
            end: ISO date string (defaults to today)
            use_cache: read/write from local parquet cache

        Returns:
            pd.Series with DatetimeIndex
        """
        cache_key = f"{project}_{metric}_{start}_{end}"
        cache_path = self._cache_path(cache_key)

        if use_cache and cache_path.exists():
            df = pd.read_parquet(cache_path)
            return df.iloc[:, 0]

        params = {
            "project": project,
            "start_date": start,
        }
        if end:
            params["end_date"] = end

        data = self._get(f"/v1/metrics/{metric}", params=params)

        # Response: {"project_slug": [{"date": ..., "value": ...}, ...]}
        records = data.get(project, [])

        # Fallback: try first list-valued key in response
        if not records and isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, list) and len(val) > 0:
                    records = val
                    break

        if not records:
            logger.warning(f"No data for {project}/{metric}")
            return pd.Series(dtype=float, name=metric)

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        series = df.set_index("date")["value"].sort_index()
        series.name = metric

        # Cache
        series.to_frame().to_parquet(cache_path)
        return series

    def get_project_data(
        self,
        project: str,
        start: str = "2021-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Auto-discover and fetch ALL available metrics for a project.
        Returns DataFrame with DatetimeIndex, one column per metric name.
        """
        catalog = self.list_project_metrics(project)
        if not catalog:
            logger.warning(f"No metrics found for {project}")
            return pd.DataFrame()

        frames = {}
        for entry in catalog:
            identifier = entry["identifier"]
            name = entry["name"]
            try:
                s = self.get_timeseries(project, identifier, start=start, end=end)
                if len(s) > 0:
                    frames[name] = s
            except Exception as e:
                logger.warning(f"Failed {project}/{identifier}: {e}")

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def get_bulk_metric(
        self,
        metric: str,
        projects: list[str],
        start: str = "2021-01-01",
        end: Optional[str] = None,
        batch_size: int = 10,
    ) -> pd.DataFrame:
        """
        Fetch one metric across many projects.
        Returns DataFrame: DatetimeIndex, columns = project slugs.

        Uses comma-separated project slugs for batch queries.
        Falls back to individual requests if batch fails.
        """
        frames = {}

        for i in range(0, len(projects), batch_size):
            batch = projects[i : i + batch_size]
            batch_str = ",".join(batch)

            try:
                params = {
                    "project": batch_str,
                    "start_date": start,
                }
                if end:
                    params["end_date"] = end

                data = self._get(f"/v1/metrics/{metric}", params=params)

                # Response: {"slug1": [...], "slug2": [...], ...}
                for slug in batch:
                    records = data.get(slug, [])
                    if records:
                        df = pd.DataFrame(records)
                        df["date"] = pd.to_datetime(df["date"])
                        df["value"] = pd.to_numeric(df["value"], errors="coerce")
                        series = df.set_index("date")["value"].sort_index()
                        frames[slug] = series

            except Exception as e:
                logger.warning(f"Batch fetch failed for {metric} batch {i}: {e}")
                for proto in batch:
                    try:
                        s = self.get_timeseries(proto, metric, start=start, end=end)
                        if len(s) > 0:
                            frames[proto] = s
                    except Exception as e2:
                        logger.warning(f"Failed {proto}/{metric}: {e2}")

        if not frames:
            return pd.DataFrame()

        return pd.DataFrame(frames).sort_index()
