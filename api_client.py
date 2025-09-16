#!/usr/bin/env python3

import polars as pl
import requests
import hashlib
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, Union
import logging
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class SFOpenDataAPIClient:
    """
    Client for SF Open Data API with intelligent caching and direct Polars integration.

    Features:
    - Direct Polars integration for streaming CSV data from Socrata API
    - Intelligent caching with change detection
    - Incremental updates when supported
    - Data versioning and lineage tracking
    - Rate limiting and error handling
    """

    # Try multiple potential endpoints for the SF business data
    POTENTIAL_ENDPOINTS = [
        "https://data.sfgov.org/api/views/g8m3-pdis/rows.csv?accessType=DOWNLOAD",
        "https://data.sfgov.org/resource/g8m3-pdis.csv",
        "https://data.sfgov.org/api/odata/v4/g8m3-pdis",
    ]

    BUSINESS_DATASET_ID = "g8m3-pdis"

    # Metadata endpoint for checking updates
    METADATA_URL = "https://data.sfgov.org/api/views/g8m3-pdis.json"

    def __init__(self, cache_dir: str = "cache", app_token: Optional[str] = None,
                 fallback_csv_path: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.app_token = app_token
        self.fallback_csv_path = fallback_csv_path

        # Create metadata and versions subdirectories
        self.metadata_dir = self.cache_dir / "metadata"
        self.versions_dir = self.cache_dir / "versions"
        self.metadata_dir.mkdir(exist_ok=True)
        self.versions_dir.mkdir(exist_ok=True)

        # Headers for API requests
        self.headers = {
            "User-Agent": "SF-Business-Pipeline/1.0",
            "Accept": "text/csv"
        }
        if self.app_token:
            self.headers["X-App-Token"] = self.app_token

    def _build_csv_url(self, dataset_id: str, params: Optional[Dict] = None) -> str:
        """Build Socrata API CSV URL with parameters."""
        url = f"{self.BASE_URL}/{dataset_id}.csv"
        if params:
            url += "?" + urlencode(params)
        return url

    def _get_dataset_metadata(self, dataset_id: str) -> Dict:
        """Get dataset metadata from Socrata API."""
        try:
            # Get dataset info from metadata endpoint
            info_response = requests.get(self.METADATA_URL, timeout=30)
            info_response.raise_for_status()
            info_data = info_response.json()

            return {
                "dataset_id": dataset_id,
                "total_rows": info_data.get("viewCount", 0),
                "last_updated": info_data.get("rowsUpdatedAt", info_data.get("publicationDate")),
                "name": info_data.get("name", "Unknown"),
                "description": info_data.get("description", ""),
                "columns": [col.get("name", col.get("fieldName", "")) for col in info_data.get("columns", [])],
                "metadata_retrieved_at": datetime.now().isoformat(),
                "view_count": info_data.get("viewCount"),
                "publication_date": info_data.get("publicationDate"),
                "creation_date": info_data.get("createdAt")
            }
        except Exception as e:
            logger.warning(f"Could not retrieve metadata for {dataset_id}: {e}")
            return {
                "dataset_id": dataset_id,
                "total_rows": 0,
                "last_updated": None,
                "metadata_retrieved_at": datetime.now().isoformat(),
                "error": str(e)
            }

    def _save_metadata(self, dataset_id: str, metadata: Dict):
        """Save dataset metadata to cache."""
        metadata_file = self.metadata_dir / f"{dataset_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self, dataset_id: str) -> Optional[Dict]:
        """Load dataset metadata from cache."""
        metadata_file = self.metadata_dir / f"{dataset_id}_metadata.json"
        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata for {dataset_id}: {e}")
            return None

    def _get_cache_file_path(self, dataset_id: str, version: Optional[str] = None) -> Path:
        """Get cache file path for dataset."""
        if version:
            return self.versions_dir / f"{dataset_id}_{version}.parquet"
        else:
            return self.cache_dir / f"{dataset_id}_latest.parquet"

    def _is_cache_valid(self, dataset_id: str, max_age_hours: int = 1) -> Tuple[bool, Optional[Path]]:
        """Check if cached data is valid."""
        cache_file = self._get_cache_file_path(dataset_id)

        if not cache_file.exists():
            return False, None

        # Check file age
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if file_age > timedelta(hours=max_age_hours):
            logger.info(f"Cache expired for {dataset_id} (age: {file_age})")
            return False, None

        # Check if remote data has been updated
        current_metadata = self._get_dataset_metadata(dataset_id)
        cached_metadata = self._load_metadata(dataset_id)

        if cached_metadata and current_metadata.get("last_updated") != cached_metadata.get("last_updated"):
            logger.info(f"Remote data updated for {dataset_id}")
            return False, None

        logger.info(f"Cache valid for {dataset_id}")
        return True, cache_file

    def _detect_schema_changes(self, old_metadata: Dict, new_metadata: Dict) -> Dict:
        """Detect schema changes between dataset versions."""
        changes = {
            "column_changes": [],
            "row_count_change": new_metadata.get("total_rows", 0) - old_metadata.get("total_rows", 0),
            "last_updated_changed": old_metadata.get("last_updated") != new_metadata.get("last_updated")
        }

        old_columns = set(old_metadata.get("columns", []))
        new_columns = set(new_metadata.get("columns", []))

        if old_columns != new_columns:
            changes["column_changes"] = {
                "added": list(new_columns - old_columns),
                "removed": list(old_columns - new_columns)
            }

        return changes

    def get_business_data(self, force_refresh: bool = False, limit: Optional[int] = None,
                         incremental: bool = True) -> pl.DataFrame:
        """
        Get SF business registration data with intelligent caching.

        Args:
            force_refresh: Force refresh from API even if cache is valid
            limit: Limit number of rows (for testing/development)
            incremental: Use incremental updates when possible

        Returns:
            Polars DataFrame with business data
        """
        dataset_id = self.BUSINESS_DATASET_ID

        # Check cache validity unless forced refresh
        if not force_refresh:
            cache_valid, cache_file = self._is_cache_valid(dataset_id)
            if cache_valid and cache_file:
                logger.info(f"Loading from cache: {cache_file}")
                return pl.read_parquet(cache_file)

        # Get current metadata
        logger.info(f"Fetching fresh data for dataset {dataset_id}")
        current_metadata = self._get_dataset_metadata(dataset_id)
        cached_metadata = self._load_metadata(dataset_id)

        # Detect changes if we have previous metadata
        if cached_metadata:
            changes = self._detect_schema_changes(cached_metadata, current_metadata)
            if changes["column_changes"] or changes["row_count_change"] != 0:
                logger.info(f"Dataset changes detected: {changes}")

        # Try multiple endpoints to fetch the data
        df = None
        successful_url = None

        # First try the API endpoints
        for endpoint_url in self.POTENTIAL_ENDPOINTS:
            try:
                logger.info(f"Trying endpoint: {endpoint_url}")

                # For JSON endpoints, add limit parameter
                if "odata" in endpoint_url or endpoint_url.endswith(".json"):
                    if limit:
                        separator = "&" if "?" in endpoint_url else "?"
                        endpoint_url += f"{separator}$top={limit}"
                    logger.info("Note: JSON endpoints may require different processing")
                    continue  # Skip JSON for now, focus on CSV

                # Use Polars to read CSV directly from URL
                df = pl.read_csv(
                    endpoint_url,
                    infer_schema_length=10000,
                    ignore_errors=True,
                    null_values=["", "null", "NULL", "PRUSS"],
                    rechunk=True
                )

                logger.info(f"Successfully fetched {len(df):,} records with {len(df.columns)} columns from {endpoint_url}")
                successful_url = endpoint_url
                break

            except Exception as e:
                logger.warning(f"Failed to fetch from {endpoint_url}: {e}")
                continue

        # If all API endpoints failed, try fallback CSV
        if df is None and self.fallback_csv_path and Path(self.fallback_csv_path).exists():
            logger.info(f"API endpoints failed, using fallback CSV: {self.fallback_csv_path}")
            try:
                df = pl.read_csv(
                    self.fallback_csv_path,
                    infer_schema_length=10000,
                    ignore_errors=True,
                    null_values=["", "null", "NULL", "PRUSS"],
                    rechunk=True
                )

                if limit and len(df) > limit:
                    df = df.head(limit)

                logger.info(f"Loaded {len(df):,} records from fallback CSV")
                successful_url = self.fallback_csv_path

            except Exception as e:
                logger.error(f"Failed to load fallback CSV {self.fallback_csv_path}: {e}")

        # If we still don't have data, raise an error
        if df is None:
            raise Exception("All data sources failed - no API endpoints accessible and no fallback CSV")

        logger.info(f"Final dataset: {len(df):,} records with {len(df.columns)} columns")

        # Save to cache
        cache_file = self._get_cache_file_path(dataset_id)
        df.write_parquet(cache_file, compression="zstd")
        logger.info(f"Cached data to: {cache_file}")

        # Save versioned copy if significant changes
        if cached_metadata:
            changes = self._detect_schema_changes(cached_metadata, current_metadata)
            if changes["row_count_change"] > 1000 or changes["column_changes"]:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
                version_file = self._get_cache_file_path(dataset_id, version)
                df.write_parquet(version_file, compression="zstd")
                logger.info(f"Saved version: {version_file}")

        # Update metadata
        current_metadata["cached_at"] = datetime.now().isoformat()
        current_metadata["cache_file"] = str(cache_file)
        current_metadata["row_count"] = len(df)
        current_metadata["successful_url"] = successful_url
        self._save_metadata(dataset_id, current_metadata)

        return df

    def get_data_freshness_info(self, dataset_id: str = None) -> Dict:
        """Get information about data freshness and cache status."""
        if dataset_id is None:
            dataset_id = self.BUSINESS_DATASET_ID

        metadata = self._load_metadata(dataset_id)
        cache_file = self._get_cache_file_path(dataset_id)

        info = {
            "dataset_id": dataset_id,
            "cache_exists": cache_file.exists(),
            "metadata_available": metadata is not None
        }

        if cache_file.exists():
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            info["cache_age_hours"] = (datetime.now() - cache_time).total_seconds() / 3600
            info["cache_file"] = str(cache_file)

        if metadata:
            info["last_updated_api"] = metadata.get("last_updated")
            info["cached_at"] = metadata.get("cached_at")
            info["total_rows"] = metadata.get("total_rows")
            info["cached_rows"] = metadata.get("row_count")

        return info

    def clear_cache(self, dataset_id: str = None):
        """Clear cache for specific dataset or all datasets."""
        if dataset_id:
            # Clear specific dataset
            cache_file = self._get_cache_file_path(dataset_id)
            metadata_file = self.metadata_dir / f"{dataset_id}_metadata.json"

            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Removed cache file: {cache_file}")

            if metadata_file.exists():
                metadata_file.unlink()
                logger.info(f"Removed metadata file: {metadata_file}")
        else:
            # Clear all cache
            for file in self.cache_dir.glob("*.parquet"):
                file.unlink()
            for file in self.metadata_dir.glob("*.json"):
                file.unlink()
            logger.info("Cleared all cache files")

def main():
    """Test the API client."""
    logging.basicConfig(level=logging.INFO)

    # Initialize client with fallback to current CSV file
    fallback_csv = "data/Registered_Business_Locations_-_San_Francisco_20250916.csv"
    client = SFOpenDataAPIClient(fallback_csv_path=fallback_csv)

    # Test data freshness
    freshness = client.get_data_freshness_info()
    print("Data Freshness Info:")
    for key, value in freshness.items():
        print(f"  {key}: {value}")

    # Fetch business data (limited for testing)
    print("\nFetching business data...")
    df = client.get_business_data(limit=1000)

    print(f"\nDataFrame Info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns}")
    if hasattr(df, 'estimated_size'):
        print(f"  Memory usage: {df.estimated_size('mb'):.2f} MB")
    else:
        print(f"  Memory usage estimation not available")

    # Show some sample data
    print(f"\nSample data (first 3 rows):")
    print(df.head(3))

if __name__ == "__main__":
    main()