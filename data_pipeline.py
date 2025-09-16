#!/usr/bin/env python3

import polars as pl
import re
from pathlib import Path
import logging
import math
import hashlib
import json
from datetime import datetime, timedelta
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Optional, Tuple, Dict, List
from validation import DataValidator
from quality_metrics import QualityMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SFBusinessPipeline:
    """Data ingestion pipeline for SF business data using Polars native operations."""

    def __init__(self, output_dir: str = "output", enable_validation: bool = True,
                 enable_caching: bool = True, enable_parallel: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

        self.enable_validation = enable_validation
        self.enable_caching = enable_caching
        self.enable_parallel = enable_parallel
        self.cpu_count = mp.cpu_count()

        if self.enable_validation:
            self.validator = DataValidator(output_dir)
            self.quality_metrics = QualityMetrics(output_dir)

    def setup_database(self):
        """Setup output directory for data files."""
        logger.info("Setting up output directory...")
        logger.info(f"Output directory: {self.output_dir}")

    def _get_file_hash(self, file_path: str) -> str:
        """Generate SHA256 hash of a file for cache validation."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _get_cache_metadata(self, csv_path: str) -> Optional[Dict]:
        """Get cache metadata for a CSV file."""
        if not self.enable_caching:
            return None

        cache_meta_path = self.cache_dir / f"{Path(csv_path).stem}_metadata.json"
        if not cache_meta_path.exists():
            return None

        try:
            with open(cache_meta_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def _save_cache_metadata(self, csv_path: str, file_hash: str, record_count: int):
        """Save cache metadata for a CSV file."""
        if not self.enable_caching:
            return

        metadata = {
            'file_hash': file_hash,
            'timestamp': datetime.now().isoformat(),
            'record_count': record_count,
            'csv_path': str(csv_path)
        }

        cache_meta_path = self.cache_dir / f"{Path(csv_path).stem}_metadata.json"
        with open(cache_meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _is_cache_valid(self, csv_path: str) -> Tuple[bool, Optional[str]]:
        """Check if cached data is valid for the given CSV file."""
        if not self.enable_caching:
            return False, None

        cache_meta = self._get_cache_metadata(csv_path)
        if not cache_meta:
            return False, None

        # Check if CSV file has changed
        current_hash = self._get_file_hash(csv_path)
        if current_hash != cache_meta['file_hash']:
            logger.info("CSV file has changed, cache invalid")
            return False, None

        # Check if cached parquet exists
        cache_file_path = self.cache_dir / f"{Path(csv_path).stem}_cleaned.parquet"
        if not cache_file_path.exists():
            logger.info("Cached parquet file missing")
            return False, None

        # Check cache age (invalidate after 24 hours)
        cache_time = datetime.fromisoformat(cache_meta['timestamp'])
        if datetime.now() - cache_time > timedelta(hours=24):
            logger.info("Cache expired (>24 hours old)")
            return False, None

        logger.info(f"Cache valid for {csv_path}")
        return True, str(cache_file_path)

    def _save_to_cache(self, df: pl.DataFrame, csv_path: str):
        """Save cleaned dataframe to cache."""
        if not self.enable_caching:
            return

        cache_file_path = self.cache_dir / f"{Path(csv_path).stem}_cleaned.parquet"
        df.write_parquet(cache_file_path)

        file_hash = self._get_file_hash(csv_path)
        self._save_cache_metadata(csv_path, file_hash, len(df))

        logger.info(f"Saved {len(df):,} records to cache: {cache_file_path}")

    def _load_from_cache(self, cache_path: str) -> pl.DataFrame:
        """Load cleaned dataframe from cache."""
        logger.info(f"Loading from cache: {cache_path}")
        return pl.read_parquet(cache_path)

    def load_and_clean_csv(self, csv_path: str) -> pl.DataFrame:
        """Load and clean the SF business CSV data with caching support."""
        logger.info(f"Loading CSV from {csv_path}")

        # Check cache first
        cache_valid, cache_path = self._is_cache_valid(csv_path)
        if cache_valid and cache_path:
            return self._load_from_cache(cache_path)

        # Read CSV with optimized settings for large files
        df = pl.read_csv(
            csv_path,
            infer_schema_length=10000,
            ignore_errors=True,
            null_values=["", "null", "NULL", "PRUSS"],
            # Use lazy loading for better memory efficiency
            rechunk=True,
            # Use multiple threads for CSV parsing
            n_threads=self.cpu_count if self.enable_parallel else 1
        )

        logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")

        # Optimized data cleaning using streaming operations
        df_cleaned = (
            df.lazy()
            .with_columns([
                # Parse coordinates from POINT geometry strings in parallel
                pl.col("Business Location")
                .str.extract(r"POINT \(([^\s]+) ([^\s]+)\)", 1)
                .cast(pl.Float64, strict=False)
                .alias("longitude"),

                pl.col("Business Location")
                .str.extract(r"POINT \(([^\s]+) ([^\s]+)\)", 2)
                .cast(pl.Float64, strict=False)
                .alias("latitude"),

                # Clean address fields with vectorized operations
                pl.col("Street Address").str.strip_chars().alias("street_address_clean"),
                pl.col("City").str.strip_chars().str.to_uppercase().alias("city_clean"),
                pl.col("State").str.strip_chars().str.to_uppercase().alias("state_clean"),

                # Parse dates efficiently
                pl.col("Business Start Date").str.to_datetime("%m/%d/%Y", strict=False).alias("business_start_date_parsed"),
                pl.col("Business End Date").str.to_datetime("%m/%d/%Y", strict=False).alias("business_end_date_parsed"),

                # Clean business names
                pl.col("DBA Name").str.strip_chars().alias("dba_name_clean"),
                pl.col("Ownership Name").str.strip_chars().alias("ownership_name_clean"),
            ])
            .filter(
                # Only keep records with valid coordinates in SF area
                (pl.col("longitude").is_between(-122.6, -122.3)) &
                (pl.col("latitude").is_between(37.7, 37.9))
            )
            .collect(streaming=True)  # Use streaming collection for memory efficiency
        )

        logger.info(f"After cleaning: {len(df_cleaned):,} valid records")

        # Save to cache for future runs
        self._save_to_cache(df_cleaned, csv_path)

        return df_cleaned

    def save_businesses_data(self, df: pl.DataFrame):
        """Save cleaned business data to Parquet files for efficient querying."""
        logger.info("Saving businesses data...")

        # Select and rename columns for final dataset using lazy evaluation
        df_final = (
            df.lazy()
            .select([
                pl.col("UniqueID").alias("unique_id"),
                pl.col("Business Account Number").alias("business_account_number"),
                pl.col("Location Id").alias("location_id"),
                pl.col("ownership_name_clean").alias("ownership_name"),
                pl.col("dba_name_clean").alias("dba_name"),
                pl.col("street_address_clean").alias("street_address"),
                pl.col("city_clean").alias("city"),
                pl.col("state_clean").alias("state"),
                pl.col("Source Zipcode").cast(pl.String).alias("zipcode"),
                pl.col("business_start_date_parsed").alias("business_start_date"),
                pl.col("business_end_date_parsed").alias("business_end_date"),
                pl.col("Location Start Date").str.to_datetime("%m/%d/%Y", strict=False).alias("location_start_date"),
                pl.col("Location End Date").str.to_datetime("%m/%d/%Y", strict=False).alias("location_end_date"),
                pl.col("NAICS Code").alias("naics_code"),
                pl.col("NAICS Code Description").alias("naics_description"),
                pl.col("longitude"),
                pl.col("latitude"),
                pl.col("Supervisor District").alias("supervisor_district"),
                pl.col("Neighborhoods - Analysis Boundaries").alias("neighborhood")
            ])
            .collect(streaming=True)
        )

        # Save main file with compression and optimal chunk size
        output_path = self.output_dir / "businesses.parquet"
        df_final.write_parquet(
            output_path,
            compression="zstd",  # Better compression than default
            row_group_size=50000  # Optimized for query performance
        )
        logger.info(f"Businesses data saved to {output_path}")

        # Run validation and quality checks asynchronously if enabled
        validation_future = None
        if self.enable_validation and self.enable_parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                def run_validation():
                    logger.info("Running data validation and quality checks...")
                    validation_report = self.validator.run_comprehensive_validation(business_df=df_final)
                    quality_score = self.quality_metrics.calculate_overall_quality_score(df_final)
                    self.quality_metrics.save_quality_metrics(quality_score, "business_quality_metrics.json")
                    logger.info(f"Validation complete - Overall quality score: {quality_score['overall_score']:.2%}")
                    return quality_score

                validation_future = executor.submit(run_validation)

                # Save partitioned files in parallel while validation runs
                def save_naics_partition():
                    naics_output = self.output_dir / "businesses_by_naics.parquet"
                    df_final.filter(pl.col("naics_code").is_not_null()).write_parquet(
                        naics_output,
                        compression="zstd",
                        row_group_size=10000
                    )
                    logger.info(f"NAICS partitioned data saved to {naics_output}")

                def save_neighborhood_partition():
                    neighborhood_output = self.output_dir / "businesses_by_neighborhood.parquet"
                    df_final.filter(pl.col("neighborhood").is_not_null()).write_parquet(
                        neighborhood_output,
                        compression="zstd",
                        row_group_size=10000
                    )
                    logger.info(f"Neighborhood partitioned data saved to {neighborhood_output}")

                # Submit partitioning tasks
                naics_future = executor.submit(save_naics_partition)
                neighborhood_future = executor.submit(save_neighborhood_partition)

                # Wait for all tasks to complete
                naics_future.result()
                neighborhood_future.result()
                if validation_future:
                    validation_future.result()

        else:
            # Sequential execution if parallel processing is disabled
            if self.enable_validation:
                logger.info("Running data validation and quality checks...")
                validation_report = self.validator.run_comprehensive_validation(business_df=df_final)
                quality_score = self.quality_metrics.calculate_overall_quality_score(df_final)
                self.quality_metrics.save_quality_metrics(quality_score, "business_quality_metrics.json")
                logger.info(f"Validation complete - Overall quality score: {quality_score['overall_score']:.2%}")

            # Save partitioned files sequentially
            naics_output = self.output_dir / "businesses_by_naics.parquet"
            df_final.filter(pl.col("naics_code").is_not_null()).write_parquet(
                naics_output,
                compression="zstd",
                row_group_size=10000
            )

            neighborhood_output = self.output_dir / "businesses_by_neighborhood.parquet"
            df_final.filter(pl.col("neighborhood").is_not_null()).write_parquet(
                neighborhood_output,
                compression="zstd",
                row_group_size=10000
            )

        return df_final

    def get_business_stats(self, df: pl.DataFrame) -> dict:
        """Get statistics about loaded business data."""
        stats = df.select([
            pl.len().alias("total_businesses"),
            pl.col("naics_code").n_unique().alias("unique_naics_codes"),
            pl.col("neighborhood").n_unique().alias("unique_neighborhoods"),
            pl.col("supervisor_district").n_unique().alias("unique_districts"),
            pl.col("business_start_date").min().alias("earliest_start_date"),
            pl.col("business_start_date").max().alias("latest_start_date")
        ]).row(0, named=True)

        return stats

def main():
    """Main pipeline execution."""
    pipeline = SFBusinessPipeline()

    # Setup output directory
    pipeline.setup_database()

    # Load and clean data
    df = pipeline.load_and_clean_csv("data/Registered_Business_Locations_-_San_Francisco_20250916.csv")

    # Save data to Parquet files
    df_final = pipeline.save_businesses_data(df)

    # Print statistics
    stats = pipeline.get_business_stats(df_final)
    print("\n=== PIPELINE RESULTS ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()