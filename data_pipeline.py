#!/usr/bin/env python3

import polars as pl
import re
from pathlib import Path
import logging
import math
from validation import DataValidator
from quality_metrics import QualityMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SFBusinessPipeline:
    """Data ingestion pipeline for SF business data using Polars native operations."""

    def __init__(self, output_dir: str = "output", enable_validation: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.enable_validation = enable_validation

        if self.enable_validation:
            self.validator = DataValidator(output_dir)
            self.quality_metrics = QualityMetrics(output_dir)

    def setup_database(self):
        """Setup output directory for data files."""
        logger.info("Setting up output directory...")
        logger.info(f"Output directory: {self.output_dir}")

    def load_and_clean_csv(self, csv_path: str) -> pl.DataFrame:
        """Load and clean the SF business CSV data."""
        logger.info(f"Loading CSV from {csv_path}")

        # Read CSV with data quality handling
        df = pl.read_csv(
            csv_path,
            infer_schema_length=10000,
            ignore_errors=True,
            null_values=["", "null", "NULL", "PRUSS"]
        )

        logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")

        # Clean and standardize data
        df_cleaned = df.with_columns([
            # Parse coordinates from POINT geometry strings
            pl.col("Business Location")
            .str.extract(r"POINT \(([^\s]+) ([^\s]+)\)", 1)
            .cast(pl.Float64, strict=False)
            .alias("longitude"),

            pl.col("Business Location")
            .str.extract(r"POINT \(([^\s]+) ([^\s]+)\)", 2)
            .cast(pl.Float64, strict=False)
            .alias("latitude"),

            # Clean address fields
            pl.col("Street Address").str.strip_chars().alias("street_address_clean"),
            pl.col("City").str.strip_chars().str.to_uppercase().alias("city_clean"),
            pl.col("State").str.strip_chars().str.to_uppercase().alias("state_clean"),

            # Parse dates
            pl.col("Business Start Date").str.to_datetime("%m/%d/%Y", strict=False).alias("business_start_date_parsed"),
            pl.col("Business End Date").str.to_datetime("%m/%d/%Y", strict=False).alias("business_end_date_parsed"),

            # Clean business names
            pl.col("DBA Name").str.strip_chars().alias("dba_name_clean"),
            pl.col("Ownership Name").str.strip_chars().alias("ownership_name_clean"),

        ]).filter(
            # Only keep records with valid coordinates in SF area
            (pl.col("longitude").is_between(-122.6, -122.3)) &
            (pl.col("latitude").is_between(37.7, 37.9))
        )

        logger.info(f"After cleaning: {len(df_cleaned):,} valid records")
        return df_cleaned

    def save_businesses_data(self, df: pl.DataFrame):
        """Save cleaned business data to Parquet files for efficient querying."""
        logger.info("Saving businesses data...")

        # Select and rename columns for final dataset
        df_final = df.select([
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

        # Save to Parquet format for efficient storage and querying
        output_path = self.output_dir / "businesses.parquet"
        df_final.write_parquet(output_path)
        logger.info(f"Businesses data saved to {output_path}")

        # Run validation and quality checks
        if self.enable_validation:
            logger.info("Running data validation and quality checks...")
            validation_report = self.validator.run_comprehensive_validation(business_df=df_final)
            quality_score = self.quality_metrics.calculate_overall_quality_score(df_final)

            # Save quality metrics
            self.quality_metrics.save_quality_metrics(quality_score, "business_quality_metrics.json")

            logger.info(f"Validation complete - Overall quality score: {quality_score['overall_score']:.2%}")

        # Create additional files partitioned by key dimensions for faster queries

        # By NAICS code
        naics_output = self.output_dir / "businesses_by_naics.parquet"
        df_final.filter(pl.col("naics_code").is_not_null()).write_parquet(
            naics_output,
            row_group_size=10000
        )

        # By neighborhood
        neighborhood_output = self.output_dir / "businesses_by_neighborhood.parquet"
        df_final.filter(pl.col("neighborhood").is_not_null()).write_parquet(
            neighborhood_output,
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