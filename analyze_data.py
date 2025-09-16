#!/usr/bin/env python3

import polars as pl

def analyze_csv_structure():
    """Analyze the SF businesses CSV file structure and data quality."""

    # Read the CSV file with Polars, handling data quality issues
    df = pl.read_csv(
        "data/Registered_Business_Locations_-_San_Francisco_20250916.csv",
        infer_schema_length=10000,
        ignore_errors=True,
        null_values=["", "null", "NULL", "PRUSS"]  # Handle bad zipcode values
    )

    print("=== CSV DATA ANALYSIS ===")
    print(f"Total records: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    print("\n=== COLUMN INFORMATION ===")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        null_count = df[col].null_count()
        print(f"{i:2d}. {col:30s} | {str(dtype):12s} | {null_count:8,} nulls")

    print("\n=== SPATIAL DATA SAMPLE ===")
    # Look at the Business Location column which contains POINT geometries
    location_sample = df.select("Business Location").filter(
        pl.col("Business Location").is_not_null()
    ).head(5)

    for location in location_sample["Business Location"]:
        print(f"  {location}")

    print("\n=== ADDRESS DATA QUALITY ===")
    address_stats = df.select([
        pl.col("Street Address").null_count().alias("missing_addresses"),
        pl.col("City").null_count().alias("missing_cities"),
        pl.col("State").null_count().alias("missing_states"),
        pl.col("Source Zipcode").null_count().alias("missing_zipcodes"),
        pl.col("Business Location").null_count().alias("missing_coordinates")
    ])

    for row in address_stats.iter_rows(named=True):
        for field, count in row.items():
            print(f"  {field}: {count:,}")

    print("\n=== BUSINESS CATEGORIES SAMPLE ===")
    naics_sample = df.select("NAICS Code Description").filter(
        pl.col("NAICS Code Description").is_not_null()
    ).unique().head(10)

    for desc in naics_sample["NAICS Code Description"]:
        print(f"  {desc}")

if __name__ == "__main__":
    analyze_csv_structure()