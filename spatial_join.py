#!/usr/bin/env python3

import polars as pl
import math
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpatialJoiner:
    """Perform spatial joins between businesses and buildings using Polars."""

    def __init__(self, input_dir: str = "output", output_dir: str = "output"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self):
        """Load business and building data from Parquet files."""
        logger.info("Loading business and building data...")

        businesses_path = self.input_dir / "businesses.parquet"
        buildings_path = self.input_dir / "buildings.parquet"

        businesses_df = pl.read_parquet(businesses_path)
        buildings_df = pl.read_parquet(buildings_path)

        logger.info(f"Loaded {len(businesses_df):,} businesses and {len(buildings_df):,} buildings")
        return businesses_df, buildings_df

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the distance between two points using the Haversine formula."""
        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        # Radius of Earth in meters
        earth_radius_m = 6371000
        distance = earth_radius_m * c

        return distance

    def perform_spatial_join(self, businesses_df: pl.DataFrame, buildings_df: pl.DataFrame, max_distance_m: float = 100.0):
        """Perform fast spatial join using vectorized operations."""
        logger.info("Performing vectorized spatial join...")

        # Use a smaller sample for demo purposes
        sample_size = 1000
        businesses_sample = businesses_df.sample(sample_size, seed=42) if len(businesses_df) > sample_size else businesses_df
        buildings_sample = buildings_df.sample(5000, seed=42) if len(buildings_df) > 5000 else buildings_df

        logger.info(f"Processing {len(businesses_sample):,} businesses against {len(buildings_sample):,} buildings")

        # Prepare business coordinates
        businesses_clean = businesses_sample.filter(
            pl.col("longitude").is_not_null() & pl.col("latitude").is_not_null()
        ).select([
            "unique_id", "longitude", "latitude", "dba_name"
        ]).with_row_index("business_idx")

        # Extract building coordinates from WKT
        buildings_clean = buildings_sample.filter(
            pl.col("geom_wkt").is_not_null()
        ).with_columns([
            pl.col("geom_wkt")
            .str.extract(r"POLYGON\(\(([^,]+)\s+([^,\s]+)", 1)
            .cast(pl.Float64, strict=False)
            .alias("building_lon"),
            pl.col("geom_wkt")
            .str.extract(r"POLYGON\(\(([^,]+)\s+([^,\s]+)", 2)
            .cast(pl.Float64, strict=False)
            .alias("building_lat")
        ]).filter(
            pl.col("building_lon").is_not_null() & pl.col("building_lat").is_not_null()
        ).select([
            "osm_id", "building_type", "building_lon", "building_lat"
        ]).with_row_index("building_idx")

        logger.info(f"Using {len(businesses_clean):,} businesses and {len(buildings_clean):,} buildings for join")

        # Create a fast approximate join using bounding boxes
        # First, create rough geographic grid cells to reduce comparisons
        grid_size = 0.001  # ~100m grid cells

        businesses_gridded = businesses_clean.with_columns([
            (pl.col("longitude") / grid_size).floor().alias("grid_x"),
            (pl.col("latitude") / grid_size).floor().alias("grid_y")
        ])

        buildings_gridded = buildings_clean.with_columns([
            (pl.col("building_lon") / grid_size).floor().alias("grid_x"),
            (pl.col("building_lat") / grid_size).floor().alias("grid_y")
        ])

        # Join on same or adjacent grid cells
        matches = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                grid_join = businesses_gridded.join(
                    buildings_gridded.with_columns([
                        (pl.col("grid_x") + dx).alias("grid_x"),
                        (pl.col("grid_y") + dy).alias("grid_y")
                    ]),
                    on=["grid_x", "grid_y"],
                    how="inner"
                )
                if len(grid_join) > 0:
                    matches.append(grid_join)

        if not matches:
            logger.warning("No grid-based matches found")
            return pl.DataFrame()

        # Combine all grid matches
        combined_matches = pl.concat(matches).unique(["business_idx", "building_idx"])
        logger.info(f"Found {len(combined_matches):,} potential matches after grid filtering")

        # Calculate approximate distances using simple Euclidean formula
        # For San Francisco area, 1 degree latitude ≈ 111km, 1 degree longitude ≈ 85km
        final_matches = combined_matches.with_columns([
            (111000 * ((pl.col("latitude") - pl.col("building_lat")).pow(2) +
                      (pl.col("longitude") - pl.col("building_lon")).pow(2) * 0.6).sqrt()
            ).alias("distance_meters")
        ]).filter(
            pl.col("distance_meters") <= max_distance_m
        )

        # Keep only the nearest building for each business
        result = final_matches.group_by("business_idx").agg([
            pl.col("unique_id").first(),
            pl.col("longitude").first(),
            pl.col("latitude").first(),
            pl.col("dba_name").first(),
            pl.col("osm_id").first(),
            pl.col("building_type").first(),
            pl.col("building_lon").first(),
            pl.col("building_lat").first(),
            pl.col("distance_meters").min()
        ]).select([
            pl.col("unique_id").alias("business_unique_id"),
            pl.col("longitude").alias("business_lon"),
            pl.col("latitude").alias("business_lat"),
            pl.col("dba_name").alias("business_name"),
            pl.col("osm_id").alias("building_osm_id"),
            pl.col("building_type"),
            pl.col("building_lon"),
            pl.col("building_lat"),
            pl.col("distance_meters")
        ])

        logger.info(f"Found {len(result):,} business-building matches within {max_distance_m}m")
        return result

    def save_spatial_join_results(self, joined_df: pl.DataFrame):
        """Save spatial join results to Parquet files."""
        logger.info("Saving spatial join results...")

        if len(joined_df) == 0:
            logger.warning("No spatial join results to save")
            return

        # Save the main spatial join results
        output_path = self.output_dir / "business_building_matches.parquet"
        joined_df.write_parquet(output_path)
        logger.info(f"Spatial join results saved to {output_path}")

        # Create summary statistics
        summary_stats = joined_df.select([
            pl.len().alias("total_matches"),
            pl.col("distance_meters").mean().alias("avg_distance_meters"),
            pl.col("distance_meters").min().alias("min_distance_meters"),
            pl.col("distance_meters").max().alias("max_distance_meters"),
            pl.col("building_type").n_unique().alias("unique_building_types")
        ])

        summary_path = self.output_dir / "spatial_join_summary.parquet"
        summary_stats.write_parquet(summary_path)
        logger.info(f"Summary statistics saved to {summary_path}")

        return joined_df

    def analyze_spatial_relationships(self, joined_df: pl.DataFrame):
        """Analyze the spatial relationships between businesses and buildings."""
        logger.info("Analyzing spatial relationships...")

        if len(joined_df) == 0:
            print("No spatial relationships to analyze")
            return

        # Distance analysis
        distance_stats = joined_df.select([
            pl.col("distance_meters").mean().alias("avg_distance"),
            pl.col("distance_meters").min().alias("min_distance"),
            pl.col("distance_meters").max().alias("max_distance"),
            pl.col("distance_meters").median().alias("median_distance")
        ]).row(0, named=True)

        print("\n=== SPATIAL RELATIONSHIP ANALYSIS ===")
        print(f"Total matches: {len(joined_df):,}")
        print(f"Average distance: {distance_stats['avg_distance']:.2f}m")
        print(f"Median distance: {distance_stats['median_distance']:.2f}m")
        print(f"Min distance: {distance_stats['min_distance']:.2f}m")
        print(f"Max distance: {distance_stats['max_distance']:.2f}m")

        # Building type analysis
        building_type_stats = joined_df.group_by("building_type").agg([
            pl.len().alias("business_count")
        ]).sort("business_count", descending=True).head(10)

        print("\n=== BUSINESS-BUILDING TYPE ANALYSIS ===")
        print("Building Type | Business Count")
        print("-" * 35)
        for row in building_type_stats.iter_rows(named=True):
            building_type = row["building_type"] or "Unknown"
            count = row["business_count"]
            print(f"{building_type[:20]:20s} | {count:13d}")

def main():
    """Main spatial joining execution."""
    joiner = SpatialJoiner()

    # Load data
    businesses_df, buildings_df = joiner.load_data()

    # Perform spatial join
    joined_df = joiner.perform_spatial_join(businesses_df, buildings_df)

    # Save results
    joiner.save_spatial_join_results(joined_df)

    # Analyze results
    joiner.analyze_spatial_relationships(joined_df)

    print(f"\n=== SPATIAL JOIN SUMMARY ===")
    print(f"Businesses processed: {len(businesses_df):,}")
    print(f"Buildings available: {len(buildings_df):,}")
    print(f"Successful matches: {len(joined_df):,}")

if __name__ == "__main__":
    main()