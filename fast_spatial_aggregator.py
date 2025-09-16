#!/usr/bin/env python3
"""Fast spatial aggregator using KDTree and optimized algorithms."""

import polars as pl
import numpy as np
from scipy.spatial import cKDTree
import json
from pathlib import Path
import logging
import time
from typing import Dict, Tuple, Optional
from shapely import wkt
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastSpatialAggregator:
    """Fast spatial aggregator using KDTree for efficient nearest neighbor search."""

    def __init__(self, input_dir: str = "output", output_dir: str = "output"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load business and building data."""
        logger.info("Loading data...")
        businesses_df = pl.read_parquet(self.input_dir / "businesses.parquet")
        buildings_df = pl.read_parquet(self.input_dir / "buildings.parquet")
        logger.info(f"Loaded {len(businesses_df):,} businesses and {len(buildings_df):,} buildings")
        return businesses_df, buildings_df

    def extract_centroids_fast(self, buildings_df: pl.DataFrame) -> pl.DataFrame:
        """Extract centroids using vectorized operations."""
        logger.info("Extracting building centroids...")

        def extract_centroid(wkt_str):
            try:
                if not wkt_str or wkt_str == "null":
                    return None, None
                geom = wkt.loads(wkt_str)
                c = geom.centroid
                return c.x, c.y
            except:
                return None, None

        # Process in batches for memory efficiency
        centroids = []
        batch_size = 10000

        for i in range(0, len(buildings_df), batch_size):
            batch = buildings_df.slice(i, min(batch_size, len(buildings_df) - i))

            batch_centroids = [extract_centroid(wkt_str)
                             for wkt_str in batch["geom_wkt"].to_list()]

            lons, lats = zip(*batch_centroids) if batch_centroids else ([], [])

            batch = batch.with_columns([
                pl.Series("center_lon", lons),
                pl.Series("center_lat", lats)
            ])

            centroids.append(batch)

        result = pl.concat(centroids).filter(
            pl.col("center_lon").is_not_null() &
            pl.col("center_lat").is_not_null()
        )

        logger.info(f"Extracted centroids for {len(result):,} buildings")
        return result

    def kdtree_spatial_join(self, businesses_df: pl.DataFrame, buildings_df: pl.DataFrame,
                           radius_degrees: float = 0.0005) -> pl.DataFrame:
        """Fast spatial join using KDTree."""
        logger.info("Building KDTree spatial index...")

        # Filter valid business coordinates
        valid_businesses = businesses_df.filter(
            pl.col("longitude").is_not_null() &
            pl.col("latitude").is_not_null()
        )

        if len(valid_businesses) == 0:
            logger.warning("No valid business coordinates")
            return pl.DataFrame()

        # Convert to numpy arrays
        business_coords = np.column_stack([
            valid_businesses["longitude"].to_numpy(),
            valid_businesses["latitude"].to_numpy()
        ])

        # Build KDTree
        kdtree = cKDTree(business_coords)

        logger.info(f"KDTree built with {len(business_coords):,} points")

        # Query for each building
        logger.info("Performing spatial join...")
        results = []

        building_coords = np.column_stack([
            buildings_df["center_lon"].to_numpy(),
            buildings_df["center_lat"].to_numpy()
        ])

        # Query all buildings at once
        indices_list = kdtree.query_ball_point(building_coords, radius_degrees)

        # Process results
        for building_idx, business_indices in enumerate(indices_list):
            if len(business_indices) > 0:
                building_row = buildings_df.row(building_idx, named=True)
                for bus_idx in business_indices:
                    business_row = valid_businesses.row(bus_idx, named=True)
                    results.append({
                        'building_id': building_row["osm_id"],
                        'building_type': building_row.get("building_type"),
                        'building_category': building_row.get("building_category"),
                        'building_center_lon': building_row["center_lon"],
                        'building_center_lat': building_row["center_lat"],
                        'business_id': business_row.get("business_id"),
                        'naics_code': business_row.get("naics_code"),
                        'naics_description': business_row.get("naics_description"),
                        'business_start_date': business_row.get("business_start_date"),
                        'business_end_date': business_row.get("business_end_date")
                    })

            # Progress logging
            if (building_idx + 1) % 10000 == 0:
                logger.info(f"Processed {building_idx + 1:,}/{len(buildings_df):,} buildings")

        if results:
            result_df = pl.DataFrame(results)
            logger.info(f"Spatial join complete: {len(result_df):,} associations")
            return result_df
        else:
            logger.warning("No spatial associations found")
            return pl.DataFrame()

    def aggregate_by_building(self, joined_df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate business statistics by building."""
        logger.info("Aggregating by building...")

        if len(joined_df) == 0:
            return pl.DataFrame()

        aggregated = joined_df.group_by("building_id").agg([
            pl.len().alias("business_count"),
            pl.col("naics_code").n_unique().alias("unique_business_types"),
            pl.col("naics_description").mode().first().alias("primary_business_type"),
            pl.col("building_type").first().alias("building_type"),
            pl.col("building_category").first().alias("building_category"),
            pl.col("building_center_lon").first().alias("center_lon"),
            pl.col("building_center_lat").first().alias("center_lat")
        ]).with_columns([
            pl.when(pl.col("business_count") >= 10).then(pl.lit("very_high"))
            .when(pl.col("business_count") >= 5).then(pl.lit("high"))
            .when(pl.col("business_count") >= 2).then(pl.lit("medium"))
            .otherwise(pl.lit("low")).alias("density_category"),

            (pl.col("business_count") * 10).clip(0, 100).alias("intensity_score")
        ])

        logger.info(f"Aggregation complete: {len(aggregated):,} buildings with businesses")
        return aggregated

    def create_grid_aggregation(self, joined_df: pl.DataFrame, grid_size: float = 0.002) -> pl.DataFrame:
        """Create grid-based aggregation for visualization."""
        logger.info(f"Creating grid aggregation (grid size: {grid_size})...")

        if len(joined_df) == 0:
            return pl.DataFrame()

        grid_df = joined_df.with_columns([
            (pl.col("building_center_lon") / grid_size).floor().alias("grid_x"),
            (pl.col("building_center_lat") / grid_size).floor().alias("grid_y")
        ])

        grid_agg = grid_df.group_by(["grid_x", "grid_y"]).agg([
            pl.len().alias("total_businesses"),
            pl.col("building_id").n_unique().alias("building_count"),
            pl.col("building_center_lon").mean().alias("center_lon"),
            pl.col("building_center_lat").mean().alias("center_lat")
        ]).with_columns([
            pl.when(pl.col("total_businesses") >= 50).then(pl.lit("very_high"))
            .when(pl.col("total_businesses") >= 20).then(pl.lit("high"))
            .when(pl.col("total_businesses") >= 5).then(pl.lit("medium"))
            .otherwise(pl.lit("low")).alias("density_category")
        ])

        logger.info(f"Grid aggregation complete: {len(grid_agg):,} grid cells")
        return grid_agg

    def run(self, sample_size: Optional[int] = None) -> Dict:
        """Run the complete aggregation pipeline."""
        logger.info("=" * 60)
        logger.info("Starting FAST spatial aggregation pipeline")
        logger.info("=" * 60)
        start_time = time.time()

        # Load data
        businesses_df, buildings_df = self.load_data()

        # Sample for testing if requested
        if sample_size:
            logger.info(f"Sampling {sample_size} buildings for testing...")
            buildings_df = buildings_df.sample(n=min(sample_size, len(buildings_df)))

        # Extract centroids
        buildings_with_centroids = self.extract_centroids_fast(buildings_df)

        # Perform spatial join
        join_start = time.time()
        joined_df = self.kdtree_spatial_join(businesses_df, buildings_with_centroids)
        join_time = time.time() - join_start
        logger.info(f"Spatial join completed in {join_time:.2f} seconds")

        if len(joined_df) == 0:
            logger.error("No spatial associations found")
            return {"error": "No spatial associations"}

        # Aggregate by building
        building_agg = self.aggregate_by_building(joined_df)

        # Create grid aggregation
        grid_agg = self.create_grid_aggregation(joined_df)

        # Save results
        building_agg.write_parquet(self.output_dir / "buildings_businesses_fast.parquet")
        grid_agg.write_parquet(self.output_dir / "business_grid_fast.parquet")

        # Save GeoJSON for visualization
        self.save_geojson(building_agg, "buildings_businesses_fast.geojson")

        # Calculate statistics
        total_time = time.time() - start_time
        stats = {
            "total_time_seconds": round(total_time, 2),
            "spatial_join_time_seconds": round(join_time, 2),
            "buildings_processed": len(buildings_with_centroids),
            "businesses_processed": len(businesses_df),
            "total_associations": len(joined_df),
            "buildings_with_businesses": len(building_agg),
            "grid_cells": len(grid_agg),
            "avg_businesses_per_building": float(building_agg["business_count"].mean()),
            "max_businesses_per_building": int(building_agg["business_count"].max()),
            "density_distribution": building_agg.group_by("density_category").count().to_dicts()
        }

        # Save statistics
        with open(self.output_dir / "fast_aggregation_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("=" * 60)
        logger.info(f"Pipeline complete in {total_time:.2f} seconds")
        logger.info("=" * 60)

        return stats

    def save_geojson(self, df: pl.DataFrame, filename: str):
        """Save DataFrame as GeoJSON."""
        features = []
        for row in df.to_dicts():
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["center_lon"], row["center_lat"]]
                },
                "properties": {
                    k: v for k, v in row.items()
                    if k not in ["center_lon", "center_lat"]
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(self.output_dir / filename, 'w') as f:
            json.dump(geojson, f, separators=(',', ':'))

        logger.info(f"Saved GeoJSON: {filename}")

def benchmark_algorithms():
    """Compare different spatial join algorithms."""
    logger.info("Starting algorithm benchmarks...")
    results = {}

    # Test 1: KDTree with different radius values
    for radius in [0.0003, 0.0005, 0.001]:
        logger.info(f"\nTesting KDTree with radius={radius}")
        aggregator = FastSpatialAggregator()

        # Use a sample for benchmarking
        start = time.time()
        stats = aggregator.run(sample_size=5000)
        elapsed = time.time() - start

        results[f"kdtree_radius_{radius}"] = {
            "time": elapsed,
            "associations": stats.get("total_associations", 0),
            "buildings_with_businesses": stats.get("buildings_with_businesses", 0)
        }

    # Save benchmark results
    with open("output/fast_benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  Time: {data['time']:.2f} seconds")
        print(f"  Associations: {data['associations']:,}")
        print(f"  Buildings with businesses: {data['buildings_with_businesses']:,}")

    return results

def main():
    """Run the fast spatial aggregator."""
    import argparse

    parser = argparse.ArgumentParser(description="Fast spatial aggregation")
    parser.add_argument("--sample", type=int, help="Sample size for testing")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--radius", type=float, default=0.0005, help="Search radius in degrees")
    args = parser.parse_args()

    if args.benchmark:
        benchmark_algorithms()
    else:
        aggregator = FastSpatialAggregator()
        stats = aggregator.run(sample_size=args.sample)

        if stats and "error" not in stats:
            print("\n" + "=" * 60)
            print("FAST AGGREGATION RESULTS")
            print("=" * 60)
            print(f"Total time: {stats['total_time_seconds']} seconds")
            print(f"Spatial join time: {stats['spatial_join_time_seconds']} seconds")
            print(f"Total associations: {stats['total_associations']:,}")
            print(f"Buildings with businesses: {stats['buildings_with_businesses']:,}")
            print(f"Average businesses per building: {stats['avg_businesses_per_building']:.2f}")
            print(f"Maximum businesses per building: {stats['max_businesses_per_building']}")
            print("\nDensity distribution:")
            for item in stats['density_distribution']:
                print(f"  {item['density_category']}: {item['count']:,} buildings")

if __name__ == "__main__":
    main()