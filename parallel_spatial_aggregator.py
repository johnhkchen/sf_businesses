#!/usr/bin/env python3
"""Ultra-fast parallel spatial aggregator optimized to meet <30 second requirement."""

import polars as pl
import numpy as np
from scipy.spatial import cKDTree
import json
from pathlib import Path
import logging
import time
from typing import Dict, Tuple, Optional, List
from shapely import wkt
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_centroid_batch(wkt_strings):
    """Extract centroids from a batch of WKT strings (for parallel processing)."""
    results = []
    for wkt_str in wkt_strings:
        try:
            if wkt_str and wkt_str != "null":
                geom = wkt.loads(wkt_str)
                c = geom.centroid
                results.append((c.x, c.y))
            else:
                results.append((None, None))
        except:
            results.append((None, None))
    return results

def process_building_batch(args):
    """Process a batch of buildings (for parallel processing)."""
    building_coords, business_tree_data, business_data, radius = args

    # Rebuild KDTree in worker process
    business_coords, business_indices = business_tree_data
    kdtree = cKDTree(business_coords)

    results = []
    for building_idx, (lon, lat, building_info) in enumerate(building_coords):
        # Query KDTree
        business_idxs = kdtree.query_ball_point([lon, lat], radius)

        for bus_idx in business_idxs:
            results.append({
                'building_id': building_info['osm_id'],
                'building_type': building_info.get('building_type'),
                'building_category': building_info.get('building_category'),
                'building_center_lon': lon,
                'building_center_lat': lat,
                **business_data[bus_idx]
            })

    return results

class ParallelSpatialAggregator:
    """Parallel spatial aggregator optimized for speed."""

    def __init__(self, input_dir: str = "output", output_dir: str = "output", n_workers: int = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_workers = n_workers or mp.cpu_count()
        logger.info(f"Using {self.n_workers} parallel workers")

    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load data with optimizations."""
        logger.info("Loading data...")

        # Load only necessary columns
        businesses_df = pl.read_parquet(
            self.input_dir / "businesses.parquet",
            columns=["unique_id", "naics_code", "naics_description",
                    "longitude", "latitude", "business_start_date", "business_end_date"]
        ).rename({"unique_id": "business_id"})

        buildings_df = pl.read_parquet(
            self.input_dir / "buildings.parquet",
            columns=["osm_id", "building_type", "building_category", "geom_wkt"]
        )

        logger.info(f"Loaded {len(businesses_df):,} businesses and {len(buildings_df):,} buildings")
        return businesses_df, buildings_df

    def extract_centroids_parallel(self, buildings_df: pl.DataFrame) -> pl.DataFrame:
        """Extract centroids using parallel processing."""
        logger.info("Extracting building centroids in parallel...")

        # Split into chunks for parallel processing
        wkt_list = buildings_df["geom_wkt"].to_list()
        chunk_size = len(wkt_list) // self.n_workers + 1
        chunks = [wkt_list[i:i+chunk_size] for i in range(0, len(wkt_list), chunk_size)]

        # Process in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(extract_centroid_batch, chunk) for chunk in chunks]
            results = []
            for future in as_completed(futures):
                results.extend(future.result())

        # Add centroids to dataframe
        lons, lats = zip(*results) if results else ([], [])
        buildings_with_centroids = buildings_df.with_columns([
            pl.Series("center_lon", lons),
            pl.Series("center_lat", lats)
        ]).filter(
            pl.col("center_lon").is_not_null() &
            pl.col("center_lat").is_not_null()
        )

        logger.info(f"Extracted centroids for {len(buildings_with_centroids):,} buildings")
        return buildings_with_centroids

    def parallel_kdtree_join(self, businesses_df: pl.DataFrame, buildings_df: pl.DataFrame,
                            radius: float = 0.0005) -> pl.DataFrame:
        """Ultra-fast parallel spatial join using KDTree."""
        logger.info("Building KDTree...")

        # Filter valid business coordinates
        valid_businesses = businesses_df.filter(
            pl.col("longitude").is_not_null() &
            pl.col("latitude").is_not_null()
        )

        # Prepare business data
        business_coords = np.column_stack([
            valid_businesses["longitude"].to_numpy(),
            valid_businesses["latitude"].to_numpy()
        ])

        # Prepare business data for workers
        business_data = valid_businesses.to_dicts()
        business_tree_data = (business_coords, list(range(len(business_coords))))

        logger.info(f"KDTree data prepared with {len(business_coords):,} points")

        # Prepare building batches for parallel processing
        building_batch_size = max(100, len(buildings_df) // (self.n_workers * 10))
        building_batches = []

        for i in range(0, len(buildings_df), building_batch_size):
            batch = buildings_df.slice(i, min(building_batch_size, len(buildings_df) - i))
            building_coords = []

            for row in batch.iter_rows(named=True):
                building_coords.append((
                    row['center_lon'],
                    row['center_lat'],
                    {
                        'osm_id': row['osm_id'],
                        'building_type': row.get('building_type'),
                        'building_category': row.get('building_category')
                    }
                ))

            building_batches.append((
                building_coords,
                business_tree_data,
                business_data,
                radius
            ))

        logger.info(f"Processing {len(building_batches)} batches in parallel...")

        # Process batches in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {executor.submit(process_building_batch, batch): i
                      for i, batch in enumerate(building_batches)}

            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)

                    if batch_idx % 10 == 0:
                        logger.info(f"Completed batch {batch_idx}/{len(building_batches)}")
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")

        if all_results:
            result_df = pl.DataFrame(all_results)
            logger.info(f"Spatial join complete: {len(result_df):,} associations")
            return result_df
        else:
            logger.warning("No spatial associations found")
            return pl.DataFrame()

    def fast_aggregate(self, joined_df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Fast aggregation using optimized operations."""
        logger.info("Fast aggregation...")

        # Building aggregation
        building_agg = joined_df.group_by("building_id").agg([
            pl.len().alias("business_count"),
            pl.col("naics_code").n_unique().alias("unique_business_types"),
            pl.col("building_type").first().alias("building_type"),
            pl.col("building_category").first().alias("building_category"),
            pl.col("building_center_lon").first().alias("center_lon"),
            pl.col("building_center_lat").first().alias("center_lat")
        ]).with_columns([
            pl.when(pl.col("business_count") >= 10).then(pl.lit("very_high"))
            .when(pl.col("business_count") >= 5).then(pl.lit("high"))
            .when(pl.col("business_count") >= 2).then(pl.lit("medium"))
            .otherwise(pl.lit("low")).alias("density_category")
        ])

        # Grid aggregation (simplified for speed)
        grid_size = 0.002
        grid_agg = joined_df.with_columns([
            (pl.col("building_center_lon") / grid_size).floor().alias("grid_x"),
            (pl.col("building_center_lat") / grid_size).floor().alias("grid_y")
        ]).group_by(["grid_x", "grid_y"]).agg([
            pl.len().alias("total_businesses"),
            pl.col("building_center_lon").mean().alias("center_lon"),
            pl.col("building_center_lat").mean().alias("center_lat")
        ])

        logger.info(f"Aggregation complete: {len(building_agg):,} buildings, {len(grid_agg):,} grid cells")
        return building_agg, grid_agg

    def run(self) -> Dict:
        """Run the optimized pipeline."""
        logger.info("=" * 60)
        logger.info("Starting PARALLEL spatial aggregation pipeline")
        logger.info(f"Using {self.n_workers} CPU cores")
        logger.info("=" * 60)

        start_time = time.time()

        # Phase 1: Load data
        load_start = time.time()
        businesses_df, buildings_df = self.load_data()
        load_time = time.time() - load_start
        logger.info(f"Data loading: {load_time:.2f}s")

        # Phase 2: Extract centroids in parallel
        centroid_start = time.time()
        buildings_with_centroids = self.extract_centroids_parallel(buildings_df)
        centroid_time = time.time() - centroid_start
        logger.info(f"Centroid extraction: {centroid_time:.2f}s")

        # Phase 3: Parallel spatial join
        join_start = time.time()
        joined_df = self.parallel_kdtree_join(businesses_df, buildings_with_centroids)
        join_time = time.time() - join_start
        logger.info(f"Spatial join: {join_time:.2f}s")

        if len(joined_df) == 0:
            logger.error("No spatial associations found")
            return {"error": "No spatial associations"}

        # Phase 4: Fast aggregation
        agg_start = time.time()
        building_agg, grid_agg = self.fast_aggregate(joined_df)
        agg_time = time.time() - agg_start
        logger.info(f"Aggregation: {agg_time:.2f}s")

        # Phase 5: Save results
        save_start = time.time()
        building_agg.write_parquet(self.output_dir / "buildings_businesses_parallel.parquet")
        grid_agg.write_parquet(self.output_dir / "business_grid_parallel.parquet")
        save_time = time.time() - save_start
        logger.info(f"Save results: {save_time:.2f}s")

        # Statistics
        total_time = time.time() - start_time
        stats = {
            "total_time_seconds": round(total_time, 2),
            "phase_times": {
                "data_loading": round(load_time, 2),
                "centroid_extraction": round(centroid_time, 2),
                "spatial_join": round(join_time, 2),
                "aggregation": round(agg_time, 2),
                "save_results": round(save_time, 2)
            },
            "buildings_processed": len(buildings_with_centroids),
            "businesses_processed": len(businesses_df),
            "total_associations": len(joined_df),
            "buildings_with_businesses": len(building_agg),
            "grid_cells": len(grid_agg),
            "avg_businesses_per_building": float(building_agg["business_count"].mean()),
            "max_businesses_per_building": int(building_agg["business_count"].max()),
            "workers_used": self.n_workers,
            "meets_30s_requirement": total_time < 30
        }

        # Save statistics
        with open(self.output_dir / "parallel_aggregation_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("=" * 60)
        if total_time < 30:
            logger.info(f"✅ SUCCESS: Pipeline complete in {total_time:.2f} seconds (< 30s requirement)")
        else:
            logger.info(f"⚠️ Pipeline complete in {total_time:.2f} seconds (exceeds 30s requirement)")
        logger.info("=" * 60)

        return stats

def main():
    """Run the parallel spatial aggregator."""
    import argparse

    parser = argparse.ArgumentParser(description="Parallel spatial aggregation")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    args = parser.parse_args()

    aggregator = ParallelSpatialAggregator(n_workers=args.workers)
    stats = aggregator.run()

    if stats and "error" not in stats:
        print("\n" + "=" * 60)
        print("PARALLEL AGGREGATION RESULTS")
        print("=" * 60)
        print(f"Total time: {stats['total_time_seconds']} seconds")
        print(f"Meets <30s requirement: {stats['meets_30s_requirement']}")
        print("\nPhase breakdown:")
        for phase, time in stats['phase_times'].items():
            print(f"  {phase}: {time}s")
        print(f"\nTotal associations: {stats['total_associations']:,}")
        print(f"Buildings with businesses: {stats['buildings_with_businesses']:,}")
        print(f"Workers used: {stats['workers_used']}")

if __name__ == "__main__":
    main()