#!/usr/bin/env python3
"""
Robust spatial aggregator optimized for quality and performance.
Uses KDTree spatial indexing with caching and incremental processing.
"""

import polars as pl
import numpy as np
from scipy.spatial import cKDTree
import json
from pathlib import Path
import logging
import time
from typing import Dict, Tuple, Optional, List
from shapely import wkt
import hashlib
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustSpatialAggregator:
    """
    Production-ready spatial aggregator with the following optimizations:
    - KDTree spatial indexing for O(log n) proximity searches
    - Efficient centroid extraction with error handling
    - Memory-optimized processing with column selection
    - Caching for incremental updates
    - Comprehensive error handling and logging
    - Quality metrics and validation
    """

    def __init__(self, input_dir: str = "output", output_dir: str = "output",
                 proximity_radius: float = 0.0005, enable_cache: bool = True):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.cache_dir = self.output_dir / "cache"
        if enable_cache:
            self.cache_dir.mkdir(exist_ok=True)

        # Configuration
        self.proximity_radius = proximity_radius  # ~50 meters at SF latitude
        self.enable_cache = enable_cache

        logger.info(f"Initialized RobustSpatialAggregator")
        logger.info(f"  Input: {self.input_dir}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Proximity radius: {proximity_radius} degrees (~{proximity_radius * 111000:.0f}m)")
        logger.info(f"  Caching: {'enabled' if enable_cache else 'disabled'}")

    def _get_cache_key(self, operation: str, **params) -> str:
        """Generate cache key for operation."""
        params_str = json.dumps(params, sort_keys=True)
        key = f"{operation}_{hashlib.md5(params_str.encode()).hexdigest()}"
        return key

    def _load_cache(self, cache_key: str) -> Optional[pl.DataFrame]:
        """Load cached result if available and valid."""
        if not self.enable_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            # Check age (cache valid for 1 hour)
            age = time.time() - cache_file.stat().st_mtime
            if age < 3600:
                logger.info(f"Loading cached result: {cache_key}")
                return pl.read_parquet(cache_file)
        return None

    def _save_cache(self, df: pl.DataFrame, cache_key: str):
        """Save result to cache."""
        if not self.enable_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.parquet"
        df.write_parquet(cache_file)
        logger.info(f"Saved to cache: {cache_key}")

    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load business and building data with only required columns."""
        logger.info("Loading business and building data...")

        start_time = time.time()

        # Load businesses with essential columns only
        businesses_df = pl.read_parquet(
            self.input_dir / "businesses.parquet",
            columns=["unique_id", "naics_code", "naics_description",
                    "longitude", "latitude", "business_start_date", "business_end_date"]
        ).rename({"unique_id": "business_id"})

        # Load buildings with essential columns only
        buildings_df = pl.read_parquet(
            self.input_dir / "buildings.parquet",
            columns=["osm_id", "building_type", "building_category", "geom_wkt"]
        )

        load_time = time.time() - start_time
        logger.info(f"Data loaded in {load_time:.2f}s: {len(businesses_df):,} businesses, {len(buildings_df):,} buildings")

        return businesses_df, buildings_df

    def extract_centroids(self, buildings_df: pl.DataFrame) -> pl.DataFrame:
        """Extract building centroids with robust error handling."""
        cache_key = self._get_cache_key("centroids", building_count=len(buildings_df))
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        logger.info("Extracting building centroids...")
        start_time = time.time()

        def safe_centroid_extraction(wkt_str: str) -> Tuple[Optional[float], Optional[float]]:
            """Safely extract centroid with comprehensive error handling."""
            try:
                if not wkt_str or wkt_str == "null" or wkt_str.strip() == "":
                    return None, None

                geom = wkt.loads(wkt_str)
                if geom.is_empty:
                    return None, None

                centroid = geom.centroid
                if centroid.is_empty:
                    return None, None

                return centroid.x, centroid.y
            except Exception as e:
                # Log only first few errors to avoid spam
                if not hasattr(safe_centroid_extraction, 'error_count'):
                    safe_centroid_extraction.error_count = 0
                if safe_centroid_extraction.error_count < 5:
                    logger.warning(f"Centroid extraction error: {e}")
                safe_centroid_extraction.error_count += 1
                return None, None

        # Process in batches for memory efficiency
        batch_size = 10000
        results = []

        for i in range(0, len(buildings_df), batch_size):
            batch = buildings_df.slice(i, min(batch_size, len(buildings_df) - i))

            # Extract centroids for batch
            batch_centroids = [
                safe_centroid_extraction(wkt_str)
                for wkt_str in batch["geom_wkt"].to_list()
            ]

            lons, lats = zip(*batch_centroids) if batch_centroids else ([], [])

            batch_with_centroids = batch.with_columns([
                pl.Series("center_lon", lons),
                pl.Series("center_lat", lats)
            ])

            results.append(batch_with_centroids)

            if (i // batch_size + 1) % 5 == 0:
                logger.info(f"Processed {i + batch_size:,}/{len(buildings_df):,} buildings")

        # Combine results and filter valid centroids
        buildings_with_centroids = pl.concat(results).filter(
            pl.col("center_lon").is_not_null() &
            pl.col("center_lat").is_not_null()
        )

        extraction_time = time.time() - start_time
        logger.info(f"Centroid extraction complete in {extraction_time:.2f}s")
        logger.info(f"  Valid centroids: {len(buildings_with_centroids):,}/{len(buildings_df):,}")
        logger.info(f"  Success rate: {len(buildings_with_centroids)/len(buildings_df)*100:.1f}%")

        # Cache result
        self._save_cache(buildings_with_centroids, cache_key)

        return buildings_with_centroids

    def spatial_join_kdtree(self, businesses_df: pl.DataFrame, buildings_df: pl.DataFrame) -> pl.DataFrame:
        """Perform spatial join using KDTree for optimal performance."""
        cache_key = self._get_cache_key(
            "spatial_join",
            business_count=len(businesses_df),
            building_count=len(buildings_df),
            radius=self.proximity_radius
        )
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        logger.info("Starting KDTree spatial join...")
        start_time = time.time()

        # Filter businesses with valid coordinates
        valid_businesses = businesses_df.filter(
            pl.col("longitude").is_not_null() &
            pl.col("latitude").is_not_null()
        )

        if len(valid_businesses) == 0:
            logger.error("No businesses with valid coordinates")
            return pl.DataFrame()

        logger.info(f"Building KDTree with {len(valid_businesses):,} business locations...")

        # Build KDTree from business coordinates
        business_coords = np.column_stack([
            valid_businesses["longitude"].to_numpy(),
            valid_businesses["latitude"].to_numpy()
        ])

        kdtree = cKDTree(business_coords)
        logger.info("KDTree construction complete")

        # Query KDTree for each building
        logger.info(f"Querying spatial relationships for {len(buildings_df):,} buildings...")

        building_coords = np.column_stack([
            buildings_df["center_lon"].to_numpy(),
            buildings_df["center_lat"].to_numpy()
        ])

        # Query all buildings at once (vectorized operation)
        business_indices_list = kdtree.query_ball_point(building_coords, self.proximity_radius)

        # Build results efficiently
        associations = []
        buildings_list = buildings_df.to_dicts()
        businesses_list = valid_businesses.to_dicts()

        for building_idx, business_indices in enumerate(business_indices_list):
            if len(business_indices) > 0:
                building = buildings_list[building_idx]

                for bus_idx in business_indices:
                    business = businesses_list[bus_idx]

                    # Calculate actual distance for quality metrics
                    dx = business["longitude"] - building["center_lon"]
                    dy = business["latitude"] - building["center_lat"]
                    distance = np.sqrt(dx*dx + dy*dy)

                    associations.append({
                        'building_id': building["osm_id"],
                        'building_type': building.get("building_type"),
                        'building_category': building.get("building_category"),
                        'building_center_lon': building["center_lon"],
                        'building_center_lat': building["center_lat"],
                        'business_id': business["business_id"],
                        'naics_code': business.get("naics_code"),
                        'naics_description': business.get("naics_description"),
                        'distance_degrees': distance
                    })

            # Progress logging
            if (building_idx + 1) % 20000 == 0:
                logger.info(f"Processed {building_idx + 1:,}/{len(buildings_df):,} buildings")

        if associations:
            result_df = pl.DataFrame(associations)

            join_time = time.time() - start_time
            logger.info(f"Spatial join complete in {join_time:.2f}s")
            logger.info(f"  Found {len(result_df):,} business-building associations")
            logger.info(f"  Average distance: {result_df['distance_degrees'].mean():.6f} degrees")
            logger.info(f"  Max distance: {result_df['distance_degrees'].max():.6f} degrees")

            # Cache result
            self._save_cache(result_df, cache_key)

            return result_df
        else:
            logger.warning("No spatial associations found")
            return pl.DataFrame()

    def aggregate_by_building(self, joined_df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate business data by building with comprehensive metrics."""
        if len(joined_df) == 0:
            return pl.DataFrame()

        logger.info("Aggregating business data by building...")
        start_time = time.time()

        # Comprehensive building-level aggregation
        building_aggregated = joined_df.group_by("building_id").agg([
            pl.len().alias("business_count"),
            pl.col("naics_code").n_unique().alias("unique_business_types"),
            pl.col("naics_description").mode().first().alias("primary_business_type"),
            pl.col("naics_description").unique().len().alias("business_diversity"),
            pl.col("building_type").first().alias("building_type"),
            pl.col("building_category").first().alias("building_category"),
            pl.col("building_center_lon").first().alias("center_lon"),
            pl.col("building_center_lat").first().alias("center_lat"),
            pl.col("distance_degrees").mean().alias("avg_distance"),
            pl.col("distance_degrees").min().alias("min_distance"),
            pl.col("distance_degrees").max().alias("max_distance")
        ])

        # Add derived metrics
        building_final = building_aggregated.with_columns([
            # Density categories
            pl.when(pl.col("business_count") >= 20).then(pl.lit("very_high"))
            .when(pl.col("business_count") >= 10).then(pl.lit("high"))
            .when(pl.col("business_count") >= 5).then(pl.lit("medium"))
            .when(pl.col("business_count") >= 2).then(pl.lit("low"))
            .otherwise(pl.lit("minimal")).alias("density_category"),

            # Business intensity score (0-100)
            (pl.col("business_count") * 5 + pl.col("business_diversity") * 3)
            .clip(0, 100).alias("intensity_score"),

            # Mixed use indicator
            (pl.col("business_diversity") >= 3).alias("mixed_use"),

            # Quality metrics
            (1.0 / (1.0 + pl.col("avg_distance") * 1000)).alias("proximity_quality"),
            pl.col("business_count").cast(pl.Float64).alias("business_count_float")
        ])

        agg_time = time.time() - start_time
        logger.info(f"Building aggregation complete in {agg_time:.2f}s")
        logger.info(f"  Buildings with businesses: {len(building_final):,}")
        logger.info(f"  Average businesses per building: {building_final['business_count'].mean():.2f}")
        logger.info(f"  Max businesses per building: {building_final['business_count'].max()}")

        return building_final

    def create_grid_aggregation(self, joined_df: pl.DataFrame, grid_size: float = 0.002) -> pl.DataFrame:
        """Create grid-based aggregation for multi-scale visualization."""
        if len(joined_df) == 0:
            return pl.DataFrame()

        logger.info(f"Creating grid aggregation (grid size: {grid_size} degrees)")

        # Create grid coordinates
        grid_df = joined_df.with_columns([
            (pl.col("building_center_lon") / grid_size).floor().alias("grid_x"),
            (pl.col("building_center_lat") / grid_size).floor().alias("grid_y")
        ])

        # Aggregate by grid cell
        grid_aggregated = grid_df.group_by(["grid_x", "grid_y"]).agg([
            pl.len().alias("total_businesses"),
            pl.col("building_id").n_unique().alias("building_count"),
            pl.col("naics_code").n_unique().alias("business_type_diversity"),
            pl.col("building_center_lon").mean().alias("center_lon"),
            pl.col("building_center_lat").mean().alias("center_lat"),
            pl.col("building_category").mode().first().alias("dominant_building_type")
        ]).with_columns([
            # Grid metrics
            (pl.col("total_businesses") / pl.col("building_count")).alias("density_ratio"),
            pl.lit(grid_size).alias("grid_size"),

            # Grid density categories
            pl.when(pl.col("total_businesses") >= 100).then(pl.lit("very_high"))
            .when(pl.col("total_businesses") >= 50).then(pl.lit("high"))
            .when(pl.col("total_businesses") >= 20).then(pl.lit("medium"))
            .when(pl.col("total_businesses") >= 5).then(pl.lit("low"))
            .otherwise(pl.lit("minimal")).alias("density_category")
        ])

        logger.info(f"Grid aggregation complete: {len(grid_aggregated):,} grid cells")
        return grid_aggregated

    def save_results(self, building_agg: pl.DataFrame, grid_agg: pl.DataFrame,
                    stats: Dict) -> None:
        """Save all results with comprehensive metadata."""
        logger.info("Saving aggregation results...")

        # Save parquet files
        building_agg.write_parquet(self.output_dir / "buildings_businesses_robust.parquet")
        grid_agg.write_parquet(self.output_dir / "business_grid_robust.parquet")

        # Save GeoJSON for visualization
        self._save_geojson(building_agg, "buildings_businesses_robust.geojson")

        # Save comprehensive statistics with datetime handling
        def serialize_stats(obj):
            """Custom JSON serializer for datetime objects."""
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(self.output_dir / "robust_aggregation_stats.json", 'w') as f:
            json.dump(stats, f, indent=2, default=serialize_stats)

        logger.info("Results saved successfully")

    def _save_geojson(self, df: pl.DataFrame, filename: str):
        """Save DataFrame as GeoJSON for visualization."""
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
                    if k not in ["center_lon", "center_lat"] and v is not None
                }
            }
            features.append(feature)

        geojson_data = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {
                "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "feature_count": len(features),
                "generator": "RobustSpatialAggregator"
            }
        }

        with open(self.output_dir / filename, 'w') as f:
            json.dump(geojson_data, f, separators=(',', ':'))

    def run_complete_pipeline(self) -> Dict:
        """Run the complete robust spatial aggregation pipeline."""
        logger.info("=" * 70)
        logger.info("ROBUST SPATIAL AGGREGATION PIPELINE")
        logger.info("=" * 70)

        pipeline_start = time.time()
        phase_times = {}

        try:
            # Phase 1: Data Loading
            phase_start = time.time()
            businesses_df, buildings_df = self.load_data()
            phase_times["data_loading"] = time.time() - phase_start

            # Phase 2: Centroid Extraction
            phase_start = time.time()
            buildings_with_centroids = self.extract_centroids(buildings_df)
            phase_times["centroid_extraction"] = time.time() - phase_start

            if len(buildings_with_centroids) == 0:
                raise ValueError("No valid building centroids extracted")

            # Phase 3: Spatial Join
            phase_start = time.time()
            joined_df = self.spatial_join_kdtree(businesses_df, buildings_with_centroids)
            phase_times["spatial_join"] = time.time() - phase_start

            if len(joined_df) == 0:
                raise ValueError("No spatial associations found")

            # Phase 4: Building Aggregation
            phase_start = time.time()
            building_agg = self.aggregate_by_building(joined_df)
            phase_times["building_aggregation"] = time.time() - phase_start

            # Phase 5: Grid Aggregation
            phase_start = time.time()
            grid_agg = self.create_grid_aggregation(joined_df)
            phase_times["grid_aggregation"] = time.time() - phase_start

            # Calculate comprehensive statistics
            total_time = time.time() - pipeline_start

            # Convert any potential datetime objects to strings
            def safe_value(val):
                """Convert value to JSON-safe format."""
                if hasattr(val, 'isoformat'):
                    return val.isoformat()
                elif hasattr(val, 'item'):  # numpy scalar
                    return val.item()
                else:
                    return val

            stats = {
                "pipeline_success": True,
                "total_time_seconds": round(total_time, 2),
                "performance_rating": "excellent" if total_time < 30 else "good" if total_time < 60 else "acceptable",
                "phase_times": {k: round(v, 2) for k, v in phase_times.items()},
                "data_quality": {
                    "buildings_loaded": len(buildings_df),
                    "businesses_loaded": len(businesses_df),
                    "buildings_with_centroids": len(buildings_with_centroids),
                    "centroid_success_rate": round(len(buildings_with_centroids)/len(buildings_df)*100, 2),
                    "spatial_associations": len(joined_df),
                    "buildings_with_businesses": len(building_agg),
                    "coverage_rate": round(len(building_agg)/len(buildings_with_centroids)*100, 2)
                },
                "business_metrics": {
                    "avg_businesses_per_building": round(safe_value(building_agg["business_count"].mean()), 2),
                    "max_businesses_per_building": safe_value(building_agg["business_count"].max()),
                    "total_unique_business_types": safe_value(joined_df["naics_code"].n_unique()),
                    "mixed_use_buildings": safe_value(building_agg.filter(pl.col("mixed_use") == True).height)
                },
                "spatial_quality": {
                    "proximity_radius_degrees": self.proximity_radius,
                    "proximity_radius_meters": round(self.proximity_radius * 111000, 1),
                    "avg_association_distance_meters": round(safe_value(joined_df["distance_degrees"].mean()) * 111000, 1),
                    "max_association_distance_meters": round(safe_value(joined_df["distance_degrees"].max()) * 111000, 1)
                },
                "output_files": {
                    "buildings_parquet": "buildings_businesses_robust.parquet",
                    "grid_parquet": "business_grid_robust.parquet",
                    "buildings_geojson": "buildings_businesses_robust.geojson",
                    "stats_json": "robust_aggregation_stats.json"
                },
                "grid_summary": {
                    "total_grid_cells": len(grid_agg),
                    "grid_size_degrees": 0.002,
                    "grid_size_meters": round(0.002 * 111000, 1)
                },
                "density_distribution": [
                    {k: safe_value(v) for k, v in item.items()}
                    for item in building_agg.group_by("density_category").len().to_dicts()
                ]
            }

            # Phase 6: Save Results
            phase_start = time.time()
            self.save_results(building_agg, grid_agg, stats)
            phase_times["save_results"] = time.time() - phase_start

            # Final logging
            logger.info("=" * 70)
            logger.info(f"✅ PIPELINE COMPLETE - {total_time:.2f} seconds")
            logger.info(f"   Performance: {stats['performance_rating'].upper()}")
            logger.info(f"   Spatial associations: {len(joined_df):,}")
            logger.info(f"   Buildings with businesses: {len(building_agg):,}")
            logger.info(f"   Coverage rate: {stats['data_quality']['coverage_rate']:.1f}%")
            logger.info("=" * 70)

            return stats

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                "pipeline_success": False,
                "error": str(e),
                "total_time_seconds": time.time() - pipeline_start
            }

def main():
    """Run the robust spatial aggregator."""
    import argparse

    parser = argparse.ArgumentParser(description="Robust spatial aggregation pipeline")
    parser.add_argument("--radius", type=float, default=0.0005,
                       help="Proximity radius in degrees (default: 0.0005 ≈ 50m)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--input-dir", default="output", help="Input directory")
    parser.add_argument("--output-dir", default="output", help="Output directory")

    args = parser.parse_args()

    aggregator = RobustSpatialAggregator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        proximity_radius=args.radius,
        enable_cache=not args.no_cache
    )

    stats = aggregator.run_complete_pipeline()

    if stats.get("pipeline_success"):
        print("\n" + "=" * 70)
        print("ROBUST SPATIAL AGGREGATION - SUCCESS")
        print("=" * 70)
        print(f"Total time: {stats['total_time_seconds']}s ({stats['performance_rating']})")
        print(f"Spatial associations: {stats['data_quality']['spatial_associations']:,}")
        print(f"Buildings with businesses: {stats['data_quality']['buildings_with_businesses']:,}")
        print(f"Coverage rate: {stats['data_quality']['coverage_rate']:.1f}%")
        print(f"Avg businesses per building: {stats['business_metrics']['avg_businesses_per_building']}")
        print(f"Mixed-use buildings: {stats['business_metrics']['mixed_use_buildings']:,}")
        print("\nDensity distribution:")
        for item in stats['density_distribution']:
            print(f"  {item['density_category']}: {item['len']:,} buildings")
    else:
        print("\n" + "=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)
        print(f"Error: {stats.get('error', 'Unknown error')}")
        print(f"Time before failure: {stats['total_time_seconds']:.2f}s")

if __name__ == "__main__":
    main()