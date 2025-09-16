#!/usr/bin/env python3
"""Optimized building-business aggregator with spatial indexing and parallel processing."""

import polars as pl
import json
from pathlib import Path
import logging
from shapely import wkt
from shapely.geometry import Point
import geojson
from typing import Dict, List, Tuple, Optional
import math
import time
from functools import lru_cache
import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
from dataclasses import dataclass
import pickle
import psycopg2
from psycopg2 import sql
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpatialConfig:
    """Configuration for spatial join operations."""
    use_postgis: bool = True
    use_rtree: bool = True
    use_parallel: bool = True
    parallel_workers: int = mp.cpu_count()
    cache_enabled: bool = True
    cache_dir: Path = Path("output/cache")
    proximity_radius: float = 0.0005  # ~50 meters at SF latitude

class SpatialIndex:
    """R-tree spatial index for efficient proximity searches."""

    def __init__(self):
        self.index = None
        self.id_to_coords = {}
        self._init_rtree()

    def _init_rtree(self):
        """Initialize R-tree index if available."""
        try:
            from rtree import index
            p = index.Property()
            p.dimension = 2
            self.index = index.Index(properties=p)
            self.has_rtree = True
            logger.info("R-tree spatial index initialized")
        except ImportError:
            self.has_rtree = False
            logger.warning("R-tree not available, falling back to basic spatial search")

    def add_point(self, idx: int, lon: float, lat: float):
        """Add a point to the spatial index."""
        if self.has_rtree:
            self.index.insert(idx, (lon, lat, lon, lat))
        self.id_to_coords[idx] = (lon, lat)

    def query_radius(self, lon: float, lat: float, radius: float) -> List[int]:
        """Query points within radius."""
        if self.has_rtree:
            bbox = (lon - radius, lat - radius, lon + radius, lat + radius)
            candidates = list(self.index.intersection(bbox))

            # Filter by actual distance
            results = []
            for idx in candidates:
                cx, cy = self.id_to_coords[idx]
                dist = math.sqrt((cx - lon) ** 2 + (cy - lat) ** 2)
                if dist <= radius:
                    results.append(idx)
            return results
        else:
            # Fallback to brute force
            results = []
            for idx, (cx, cy) in self.id_to_coords.items():
                if abs(cx - lon) <= radius and abs(cy - lat) <= radius:
                    dist = math.sqrt((cx - lon) ** 2 + (cy - lat) ** 2)
                    if dist <= radius:
                        results.append(idx)
            return results

class PostGISHandler:
    """Handle PostGIS spatial operations."""

    def __init__(self):
        self.conn = None
        self.enabled = False
        self._init_connection()

    def _init_connection(self):
        """Initialize PostGIS connection."""
        try:
            # Check for PostGIS availability
            self.conn = psycopg2.connect(
                host=os.environ.get("PGHOST", "localhost"),
                port=os.environ.get("PGPORT", 5432),
                database=os.environ.get("PGDATABASE", "postgres"),  # Use default postgres database
                user=os.environ.get("PGUSER", os.environ.get("USER", "postgres")),
                password=os.environ.get("PGPASSWORD", "")
            )

            # Try to create database if it doesn't exist
            try:
                with self.conn.cursor() as cur:
                    self.conn.autocommit = True
                    cur.execute("CREATE DATABASE sf_businesses;")
                    logger.info("Created sf_businesses database")
            except:
                pass  # Database might already exist

            # Reconnect to sf_businesses database
            self.conn.close()
            self.conn = psycopg2.connect(
                host=os.environ.get("PGHOST", "localhost"),
                port=os.environ.get("PGPORT", 5432),
                database="sf_businesses",
                user=os.environ.get("PGUSER", os.environ.get("USER", "postgres")),
                password=os.environ.get("PGPASSWORD", "")
            )

            # Try to enable PostGIS
            with self.conn.cursor() as cur:
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                    self.conn.commit()
                except:
                    pass  # PostGIS might not be available

                # Check for PostGIS extension
                cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'postgis');")
                has_postgis = cur.fetchone()[0]

                if has_postgis:
                    cur.execute("SELECT PostGIS_version();")
                    version = cur.fetchone()
                    logger.info(f"PostGIS available: {version[0]}")
                    self.enabled = True
                else:
                    logger.warning("PostGIS extension not available")
                    self.enabled = False
        except Exception as e:
            logger.warning(f"PostGIS not available: {e}")
            self.enabled = False

    def create_spatial_tables(self, businesses_df: pl.DataFrame, buildings_df: pl.DataFrame):
        """Create PostGIS tables for spatial operations."""
        if not self.enabled:
            return False

        try:
            with self.conn.cursor() as cur:
                # Create businesses table
                cur.execute("""
                    DROP TABLE IF EXISTS temp_businesses CASCADE;
                    CREATE TABLE temp_businesses (
                        id SERIAL PRIMARY KEY,
                        business_id TEXT,
                        naics_code TEXT,
                        naics_description TEXT,
                        location GEOMETRY(Point, 4326)
                    );
                """)

                # Create buildings table
                cur.execute("""
                    DROP TABLE IF EXISTS temp_buildings CASCADE;
                    CREATE TABLE temp_buildings (
                        id SERIAL PRIMARY KEY,
                        building_id TEXT,
                        building_type TEXT,
                        building_category TEXT,
                        location GEOMETRY(Point, 4326)
                    );
                """)

                # Insert businesses
                for row in businesses_df.filter(
                    pl.col("longitude").is_not_null() & pl.col("latitude").is_not_null()
                ).iter_rows(named=True):
                    cur.execute("""
                        INSERT INTO temp_businesses (business_id, naics_code, naics_description, location)
                        VALUES (%s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
                    """, (
                        row.get("business_id", ""),
                        row.get("naics_code", ""),
                        row.get("naics_description", ""),
                        row["longitude"],
                        row["latitude"]
                    ))

                # Insert buildings with centroids
                for row in buildings_df.iter_rows(named=True):
                    if row["center_lon"] is not None and row["center_lat"] is not None:
                        cur.execute("""
                            INSERT INTO temp_buildings (building_id, building_type, building_category, location)
                            VALUES (%s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
                        """, (
                            str(row["osm_id"]),
                            row.get("building_type", ""),
                            row.get("building_category", ""),
                            row["center_lon"],
                            row["center_lat"]
                        ))

                # Create spatial indexes
                cur.execute("CREATE INDEX idx_businesses_location ON temp_businesses USING GIST(location);")
                cur.execute("CREATE INDEX idx_buildings_location ON temp_buildings USING GIST(location);")

                self.conn.commit()
                logger.info("PostGIS tables created successfully")
                return True

        except Exception as e:
            logger.error(f"Failed to create PostGIS tables: {e}")
            if self.conn:
                self.conn.rollback()
            return False

    def spatial_join_postgis(self, radius_meters: float = 50) -> pl.DataFrame:
        """Perform spatial join using PostGIS."""
        if not self.enabled:
            return pl.DataFrame()

        try:
            query = """
                SELECT
                    b.building_id,
                    b.building_type,
                    b.building_category,
                    ST_X(b.location) as building_lon,
                    ST_Y(b.location) as building_lat,
                    bus.business_id,
                    bus.naics_code,
                    bus.naics_description,
                    ST_Distance(b.location::geography, bus.location::geography) as distance_meters
                FROM temp_buildings b
                JOIN temp_businesses bus
                ON ST_DWithin(b.location::geography, bus.location::geography, %s)
                ORDER BY b.building_id, distance_meters;
            """

            with self.conn.cursor() as cur:
                cur.execute(query, (radius_meters,))
                results = cur.fetchall()

                # Convert to Polars DataFrame
                if results:
                    df = pl.DataFrame(
                        results,
                        schema=[
                            "building_id", "building_type", "building_category",
                            "building_lon", "building_lat", "business_id",
                            "naics_code", "naics_description", "distance_meters"
                        ]
                    )
                    logger.info(f"PostGIS spatial join found {len(df):,} associations")
                    return df
                else:
                    return pl.DataFrame()

        except Exception as e:
            logger.error(f"PostGIS spatial join failed: {e}")
            return pl.DataFrame()

    def close(self):
        """Close PostGIS connection."""
        if self.conn:
            self.conn.close()

class OptimizedBuildingBusinessAggregator:
    """Optimized aggregator with multiple performance improvements."""

    def __init__(self, input_dir: str = "output", output_dir: str = "output", config: Optional[SpatialConfig] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.config = config or SpatialConfig()
        if self.config.cache_enabled:
            self.config.cache_dir.mkdir(exist_ok=True)

        self.postgis = PostGISHandler() if self.config.use_postgis else None

    def _get_cache_key(self, operation: str, **params) -> str:
        """Generate cache key for operation."""
        key_str = f"{operation}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[pl.DataFrame]:
        """Load DataFrame from cache."""
        if not self.config.cache_enabled:
            return None

        cache_file = self.config.cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < 3600:  # 1 hour cache validity
                logger.info(f"Loading from cache: {cache_key}")
                return pl.read_parquet(cache_file)
        return None

    def _save_to_cache(self, df: pl.DataFrame, cache_key: str):
        """Save DataFrame to cache."""
        if not self.config.cache_enabled:
            return

        cache_file = self.config.cache_dir / f"{cache_key}.parquet"
        df.write_parquet(cache_file)
        logger.info(f"Saved to cache: {cache_key}")

    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load business and building data."""
        logger.info("Loading business and building data for aggregation...")

        businesses_df = pl.read_parquet(self.input_dir / "businesses.parquet")
        buildings_df = pl.read_parquet(self.input_dir / "buildings.parquet")

        logger.info(f"Loaded {len(businesses_df):,} businesses and {len(buildings_df):,} buildings")
        return businesses_df, buildings_df

    def extract_centroids(self, buildings_df: pl.DataFrame) -> pl.DataFrame:
        """Extract centroids from building WKT geometries with parallel processing."""
        logger.info("Extracting building centroids...")

        def get_centroid_coords(wkt_str: str) -> Tuple[Optional[float], Optional[float]]:
            """Extract centroid from WKT geometry."""
            try:
                if not wkt_str or wkt_str == "null":
                    return None, None
                geom = wkt.loads(wkt_str)
                centroid = geom.centroid
                return centroid.x, centroid.y
            except Exception:
                return None, None

        # Add centroids to buildings
        buildings_with_centroids = buildings_df.with_columns([
            pl.col("geom_wkt").map_elements(
                lambda x: get_centroid_coords(x)[0] if x else None,
                return_dtype=pl.Float64
            ).alias("center_lon"),
            pl.col("geom_wkt").map_elements(
                lambda x: get_centroid_coords(x)[1] if x else None,
                return_dtype=pl.Float64
            ).alias("center_lat")
        ]).filter(
            pl.col("center_lon").is_not_null() & pl.col("center_lat").is_not_null()
        )

        logger.info(f"Successfully extracted centroids for {len(buildings_with_centroids):,} buildings")
        return buildings_with_centroids

    def _process_building_batch(self, args):
        """Process a batch of buildings for spatial join (for parallel processing)."""
        buildings_batch, businesses_df, radius = args
        results = []

        for building in buildings_batch:
            center_lon = building['center_lon']
            center_lat = building['center_lat']

            # Find businesses within radius
            nearby_mask = (
                (businesses_df['longitude'] >= center_lon - radius) &
                (businesses_df['longitude'] <= center_lon + radius) &
                (businesses_df['latitude'] >= center_lat - radius) &
                (businesses_df['latitude'] <= center_lat + radius)
            )

            nearby_businesses = businesses_df[nearby_mask]

            if len(nearby_businesses) > 0:
                # Calculate actual distances
                distances = np.sqrt(
                    (nearby_businesses['longitude'] - center_lon) ** 2 +
                    (nearby_businesses['latitude'] - center_lat) ** 2
                )

                within_radius = distances <= radius

                for i, is_within in enumerate(within_radius):
                    if is_within:
                        results.append({
                            'building_id': building['osm_id'],
                            'building_type': building['building_type'],
                            'building_category': building['building_category'],
                            'building_center_lon': center_lon,
                            'building_center_lat': center_lat,
                            'business_index': nearby_businesses.iloc[i]['business_index'],
                            'distance_degrees': distances[i]
                        })

        return results

    def spatial_join_optimized(self, businesses_df: pl.DataFrame, buildings_df: pl.DataFrame) -> pl.DataFrame:
        """Optimized spatial join using multiple strategies."""
        start_time = time.time()

        # Check cache first
        cache_key = self._get_cache_key(
            "spatial_join",
            businesses_count=len(businesses_df),
            buildings_count=len(buildings_df)
        )
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        # Try PostGIS first if available
        if self.postgis and self.postgis.enabled:
            logger.info("Using PostGIS for spatial join...")
            if self.postgis.create_spatial_tables(businesses_df, buildings_df):
                result = self.postgis.spatial_join_postgis()
                if len(result) > 0:
                    elapsed = time.time() - start_time
                    logger.info(f"PostGIS spatial join completed in {elapsed:.2f} seconds")
                    self._save_to_cache(result, cache_key)
                    return result

        # Fallback to optimized in-memory spatial join
        logger.info("Using optimized in-memory spatial join...")

        # Build spatial index if available
        if self.config.use_rtree:
            spatial_index = SpatialIndex()

            # Add businesses to spatial index
            businesses_indexed = businesses_df.with_row_index("business_index")
            for row in businesses_indexed.filter(
                pl.col("longitude").is_not_null() & pl.col("latitude").is_not_null()
            ).iter_rows(named=True):
                spatial_index.add_point(
                    row['business_index'],
                    row['longitude'],
                    row['latitude']
                )

            # Process buildings using spatial index
            joined_data = []
            for building in buildings_df.iter_rows(named=True):
                center_lon = building['center_lon']
                center_lat = building['center_lat']

                # Query spatial index
                nearby_indices = spatial_index.query_radius(
                    center_lon, center_lat, self.config.proximity_radius
                )

                for idx in nearby_indices:
                    business = businesses_indexed.filter(pl.col("business_index") == idx).to_dicts()[0]
                    joined_data.append({
                        'building_id': building['osm_id'],
                        'building_type': building['building_type'],
                        'building_category': building['building_category'],
                        'building_center_lon': center_lon,
                        'building_center_lat': center_lat,
                        **business
                    })
        else:
            # Use parallel processing for batch spatial join
            logger.info(f"Using parallel processing with {self.config.parallel_workers} workers...")

            businesses_indexed = businesses_df.with_row_index("business_index")

            # Convert to numpy for faster processing
            businesses_np = {
                'longitude': businesses_indexed['longitude'].to_numpy(),
                'latitude': businesses_indexed['latitude'].to_numpy(),
                'business_index': businesses_indexed['business_index'].to_numpy()
            }

            # Add other columns
            for col in businesses_indexed.columns:
                if col not in ['longitude', 'latitude', 'business_index']:
                    businesses_np[col] = businesses_indexed[col].to_list()

            # Split buildings into batches
            batch_size = max(100, len(buildings_df) // (self.config.parallel_workers * 10))
            building_batches = []

            for i in range(0, len(buildings_df), batch_size):
                batch = buildings_df.slice(i, batch_size).to_dicts()
                building_batches.append((batch, businesses_np, self.config.proximity_radius))

            # Process batches in parallel
            joined_data = []
            with ProcessPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                future_to_batch = {
                    executor.submit(self._process_building_batch, batch): i
                    for i, batch in enumerate(building_batches)
                }

                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        joined_data.extend(batch_results)

                        if batch_idx % 10 == 0:
                            logger.info(f"Processed batch {batch_idx}/{len(building_batches)}")
                    except Exception as e:
                        logger.error(f"Batch {batch_idx} failed: {e}")

        if joined_data:
            result = pl.DataFrame(joined_data)

            # Merge with business data
            result = result.join(
                businesses_indexed,
                on="business_index",
                how="left"
            )

            elapsed = time.time() - start_time
            logger.info(f"Spatial join completed in {elapsed:.2f} seconds: {len(result):,} associations")

            # Save to cache
            self._save_to_cache(result, cache_key)
            return result
        else:
            logger.warning("No spatial joins found!")
            return pl.DataFrame()

    def aggregate_by_building(self, joined_df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate business statistics by building."""
        logger.info("Aggregating business statistics by building...")

        # Aggregate business metrics per building
        building_aggregated = joined_df.group_by("building_id").agg([
            pl.count().alias("business_count"),
            pl.col("naics_code").n_unique().alias("unique_business_types"),
            pl.col("naics_description").first().alias("primary_business_type"),
            pl.col("naics_description").unique().len().alias("business_diversity"),
            pl.col("building_type").first().alias("building_type"),
            pl.col("building_category").first().alias("building_category"),
            pl.col("building_center_lon").first().alias("center_lon"),
            pl.col("building_center_lat").first().alias("center_lat")
        ])

        # Add derived metrics
        building_final = building_aggregated.with_columns([
            # Business density category
            pl.when(pl.col("business_count") >= 10).then("very_high")
            .when(pl.col("business_count") >= 5).then("high")
            .when(pl.col("business_count") >= 2).then("medium")
            .when(pl.col("business_count") >= 1).then("low")
            .otherwise("none").alias("density_category"),

            # Business intensity score (0-100)
            (pl.col("business_count") * 10 + pl.col("business_diversity") * 5)
            .clip(0, 100).alias("intensity_score"),

            # Mixed use indicator
            (pl.col("business_diversity") >= 3).alias("mixed_use")
        ])

        logger.info(f"Building aggregation complete: {len(building_final):,} buildings with businesses")
        return building_final

    def run_full_aggregation(self) -> Dict:
        """Run the complete optimized building-business aggregation pipeline."""
        logger.info("Starting OPTIMIZED building-business aggregation pipeline...")
        overall_start = time.time()

        # Load data
        businesses_df, buildings_df = self.load_data()

        # Extract building centroids
        buildings_with_centroids = self.extract_centroids(buildings_df)

        # Perform optimized spatial join
        joined_df = self.spatial_join_optimized(businesses_df, buildings_with_centroids)

        if len(joined_df) == 0:
            logger.error("No spatial joins found - cannot proceed with aggregation")
            return {}

        # Aggregate by building
        building_aggregated = self.aggregate_by_building(joined_df)

        # Save results
        output_path = self.output_dir / "buildings_businesses_optimized.parquet"
        building_aggregated.write_parquet(output_path)

        # Generate statistics
        elapsed_total = time.time() - overall_start
        stats = {
            "total_time_seconds": round(elapsed_total, 2),
            "total_buildings_with_businesses": len(building_aggregated),
            "total_associations": len(joined_df),
            "avg_businesses_per_building": float(building_aggregated.select("business_count").mean().item()),
            "max_businesses_per_building": int(building_aggregated.select("business_count").max().item()),
            "density_distribution": building_aggregated.group_by("density_category").count().to_dicts(),
            "optimization_methods_used": {
                "postgis": self.postgis.enabled if self.postgis else False,
                "rtree": self.config.use_rtree,
                "parallel": self.config.use_parallel,
                "cache": self.config.cache_enabled
            }
        }

        # Save statistics
        stats_path = self.output_dir / "optimized_aggregation_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"OPTIMIZED aggregation pipeline complete in {elapsed_total:.2f} seconds!")

        # Clean up PostGIS connection
        if self.postgis:
            self.postgis.close()

        return stats

def benchmark_performance():
    """Run performance benchmarks comparing different optimization strategies."""
    logger.info("Starting performance benchmarks...")

    results = {}

    # Test 1: Default (no optimizations)
    config_default = SpatialConfig(
        use_postgis=False,
        use_rtree=False,
        use_parallel=False,
        cache_enabled=False
    )
    aggregator = OptimizedBuildingBusinessAggregator(config=config_default)
    start = time.time()
    stats = aggregator.run_full_aggregation()
    results["baseline"] = {
        "time": time.time() - start,
        "stats": stats
    }

    # Test 2: With R-tree only
    config_rtree = SpatialConfig(
        use_postgis=False,
        use_rtree=True,
        use_parallel=False,
        cache_enabled=False
    )
    aggregator = OptimizedBuildingBusinessAggregator(config=config_rtree)
    start = time.time()
    stats = aggregator.run_full_aggregation()
    results["rtree_only"] = {
        "time": time.time() - start,
        "stats": stats
    }

    # Test 3: With parallel processing
    config_parallel = SpatialConfig(
        use_postgis=False,
        use_rtree=False,
        use_parallel=True,
        cache_enabled=False
    )
    aggregator = OptimizedBuildingBusinessAggregator(config=config_parallel)
    start = time.time()
    stats = aggregator.run_full_aggregation()
    results["parallel_only"] = {
        "time": time.time() - start,
        "stats": stats
    }

    # Test 4: All optimizations
    config_all = SpatialConfig(
        use_postgis=True,
        use_rtree=True,
        use_parallel=True,
        cache_enabled=True
    )
    aggregator = OptimizedBuildingBusinessAggregator(config=config_all)
    start = time.time()
    stats = aggregator.run_full_aggregation()
    results["all_optimizations"] = {
        "time": time.time() - start,
        "stats": stats
    }

    # Save benchmark results
    with open("output/benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n=== PERFORMANCE BENCHMARK RESULTS ===")
    for name, data in results.items():
        print(f"\n{name}:")
        print(f"  Time: {data['time']:.2f} seconds")
        if data['stats']:
            print(f"  Associations: {data['stats'].get('total_associations', 'N/A')}")

    return results

def main():
    """Run the optimized aggregation."""
    import argparse

    parser = argparse.ArgumentParser(description="Optimized building-business aggregation")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--no-postgis", action="store_true", help="Disable PostGIS")
    parser.add_argument("--no-rtree", action="store_true", help="Disable R-tree")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    args = parser.parse_args()

    if args.benchmark:
        benchmark_performance()
    else:
        config = SpatialConfig(
            use_postgis=not args.no_postgis,
            use_rtree=not args.no_rtree,
            use_parallel=not args.no_parallel,
            cache_enabled=not args.no_cache
        )

        aggregator = OptimizedBuildingBusinessAggregator(config=config)
        stats = aggregator.run_full_aggregation()

        if stats:
            print("\n=== OPTIMIZED AGGREGATION RESULTS ===")
            for key, value in stats.items():
                print(f"{key}: {value}")
        else:
            print("Aggregation failed!")

if __name__ == "__main__":
    main()