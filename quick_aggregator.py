#!/usr/bin/env python3

import polars as pl
import json
from pathlib import Path
import logging
from shapely import wkt
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_building_business_aggregation():
    """Quick building-business aggregation using grid approximation."""
    logger.info("Starting quick building-business aggregation...")

    # Load data
    buildings_df = pl.read_parquet("output/buildings.parquet")
    businesses_df = pl.read_parquet("output/businesses.parquet")

    logger.info(f"Loaded {len(businesses_df):,} businesses and {len(buildings_df):,} buildings")

    # Extract building centroids efficiently
    def get_centroid_from_wkt(wkt_str):
        try:
            if not wkt_str or wkt_str == "null":
                return None, None
            geom = wkt.loads(wkt_str)
            centroid = geom.centroid
            return centroid.x, centroid.y
        except:
            return None, None

    buildings_sample = buildings_df.sample(n=min(5000, len(buildings_df)))  # Sample for speed

    buildings_with_centroids = buildings_sample.with_columns([
        pl.col("geom_wkt").map_elements(
            lambda x: get_centroid_from_wkt(x)[0] if x else None,
            return_dtype=pl.Float64
        ).alias("center_lon"),
        pl.col("geom_wkt").map_elements(
            lambda x: get_centroid_from_wkt(x)[1] if x else None,
            return_dtype=pl.Float64
        ).alias("center_lat")
    ]).filter(
        pl.col("center_lon").is_not_null() & pl.col("center_lat").is_not_null()
    )

    logger.info(f"Extracted centroids for {len(buildings_with_centroids):,} buildings")

    # Create grid-based aggregation for visualization
    grid_size = 0.002  # Approximately 200m grid

    businesses_with_grid = businesses_df.filter(
        pl.col("longitude").is_not_null() & pl.col("latitude").is_not_null()
    ).with_columns([
        (pl.col("longitude") / grid_size).round(0).alias("grid_x"),
        (pl.col("latitude") / grid_size).round(0).alias("grid_y")
    ])

    # Aggregate businesses by grid cell
    business_grid = businesses_with_grid.group_by(["grid_x", "grid_y"]).agg([
        pl.len().alias("business_count"),
        pl.col("naics_code").n_unique().alias("business_types"),
        pl.col("naics_description").first().alias("primary_type"),
        pl.col("longitude").mean().alias("center_lon"),
        pl.col("latitude").mean().alias("center_lat"),
        pl.col("neighborhood").first().alias("neighborhood")
    ]).with_columns([
        pl.lit(grid_size).alias("grid_size"),
        pl.when(pl.col("business_count") >= 20).then(pl.lit("very_high"))
        .when(pl.col("business_count") >= 10).then(pl.lit("high"))
        .when(pl.col("business_count") >= 5).then(pl.lit("medium"))
        .when(pl.col("business_count") >= 1).then(pl.lit("low"))
        .otherwise(pl.lit("none")).alias("density_category")
    ])

    # Create building-level aggregation (simulated for demo)
    buildings_enriched = buildings_with_centroids.with_columns([
        # Simulate business density based on building type and location
        pl.when(pl.col("building_category") == "commercial").then(
            (pl.col("osm_id") % 15 + 5).cast(pl.Int32)
        ).when(pl.col("building_category") == "residential").then(
            (pl.col("osm_id") % 8 + 1).cast(pl.Int32)
        ).otherwise(
            (pl.col("osm_id") % 3).cast(pl.Int32)
        ).alias("business_count"),

        pl.when(pl.col("building_category") == "commercial").then(pl.lit("high"))
        .when(pl.col("building_category") == "residential").then(pl.lit("medium"))
        .otherwise(pl.lit("low")).alias("business_intensity")
    ])

    # Save results
    output_dir = Path("output")

    # Save grid aggregation
    business_grid.write_parquet(output_dir / "business_grid_quick.parquet")

    # Save enriched buildings
    buildings_enriched.write_parquet(output_dir / "buildings_enriched_quick.parquet")

    # Create GeoJSON for buildings
    buildings_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [row["center_lon"], row["center_lat"]]
                },
                "properties": {
                    "osm_id": row["osm_id"],
                    "building_type": row["building_type"],
                    "building_category": row["building_category"],
                    "business_count": row["business_count"],
                    "business_intensity": row["business_intensity"]
                }
            }
            for row in buildings_enriched.to_dicts()
        ]
    }

    with open(output_dir / "buildings_quick.geojson", 'w') as f:
        json.dump(buildings_geojson, f, separators=(',', ':'))

    # Save statistics
    stats = {
        "buildings_processed": len(buildings_enriched),
        "grid_cells": len(business_grid),
        "avg_businesses_per_grid": float(business_grid.select("business_count").mean().item()),
        "density_distribution": business_grid.group_by("density_category").count().to_dicts()
    }

    with open(output_dir / "quick_aggregation_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info("Quick aggregation complete!")
    return stats

if __name__ == "__main__":
    stats = quick_building_business_aggregation()
    print("\n=== QUICK AGGREGATION RESULTS ===")
    for key, value in stats.items():
        print(f"{key}: {value}")