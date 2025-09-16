#!/usr/bin/env python3

import polars as pl
import json
from pathlib import Path
import logging
from shapely import wkt
from shapely.geometry import Point
import geojson
from typing import Dict, List, Tuple, Optional
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BuildingBusinessAggregator:
    """Aggregate business data by building for visualization performance."""

    def __init__(self, input_dir: str = "output", output_dir: str = "output"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Load business and building data."""
        logger.info("Loading business and building data for aggregation...")

        businesses_df = pl.read_parquet(self.input_dir / "businesses.parquet")
        buildings_df = pl.read_parquet(self.input_dir / "buildings.parquet")

        logger.info(f"Loaded {len(businesses_df):,} businesses and {len(buildings_df):,} buildings")
        return businesses_df, buildings_df

    def extract_centroids(self, buildings_df: pl.DataFrame) -> pl.DataFrame:
        """Extract centroids from building WKT geometries."""
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

    def spatial_join_proximity(self, businesses_df: pl.DataFrame, buildings_df: pl.DataFrame,
                              radius: float = 0.0005) -> pl.DataFrame:
        """
        Join businesses to nearest buildings using proximity search.

        Args:
            radius: Search radius in degrees (approximately 50 meters at SF latitude)
        """
        logger.info(f"Performing spatial join with {radius} degree radius...")

        # Add row index to businesses for later joins
        businesses_indexed = businesses_df.with_row_index("business_index")

        joined_data = []
        batch_size = 1000

        for i in range(0, len(buildings_df), batch_size):
            batch = buildings_df.slice(i, batch_size)

            for building in batch.iter_rows(named=True):
                center_lon = building['center_lon']
                center_lat = building['center_lat']

                # Find businesses within radius
                nearby_businesses = businesses_indexed.filter(
                    (pl.col("longitude") >= center_lon - radius) &
                    (pl.col("longitude") <= center_lon + radius) &
                    (pl.col("latitude") >= center_lat - radius) &
                    (pl.col("latitude") <= center_lat + radius)
                )

                # Calculate actual distances for better accuracy
                if len(nearby_businesses) > 0:
                    nearby_with_distance = nearby_businesses.with_columns([
                        pl.lit(building['osm_id']).alias("building_id"),
                        pl.lit(building['building_type']).alias("building_type"),
                        pl.lit(building['building_category']).alias("building_category"),
                        pl.lit(center_lon).alias("building_center_lon"),
                        pl.lit(center_lat).alias("building_center_lat"),
                        (
                            (pl.col("longitude") - center_lon).pow(2) +
                            (pl.col("latitude") - center_lat).pow(2)
                        ).sqrt().alias("distance_degrees")
                    ]).filter(
                        pl.col("distance_degrees") <= radius
                    )

                    joined_data.append(nearby_with_distance)

            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + batch_size:,}/{len(buildings_df):,} buildings")

        if joined_data:
            result = pl.concat(joined_data)
            logger.info(f"Spatial join complete: {len(result):,} business-building associations")
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
            pl.col("building_center_lat").first().alias("center_lat"),
            pl.col("business_start_date").min().alias("earliest_business"),
            pl.col("business_end_date").max().alias("latest_business")
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

    def create_grid_aggregation(self, joined_df: pl.DataFrame, grid_size: float = 0.001) -> pl.DataFrame:
        """Create grid-based aggregation for lower zoom levels."""
        logger.info(f"Creating grid aggregation with {grid_size} degree grid size...")

        # Create grid coordinates
        grid_df = joined_df.with_columns([
            (pl.col("building_center_lon") / grid_size).floor().alias("grid_x"),
            (pl.col("building_center_lat") / grid_size).floor().alias("grid_y")
        ])

        # Aggregate by grid cell
        grid_aggregated = grid_df.group_by(["grid_x", "grid_y"]).agg([
            pl.count().alias("total_businesses"),
            pl.col("building_id").n_unique().alias("building_count"),
            pl.col("naics_code").n_unique().alias("business_type_diversity"),
            pl.col("building_center_lon").mean().alias("center_lon"),
            pl.col("building_center_lat").mean().alias("center_lat"),
            pl.col("building_category").mode().first().alias("dominant_building_type")
        ]).with_columns([
            # Grid-level metrics
            (pl.col("total_businesses") / pl.col("building_count")).alias("avg_businesses_per_building"),
            pl.lit(grid_size).alias("grid_size"),

            # Density category for grid
            pl.when(pl.col("total_businesses") >= 50).then("very_high")
            .when(pl.col("total_businesses") >= 25).then("high")
            .when(pl.col("total_businesses") >= 10).then("medium")
            .when(pl.col("total_businesses") >= 1).then("low")
            .otherwise("none").alias("density_category")
        ])

        logger.info(f"Grid aggregation complete: {len(grid_aggregated):,} grid cells")
        return grid_aggregated

    def generate_vector_tiles_data(self, building_aggregated: pl.DataFrame) -> Dict:
        """Generate data structure optimized for vector tile serving."""
        logger.info("Generating vector tile data structure...")

        # Convert to GeoJSON format for tiles
        features = []

        for building in building_aggregated.iter_rows(named=True):
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [building['center_lon'], building['center_lat']]
                },
                "properties": {
                    "building_id": building['building_id'],
                    "business_count": building['business_count'],
                    "density_category": building['density_category'],
                    "intensity_score": building['intensity_score'],
                    "building_type": building['building_type'],
                    "building_category": building['building_category'],
                    "mixed_use": building['mixed_use'],
                    "unique_business_types": building['unique_business_types']
                }
            }
            features.append(feature)

        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }

        return geojson_data

    def save_aggregated_data(self, building_aggregated: pl.DataFrame, grid_aggregated: pl.DataFrame):
        """Save aggregated data for web API use."""
        logger.info("Saving aggregated data...")

        # Save building-level aggregation
        building_path = self.output_dir / "buildings_with_businesses.parquet"
        building_aggregated.write_parquet(building_path)

        # Save grid-level aggregation
        grid_path = self.output_dir / "business_grid_aggregation.parquet"
        grid_aggregated.write_parquet(grid_path)

        # Generate and save GeoJSON for vector tiles
        geojson_data = self.generate_vector_tiles_data(building_aggregated)
        geojson_path = self.output_dir / "buildings_businesses.geojson"
        with open(geojson_path, 'w') as f:
            json.dump(geojson_data, f, separators=(',', ':'))

        # Save summary statistics
        stats = {
            "total_buildings_with_businesses": len(building_aggregated),
            "total_grid_cells": len(grid_aggregated),
            "avg_businesses_per_building": float(building_aggregated.select("business_count").mean().item()),
            "max_businesses_per_building": int(building_aggregated.select("business_count").max().item()),
            "density_distribution": building_aggregated.group_by("density_category").count().to_dicts()
        }

        stats_path = self.output_dir / "building_business_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved aggregated data to {building_path}, {grid_path}, and {geojson_path}")
        return stats

    def run_full_aggregation(self) -> Dict:
        """Run the complete building-business aggregation pipeline."""
        logger.info("Starting building-business aggregation pipeline...")

        # Load data
        businesses_df, buildings_df = self.load_data()

        # Extract building centroids
        buildings_with_centroids = self.extract_centroids(buildings_df)

        # Perform spatial join
        joined_df = self.spatial_join_proximity(businesses_df, buildings_with_centroids)

        if len(joined_df) == 0:
            logger.error("No spatial joins found - cannot proceed with aggregation")
            return {}

        # Aggregate by building
        building_aggregated = self.aggregate_by_building(joined_df)

        # Create grid aggregation for zoom levels
        grid_aggregated = self.create_grid_aggregation(joined_df)

        # Save results
        stats = self.save_aggregated_data(building_aggregated, grid_aggregated)

        logger.info("Building-business aggregation pipeline complete!")
        return stats

def main():
    """Run the aggregation as a standalone script."""
    aggregator = BuildingBusinessAggregator()
    stats = aggregator.run_full_aggregation()

    if stats:
        print("\n=== BUILDING-BUSINESS AGGREGATION RESULTS ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
    else:
        print("Aggregation failed!")

if __name__ == "__main__":
    main()