#!/usr/bin/env python3

import polars as pl
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging
from dataclasses import dataclass
import time
import json

logger = logging.getLogger(__name__)


@dataclass
class ZoomConfig:
    """Configuration for different zoom levels."""
    min_zoom: int
    max_zoom: int
    aggregation_type: str
    target_size_kb: int
    grid_size: float
    max_features: int


class ProgressiveDataLoader:
    """
    Progressive data loading system with multi-level aggregation strategy.

    Implements the data loading strategy:
    - Zoom 1-8: Hexagonal aggregation bins (<1KB)
    - Zoom 9-12: Statistical clusters (<10KB)
    - Zoom 13+: Individual buildings with businesses
    """

    def __init__(self, output_dir: Path = Path("output")):
        self.output_dir = output_dir
        self.cache_dir = output_dir / "cache" / "progressive"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Define zoom level configurations
        self.zoom_configs = {
            'hexagonal': ZoomConfig(1, 8, 'hexagonal', 1, 0.01, 200),
            'clusters': ZoomConfig(9, 12, 'clusters', 10, 0.001, 1000),
            'buildings': ZoomConfig(13, 18, 'buildings', 100, 0.0001, 5000)
        }

        self.businesses_df = None
        self.buildings_df = None
        self.load_base_data()

    def load_base_data(self):
        """Load base business and building data."""
        try:
            self.businesses_df = pl.read_parquet(self.output_dir / "businesses.parquet")
            self.buildings_df = pl.read_parquet(self.output_dir / "buildings.parquet")
            logger.info(f"Loaded {len(self.businesses_df)} businesses and {len(self.buildings_df)} buildings")
        except Exception as e:
            logger.error(f"Failed to load base data: {e}")
            raise

    def get_hexagonal_coordinates(self, lon: float, lat: float, size: float) -> Tuple[int, int]:
        """
        Convert latitude/longitude to hexagonal grid coordinates.

        Args:
            lon: Longitude
            lat: Latitude
            size: Hexagon size

        Returns:
            Tuple of hexagonal grid coordinates (q, r)
        """
        # Convert to hexagonal coordinates using axial coordinate system
        # Adapted from: https://www.redblobgames.com/grids/hexagons/

        # Scale and offset
        x = lon / size
        y = lat / size

        # Convert to cube coordinates
        q = (math.sqrt(3)/3 * x - 1/3 * y)
        r = (2/3 * y)

        # Round to nearest hex
        return (round(q), round(r))

    def create_hexagonal_aggregation(self, zoom_level: int, bbox: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
        """
        Create hexagonal aggregation for low zoom levels (1-8).

        Args:
            zoom_level: Map zoom level
            bbox: Optional bounding box filter

        Returns:
            Dict with hexagonal aggregated data
        """
        cache_file = self.cache_dir / f"hex_zoom_{zoom_level}.parquet"

        # Check cache first
        if cache_file.exists() and not bbox:
            try:
                cached_data = pl.read_parquet(cache_file)
                return self._format_hexagonal_response(cached_data, zoom_level)
            except Exception as e:
                logger.warning(f"Failed to load cached hexagonal data: {e}")

        # Calculate hexagon size based on zoom level
        hex_size = 0.1 / (2 ** (zoom_level - 1))  # Smaller hexagons at higher zoom

        businesses = self.businesses_df

        # Apply bounding box filter if provided
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            businesses = businesses.filter(
                (pl.col("longitude") >= min_lon) &
                (pl.col("longitude") <= max_lon) &
                (pl.col("latitude") >= min_lat) &
                (pl.col("latitude") <= max_lat)
            )

        # Add hexagonal coordinates
        hex_coords = []
        center_coords = []

        for row in businesses.select(["longitude", "latitude"]).iter_rows():
            lon, lat = row
            q, r = self.get_hexagonal_coordinates(lon, lat, hex_size)
            hex_coords.append((q, r))

            # Calculate hex center
            center_lon = q * hex_size * math.sqrt(3)
            center_lat = r * hex_size * 3/2
            center_coords.append((center_lon, center_lat))

        # Add coordinates to dataframe
        businesses_with_hex = businesses.with_columns([
            pl.Series("hex_q", [coord[0] for coord in hex_coords]),
            pl.Series("hex_r", [coord[1] for coord in hex_coords]),
            pl.Series("hex_center_lon", [coord[0] for coord in center_coords]),
            pl.Series("hex_center_lat", [coord[1] for coord in center_coords])
        ])

        # Aggregate by hexagon
        hex_aggregated = businesses_with_hex.group_by(["hex_q", "hex_r"]).agg([
            pl.len().alias("business_count"),
            pl.col("naics_code").n_unique().alias("business_types"),
            pl.col("naics_description").first().alias("primary_naics"),
            pl.col("hex_center_lon").first().alias("center_lon"),
            pl.col("hex_center_lat").first().alias("center_lat"),
            pl.col("longitude").mean().alias("avg_lon"),
            pl.col("latitude").mean().alias("avg_lat")
        ]).with_columns([
            pl.lit(hex_size).alias("hex_size"),
            pl.lit(zoom_level).alias("zoom_level"),
            pl.lit("hexagonal").alias("aggregation_type")
        ])

        # Limit results for performance
        config = self.zoom_configs['hexagonal']
        hex_aggregated = hex_aggregated.head(config.max_features)

        # Cache if no bbox filter
        if not bbox:
            try:
                hex_aggregated.write_parquet(cache_file)
                logger.info(f"Cached hexagonal data for zoom {zoom_level}")
            except Exception as e:
                logger.warning(f"Failed to cache hexagonal data: {e}")

        return self._format_hexagonal_response(hex_aggregated, zoom_level)

    def _format_hexagonal_response(self, data: pl.DataFrame, zoom_level: int) -> Dict[str, Any]:
        """Format hexagonal aggregated data for API response."""
        features = []

        for row in data.iter_rows(named=True):
            features.append({
                "type": "hexagon",
                "hex_q": row["hex_q"],
                "hex_r": row["hex_r"],
                "center_lon": row["avg_lon"],  # Use actual business center
                "center_lat": row["avg_lat"],
                "business_count": row["business_count"],
                "business_types": row["business_types"],
                "primary_naics": row["primary_naics"],
                "hex_size": row["hex_size"]
            })

        return {
            "type": "hexagonal",
            "zoom_level": zoom_level,
            "features": features,
            "total_features": len(features),
            "data_size_estimate_kb": len(features) * 0.05  # Rough estimate
        }

    def create_statistical_clusters(self, zoom_level: int, bbox: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
        """
        Create statistical clusters for medium zoom levels (9-12).

        Args:
            zoom_level: Map zoom level
            bbox: Optional bounding box filter

        Returns:
            Dict with clustered data
        """
        cache_file = self.cache_dir / f"clusters_zoom_{zoom_level}.parquet"

        # Check cache first
        if cache_file.exists() and not bbox:
            try:
                cached_data = pl.read_parquet(cache_file)
                return self._format_clusters_response(cached_data, zoom_level)
            except Exception as e:
                logger.warning(f"Failed to load cached cluster data: {e}")

        # Use existing grid logic but with smaller cells
        config = self.zoom_configs['clusters']
        grid_size = config.grid_size * (15 - zoom_level) / 10  # Adaptive grid size

        businesses = self.businesses_df

        # Apply bounding box filter if provided
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            businesses = businesses.filter(
                (pl.col("longitude") >= min_lon) &
                (pl.col("longitude") <= max_lon) &
                (pl.col("latitude") >= min_lat) &
                (pl.col("latitude") <= max_lat)
            )

        # Create grid-based clusters with statistical metrics
        businesses_with_grid = businesses.with_columns([
            (pl.col("longitude") / grid_size).round(0).alias("grid_x"),
            (pl.col("latitude") / grid_size).round(0).alias("grid_y")
        ])

        clusters = businesses_with_grid.group_by(["grid_x", "grid_y"]).agg([
            pl.len().alias("business_count"),
            pl.col("naics_code").n_unique().alias("business_types"),
            pl.col("naics_description").first().alias("primary_naics"),
            pl.col("longitude").mean().alias("center_lon"),
            pl.col("latitude").mean().alias("center_lat"),
            pl.col("longitude").std().alias("lon_std"),
            pl.col("latitude").std().alias("lat_std"),
            pl.col("naics_code").mode().first().alias("dominant_naics"),
            pl.col("business_start_date").max().alias("newest_business"),
            pl.col("business_start_date").min().alias("oldest_business")
        ]).with_columns([
            pl.lit(grid_size).alias("grid_size"),
            pl.lit(zoom_level).alias("zoom_level"),
            pl.lit("statistical").alias("aggregation_type"),
            (pl.col("lon_std").fill_null(0) + pl.col("lat_std").fill_null(0)).alias("spatial_variance")
        ])

        # Limit results
        clusters = clusters.head(config.max_features)

        # Cache if no bbox filter
        if not bbox:
            try:
                clusters.write_parquet(cache_file)
                logger.info(f"Cached cluster data for zoom {zoom_level}")
            except Exception as e:
                logger.warning(f"Failed to cache cluster data: {e}")

        return self._format_clusters_response(clusters, zoom_level)

    def _format_clusters_response(self, data: pl.DataFrame, zoom_level: int) -> Dict[str, Any]:
        """Format statistical cluster data for API response."""
        features = []

        for row in data.iter_rows(named=True):
            features.append({
                "type": "cluster",
                "grid_x": row["grid_x"],
                "grid_y": row["grid_y"],
                "center_lon": row["center_lon"],
                "center_lat": row["center_lat"],
                "business_count": row["business_count"],
                "business_types": row["business_types"],
                "primary_naics": row["primary_naics"],
                "dominant_naics": row["dominant_naics"],
                "spatial_variance": row.get("spatial_variance", 0),
                "newest_business": str(row.get("newest_business", "")),
                "oldest_business": str(row.get("oldest_business", "")),
                "grid_size": row["grid_size"]
            })

        return {
            "type": "statistical_clusters",
            "zoom_level": zoom_level,
            "features": features,
            "total_features": len(features),
            "data_size_estimate_kb": len(features) * 0.5  # Rough estimate
        }

    def get_progressive_data(self, zoom_level: int, bbox: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
        """
        Main method to get appropriately aggregated data based on zoom level.

        Args:
            zoom_level: Map zoom level
            bbox: Optional bounding box filter

        Returns:
            Dict with appropriately aggregated data
        """
        start_time = time.time()

        try:
            if zoom_level <= 8:
                # Hexagonal aggregation for low zoom
                result = self.create_hexagonal_aggregation(zoom_level, bbox)
            elif zoom_level <= 12:
                # Statistical clusters for medium zoom
                result = self.create_statistical_clusters(zoom_level, bbox)
            else:
                # Individual buildings for high zoom (fallback to existing logic)
                result = self._get_individual_buildings(zoom_level, bbox)

            # Add performance metrics
            processing_time = time.time() - start_time
            result["performance"] = {
                "processing_time_ms": round(processing_time * 1000, 2),
                "target_met": processing_time < 1.0,  # 1 second target
                "cache_hit": processing_time < 0.1
            }

            return result

        except Exception as e:
            logger.error(f"Error in progressive data loading: {e}")
            raise

    def _get_individual_buildings(self, zoom_level: int, bbox: Optional[Tuple[float, float, float, float]] = None) -> Dict[str, Any]:
        """Get individual building data for high zoom levels."""
        # This uses existing logic from web_api.py but with enhancements
        config = self.zoom_configs['buildings']

        buildings = self.buildings_df

        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            # Use pre-computed centroids if available
            try:
                buildings_enriched = pl.read_parquet(self.output_dir / "buildings_enriched_quick.parquet")
                buildings = buildings_enriched.filter(
                    (pl.col("center_lon") >= min_lon) &
                    (pl.col("center_lon") <= max_lon) &
                    (pl.col("center_lat") >= min_lat) &
                    (pl.col("center_lat") <= max_lat)
                ).head(config.max_features)

                return {
                    "type": "individual_buildings",
                    "zoom_level": zoom_level,
                    "features": buildings.to_dicts(),
                    "total_features": len(buildings),
                    "data_size_estimate_kb": len(buildings) * 0.02
                }
            except:
                pass

        # Fallback to basic building data
        return {
            "type": "individual_buildings",
            "zoom_level": zoom_level,
            "features": [],
            "total_features": 0,
            "data_size_estimate_kb": 0
        }

    def precompute_all_zoom_levels(self):
        """Precompute aggregations for all zoom levels."""
        logger.info("Starting precomputation of all zoom levels...")

        results = {}

        # Precompute hexagonal aggregations
        for zoom in range(1, 9):
            logger.info(f"Precomputing hexagonal aggregation for zoom {zoom}")
            result = self.create_hexagonal_aggregation(zoom)
            results[f"hex_zoom_{zoom}"] = {
                "features": result["total_features"],
                "size_kb": result["data_size_estimate_kb"]
            }

        # Precompute statistical clusters
        for zoom in range(9, 13):
            logger.info(f"Precomputing statistical clusters for zoom {zoom}")
            result = self.create_statistical_clusters(zoom)
            results[f"clusters_zoom_{zoom}"] = {
                "features": result["total_features"],
                "size_kb": result["data_size_estimate_kb"]
            }

        # Save precomputation summary
        summary_file = self.cache_dir / "precomputation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results,
                "total_cache_files": len(results)
            }, f, indent=2)

        logger.info(f"Precomputation complete. Summary saved to {summary_file}")
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize progressive loader
    loader = ProgressiveDataLoader()

    # Test different zoom levels
    test_bbox = (-122.5, 37.7, -122.3, 37.8)  # SF area

    print("Testing Progressive Data Loading...")

    for zoom in [3, 6, 10, 14]:
        print(f"\n=== Zoom Level {zoom} ===")
        result = loader.get_progressive_data(zoom, test_bbox)
        print(f"Type: {result['type']}")
        print(f"Features: {result['total_features']}")
        print(f"Estimated size: {result['data_size_estimate_kb']:.2f} KB")
        print(f"Processing time: {result['performance']['processing_time_ms']:.2f} ms")
        print(f"Target met: {result['performance']['target_met']}")