#!/usr/bin/env python3

import requests
import polars as pl
import json
import pickle
import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
from shapely.geometry import Polygon
from shapely import wkt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OSMBuildingExtractor:
    """Extract building footprints from OpenStreetMap for San Francisco."""

    def __init__(self, output_dir: str = "output", cache_dir: str = "output/cache", simplify_tolerance: float = 0.00001):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.simplify_tolerance = simplify_tolerance
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        # Enhanced building type classification
        self.building_type_mapping = {
            'house': 'residential',
            'apartments': 'residential',
            'residential': 'residential',
            'detached': 'residential',
            'terrace': 'residential',
            'dormitory': 'residential',
            'hotel': 'commercial',
            'office': 'commercial',
            'commercial': 'commercial',
            'retail': 'commercial',
            'shop': 'commercial',
            'supermarket': 'commercial',
            'warehouse': 'industrial',
            'industrial': 'industrial',
            'factory': 'industrial',
            'manufacture': 'industrial',
            'school': 'institutional',
            'hospital': 'institutional',
            'church': 'institutional',
            'university': 'institutional',
            'public': 'institutional',
            'government': 'institutional',
            'civic': 'institutional',
            'garage': 'utility',
            'garages': 'utility',
            'parking': 'utility',
            'service': 'utility',
            'utility': 'utility',
            'yes': 'unknown'
        }

    def setup_output(self):
        """Setup output directory for building data."""
        logger.info(f"Setup output directory: {self.output_dir}")

    def _get_cache_key(self, bbox: str) -> str:
        """Generate cache key for OSM query."""
        query_hash = hashlib.md5(bbox.encode()).hexdigest()
        return f"osm_buildings_{query_hash}"

    def _is_cache_valid(self, cache_file: Path, max_age_hours: int = 24) -> bool:
        """Check if cache file is valid and not expired."""
        if not cache_file.exists():
            return False

        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < timedelta(hours=max_age_hours)

    def _save_to_cache(self, data: dict, cache_key: str):
        """Save OSM data to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamp': datetime.now(),
                    'bbox': data.get('bbox', 'unknown')
                }, f)
            logger.info(f"Cached OSM data to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")

    def _load_from_cache(self, cache_key: str) -> dict:
        """Load OSM data from cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                logger.info(f"Loaded OSM data from cache ({cache_file})")
                return cached_data['data']
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return {}

    def fetch_sf_buildings(self, use_cache: bool = True, max_cache_age_hours: int = 24):
        """Fetch building data from Overpass API for San Francisco with caching support."""
        logger.info("Fetching building data from OpenStreetMap...")

        # San Francisco bounding box
        bbox = "37.7049,-122.5110,37.8089,-122.3816"
        cache_key = self._get_cache_key(bbox)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        # Try to load from cache first
        if use_cache and self._is_cache_valid(cache_file, max_cache_age_hours):
            cached_data = self._load_from_cache(cache_key)
            if cached_data and 'elements' in cached_data:
                logger.info(f"Using cached data with {len(cached_data['elements'])} building elements")
                return cached_data['elements']

        # Fetch fresh data from API
        overpass_query = f"""
        [out:json][timeout:300];
        (
          way["building"]({bbox});
          relation["building"]({bbox});
        );
        out geom meta;
        """

        overpass_url = "http://overpass-api.de/api/interpreter"

        try:
            response = requests.post(
                overpass_url,
                data=overpass_query,
                timeout=300
            )
            response.raise_for_status()
            data = response.json()
            data['bbox'] = bbox  # Store bbox for cache metadata
            logger.info(f"Fetched {len(data['elements'])} building elements")

            # Cache the fresh data
            if use_cache:
                self._save_to_cache(data, cache_key)

            return data['elements']

        except requests.RequestException as e:
            logger.error(f"Failed to fetch OSM data: {e}")
            # Try to fall back to cache even if expired
            if use_cache and cache_file.exists():
                logger.info("Falling back to expired cache data")
                cached_data = self._load_from_cache(cache_key)
                if cached_data and 'elements' in cached_data:
                    return cached_data['elements']
            return []

    def _simplify_polygon(self, coords: list) -> str:
        """Simplify polygon geometry using Shapely with Douglas-Peucker algorithm."""
        try:
            # Create Shapely polygon
            polygon = Polygon(coords)

            # Simplify using Douglas-Peucker algorithm
            simplified = polygon.simplify(self.simplify_tolerance, preserve_topology=True)

            # Convert back to WKT
            return simplified.wkt
        except Exception as e:
            # Fallback to original polygon if simplification fails
            logger.warning(f"Polygon simplification failed: {e}, using original")
            return f"POLYGON(({','.join([f'{lon} {lat}' for lon, lat in coords])}))"

    def _classify_building_type(self, building_type: str) -> tuple:
        """Enhanced building type classification."""
        if not building_type:
            return 'unknown', 'unknown'

        building_type_lower = building_type.lower()
        category = self.building_type_mapping.get(building_type_lower, 'other')
        return building_type, category

    def _estimate_building_age(self, tags: dict, timestamp: str = None) -> dict:
        """Estimate building age from OSM metadata."""
        age_info = {
            'construction_year': None,
            'estimated_age_years': None,
            'age_source': None
        }

        # Direct construction year from tags
        construction_tags = ['start_date', 'construction_date', 'building:construction_date']
        for tag in construction_tags:
            if tag in tags:
                try:
                    year_str = tags[tag]
                    # Extract year from various formats (YYYY, YYYY-MM-DD, etc.)
                    import re
                    year_match = re.search(r'\b(19|20)\d{2}\b', year_str)
                    if year_match:
                        construction_year = int(year_match.group())
                        age_info['construction_year'] = construction_year
                        age_info['estimated_age_years'] = 2024 - construction_year
                        age_info['age_source'] = tag
                        break
                except (ValueError, TypeError):
                    continue

        # Estimate from building style or historical context
        if not age_info['construction_year']:
            style_age_estimates = {
                'victorian': 1890,
                'edwardian': 1905,
                'art_deco': 1925,
                'modernist': 1955,
                'contemporary': 1990
            }

            building_style = tags.get('architectural_style', '').lower()
            if building_style in style_age_estimates:
                estimated_year = style_age_estimates[building_style]
                age_info['construction_year'] = estimated_year
                age_info['estimated_age_years'] = 2024 - estimated_year
                age_info['age_source'] = 'architectural_style'

        return age_info

    def process_osm_buildings(self, elements, enable_simplification: bool = True):
        """Process OSM building elements into structured data with enhancements."""
        logger.info("Processing OSM building data with enhancements...")

        buildings = []

        for element in elements:
            if element['type'] not in ['way', 'relation']:
                continue

            # Extract geometry
            if element['type'] == 'way' and 'geometry' in element:
                coords = [[node['lon'], node['lat']] for node in element['geometry']]
                if len(coords) >= 4:  # Valid polygon needs at least 4 points
                    # Close the polygon if not already closed
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])

                    # Apply polygon simplification if enabled
                    if enable_simplification:
                        geom_wkt = self._simplify_polygon(coords)
                    else:
                        geom_wkt = f"POLYGON(({','.join([f'{lon} {lat}' for lon, lat in coords])}))"

                    # Extract tags
                    tags = element.get('tags', {})

                    # Enhanced building type classification
                    building_type, building_category = self._classify_building_type(tags.get('building', 'yes'))

                    # Building age estimation
                    age_info = self._estimate_building_age(tags, element.get('timestamp'))

                    building = {
                        'osm_id': element['id'],
                        'osm_type': element['type'],
                        'building_type': building_type,
                        'building_category': building_category,
                        'name': tags.get('name'),
                        'addr_housenumber': tags.get('addr:housenumber'),
                        'addr_street': tags.get('addr:street'),
                        'addr_city': tags.get('addr:city'),
                        'addr_postcode': tags.get('addr:postcode'),
                        'height': self._parse_height(tags.get('height')),
                        'levels': self._parse_levels(tags.get('building:levels')),
                        'construction_year': age_info['construction_year'],
                        'estimated_age_years': age_info['estimated_age_years'],
                        'age_source': age_info['age_source'],
                        'architectural_style': tags.get('architectural_style'),
                        'last_modified': element.get('timestamp'),
                        'version': element.get('version'),
                        'geom_wkt': geom_wkt
                    }

                    buildings.append(building)

        logger.info(f"Processed {len(buildings)} valid building polygons")
        return buildings

    def _parse_height(self, height_str):
        """Parse height string to float."""
        if not height_str:
            return None
        try:
            # Handle formats like "10.5 m", "35 ft", "12"
            import re
            match = re.search(r'(\d+\.?\d*)', height_str)
            if match:
                height = float(match.group(1))
                if 'ft' in height_str.lower():
                    height = height * 0.3048  # Convert feet to meters
                return height
        except (ValueError, AttributeError):
            pass
        return None

    def _parse_levels(self, levels_str):
        """Parse building levels to integer."""
        if not levels_str:
            return None
        try:
            return int(float(levels_str))
        except (ValueError, TypeError):
            return None

    def save_buildings_data(self, buildings, optimize_storage: bool = True):
        """Save building data to optimized Parquet files."""
        logger.info("Saving buildings data with optimizations...")

        if not buildings:
            logger.warning("No buildings to save")
            return None

        # Create DataFrame
        df = pl.DataFrame(buildings)

        # Optimize storage format
        if optimize_storage:
            # Use compression and optimize column types
            compression_options = {
                'compression': 'zstd',  # Better compression than snappy
                'compression_level': 3   # Good balance of speed vs compression
            }

            # Optimize data types for storage
            df = df.with_columns([
                pl.col('osm_id').cast(pl.Int64),
                pl.col('version').cast(pl.Int32),
                pl.col('construction_year').cast(pl.Int32),
                pl.col('estimated_age_years').cast(pl.Int32),
                pl.col('levels').cast(pl.Int32),
                pl.col('height').cast(pl.Float32)
            ])
        else:
            compression_options = {}

        # Save main buildings file
        output_path = self.output_dir / "buildings.parquet"
        df.write_parquet(output_path, **compression_options)
        logger.info(f"Buildings data saved to {output_path}")

        # Create partitioned files for efficient querying
        building_types_path = self.output_dir / "buildings_by_type.parquet"
        df.filter(pl.col("building_type").is_not_null()).write_parquet(
            building_types_path,
            row_group_size=5000,
            **compression_options
        )

        # Save buildings by category for faster category-based queries
        building_categories_path = self.output_dir / "buildings_by_category.parquet"
        df.filter(pl.col("building_category").is_not_null()).write_parquet(
            building_categories_path,
            row_group_size=5000,
            **compression_options
        )

        # Save timestamped incremental data for future updates
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        incremental_path = self.output_dir / f"buildings_incremental_{timestamp}.parquet"
        df.write_parquet(incremental_path, **compression_options)
        logger.info(f"Incremental snapshot saved to {incremental_path}")

        return df

    def get_building_stats(self, df: pl.DataFrame):
        """Get enhanced statistics about loaded building data."""
        stats = df.select([
            pl.len().alias("total_buildings"),
            pl.col("building_type").n_unique().alias("unique_building_types"),
            pl.col("building_category").n_unique().alias("unique_building_categories"),
            pl.col("addr_street").filter(pl.col("addr_street").is_not_null()).len().alias("buildings_with_addresses"),
            pl.col("height").mean().alias("avg_height_meters"),
            pl.col("levels").mean().alias("avg_levels"),
            pl.col("construction_year").filter(pl.col("construction_year").is_not_null()).len().alias("buildings_with_age_data"),
            pl.col("estimated_age_years").mean().alias("avg_age_years"),
            pl.col("architectural_style").filter(pl.col("architectural_style").is_not_null()).len().alias("buildings_with_style_data")
        ]).row(0, named=True)

        # Round floating point values
        for key in ["avg_height_meters", "avg_levels", "avg_age_years"]:
            if stats[key] is not None:
                stats[key] = round(stats[key], 2)

        # Add category breakdown
        category_stats = df.group_by("building_category").agg([
            pl.len().alias("count")
        ]).sort("count", descending=True)

        stats["category_breakdown"] = category_stats.to_dicts()

        return stats

    def check_for_updates(self, last_update_file: str = "last_update.txt") -> bool:
        """Check if incremental updates are needed based on timestamp."""
        last_update_path = self.output_dir / last_update_file

        if not last_update_path.exists():
            return True  # First run, need full update

        try:
            with open(last_update_path, 'r') as f:
                last_update_str = f.read().strip()
                last_update = datetime.fromisoformat(last_update_str)

            # Check if more than 7 days since last update
            time_diff = datetime.now() - last_update
            return time_diff > timedelta(days=7)

        except (ValueError, FileNotFoundError):
            return True

    def update_timestamp(self, last_update_file: str = "last_update.txt"):
        """Update the last update timestamp."""
        last_update_path = self.output_dir / last_update_file

        with open(last_update_path, 'w') as f:
            f.write(datetime.now().isoformat())

        logger.info(f"Updated timestamp to {last_update_path}")

def main():
    """Main building extraction execution with enhancements."""
    # Initialize with enhanced configuration
    extractor = OSMBuildingExtractor(
        simplify_tolerance=0.00001  # ~1 meter tolerance for SF
    )

    # Setup output
    extractor.setup_output()

    # Check if updates are needed (for incremental sync)
    needs_update = extractor.check_for_updates()

    if needs_update:
        logger.info("Updates needed, fetching fresh data...")
        # Fetch and process building data with caching
        elements = extractor.fetch_sf_buildings(use_cache=True, max_cache_age_hours=24)
        buildings = extractor.process_osm_buildings(elements, enable_simplification=True)

        # Save to optimized Parquet files
        df = extractor.save_buildings_data(buildings, optimize_storage=True)

        # Update timestamp
        extractor.update_timestamp()
    else:
        logger.info("Data is up to date, loading from existing files...")
        # Load existing data
        buildings_path = extractor.output_dir / "buildings.parquet"
        if buildings_path.exists():
            df = pl.read_parquet(buildings_path)
        else:
            logger.warning("No existing data found, forcing fresh extraction...")
            elements = extractor.fetch_sf_buildings(use_cache=True)
            buildings = extractor.process_osm_buildings(elements, enable_simplification=True)
            df = extractor.save_buildings_data(buildings, optimize_storage=True)
            extractor.update_timestamp()

    if df is not None:
        # Print enhanced statistics
        stats = extractor.get_building_stats(df)
        print("\n=== ENHANCED BUILDING EXTRACTION RESULTS ===")
        for key, value in stats.items():
            if key == "category_breakdown":
                print(f"\n{key}:")
                for category in value[:5]:  # Show top 5 categories
                    print(f"  {category['building_category']}: {category['count']}")
            else:
                print(f"{key}: {value}")

if __name__ == "__main__":
    main()