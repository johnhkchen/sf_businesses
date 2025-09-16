#!/usr/bin/env python3

import requests
import polars as pl
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OSMBuildingExtractor:
    """Extract building footprints from OpenStreetMap for San Francisco."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def setup_output(self):
        """Setup output directory for building data."""
        logger.info(f"Setup output directory: {self.output_dir}")

    def fetch_sf_buildings(self):
        """Fetch building data from Overpass API for San Francisco."""
        logger.info("Fetching building data from OpenStreetMap...")

        # San Francisco bounding box
        bbox = "37.7049,-122.5110,37.8089,-122.3816"

        overpass_query = f"""
        [out:json][timeout:300];
        (
          way["building"]({bbox});
          relation["building"]({bbox});
        );
        out geom;
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
            logger.info(f"Fetched {len(data['elements'])} building elements")
            return data['elements']

        except requests.RequestException as e:
            logger.error(f"Failed to fetch OSM data: {e}")
            return []

    def process_osm_buildings(self, elements):
        """Process OSM building elements into structured data."""
        logger.info("Processing OSM building data...")

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

                    geom_wkt = f"POLYGON(({','.join([f'{lon} {lat}' for lon, lat in coords])}))"

                    # Extract tags
                    tags = element.get('tags', {})

                    building = {
                        'osm_id': element['id'],
                        'osm_type': element['type'],
                        'building_type': tags.get('building', 'yes'),
                        'name': tags.get('name'),
                        'addr_housenumber': tags.get('addr:housenumber'),
                        'addr_street': tags.get('addr:street'),
                        'addr_city': tags.get('addr:city'),
                        'addr_postcode': tags.get('addr:postcode'),
                        'height': self._parse_height(tags.get('height')),
                        'levels': self._parse_levels(tags.get('building:levels')),
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

    def save_buildings_data(self, buildings):
        """Save building data to Parquet files."""
        logger.info("Saving buildings data...")

        if not buildings:
            logger.warning("No buildings to save")
            return None

        # Create DataFrame
        df = pl.DataFrame(buildings)

        # Save to Parquet format
        output_path = self.output_dir / "buildings.parquet"
        df.write_parquet(output_path)
        logger.info(f"Buildings data saved to {output_path}")

        # Create additional partitioned files
        building_types_path = self.output_dir / "buildings_by_type.parquet"
        df.filter(pl.col("building_type").is_not_null()).write_parquet(
            building_types_path,
            row_group_size=5000
        )

        return df

    def get_building_stats(self, df: pl.DataFrame):
        """Get statistics about loaded building data."""
        stats = df.select([
            pl.len().alias("total_buildings"),
            pl.col("building_type").n_unique().alias("unique_building_types"),
            pl.col("addr_street").filter(pl.col("addr_street").is_not_null()).len().alias("buildings_with_addresses"),
            pl.col("height").mean().alias("avg_height_meters"),
            pl.col("levels").mean().alias("avg_levels")
        ]).row(0, named=True)

        # Round floating point values
        if stats["avg_height_meters"] is not None:
            stats["avg_height_meters"] = round(stats["avg_height_meters"], 2)
        if stats["avg_levels"] is not None:
            stats["avg_levels"] = round(stats["avg_levels"], 2)

        return stats

def main():
    """Main building extraction execution."""
    extractor = OSMBuildingExtractor()

    # Setup output
    extractor.setup_output()

    # Fetch and process building data
    elements = extractor.fetch_sf_buildings()
    buildings = extractor.process_osm_buildings(elements)

    # Save to Parquet files
    df = extractor.save_buildings_data(buildings)

    if df is not None:
        # Print statistics
        stats = extractor.get_building_stats(df)
        print("\n=== BUILDING EXTRACTION RESULTS ===")
        for key, value in stats.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main()