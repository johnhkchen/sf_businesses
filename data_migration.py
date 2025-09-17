#!/usr/bin/env python3
"""Data migration from Parquet files to PostgreSQL."""

import polars as pl
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from postgres_connection import PostgreSQLConnection
import time
from shapely import wkt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataMigrator:
    """Migrate data from Parquet files to PostgreSQL database."""

    def __init__(self, input_dir: str = "output"):
        self.input_dir = Path(input_dir)
        self.db = PostgreSQLConnection()

    def load_parquet_data(self) -> tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """Load business and building data from Parquet files."""
        logger.info("Loading data from Parquet files...")

        businesses_path = self.input_dir / "businesses.parquet"
        buildings_path = self.input_dir / "buildings.parquet"

        businesses_df = None
        buildings_df = None

        if businesses_path.exists():
            businesses_df = pl.read_parquet(businesses_path)
            logger.info(f"Loaded {len(businesses_df):,} businesses from {businesses_path}")
        else:
            logger.warning(f"Businesses file not found: {businesses_path}")

        if buildings_path.exists():
            buildings_df = pl.read_parquet(buildings_path)
            logger.info(f"Loaded {len(buildings_df):,} buildings from {buildings_path}")
        else:
            logger.warning(f"Buildings file not found: {buildings_path}")

        return businesses_df, buildings_df

    def extract_building_centroid(self, wkt_str: str) -> tuple[Optional[float], Optional[float]]:
        """Extract centroid from building WKT geometry."""
        try:
            if not wkt_str or wkt_str == "null":
                return None, None
            geom = wkt.loads(wkt_str)
            c = geom.centroid
            return float(c.x), float(c.y)
        except Exception:
            return None, None

    def prepare_business_data(self, df: pl.DataFrame) -> List[tuple]:
        """Prepare business data for PostgreSQL insertion."""
        logger.info("Preparing business data for insertion...")

        # Convert Polars DataFrame to list of tuples for batch insertion
        records = []

        for row in df.iter_rows(named=True):
            record = (
                row.get('unique_id'),
                row.get('business_account_number'),
                row.get('location_id'),
                row.get('ownership_name'),
                row.get('dba_name'),
                row.get('street_address'),
                row.get('city'),
                row.get('state'),
                row.get('zipcode'),
                row.get('business_start_date'),
                row.get('business_end_date'),
                row.get('location_start_date'),
                row.get('location_end_date'),
                row.get('naics_code'),
                row.get('naics_description'),
                row.get('supervisor_district'),
                row.get('neighborhood'),
                float(row.get('longitude')) if row.get('longitude') is not None else None,
                float(row.get('latitude')) if row.get('latitude') is not None else None
            )
            records.append(record)

        logger.info(f"Prepared {len(records):,} business records")
        return records

    def prepare_building_data(self, df: pl.DataFrame) -> List[tuple]:
        """Prepare building data for PostgreSQL insertion."""
        logger.info("Preparing building data for insertion...")

        records = []

        for row in df.iter_rows(named=True):
            # Extract centroid from geometry
            geom_wkt = row.get('geom_wkt')
            centroid_lon, centroid_lat = self.extract_building_centroid(geom_wkt)

            record = (
                row.get('osm_id'),
                row.get('building_type'),
                row.get('address'),
                row.get('levels'),
                row.get('height'),
                row.get('area'),
                geom_wkt,
                centroid_lon,
                centroid_lat
            )
            records.append(record)

        logger.info(f"Prepared {len(records):,} building records")
        return records

    def migrate_businesses(self, df: pl.DataFrame) -> int:
        """Migrate business data to PostgreSQL."""
        logger.info("Migrating business data to PostgreSQL...")

        if len(df) == 0:
            logger.warning("No business data to migrate")
            return 0

        # Prepare data
        records = self.prepare_business_data(df)

        # Define column order
        columns = [
            'unique_id', 'business_account_number', 'location_id',
            'ownership_name', 'dba_name', 'street_address', 'city', 'state', 'zipcode',
            'business_start_date', 'business_end_date', 'location_start_date', 'location_end_date',
            'naics_code', 'naics_description', 'supervisor_district', 'neighborhood',
            'longitude', 'latitude'
        ]

        # Insert data
        start_time = time.time()
        inserted_count = self.db.execute_insert_batch(
            table='businesses',
            columns=columns,
            data=records,
            batch_size=1000,
            on_conflict='ignore'
        )
        elapsed_time = time.time() - start_time

        logger.info(f"✅ Migrated {inserted_count:,} businesses in {elapsed_time:.2f} seconds")
        return inserted_count

    def migrate_buildings(self, df: pl.DataFrame) -> int:
        """Migrate building data to PostgreSQL."""
        logger.info("Migrating building data to PostgreSQL...")

        if len(df) == 0:
            logger.warning("No building data to migrate")
            return 0

        # Prepare data
        records = self.prepare_building_data(df)

        # Define column order
        columns = [
            'osm_id', 'building_type', 'address', 'levels', 'height', 'area',
            'geom_wkt', 'centroid_lon', 'centroid_lat'
        ]

        # Insert data
        start_time = time.time()
        inserted_count = self.db.execute_insert_batch(
            table='buildings',
            columns=columns,
            data=records,
            batch_size=1000,
            on_conflict='ignore'
        )
        elapsed_time = time.time() - start_time

        logger.info(f"✅ Migrated {inserted_count:,} buildings in {elapsed_time:.2f} seconds")
        return inserted_count

    def create_spatial_indexes(self):
        """Create additional spatial indexes for performance."""
        logger.info("Creating spatial indexes...")

        indexes = [
            # Business indexes
            "CREATE INDEX IF NOT EXISTS idx_businesses_bbox ON businesses (latitude, longitude)",
            "CREATE INDEX IF NOT EXISTS idx_businesses_lat ON businesses (latitude)",
            "CREATE INDEX IF NOT EXISTS idx_businesses_lon ON businesses (longitude)",

            # Building indexes
            "CREATE INDEX IF NOT EXISTS idx_buildings_bbox ON buildings (centroid_lat, centroid_lon)",
            "CREATE INDEX IF NOT EXISTS idx_buildings_lat ON buildings (centroid_lat)",
            "CREATE INDEX IF NOT EXISTS idx_buildings_lon ON buildings (centroid_lon)",

            # Attribute indexes for filtering
            "CREATE INDEX IF NOT EXISTS idx_businesses_naics_hash ON businesses USING HASH (naics_code)",
            "CREATE INDEX IF NOT EXISTS idx_businesses_neighborhood_hash ON businesses USING HASH (neighborhood)",
            "CREATE INDEX IF NOT EXISTS idx_buildings_type_hash ON buildings USING HASH (building_type)",
        ]

        for index_query in indexes:
            try:
                self.db.execute_query(index_query)
                logger.info(f"Created index: {index_query.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                logger.warning(f"Failed to create index: {e}")

    def verify_migration(self) -> Dict[str, Any]:
        """Verify the migration was successful."""
        logger.info("Verifying migration...")

        verification = {}

        # Check row counts
        verification['businesses_count'] = self.db.get_table_row_count('businesses')
        verification['buildings_count'] = self.db.get_table_row_count('buildings')

        # Check sample data
        sample_business = self.db.execute_query(
            "SELECT * FROM businesses WHERE longitude IS NOT NULL AND latitude IS NOT NULL LIMIT 1",
            fetch='one'
        )
        verification['has_business_coordinates'] = sample_business is not None

        sample_building = self.db.execute_query(
            "SELECT * FROM buildings WHERE centroid_lon IS NOT NULL AND centroid_lat IS NOT NULL LIMIT 1",
            fetch='one'
        )
        verification['has_building_coordinates'] = sample_building is not None

        # Check spatial function
        try:
            distance_test = self.db.execute_query(
                "SELECT calculate_distance(37.7749, -122.4194, 37.7849, -122.4094) as distance",
                fetch='one'
            )
            verification['spatial_functions_working'] = distance_test is not None
            verification['distance_test_result'] = float(distance_test['distance']) if distance_test else None
        except Exception as e:
            verification['spatial_functions_working'] = False
            verification['spatial_function_error'] = str(e)

        return verification

    def run_migration(self, force: bool = False):
        """Run the complete data migration process."""
        logger.info("🚀 Starting data migration from Parquet to PostgreSQL...")

        # Check if data already exists
        existing_businesses = self.db.get_table_row_count('businesses')
        existing_buildings = self.db.get_table_row_count('buildings')

        if (existing_businesses > 0 or existing_buildings > 0) and not force:
            logger.warning(f"Data already exists (businesses: {existing_businesses:,}, buildings: {existing_buildings:,})")
            logger.info("Use force=True to overwrite existing data")
            return

        # Load source data
        businesses_df, buildings_df = self.load_parquet_data()

        migration_results = {}

        # Migrate businesses
        if businesses_df is not None:
            migration_results['businesses'] = self.migrate_businesses(businesses_df)
        else:
            migration_results['businesses'] = 0

        # Migrate buildings
        if buildings_df is not None:
            migration_results['buildings'] = self.migrate_buildings(buildings_df)
        else:
            migration_results['buildings'] = 0

        # Create spatial indexes
        self.create_spatial_indexes()

        # Verify migration
        verification = self.verify_migration()

        # Report results
        logger.info("🎉 Migration completed!")
        logger.info(f"📊 Migration Results:")
        logger.info(f"   Businesses migrated: {migration_results['businesses']:,}")
        logger.info(f"   Buildings migrated: {migration_results['buildings']:,}")

        logger.info(f"✅ Verification Results:")
        for key, value in verification.items():
            logger.info(f"   {key}: {value}")

        return migration_results, verification


def main():
    """Main migration execution."""
    migrator = DataMigrator()

    # Run migration
    migrator.run_migration(force=False)

if __name__ == "__main__":
    main()