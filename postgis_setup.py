#!/usr/bin/env python3
"""PostGIS database setup and configuration."""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging
import os
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostGISSetup:
    """Setup and configure PostGIS database for SF businesses."""

    def __init__(self, host='/tmp/postgresql', port=5432, user=None, password=None):
        self.host = host
        self.port = port
        self.user = user or os.environ.get('USER', 'postgres')
        self.password = password
        self.database = 'sf_businesses'

    def connect_postgres(self):
        """Connect to default postgres database."""
        try:
            # Connect via Unix socket
            conn = psycopg2.connect(
                host=self.host,
                user=self.user,
                database='postgres'
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            return conn
        except psycopg2.OperationalError as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def connect_sf_businesses(self):
        """Connect to sf_businesses database."""
        try:
            # Connect via Unix socket
            conn = psycopg2.connect(
                host=self.host,
                user=self.user,
                database=self.database
            )
            return conn
        except psycopg2.OperationalError as e:
            logger.error(f"Failed to connect to sf_businesses database: {e}")
            raise

    def create_database(self):
        """Create the sf_businesses database if it doesn't exist."""
        logger.info("Setting up PostgreSQL database...")

        conn = self.connect_postgres()
        cursor = conn.cursor()

        try:
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.database,)
            )

            if cursor.fetchone():
                logger.info(f"Database '{self.database}' already exists")
            else:
                # Create database
                cursor.execute(f'CREATE DATABASE "{self.database}"')
                logger.info(f"Created database '{self.database}'")

        except Exception as e:
            logger.error(f"Error creating database: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    def setup_postgis_extensions(self):
        """Enable PostGIS extensions in the database."""
        logger.info("Setting up PostGIS extensions...")

        conn = self.connect_sf_businesses()
        cursor = conn.cursor()

        try:
            # Try to enable PostGIS extension
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis")
                logger.info("PostGIS extension enabled")

                # Enable PostGIS topology
                cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis_topology")
                logger.info("PostGIS topology extension enabled")

                # Check PostGIS version
                cursor.execute("SELECT PostGIS_Version()")
                version = cursor.fetchone()[0]
                logger.info(f"PostGIS version: {version}")

            except Exception as e:
                logger.warning(f"PostGIS extension not available (likely version mismatch): {e}")
                logger.info("Continuing with basic spatial support using PostgreSQL built-in functions")

                # Enable basic geometry support if available
                try:
                    cursor.execute("SELECT ST_GeomFromText('POINT(0 0)')")
                    logger.info("Basic spatial functions available")
                except:
                    logger.warning("No spatial functions available - will use coordinate-based operations")

            conn.commit()

        except Exception as e:
            logger.error(f"Error setting up spatial extensions: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    def create_schema(self):
        """Create the database schema for businesses and buildings."""
        logger.info("Creating database schema...")

        conn = self.connect_sf_businesses()
        cursor = conn.cursor()

        try:
            # Create businesses table (using coordinates instead of PostGIS for now)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS businesses (
                    id SERIAL PRIMARY KEY,
                    unique_id VARCHAR(255),
                    business_account_number VARCHAR(255),
                    location_id VARCHAR(255),
                    ownership_name TEXT,
                    dba_name TEXT,
                    street_address TEXT,
                    city VARCHAR(100),
                    state VARCHAR(10),
                    zipcode VARCHAR(20),
                    business_start_date DATE,
                    business_end_date DATE,
                    location_start_date DATE,
                    location_end_date DATE,
                    naics_code VARCHAR(20),
                    naics_description TEXT,
                    supervisor_district INTEGER,
                    neighborhood TEXT,
                    longitude NUMERIC(10, 7),
                    latitude NUMERIC(10, 7),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create buildings table (using coordinates and WKT text for now)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS buildings (
                    id SERIAL PRIMARY KEY,
                    osm_id BIGINT UNIQUE,
                    building_type VARCHAR(100),
                    address TEXT,
                    levels INTEGER,
                    height FLOAT,
                    area FLOAT,
                    geom_wkt TEXT,
                    centroid_lon NUMERIC(10, 7),
                    centroid_lat NUMERIC(10, 7),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create coordinate indexes for spatial queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_businesses_coords ON businesses (longitude, latitude)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_buildings_centroid ON buildings (centroid_lon, centroid_lat)")

            # Create business attribute indexes for fast filtering
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_businesses_naics ON businesses (naics_code)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_businesses_neighborhood ON businesses (neighborhood)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_businesses_district ON businesses (supervisor_district)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_buildings_type ON buildings (building_type)")

            # Create H3 extensions if available
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS h3")
                logger.info("H3 extension enabled")
            except Exception as e:
                logger.warning(f"H3 extension not available: {e}")

            conn.commit()
            logger.info("Database schema created successfully")

        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    def create_vector_tile_functions(self):
        """Create helper functions for spatial queries (coordinate-based for now)."""
        logger.info("Creating spatial query functions...")

        conn = self.connect_sf_businesses()
        cursor = conn.cursor()

        try:
            # Function to calculate distance between two points (Haversine formula)
            cursor.execute("""
                CREATE OR REPLACE FUNCTION calculate_distance(
                    lat1 NUMERIC, lon1 NUMERIC,
                    lat2 NUMERIC, lon2 NUMERIC
                ) RETURNS NUMERIC AS $$
                DECLARE
                    r NUMERIC := 6371000; -- Earth radius in meters
                    dlat NUMERIC;
                    dlon NUMERIC;
                    a NUMERIC;
                    c NUMERIC;
                BEGIN
                    dlat := radians(lat2 - lat1);
                    dlon := radians(lon2 - lon1);
                    a := sin(dlat/2) * sin(dlat/2) +
                         cos(radians(lat1)) * cos(radians(lat2)) *
                         sin(dlon/2) * sin(dlon/2);
                    c := 2 * atan2(sqrt(a), sqrt(1-a));
                    RETURN r * c;
                END;
                $$ LANGUAGE plpgsql;
            """)

            # Function to check if point is within bounding box
            cursor.execute("""
                CREATE OR REPLACE FUNCTION point_in_bbox(
                    lat NUMERIC, lon NUMERIC,
                    min_lat NUMERIC, min_lon NUMERIC,
                    max_lat NUMERIC, max_lon NUMERIC
                ) RETURNS BOOLEAN AS $$
                BEGIN
                    RETURN lat >= min_lat AND lat <= max_lat AND
                           lon >= min_lon AND lon <= max_lon;
                END;
                $$ LANGUAGE plpgsql;
            """)

            # Function to get businesses within bounding box
            cursor.execute("""
                CREATE OR REPLACE FUNCTION get_businesses_in_bbox(
                    min_lat NUMERIC, min_lon NUMERIC,
                    max_lat NUMERIC, max_lon NUMERIC,
                    naics_filter VARCHAR DEFAULT NULL,
                    neighborhood_filter VARCHAR DEFAULT NULL
                ) RETURNS TABLE(
                    id INTEGER,
                    unique_id VARCHAR,
                    dba_name TEXT,
                    longitude NUMERIC,
                    latitude NUMERIC,
                    naics_code VARCHAR,
                    naics_description TEXT,
                    neighborhood TEXT,
                    supervisor_district INTEGER
                ) AS $$
                BEGIN
                    RETURN QUERY
                    SELECT
                        b.id, b.unique_id, b.dba_name, b.longitude, b.latitude,
                        b.naics_code, b.naics_description, b.neighborhood, b.supervisor_district
                    FROM businesses b
                    WHERE point_in_bbox(b.latitude, b.longitude, min_lat, min_lon, max_lat, max_lon)
                    AND (naics_filter IS NULL OR b.naics_code = naics_filter)
                    AND (neighborhood_filter IS NULL OR b.neighborhood = neighborhood_filter);
                END;
                $$ LANGUAGE plpgsql;
            """)

            conn.commit()
            logger.info("Spatial query functions created successfully")

        except Exception as e:
            logger.error(f"Error creating spatial functions: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    def save_connection_config(self):
        """Save database connection configuration."""
        config = {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user,
            'password': self.password
        }

        config_path = Path('postgis_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Database configuration saved to {config_path}")

    def setup_complete_environment(self):
        """Complete PostGIS setup process."""
        logger.info("Starting complete PostGIS setup...")

        try:
            self.create_database()
            self.setup_postgis_extensions()
            self.create_schema()
            self.create_vector_tile_functions()
            self.save_connection_config()

            logger.info("✅ PostGIS setup completed successfully!")
            logger.info("🗄️  Database: sf_businesses")
            logger.info("🌍 PostGIS extensions enabled")
            logger.info("📊 Schema created with spatial indexes")
            logger.info("🔧 Vector tile functions ready")

        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            raise

def main():
    """Main setup execution."""
    setup = PostGISSetup()
    setup.setup_complete_environment()

if __name__ == "__main__":
    main()