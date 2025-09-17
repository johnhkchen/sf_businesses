#!/usr/bin/env python3
"""PostgreSQL connection management for SF businesses."""

import psycopg2
import psycopg2.extras
from contextlib import contextmanager
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PostgreSQLConnection:
    """Manage PostgreSQL connections for SF businesses database."""

    def __init__(self, config_path: str = "postgis_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load database configuration from file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration using socket connection
            return {
                'host': '/tmp/postgresql',
                'port': 5432,
                'database': 'sf_businesses',
                'user': 'johnchen',
                'password': None
            }

    @contextmanager
    def get_connection(self, autocommit: bool = False):
        """Get a database connection context manager."""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config.get('port', 5432),
                database=self.config['database'],
                user=self.config['user'],
                password=self.config.get('password')
            )

            if autocommit:
                conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

            yield conn

            if not autocommit:
                conn.commit()

        except Exception as e:
            if conn and not autocommit:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    @contextmanager
    def get_cursor(self, cursor_factory=None, autocommit: bool = False):
        """Get a database cursor context manager."""
        with self.get_connection(autocommit=autocommit) as conn:
            cursor = conn.cursor(cursor_factory=cursor_factory)
            try:
                yield cursor
            finally:
                cursor.close()

    @contextmanager
    def get_dict_cursor(self, autocommit: bool = False):
        """Get a dictionary cursor context manager."""
        with self.get_cursor(
            cursor_factory=psycopg2.extras.RealDictCursor,
            autocommit=autocommit
        ) as cursor:
            yield cursor

    def execute_query(self, query: str, params: Optional[tuple] = None, fetch: str = 'none'):
        """Execute a query and return results."""
        with self.get_dict_cursor() as cursor:
            cursor.execute(query, params)

            if fetch == 'one':
                return cursor.fetchone()
            elif fetch == 'all':
                return cursor.fetchall()
            elif fetch == 'many':
                return cursor.fetchmany()
            else:
                return None

    def execute_insert_batch(self, table: str, columns: list, data: list,
                           batch_size: int = 1000, on_conflict: str = 'ignore'):
        """Execute batch insert with conflict resolution."""
        if not data:
            return 0

        placeholders = ', '.join(['%s'] * len(columns))
        columns_str = ', '.join(columns)

        # Build conflict resolution clause
        if on_conflict == 'ignore':
            conflict_clause = "ON CONFLICT DO NOTHING"
        elif on_conflict == 'update':
            update_cols = ', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col != 'id'])
            conflict_clause = f"ON CONFLICT (unique_id) DO UPDATE SET {update_cols}"
        else:
            conflict_clause = ""

        query = f"""
            INSERT INTO {table} ({columns_str})
            VALUES ({placeholders})
            {conflict_clause}
        """

        total_inserted = 0

        with self.get_cursor() as cursor:
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                try:
                    cursor.executemany(query, batch)
                    total_inserted += cursor.rowcount
                    logger.info(f"Inserted batch {i//batch_size + 1}: {cursor.rowcount} rows")
                except Exception as e:
                    logger.error(f"Error inserting batch {i//batch_size + 1}: {e}")
                    raise

        return total_inserted

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = %s
            )
        """
        result = self.execute_query(query, (table_name,), fetch='one')
        return result['exists'] if result else False

    def get_table_row_count(self, table_name: str) -> int:
        """Get row count for a table."""
        if not self.table_exists(table_name):
            return 0

        query = f"SELECT COUNT(*) as count FROM {table_name}"
        result = self.execute_query(query, fetch='one')
        return result['count'] if result else 0

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information."""
        queries = {
            'version': "SELECT version()",
            'current_database': "SELECT current_database()",
            'current_user': "SELECT current_user",
            'business_count': "SELECT COUNT(*) FROM businesses",
            'building_count': "SELECT COUNT(*) FROM buildings"
        }

        info = {}
        for key, query in queries.items():
            try:
                if key in ['business_count', 'building_count']:
                    if self.table_exists(key.split('_')[0] + 's'):
                        result = self.execute_query(query, fetch='one')
                        info[key] = list(result.values())[0] if result else 0
                    else:
                        info[key] = 0
                else:
                    result = self.execute_query(query, fetch='one')
                    info[key] = list(result.values())[0] if result else None
            except Exception as e:
                info[key] = f"Error: {e}"

        return info


def main():
    """Test database connection."""
    db = PostgreSQLConnection()

    if db.test_connection():
        print("✅ Database connection successful!")

        info = db.get_database_info()
        print("\n📊 Database Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print("❌ Database connection failed!")

if __name__ == "__main__":
    main()