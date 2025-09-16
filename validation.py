#!/usr/bin/env python3

import polars as pl
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation engine for SF business and building data."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # SF geographic boundaries
        self.sf_bounds = {
            'lat_min': 37.7049,
            'lat_max': 37.8089,
            'lon_min': -122.5110,
            'lon_max': -122.3816
        }

        # Data quality thresholds
        self.quality_thresholds = {
            'min_completeness_rate': 0.80,
            'max_duplicate_rate': 0.05,
            'max_outlier_rate': 0.02
        }

    def validate_business_data_completeness(self, df: pl.DataFrame) -> Dict:
        """Check data completeness for required business fields."""
        logger.info("Validating business data completeness...")

        required_fields = ['unique_id', 'ownership_name', 'dba_name', 'street_address',
                          'longitude', 'latitude', 'naics_code']

        total_rows = len(df)
        completeness_results = {}

        for field in required_fields:
            if field in df.columns:
                non_null_count = df.select(pl.col(field).is_not_null().sum()).item()
                completeness_rate = non_null_count / total_rows if total_rows > 0 else 0
                completeness_results[field] = {
                    'non_null_count': non_null_count,
                    'total_count': total_rows,
                    'completeness_rate': completeness_rate,
                    'passes_threshold': completeness_rate >= self.quality_thresholds['min_completeness_rate']
                }

        return {
            'field_completeness': completeness_results,
            'overall_completeness': sum(r['completeness_rate'] for r in completeness_results.values()) / len(completeness_results)
        }

    def validate_coordinates(self, df: pl.DataFrame) -> Dict:
        """Validate geographic coordinates for San Francisco area."""
        logger.info("Validating coordinate data...")

        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return {'error': 'Latitude or longitude columns missing'}

        # Check for valid coordinate format
        coord_validation = df.select([
            pl.len().alias('total_records'),
            pl.col('latitude').is_not_null().sum().alias('lat_non_null'),
            pl.col('longitude').is_not_null().sum().alias('lon_non_null'),
            (pl.col('latitude').is_between(self.sf_bounds['lat_min'], self.sf_bounds['lat_max'])).sum().alias('lat_in_sf'),
            (pl.col('longitude').is_between(self.sf_bounds['lon_min'], self.sf_bounds['lon_max'])).sum().alias('lon_in_sf'),
            ((pl.col('latitude').is_between(self.sf_bounds['lat_min'], self.sf_bounds['lat_max'])) &
             (pl.col('longitude').is_between(self.sf_bounds['lon_min'], self.sf_bounds['lon_max']))).sum().alias('coords_in_sf')
        ]).row(0, named=True)

        total = coord_validation['total_records']
        coords_in_sf = coord_validation['coords_in_sf']
        sf_coverage_rate = coords_in_sf / total if total > 0 else 0

        return {
            'total_records': total,
            'coordinates_in_sf': coords_in_sf,
            'sf_coverage_rate': sf_coverage_rate,
            'latitude_completeness': coord_validation['lat_non_null'] / total if total > 0 else 0,
            'longitude_completeness': coord_validation['lon_non_null'] / total if total > 0 else 0,
            'passes_geographic_validation': sf_coverage_rate >= 0.95
        }

    def validate_naics_codes(self, df: pl.DataFrame) -> Dict:
        """Validate NAICS code format and coverage."""
        logger.info("Validating NAICS codes...")

        if 'naics_code' not in df.columns:
            return {'error': 'NAICS code column missing'}

        naics_stats = df.select([
            pl.len().alias('total_records'),
            pl.col('naics_code').is_not_null().sum().alias('naics_non_null'),
            pl.col('naics_code').n_unique().alias('unique_naics_codes'),
            # NAICS codes should be 2-6 digits
            pl.col('naics_code').str.len_chars().is_between(2, 6).sum().alias('valid_naics_format')
        ]).row(0, named=True)

        total = naics_stats['total_records']
        valid_naics = naics_stats['valid_naics_format']
        naics_coverage = naics_stats['naics_non_null'] / total if total > 0 else 0

        return {
            'total_records': total,
            'naics_coverage_rate': naics_coverage,
            'unique_naics_count': naics_stats['unique_naics_codes'],
            'valid_format_rate': valid_naics / total if total > 0 else 0,
            'passes_naics_validation': naics_coverage >= 0.80
        }

    def detect_business_duplicates(self, df: pl.DataFrame) -> Dict:
        """Detect potential duplicate business records."""
        logger.info("Detecting duplicate business records...")

        # Check for exact duplicates by key fields
        duplicate_checks = {}

        # Duplicate by business account number
        if 'business_account_number' in df.columns:
            account_dupes = df.select([
                pl.len().alias('total'),
                pl.col('business_account_number').n_unique().alias('unique_accounts')
            ]).row(0, named=True)
            duplicate_checks['account_duplicates'] = {
                'total_records': account_dupes['total'],
                'unique_accounts': account_dupes['unique_accounts'],
                'duplicate_rate': 1 - (account_dupes['unique_accounts'] / account_dupes['total']) if account_dupes['total'] > 0 else 0
            }

        # Duplicate by name + address combination
        if all(col in df.columns for col in ['dba_name', 'street_address']):
            name_address_dupes = df.select([
                pl.len().alias('total'),
                pl.concat_str([pl.col('dba_name'), pl.col('street_address')], separator='|').n_unique().alias('unique_name_address')
            ]).row(0, named=True)
            duplicate_checks['name_address_duplicates'] = {
                'total_records': name_address_dupes['total'],
                'unique_combinations': name_address_dupes['unique_name_address'],
                'duplicate_rate': 1 - (name_address_dupes['unique_name_address'] / name_address_dupes['total']) if name_address_dupes['total'] > 0 else 0
            }

        return duplicate_checks

    def detect_coordinate_outliers(self, df: pl.DataFrame) -> Dict:
        """Detect coordinate outliers and anomalies."""
        logger.info("Detecting coordinate outliers...")

        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return {'error': 'Coordinate columns missing'}

        # Statistical outlier detection using IQR method
        coord_stats = df.select([
            pl.col('latitude').quantile(0.25).alias('lat_q1'),
            pl.col('latitude').quantile(0.75).alias('lat_q3'),
            pl.col('longitude').quantile(0.25).alias('lon_q1'),
            pl.col('longitude').quantile(0.75).alias('lon_q3'),
            pl.col('latitude').mean().alias('lat_mean'),
            pl.col('longitude').mean().alias('lon_mean'),
            pl.len().alias('total_records')
        ]).row(0, named=True)

        # Calculate IQR bounds
        lat_iqr = coord_stats['lat_q3'] - coord_stats['lat_q1']
        lon_iqr = coord_stats['lon_q3'] - coord_stats['lon_q1']

        lat_lower = coord_stats['lat_q1'] - 1.5 * lat_iqr
        lat_upper = coord_stats['lat_q3'] + 1.5 * lat_iqr
        lon_lower = coord_stats['lon_q1'] - 1.5 * lon_iqr
        lon_upper = coord_stats['lon_q3'] + 1.5 * lon_iqr

        outliers = df.select([
            ((pl.col('latitude') < lat_lower) | (pl.col('latitude') > lat_upper) |
             (pl.col('longitude') < lon_lower) | (pl.col('longitude') > lon_upper)).sum().alias('outlier_count')
        ]).item()

        outlier_rate = outliers / coord_stats['total_records'] if coord_stats['total_records'] > 0 else 0

        return {
            'total_records': coord_stats['total_records'],
            'outlier_count': outliers,
            'outlier_rate': outlier_rate,
            'latitude_bounds': (lat_lower, lat_upper),
            'longitude_bounds': (lon_lower, lon_upper),
            'coordinate_center': (coord_stats['lat_mean'], coord_stats['lon_mean']),
            'passes_outlier_check': outlier_rate <= self.quality_thresholds['max_outlier_rate']
        }

    def validate_building_geometries(self, df: pl.DataFrame) -> Dict:
        """Validate building polygon geometries."""
        logger.info("Validating building geometries...")

        if 'geometry' not in df.columns:
            return {'error': 'Geometry column missing'}

        geometry_stats = df.select([
            pl.len().alias('total_buildings'),
            pl.col('geometry').is_not_null().sum().alias('non_null_geometries'),
            # Check for basic WKT format (starts with POLYGON)
            pl.col('geometry').str.starts_with('POLYGON').sum().alias('polygon_format_count')
        ]).row(0, named=True)

        total = geometry_stats['total_buildings']
        valid_polygons = geometry_stats['polygon_format_count']

        return {
            'total_buildings': total,
            'geometry_completeness': geometry_stats['non_null_geometries'] / total if total > 0 else 0,
            'valid_polygon_rate': valid_polygons / total if total > 0 else 0,
            'passes_geometry_validation': (valid_polygons / total) >= 0.90 if total > 0 else False
        }

    def validate_address_format(self, df: pl.DataFrame) -> Dict:
        """Validate address format and standardization."""
        logger.info("Validating address formats...")

        if 'street_address' not in df.columns:
            return {'error': 'Street address column missing'}

        address_patterns = df.select([
            pl.len().alias('total_records'),
            pl.col('street_address').is_not_null().sum().alias('non_null_addresses'),
            # Basic pattern: starts with number
            pl.col('street_address').str.contains(r'^\d+').sum().alias('starts_with_number'),
            # Contains street type (ST, AVE, BLVD, etc.)
            pl.col('street_address').str.contains(r'(?i)(street|st|avenue|ave|boulevard|blvd|road|rd|drive|dr|lane|ln|way|ct|circle|pl|place)').sum().alias('has_street_type')
        ]).row(0, named=True)

        total = address_patterns['total_records']
        valid_format = address_patterns['starts_with_number']
        has_street_type = address_patterns['has_street_type']

        return {
            'total_records': total,
            'address_completeness': address_patterns['non_null_addresses'] / total if total > 0 else 0,
            'valid_format_rate': valid_format / total if total > 0 else 0,
            'street_type_coverage': has_street_type / total if total > 0 else 0,
            'passes_format_validation': (valid_format / total) >= 0.85 if total > 0 else False
        }

    def run_comprehensive_validation(self, business_df: Optional[pl.DataFrame] = None,
                                   building_df: Optional[pl.DataFrame] = None) -> Dict:
        """Run all validation checks and return comprehensive report."""
        logger.info("Running comprehensive data validation...")

        from datetime import datetime
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'business_validation': {},
            'building_validation': {},
            'overall_score': 0.0
        }

        scores = []

        # Business data validation
        if business_df is not None:
            logger.info("Validating business data...")
            business_results = {}

            # Completeness
            completeness = self.validate_business_data_completeness(business_df)
            business_results['completeness'] = completeness
            scores.append(completeness['overall_completeness'])

            # Coordinates
            coordinates = self.validate_coordinates(business_df)
            business_results['coordinates'] = coordinates
            if 'sf_coverage_rate' in coordinates:
                scores.append(coordinates['sf_coverage_rate'])

            # NAICS codes
            naics = self.validate_naics_codes(business_df)
            business_results['naics'] = naics
            if 'naics_coverage_rate' in naics:
                scores.append(naics['naics_coverage_rate'])

            # Duplicates
            duplicates = self.detect_business_duplicates(business_df)
            business_results['duplicates'] = duplicates

            # Outliers
            outliers = self.detect_coordinate_outliers(business_df)
            business_results['outliers'] = outliers
            if 'passes_outlier_check' in outliers:
                scores.append(1.0 if outliers['passes_outlier_check'] else 0.5)

            # Address format
            addresses = self.validate_address_format(business_df)
            business_results['addresses'] = addresses
            if 'address_completeness' in addresses:
                scores.append(addresses['address_completeness'])

            validation_report['business_validation'] = business_results

        # Building data validation
        if building_df is not None:
            logger.info("Validating building data...")
            building_results = {}

            # Geometries
            geometries = self.validate_building_geometries(building_df)
            building_results['geometries'] = geometries
            if 'valid_polygon_rate' in geometries:
                scores.append(geometries['valid_polygon_rate'])

            validation_report['building_validation'] = building_results

        # Calculate overall score
        validation_report['overall_score'] = sum(scores) / len(scores) if scores else 0.0
        validation_report['total_checks'] = len(scores)

        # Save validation report
        self.save_validation_report(validation_report)

        return validation_report

    def save_validation_report(self, report: Dict):
        """Save validation report to JSON file."""
        import json

        report_path = self.output_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Validation report saved to {report_path}")

def main():
    """Standalone validation execution."""
    validator = DataValidator()

    # Load data if available
    business_df = None
    building_df = None

    business_path = Path("output/businesses.parquet")
    building_path = Path("output/buildings.parquet")

    if business_path.exists():
        business_df = pl.read_parquet(business_path)
        logger.info(f"Loaded {len(business_df):,} business records")

    if building_path.exists():
        building_df = pl.read_parquet(building_path)
        logger.info(f"Loaded {len(building_df):,} building records")

    # Run validation
    report = validator.run_comprehensive_validation(business_df, building_df)

    print("\n=== VALIDATION RESULTS ===")
    print(f"Overall Quality Score: {report['overall_score']:.2%}")
    print(f"Total Checks: {report['total_checks']}")

    if business_df is not None:
        print(f"\nBusiness Data Completeness: {report['business_validation']['completeness']['overall_completeness']:.2%}")
        print(f"SF Geographic Coverage: {report['business_validation']['coordinates']['sf_coverage_rate']:.2%}")

    if building_df is not None and 'geometries' in report['building_validation']:
        geometry_results = report['building_validation']['geometries']
        if 'valid_polygon_rate' in geometry_results:
            print(f"\nBuilding Geometry Validity: {geometry_results['valid_polygon_rate']:.2%}")
        else:
            print(f"\nBuilding Geometry Validation: {geometry_results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()