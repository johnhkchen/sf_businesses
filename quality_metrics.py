#!/usr/bin/env python3

import polars as pl
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityMetrics:
    """Data quality scoring and metrics system for SF business pipeline."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Quality scoring weights
        self.scoring_weights = {
            'completeness': 0.25,
            'accuracy': 0.25,
            'consistency': 0.20,
            'validity': 0.20,
            'uniqueness': 0.10
        }

        # Quality thresholds for different grades
        self.quality_grades = {
            'excellent': 0.95,
            'good': 0.85,
            'fair': 0.70,
            'poor': 0.50
        }

    def calculate_completeness_score(self, df: pl.DataFrame, required_fields: List[str]) -> Dict:
        """Calculate data completeness score."""
        logger.info("Calculating completeness score...")

        if not required_fields:
            return {'score': 0.0, 'details': 'No required fields specified'}

        field_scores = {}
        total_rows = len(df)

        for field in required_fields:
            if field in df.columns:
                non_null_count = df.select(pl.col(field).is_not_null().sum()).item()
                completeness_rate = non_null_count / total_rows if total_rows > 0 else 0
                field_scores[field] = completeness_rate

        overall_score = sum(field_scores.values()) / len(field_scores) if field_scores else 0.0

        return {
            'score': overall_score,
            'field_scores': field_scores,
            'total_fields': len(required_fields),
            'average_completeness': overall_score
        }

    def calculate_accuracy_score(self, df: pl.DataFrame) -> Dict:
        """Calculate data accuracy score based on format validation and constraints."""
        logger.info("Calculating accuracy score...")

        accuracy_checks = []
        total_rows = len(df)

        # Geographic accuracy (coordinates within SF bounds)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            sf_bounds = {
                'lat_min': 37.7049, 'lat_max': 37.8089,
                'lon_min': -122.5110, 'lon_max': -122.3816
            }

            valid_coords = df.select([
                ((pl.col('latitude').is_between(sf_bounds['lat_min'], sf_bounds['lat_max'])) &
                 (pl.col('longitude').is_between(sf_bounds['lon_min'], sf_bounds['lon_max']))).sum()
            ]).item()

            geo_accuracy = valid_coords / total_rows if total_rows > 0 else 0
            accuracy_checks.append(('geographic_accuracy', geo_accuracy))

        # NAICS code format accuracy
        if 'naics_code' in df.columns:
            valid_naics = df.select([
                pl.col('naics_code').str.len_chars().is_between(2, 6).sum()
            ]).item()

            naics_accuracy = valid_naics / total_rows if total_rows > 0 else 0
            accuracy_checks.append(('naics_format_accuracy', naics_accuracy))

        # Address format accuracy
        if 'street_address' in df.columns:
            valid_addresses = df.select([
                pl.col('street_address').str.contains(r'^\d+').sum()
            ]).item()

            address_accuracy = valid_addresses / total_rows if total_rows > 0 else 0
            accuracy_checks.append(('address_format_accuracy', address_accuracy))

        # Email format accuracy (if exists)
        if 'email' in df.columns:
            valid_emails = df.select([
                pl.col('email').str.contains(r'^[^@]+@[^@]+\.[^@]+$').sum()
            ]).item()

            email_accuracy = valid_emails / total_rows if total_rows > 0 else 0
            accuracy_checks.append(('email_format_accuracy', email_accuracy))

        overall_accuracy = sum(score for _, score in accuracy_checks) / len(accuracy_checks) if accuracy_checks else 0.0

        return {
            'score': overall_accuracy,
            'individual_checks': dict(accuracy_checks),
            'total_checks': len(accuracy_checks)
        }

    def calculate_consistency_score(self, df: pl.DataFrame) -> Dict:
        """Calculate data consistency score."""
        logger.info("Calculating consistency score...")

        consistency_checks = []

        # Name consistency (DBA vs Ownership Name)
        if 'dba_name' in df.columns and 'ownership_name' in df.columns:
            name_consistency = df.select([
                (pl.col('dba_name').str.to_lowercase() == pl.col('ownership_name').str.to_lowercase()).sum()
            ]).item()

            total_with_both = df.select([
                (pl.col('dba_name').is_not_null() & pl.col('ownership_name').is_not_null()).sum()
            ]).item()

            name_consistency_rate = name_consistency / total_with_both if total_with_both > 0 else 1.0
            consistency_checks.append(('name_consistency', name_consistency_rate))

        # Date consistency (start date <= end date)
        if 'business_start_date' in df.columns and 'business_end_date' in df.columns:
            valid_date_ranges = df.select([
                (pl.col('business_start_date') <= pl.col('business_end_date')).sum()
            ]).item()

            total_with_both_dates = df.select([
                (pl.col('business_start_date').is_not_null() & pl.col('business_end_date').is_not_null()).sum()
            ]).item()

            date_consistency_rate = valid_date_ranges / total_with_both_dates if total_with_both_dates > 0 else 1.0
            consistency_checks.append(('date_consistency', date_consistency_rate))

        # Address-coordinate consistency (rough check)
        if all(col in df.columns for col in ['street_address', 'latitude', 'longitude', 'zipcode']):
            # Check for same zipcode having reasonable coordinate proximity
            zipcode_groups = df.group_by('zipcode').agg([
                pl.col('latitude').std().alias('lat_std'),
                pl.col('longitude').std().alias('lon_std'),
                pl.len().alias('count')
            ]).filter(pl.col('count') >= 5)  # Only check zipcodes with 5+ businesses

            if len(zipcode_groups) > 0:
                # Good consistency = low standard deviation within zipcodes
                avg_lat_std = zipcode_groups.select(pl.col('lat_std').mean()).item()
                avg_lon_std = zipcode_groups.select(pl.col('lon_std').mean()).item()

                # Normalize: lower std = higher consistency (rough heuristic)
                coord_consistency = max(0, 1 - (avg_lat_std + avg_lon_std) * 100)
                consistency_checks.append(('coordinate_consistency', coord_consistency))

        overall_consistency = sum(score for _, score in consistency_checks) / len(consistency_checks) if consistency_checks else 1.0

        return {
            'score': overall_consistency,
            'individual_checks': dict(consistency_checks),
            'total_checks': len(consistency_checks)
        }

    def calculate_validity_score(self, df: pl.DataFrame) -> Dict:
        """Calculate data validity score based on business rules."""
        logger.info("Calculating validity score...")

        validity_checks = []
        total_rows = len(df)

        # Business dates validity (not in future, reasonable start dates)
        if 'business_start_date' in df.columns:
            current_date = datetime.now()
            # Reasonable business start date range (not before 1850, not in future)
            min_date = datetime(1850, 1, 1)

            valid_start_dates = df.select([
                (pl.col('business_start_date').is_between(min_date, current_date)).sum()
            ]).item()

            total_with_start_date = df.select([
                pl.col('business_start_date').is_not_null().sum()
            ]).item()

            start_date_validity = valid_start_dates / total_with_start_date if total_with_start_date > 0 else 1.0
            validity_checks.append(('start_date_validity', start_date_validity))

        # NAICS code validity (exists in standard list - simplified check)
        if 'naics_code' in df.columns:
            # Basic validity: numeric codes in reasonable range
            valid_naics_format = df.select([
                pl.col('naics_code').str.contains(r'^\d{2,6}$').sum()
            ]).item()

            total_with_naics = df.select([
                pl.col('naics_code').is_not_null().sum()
            ]).item()

            naics_validity = valid_naics_format / total_with_naics if total_with_naics > 0 else 1.0
            validity_checks.append(('naics_validity', naics_validity))

        # Coordinate precision validity (not overly precise, which indicates fake data)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Check for suspiciously precise coordinates (too many decimal places)
            reasonable_precision = df.select([
                (pl.col('latitude').round(6) == pl.col('latitude')).sum(),
                (pl.col('longitude').round(6) == pl.col('longitude')).sum()
            ])

            # Most GPS coordinates shouldn't need more than 6 decimal places
            lat_precision_ok = reasonable_precision.item(0, 0)
            lon_precision_ok = reasonable_precision.item(0, 1)

            coord_validity = min(lat_precision_ok, lon_precision_ok) / total_rows if total_rows > 0 else 1.0
            validity_checks.append(('coordinate_precision_validity', coord_validity))

        overall_validity = sum(score for _, score in validity_checks) / len(validity_checks) if validity_checks else 1.0

        return {
            'score': overall_validity,
            'individual_checks': dict(validity_checks),
            'total_checks': len(validity_checks)
        }

    def calculate_uniqueness_score(self, df: pl.DataFrame) -> Dict:
        """Calculate data uniqueness score (inverse of duplicate rate)."""
        logger.info("Calculating uniqueness score...")

        uniqueness_checks = []
        total_rows = len(df)

        # Business account number uniqueness
        if 'business_account_number' in df.columns:
            unique_accounts = df.select(pl.col('business_account_number').n_unique()).item()
            account_uniqueness = unique_accounts / total_rows if total_rows > 0 else 1.0
            uniqueness_checks.append(('account_uniqueness', account_uniqueness))

        # Name + address combination uniqueness
        if 'dba_name' in df.columns and 'street_address' in df.columns:
            unique_combinations = df.select([
                pl.concat_str([pl.col('dba_name'), pl.col('street_address')], separator='|').n_unique()
            ]).item()

            name_address_uniqueness = unique_combinations / total_rows if total_rows > 0 else 1.0
            uniqueness_checks.append(('name_address_uniqueness', name_address_uniqueness))

        # Coordinate uniqueness (to detect bulk/fake data)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            unique_coordinates = df.select([
                pl.concat_str([pl.col('latitude'), pl.col('longitude')], separator=',').n_unique()
            ]).item()

            coord_uniqueness = unique_coordinates / total_rows if total_rows > 0 else 1.0
            uniqueness_checks.append(('coordinate_uniqueness', coord_uniqueness))

        overall_uniqueness = sum(score for _, score in uniqueness_checks) / len(uniqueness_checks) if uniqueness_checks else 1.0

        return {
            'score': overall_uniqueness,
            'individual_checks': dict(uniqueness_checks),
            'total_checks': len(uniqueness_checks)
        }

    def calculate_overall_quality_score(self, df: pl.DataFrame, required_fields: List[str] = None) -> Dict:
        """Calculate comprehensive quality score using all dimensions."""
        logger.info("Calculating overall quality score...")

        if required_fields is None:
            required_fields = ['unique_id', 'dba_name', 'street_address', 'latitude', 'longitude', 'naics_code']

        # Calculate individual dimension scores
        completeness = self.calculate_completeness_score(df, required_fields)
        accuracy = self.calculate_accuracy_score(df)
        consistency = self.calculate_consistency_score(df)
        validity = self.calculate_validity_score(df)
        uniqueness = self.calculate_uniqueness_score(df)

        # Calculate weighted overall score
        overall_score = (
            completeness['score'] * self.scoring_weights['completeness'] +
            accuracy['score'] * self.scoring_weights['accuracy'] +
            consistency['score'] * self.scoring_weights['consistency'] +
            validity['score'] * self.scoring_weights['validity'] +
            uniqueness['score'] * self.scoring_weights['uniqueness']
        )

        # Determine quality grade
        quality_grade = 'poor'
        for grade, threshold in sorted(self.quality_grades.items(), key=lambda x: x[1], reverse=True):
            if overall_score >= threshold:
                quality_grade = grade
                break

        return {
            'overall_score': overall_score,
            'quality_grade': quality_grade,
            'dimension_scores': {
                'completeness': completeness,
                'accuracy': accuracy,
                'consistency': consistency,
                'validity': validity,
                'uniqueness': uniqueness
            },
            'scoring_weights': self.scoring_weights,
            'total_records': len(df),
            'timestamp': datetime.now().isoformat()
        }

    def generate_quality_trends(self, historical_scores: List[Dict]) -> Dict:
        """Generate quality trend analysis from historical scores."""
        logger.info("Generating quality trends...")

        if len(historical_scores) < 2:
            return {'error': 'Need at least 2 historical scores for trend analysis'}

        # Convert to DataFrame for easier analysis
        trends_df = pl.DataFrame([
            {
                'timestamp': score['timestamp'],
                'overall_score': score['overall_score'],
                'completeness': score['dimension_scores']['completeness']['score'],
                'accuracy': score['dimension_scores']['accuracy']['score'],
                'consistency': score['dimension_scores']['consistency']['score'],
                'validity': score['dimension_scores']['validity']['score'],
                'uniqueness': score['dimension_scores']['uniqueness']['score']
            }
            for score in historical_scores
        ])

        # Calculate trends (simple linear trend)
        latest_score = historical_scores[-1]['overall_score']
        previous_score = historical_scores[-2]['overall_score']
        score_change = latest_score - previous_score

        trend_direction = 'improving' if score_change > 0.01 else 'declining' if score_change < -0.01 else 'stable'

        return {
            'latest_score': latest_score,
            'previous_score': previous_score,
            'score_change': score_change,
            'trend_direction': trend_direction,
            'historical_data': trends_df.to_dicts(),
            'analysis_period': len(historical_scores)
        }

    def create_quality_alerts(self, quality_score: Dict, alert_thresholds: Dict = None) -> List[Dict]:
        """Generate quality alerts based on thresholds."""
        logger.info("Creating quality alerts...")

        if alert_thresholds is None:
            alert_thresholds = {
                'overall_score': 0.70,
                'completeness': 0.80,
                'accuracy': 0.85,
                'validity': 0.80
            }

        alerts = []

        # Overall score alert
        if quality_score['overall_score'] < alert_thresholds['overall_score']:
            alerts.append({
                'type': 'overall_quality',
                'severity': 'high',
                'message': f"Overall quality score ({quality_score['overall_score']:.2%}) is below threshold ({alert_thresholds['overall_score']:.2%})",
                'current_value': quality_score['overall_score'],
                'threshold': alert_thresholds['overall_score']
            })

        # Dimension-specific alerts
        dimensions = quality_score['dimension_scores']
        for dimension, threshold in alert_thresholds.items():
            if dimension in dimensions and dimensions[dimension]['score'] < threshold:
                alerts.append({
                    'type': dimension,
                    'severity': 'medium',
                    'message': f"{dimension.title()} score ({dimensions[dimension]['score']:.2%}) is below threshold ({threshold:.2%})",
                    'current_value': dimensions[dimension]['score'],
                    'threshold': threshold
                })

        return alerts

    def save_quality_metrics(self, quality_score: Dict, filename: str = None):
        """Save quality metrics to JSON file."""
        import json

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quality_metrics_{timestamp}.json"

        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(quality_score, f, indent=2, default=str)

        logger.info(f"Quality metrics saved to {output_path}")
        return output_path

def main():
    """Standalone quality metrics execution."""
    metrics = QualityMetrics()

    # Load business data if available
    business_path = Path("output/businesses.parquet")
    if not business_path.exists():
        logger.error("No business data found. Run the pipeline first.")
        return

    df = pl.read_parquet(business_path)
    logger.info(f"Loaded {len(df):,} business records for quality analysis")

    # Calculate quality metrics
    quality_score = metrics.calculate_overall_quality_score(df)

    # Generate alerts
    alerts = metrics.create_quality_alerts(quality_score)

    # Save metrics
    metrics.save_quality_metrics(quality_score)

    # Print results
    print("\n=== DATA QUALITY METRICS ===")
    print(f"Overall Quality Score: {quality_score['overall_score']:.2%}")
    print(f"Quality Grade: {quality_score['quality_grade'].upper()}")
    print(f"Total Records: {quality_score['total_records']:,}")

    print("\n=== DIMENSION SCORES ===")
    for dimension, details in quality_score['dimension_scores'].items():
        print(f"{dimension.title()}: {details['score']:.2%}")

    if alerts:
        print(f"\n=== QUALITY ALERTS ({len(alerts)}) ===")
        for alert in alerts:
            print(f"[{alert['severity'].upper()}] {alert['message']}")

if __name__ == "__main__":
    main()