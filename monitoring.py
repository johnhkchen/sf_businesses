#!/usr/bin/env python3

import polars as pl
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataMonitoringDashboard:
    """Data monitoring dashboard and alerting system for SF business pipeline."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create monitoring subdirectory
        self.monitoring_dir = self.output_dir / "monitoring"
        self.monitoring_dir.mkdir(exist_ok=True)

        # Alert configuration
        self.alert_config = {
            'email_enabled': False,
            'slack_enabled': False,
            'file_alerts': True,
            'alert_cooldown_hours': 1
        }

        # Data freshness thresholds
        self.freshness_thresholds = {
            'business_data': timedelta(days=7),
            'building_data': timedelta(days=30),
            'spatial_joins': timedelta(days=7)
        }

    def collect_pipeline_metrics(self) -> Dict:
        """Collect comprehensive pipeline metrics."""
        logger.info("Collecting pipeline metrics...")

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'data_freshness': self.check_data_freshness(),
            'processing_stats': self.get_processing_statistics(),
            'quality_trends': self.analyze_quality_trends(),
            'system_health': self.check_system_health()
        }

        return metrics

    def check_data_freshness(self) -> Dict:
        """Check freshness of data files."""
        logger.info("Checking data freshness...")

        freshness_results = {}
        current_time = datetime.now()

        data_files = {
            'business_data': 'businesses.parquet',
            'building_data': 'buildings.parquet',
            'spatial_joins': 'spatial_join_results.parquet'
        }

        for data_type, filename in data_files.items():
            file_path = self.output_dir / filename
            if file_path.exists():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                age = current_time - file_mtime
                threshold = self.freshness_thresholds[data_type]

                freshness_results[data_type] = {
                    'last_updated': file_mtime.isoformat(),
                    'age_hours': age.total_seconds() / 3600,
                    'is_fresh': age <= threshold,
                    'threshold_hours': threshold.total_seconds() / 3600,
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024)
                }
            else:
                freshness_results[data_type] = {
                    'last_updated': None,
                    'age_hours': None,
                    'is_fresh': False,
                    'file_exists': False
                }

        return freshness_results

    def get_processing_statistics(self) -> Dict:
        """Get processing statistics from data files."""
        logger.info("Getting processing statistics...")

        stats = {}

        # Business data statistics
        business_path = self.output_dir / "businesses.parquet"
        if business_path.exists():
            try:
                business_df = pl.read_parquet(business_path)
                stats['business_data'] = {
                    'total_records': len(business_df),
                    'unique_naics_codes': business_df.select(pl.col('naics_code').n_unique()).item(),
                    'unique_neighborhoods': business_df.select(pl.col('neighborhood').n_unique()).item(),
                    'date_range': {
                        'earliest': business_df.select(pl.col('business_start_date').min()).item(),
                        'latest': business_df.select(pl.col('business_start_date').max()).item()
                    },
                    'geographic_coverage': {
                        'lat_range': [
                            business_df.select(pl.col('latitude').min()).item(),
                            business_df.select(pl.col('latitude').max()).item()
                        ],
                        'lon_range': [
                            business_df.select(pl.col('longitude').min()).item(),
                            business_df.select(pl.col('longitude').max()).item()
                        ]
                    }
                }
            except Exception as e:
                stats['business_data'] = {'error': str(e)}

        # Building data statistics
        building_path = self.output_dir / "buildings.parquet"
        if building_path.exists():
            try:
                building_df = pl.read_parquet(building_path)
                stats['building_data'] = {
                    'total_buildings': len(building_df),
                    'building_types': building_df.select(pl.col('building_type').n_unique()).item() if 'building_type' in building_df.columns else 0,
                    'avg_building_area': building_df.select(pl.col('area').mean()).item() if 'area' in building_df.columns else None
                }
            except Exception as e:
                stats['building_data'] = {'error': str(e)}

        # Spatial join statistics
        spatial_path = self.output_dir / "spatial_join_results.parquet"
        if spatial_path.exists():
            try:
                spatial_df = pl.read_parquet(spatial_path)
                stats['spatial_joins'] = {
                    'total_matches': len(spatial_df),
                    'match_rate': len(spatial_df) / stats.get('business_data', {}).get('total_records', 1) if 'business_data' in stats else 0,
                    'avg_distance': spatial_df.select(pl.col('distance_meters').mean()).item() if 'distance_meters' in spatial_df.columns else None
                }
            except Exception as e:
                stats['spatial_joins'] = {'error': str(e)}

        return stats

    def analyze_quality_trends(self) -> Dict:
        """Analyze quality trends from historical metrics."""
        logger.info("Analyzing quality trends...")

        # Look for quality metrics files
        quality_files = list(self.monitoring_dir.glob("quality_metrics_*.json"))

        if len(quality_files) < 2:
            return {'status': 'insufficient_history', 'files_found': len(quality_files)}

        # Load recent quality metrics
        quality_history = []
        for file_path in sorted(quality_files)[-10:]:  # Last 10 measurements
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    quality_history.append({
                        'timestamp': data['timestamp'],
                        'overall_score': data['overall_score'],
                        'dimensions': {
                            k: v['score'] for k, v in data['dimension_scores'].items()
                        }
                    })
            except Exception as e:
                logger.warning(f"Could not load quality file {file_path}: {e}")

        if len(quality_history) < 2:
            return {'status': 'insufficient_valid_history'}

        # Calculate trends
        latest = quality_history[-1]
        previous = quality_history[-2]

        trend_analysis = {
            'latest_score': latest['overall_score'],
            'previous_score': previous['overall_score'],
            'score_change': latest['overall_score'] - previous['overall_score'],
            'trend_direction': 'improving' if latest['overall_score'] > previous['overall_score'] else 'declining',
            'dimension_trends': {}
        }

        # Dimension-level trends
        for dimension in latest['dimensions']:
            if dimension in previous['dimensions']:
                change = latest['dimensions'][dimension] - previous['dimensions'][dimension]
                trend_analysis['dimension_trends'][dimension] = {
                    'current': latest['dimensions'][dimension],
                    'previous': previous['dimensions'][dimension],
                    'change': change,
                    'trend': 'improving' if change > 0 else 'declining'
                }

        return trend_analysis

    def check_system_health(self) -> Dict:
        """Check overall system health indicators."""
        logger.info("Checking system health...")

        health_indicators = {}

        # Check disk space
        try:
            output_size = sum(f.stat().st_size for f in self.output_dir.rglob('*') if f.is_file())
            health_indicators['disk_usage'] = {
                'output_dir_size_mb': output_size / (1024 * 1024),
                'status': 'healthy' if output_size < 1024 * 1024 * 1024 else 'warning'  # 1GB threshold
            }
        except Exception as e:
            health_indicators['disk_usage'] = {'error': str(e)}

        # Check file integrity
        required_files = ['businesses.parquet', 'buildings.parquet']
        missing_files = [f for f in required_files if not (self.output_dir / f).exists()]

        health_indicators['file_integrity'] = {
            'required_files': required_files,
            'missing_files': missing_files,
            'status': 'healthy' if not missing_files else 'error'
        }

        # Check for recent errors in logs (if log files exist)
        health_indicators['error_status'] = {
            'status': 'healthy',  # Simplified - would check actual logs in production
            'last_error': None
        }

        return health_indicators

    def generate_dashboard_html(self, metrics: Dict) -> str:
        """Generate HTML dashboard for data monitoring."""
        logger.info("Generating HTML dashboard...")

        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>SF Business Data Pipeline - Monitoring Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .card {{ background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        .status-good {{ color: #27ae60; }}
        .status-warning {{ color: #f39c12; }}
        .status-error {{ color: #e74c3c; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SF Business Data Pipeline Monitor</h1>
            <p>Last updated: {timestamp}</p>
        </div>

        <div class="grid">
            <!-- Data Freshness Card -->
            <div class="card">
                <h3>Data Freshness</h3>
                {freshness_content}
            </div>

            <!-- Processing Statistics Card -->
            <div class="card">
                <h3>Processing Statistics</h3>
                {processing_content}
            </div>

            <!-- Quality Trends Card -->
            <div class="card">
                <h3>Quality Trends</h3>
                {quality_content}
            </div>

            <!-- System Health Card -->
            <div class="card">
                <h3>System Health</h3>
                {health_content}
            </div>
        </div>
    </div>
</body>
</html>
        """

        # Generate freshness content
        freshness_content = ""
        for data_type, freshness in metrics['data_freshness'].items():
            status_class = "status-good" if freshness.get('is_fresh', False) else "status-error"
            age_display = f"{freshness.get('age_hours', 0):.1f}h" if freshness.get('age_hours') else "N/A"
            freshness_content += f'<div class="{status_class}">{data_type}: {age_display} old</div>'

        # Generate processing content
        processing_content = ""
        for data_type, stats in metrics['processing_stats'].items():
            if 'error' in stats:
                processing_content += f'<div class="status-error">{data_type}: {stats["error"]}</div>'
            else:
                if data_type == 'business_data':
                    processing_content += f'<div class="metric"><span class="metric-value">{stats["total_records"]:,}</span><br><span class="metric-label">Business Records</span></div>'
                elif data_type == 'building_data':
                    processing_content += f'<div class="metric"><span class="metric-value">{stats["total_buildings"]:,}</span><br><span class="metric-label">Building Records</span></div>'
                elif data_type == 'spatial_joins':
                    processing_content += f'<div class="metric"><span class="metric-value">{stats["total_matches"]:,}</span><br><span class="metric-label">Spatial Matches</span></div>'

        # Generate quality content
        quality_content = ""
        quality_trends = metrics['quality_trends']
        if 'latest_score' in quality_trends:
            score = quality_trends['latest_score']
            trend = quality_trends['trend_direction']
            status_class = "status-good" if score > 0.85 else "status-warning" if score > 0.70 else "status-error"
            quality_content = f'<div class="{status_class}">Latest Score: {score:.2%} ({trend})</div>'
        else:
            quality_content = '<div class="status-warning">Quality history insufficient</div>'

        # Generate health content
        health_content = ""
        for component, health in metrics['system_health'].items():
            status = health.get('status', 'unknown')
            status_class = f"status-{status}" if status in ['good', 'warning', 'error'] else "status-warning"
            health_content += f'<div class="{status_class}">{component}: {status}</div>'

        return html_template.format(
            timestamp=metrics['timestamp'],
            freshness_content=freshness_content,
            processing_content=processing_content,
            quality_content=quality_content,
            health_content=health_content
        )

    def create_alert(self, alert_type: str, severity: str, message: str, details: Dict = None) -> Dict:
        """Create and log an alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'severity': severity,
            'message': message,
            'details': details or {}
        }

        # Save alert to file
        alert_file = self.monitoring_dir / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(alert_file, 'w') as f:
            json.dump(alert, f, indent=2, default=str)

        logger.warning(f"ALERT [{severity.upper()}] {alert_type}: {message}")
        return alert

    def check_and_create_alerts(self, metrics: Dict) -> List[Dict]:
        """Check metrics and create alerts if thresholds are exceeded."""
        logger.info("Checking for alert conditions...")

        alerts = []

        # Data freshness alerts
        for data_type, freshness in metrics['data_freshness'].items():
            if not freshness.get('is_fresh', False) and freshness.get('file_exists', True):
                alerts.append(self.create_alert(
                    'data_freshness',
                    'warning',
                    f'{data_type} data is stale ({freshness.get("age_hours", 0):.1f} hours old)',
                    freshness
                ))

        # System health alerts
        for component, health in metrics['system_health'].items():
            if health.get('status') == 'error':
                alerts.append(self.create_alert(
                    'system_health',
                    'high',
                    f'System health issue in {component}',
                    health
                ))

        # Quality score alerts
        quality_trends = metrics['quality_trends']
        if 'latest_score' in quality_trends:
            score = quality_trends['latest_score']
            if score < 0.70:
                alerts.append(self.create_alert(
                    'quality_degradation',
                    'high',
                    f'Data quality score critically low: {score:.2%}',
                    quality_trends
                ))
            elif score < 0.85 and quality_trends['trend_direction'] == 'declining':
                alerts.append(self.create_alert(
                    'quality_degradation',
                    'medium',
                    f'Data quality score declining: {score:.2%}',
                    quality_trends
                ))

        return alerts

    def run_monitoring_cycle(self) -> Dict:
        """Run a complete monitoring cycle."""
        logger.info("Running monitoring cycle...")

        # Collect metrics
        metrics = self.collect_pipeline_metrics()

        # Check for alerts
        alerts = self.check_and_create_alerts(metrics)

        # Generate dashboard
        dashboard_html = self.generate_dashboard_html(metrics)
        dashboard_path = self.monitoring_dir / "dashboard.html"
        with open(dashboard_path, 'w') as f:
            f.write(dashboard_html)

        # Save metrics
        metrics_file = self.monitoring_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        monitoring_result = {
            'metrics': metrics,
            'alerts': alerts,
            'dashboard_path': str(dashboard_path),
            'metrics_file': str(metrics_file)
        }

        logger.info(f"Monitoring cycle complete. Generated {len(alerts)} alerts.")
        return monitoring_result

def main():
    """Standalone monitoring execution."""
    monitor = DataMonitoringDashboard()

    # Run monitoring cycle
    result = monitor.run_monitoring_cycle()

    print("\n=== MONITORING RESULTS ===")
    print(f"Dashboard: {result['dashboard_path']}")
    print(f"Metrics saved: {result['metrics_file']}")
    print(f"Alerts generated: {len(result['alerts'])}")

    if result['alerts']:
        print("\n=== ACTIVE ALERTS ===")
        for alert in result['alerts']:
            print(f"[{alert['severity'].upper()}] {alert['type']}: {alert['message']}")

    # Print key metrics
    metrics = result['metrics']
    print(f"\n=== KEY METRICS ===")

    # Data freshness summary
    fresh_count = sum(1 for f in metrics['data_freshness'].values() if f.get('is_fresh', False))
    total_files = len(metrics['data_freshness'])
    print(f"Data Freshness: {fresh_count}/{total_files} datasets fresh")

    # Processing stats summary
    if 'business_data' in metrics['processing_stats']:
        business_count = metrics['processing_stats']['business_data'].get('total_records', 0)
        print(f"Business Records: {business_count:,}")

    # Quality score
    if 'latest_score' in metrics['quality_trends']:
        quality_score = metrics['quality_trends']['latest_score']
        print(f"Quality Score: {quality_score:.2%}")

if __name__ == "__main__":
    main()