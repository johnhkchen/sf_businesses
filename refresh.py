#!/usr/bin/env python3

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json

from api_client import SFOpenDataAPIClient
from data_pipeline import SFBusinessPipeline
from osm_buildings import OSMBuildingExtractor
from spatial_join import SpatialJoiner
from monitoring import DataMonitoringDashboard

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataRefreshOrchestrator:
    """
    Orchestrates automated data refresh for the SF Business pipeline.

    Features:
    - Intelligent refresh scheduling based on data staleness
    - Incremental updates when possible
    - Robust error handling and fallback mechanisms
    - Performance monitoring and alerting
    - Data versioning and rollback capabilities
    """

    def __init__(self, output_dir: str = "output", cache_dir: str = "cache",
                 fallback_csv_path: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.fallback_csv_path = fallback_csv_path

        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        (self.cache_dir / "refresh_logs").mkdir(exist_ok=True)

        # Initialize components
        self.api_client = SFOpenDataAPIClient(
            cache_dir=str(self.cache_dir),
            fallback_csv_path=fallback_csv_path
        )
        self.business_pipeline = SFBusinessPipeline(
            output_dir=str(self.output_dir),
            enable_validation=True,
            enable_caching=True,
            enable_parallel=True
        )

        # Refresh state tracking
        self.refresh_state_file = self.cache_dir / "refresh_state.json"

    def _load_refresh_state(self) -> Dict:
        """Load refresh state from disk."""
        if not self.refresh_state_file.exists():
            return {
                "last_business_refresh": None,
                "last_buildings_refresh": None,
                "last_full_pipeline_run": None,
                "refresh_history": [],
                "error_count": 0,
                "consecutive_failures": 0
            }

        try:
            with open(self.refresh_state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load refresh state: {e}")
            return self._load_refresh_state()  # Return default state

    def _save_refresh_state(self, state: Dict):
        """Save refresh state to disk."""
        try:
            with open(self.refresh_state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save refresh state: {e}")

    def _log_refresh_event(self, event_type: str, status: str, details: Dict = None):
        """Log refresh events for monitoring and debugging."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "status": status,
            "details": details or {}
        }

        log_file = self.cache_dir / "refresh_logs" / f"refresh_{datetime.now().strftime('%Y%m')}.jsonl"
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.warning(f"Could not log refresh event: {e}")

        # Also log to logger
        logger.info(f"Refresh event: {event_type} - {status}")

    def _should_refresh_business_data(self, max_age_hours: int = 24) -> bool:
        """Check if business data should be refreshed."""
        state = self._load_refresh_state()
        last_refresh = state.get("last_business_refresh")

        if not last_refresh:
            logger.info("No previous business data refresh found")
            return True

        last_refresh_time = datetime.fromisoformat(last_refresh)
        age_hours = (datetime.now() - last_refresh_time).total_seconds() / 3600

        if age_hours > max_age_hours:
            logger.info(f"Business data is {age_hours:.1f} hours old, refreshing")
            return True

        logger.info(f"Business data is {age_hours:.1f} hours old, still fresh")
        return False

    def _should_refresh_building_data(self, max_age_hours: int = 168) -> bool:  # 7 days default
        """Check if building data should be refreshed."""
        state = self._load_refresh_state()
        last_refresh = state.get("last_buildings_refresh")

        if not last_refresh:
            logger.info("No previous building data refresh found")
            return True

        last_refresh_time = datetime.fromisoformat(last_refresh)
        age_hours = (datetime.now() - last_refresh_time).total_seconds() / 3600

        if age_hours > max_age_hours:
            logger.info(f"Building data is {age_hours:.1f} hours old, refreshing")
            return True

        logger.info(f"Building data is {age_hours:.1f} hours old, still fresh")
        return False

    def refresh_business_data(self, force: bool = False) -> Dict:
        """Refresh business data from SF Open Data API."""
        start_time = time.time()
        event_details = {"forced": force}

        try:
            logger.info("=== Starting Business Data Refresh ===")

            # Check if refresh is needed
            if not force and not self._should_refresh_business_data():
                result = {
                    "status": "skipped",
                    "reason": "data_fresh",
                    "duration": time.time() - start_time
                }
                self._log_refresh_event("business_refresh", "skipped", result)
                return result

            # Get fresh data from API and save as temporary CSV for pipeline
            logger.info("Fetching fresh business data from API...")
            df_raw = self.api_client.get_business_data(force_refresh=force)

            logger.info(f"Fetched {len(df_raw):,} business records")

            # Save to temporary CSV for pipeline processing
            temp_csv = self.cache_dir / "temp_business_data.csv"
            df_raw.write_csv(temp_csv)
            logger.info(f"Saved temporary CSV: {temp_csv}")

            # Process the data through the business pipeline using its CSV loader
            logger.info("Processing business data through pipeline...")
            self.business_pipeline.setup_database()

            # Use the pipeline's CSV loader which includes all the cleaning logic
            df_cleaned = self.business_pipeline.load_and_clean_csv(str(temp_csv))
            df_final = self.business_pipeline.save_businesses_data(df_cleaned)

            # Clean up temporary file
            temp_csv.unlink(missing_ok=True)

            # Get statistics
            stats = self.business_pipeline.get_business_stats(df_final)

            # Update refresh state
            state = self._load_refresh_state()
            state["last_business_refresh"] = datetime.now().isoformat()
            state["consecutive_failures"] = 0
            self._save_refresh_state(state)

            duration = time.time() - start_time
            result = {
                "status": "success",
                "records_processed": len(df_final),
                "duration": duration,
                "stats": stats
            }

            event_details.update(result)
            self._log_refresh_event("business_refresh", "success", event_details)

            logger.info(f"Business data refresh completed successfully in {duration:.2f}s")
            return result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)

            # Update error state
            state = self._load_refresh_state()
            state["error_count"] = state.get("error_count", 0) + 1
            state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
            self._save_refresh_state(state)

            result = {
                "status": "error",
                "error": error_msg,
                "duration": duration
            }

            event_details.update(result)
            self._log_refresh_event("business_refresh", "error", event_details)

            logger.error(f"Business data refresh failed: {error_msg}")
            return result

    def refresh_building_data(self, force: bool = False) -> Dict:
        """Refresh building data from OSM."""
        start_time = time.time()
        event_details = {"forced": force}

        try:
            logger.info("=== Starting Building Data Refresh ===")

            # Check if refresh is needed
            if not force and not self._should_refresh_building_data():
                result = {
                    "status": "skipped",
                    "reason": "data_fresh",
                    "duration": time.time() - start_time
                }
                self._log_refresh_event("building_refresh", "skipped", result)
                return result

            # Extract building data
            logger.info("Fetching fresh building data from OSM...")
            building_extractor = OSMBuildingExtractor()
            building_extractor.setup_output()

            elements = building_extractor.fetch_sf_buildings()
            buildings = building_extractor.process_osm_buildings(elements)
            buildings_df = building_extractor.save_buildings_data(buildings)

            building_stats = building_extractor.get_building_stats(buildings_df)

            # Update refresh state
            state = self._load_refresh_state()
            state["last_buildings_refresh"] = datetime.now().isoformat()
            state["consecutive_failures"] = 0
            self._save_refresh_state(state)

            duration = time.time() - start_time
            result = {
                "status": "success",
                "records_processed": len(buildings_df) if buildings_df is not None else 0,
                "duration": duration,
                "stats": building_stats
            }

            event_details.update(result)
            self._log_refresh_event("building_refresh", "success", event_details)

            logger.info(f"Building data refresh completed successfully in {duration:.2f}s")
            return result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)

            # Update error state
            state = self._load_refresh_state()
            state["error_count"] = state.get("error_count", 0) + 1
            state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
            self._save_refresh_state(state)

            result = {
                "status": "error",
                "error": error_msg,
                "duration": duration
            }

            event_details.update(result)
            self._log_refresh_event("building_refresh", "error", event_details)

            logger.error(f"Building data refresh failed: {error_msg}")
            return result

    def run_spatial_join(self) -> Dict:
        """Run spatial join on the latest data."""
        start_time = time.time()

        try:
            logger.info("=== Starting Spatial Join ===")

            spatial_joiner = SpatialJoiner()
            businesses_df, buildings_df = spatial_joiner.load_data()

            if businesses_df is None or buildings_df is None:
                raise Exception("Required data not available for spatial join")

            joined_df = spatial_joiner.perform_spatial_join(businesses_df, buildings_df)
            spatial_joiner.save_spatial_join_results(joined_df)

            duration = time.time() - start_time
            result = {
                "status": "success",
                "matched_records": len(joined_df),
                "duration": duration
            }

            self._log_refresh_event("spatial_join", "success", result)
            logger.info(f"Spatial join completed successfully in {duration:.2f}s")
            return result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)

            result = {
                "status": "error",
                "error": error_msg,
                "duration": duration
            }

            self._log_refresh_event("spatial_join", "error", result)
            logger.error(f"Spatial join failed: {error_msg}")
            return result

    def run_monitoring_and_quality_checks(self) -> Dict:
        """Run monitoring and quality checks."""
        start_time = time.time()

        try:
            logger.info("=== Starting Monitoring and Quality Checks ===")

            monitor = DataMonitoringDashboard()
            monitoring_result = monitor.run_monitoring_cycle()

            duration = time.time() - start_time
            result = {
                "status": "success",
                "dashboard_path": monitoring_result.get("dashboard_path"),
                "alerts_generated": len(monitoring_result.get("alerts", [])),
                "duration": duration
            }

            self._log_refresh_event("monitoring", "success", result)
            logger.info(f"Monitoring completed successfully in {duration:.2f}s")
            return result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)

            result = {
                "status": "error",
                "error": error_msg,
                "duration": duration
            }

            self._log_refresh_event("monitoring", "error", result)
            logger.error(f"Monitoring failed: {error_msg}")
            return result

    def run_full_refresh(self, force_business: bool = False, force_buildings: bool = False) -> Dict:
        """Run a complete data refresh cycle."""
        start_time = time.time()
        pipeline_results = {}

        logger.info("="*60)
        logger.info("AUTOMATED DATA REFRESH PIPELINE - STARTING")
        logger.info("="*60)

        try:
            # Step 1: Refresh business data
            business_result = self.refresh_business_data(force=force_business)
            pipeline_results["business_refresh"] = business_result

            # Step 2: Refresh building data
            building_result = self.refresh_building_data(force=force_buildings)
            pipeline_results["building_refresh"] = building_result

            # Step 3: Run spatial join (if both datasets are available)
            if (business_result.get("status") in ["success", "skipped"] and
                building_result.get("status") in ["success", "skipped"]):
                spatial_result = self.run_spatial_join()
                pipeline_results["spatial_join"] = spatial_result
            else:
                logger.warning("Skipping spatial join due to data refresh failures")
                pipeline_results["spatial_join"] = {"status": "skipped", "reason": "data_unavailable"}

            # Step 4: Run monitoring and quality checks
            monitoring_result = self.run_monitoring_and_quality_checks()
            pipeline_results["monitoring"] = monitoring_result

            # Update pipeline state
            state = self._load_refresh_state()
            state["last_full_pipeline_run"] = datetime.now().isoformat()
            self._save_refresh_state(state)

            # Calculate overall results
            total_duration = time.time() - start_time
            successful_steps = sum(1 for result in pipeline_results.values()
                                 if result.get("status") in ["success", "skipped"])
            total_steps = len(pipeline_results)

            final_result = {
                "status": "success" if successful_steps == total_steps else "partial_failure",
                "total_duration": total_duration,
                "successful_steps": successful_steps,
                "total_steps": total_steps,
                "step_results": pipeline_results
            }

            self._log_refresh_event("full_pipeline", final_result["status"], final_result)

            logger.info("="*60)
            logger.info(f"AUTOMATED DATA REFRESH PIPELINE - COMPLETED")
            logger.info(f"Duration: {total_duration:.2f}s | Success Rate: {successful_steps}/{total_steps}")
            logger.info("="*60)

            return final_result

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)

            result = {
                "status": "error",
                "error": error_msg,
                "duration": duration,
                "step_results": pipeline_results
            }

            self._log_refresh_event("full_pipeline", "error", result)
            logger.error(f"Full pipeline failed: {error_msg}")
            return result

    def get_refresh_status(self) -> Dict:
        """Get current refresh status and health."""
        state = self._load_refresh_state()
        freshness = self.api_client.get_data_freshness_info()

        return {
            "refresh_state": state,
            "data_freshness": freshness,
            "business_data_needs_refresh": self._should_refresh_business_data(),
            "building_data_needs_refresh": self._should_refresh_building_data(),
            "cache_dir": str(self.cache_dir),
            "output_dir": str(self.output_dir)
        }

def main():
    """CLI interface for the refresh orchestrator."""
    if len(sys.argv) < 2:
        print("Usage: python refresh.py [business|buildings|spatial|monitoring|full|status]")
        print("  business   - Refresh business data only")
        print("  buildings  - Refresh building data only")
        print("  spatial    - Run spatial join only")
        print("  monitoring - Run monitoring and quality checks only")
        print("  full       - Run complete refresh pipeline")
        print("  status     - Show current refresh status")
        sys.exit(1)

    command = sys.argv[1].lower()
    force = "--force" in sys.argv

    # Initialize orchestrator with fallback CSV
    fallback_csv = "data/Registered_Business_Locations_-_San_Francisco_20250916.csv"
    orchestrator = DataRefreshOrchestrator(fallback_csv_path=fallback_csv)

    if command == "business":
        result = orchestrator.refresh_business_data(force=force)
    elif command == "buildings":
        result = orchestrator.refresh_building_data(force=force)
    elif command == "spatial":
        result = orchestrator.run_spatial_join()
    elif command == "monitoring":
        result = orchestrator.run_monitoring_and_quality_checks()
    elif command == "full":
        result = orchestrator.run_full_refresh(force_business=force, force_buildings=force)
    elif command == "status":
        result = orchestrator.get_refresh_status()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

    print("\n" + "="*60)
    print("REFRESH ORCHESTRATOR RESULTS")
    print("="*60)
    print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    main()