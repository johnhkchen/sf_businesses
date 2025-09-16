#!/usr/bin/env python3

import sys
import time
from data_pipeline import SFBusinessPipeline
from osm_buildings import OSMBuildingExtractor
from spatial_join import SpatialJoiner
from monitoring import DataMonitoringDashboard
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_full_pipeline():
    """Run the complete data ingestion pipeline."""
    start_time = time.time()

    logger.info("Starting SF Business Data Ingestion Pipeline")

    try:
        # Stage 1: Business Data Processing
        logger.info("=== Stage 1: Business Data Processing ===")
        business_pipeline = SFBusinessPipeline()
        business_pipeline.setup_database()

        df = business_pipeline.load_and_clean_csv("data/Registered_Business_Locations_-_San_Francisco_20250916.csv")
        df_final = business_pipeline.save_businesses_data(df)

        business_stats = business_pipeline.get_business_stats(df_final)
        logger.info(f"Business data loaded: {business_stats}")

        # Stage 2: Building Footprint Extraction
        logger.info("=== Stage 2: Building Footprint Extraction ===")
        building_extractor = OSMBuildingExtractor()
        building_extractor.setup_output()

        elements = building_extractor.fetch_sf_buildings()
        buildings = building_extractor.process_osm_buildings(elements)
        buildings_df = building_extractor.save_buildings_data(buildings)

        building_stats = building_extractor.get_building_stats(buildings_df)
        logger.info(f"Building data loaded: {building_stats}")

        # Stage 3: Spatial Joining
        logger.info("=== Stage 3: Spatial Joining ===")
        spatial_joiner = SpatialJoiner()
        businesses_df, buildings_df = spatial_joiner.load_data()
        joined_df = spatial_joiner.perform_spatial_join(businesses_df, buildings_df)
        spatial_joiner.save_spatial_join_results(joined_df)

        # Stage 4: Monitoring and Alerts
        logger.info("=== Stage 4: Monitoring and Quality Dashboard ===")
        monitor = DataMonitoringDashboard()
        monitoring_result = monitor.run_monitoring_cycle()

        logger.info(f"Monitoring complete - Dashboard: {monitoring_result['dashboard_path']}")
        if monitoring_result['alerts']:
            logger.warning(f"Generated {len(monitoring_result['alerts'])} quality alerts")

        # Final Results
        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "="*60)
        print("SF BUSINESS DATA INGESTION PIPELINE - COMPLETE")
        print("="*60)
        print(f"Pipeline Duration: {duration:.2f} seconds")
        print(f"\nBusiness Records: {business_stats['total_businesses']:,}")
        print(f"Building Footprints: {building_stats['total_buildings']:,}")
        print(f"Spatial Matches: {len(joined_df):,}")
        print(f"\nUnique NAICS codes: {business_stats['unique_naics_codes']}")
        print(f"Unique neighborhoods: {business_stats['unique_neighborhoods']}")
        print(f"Building types: {building_stats['unique_building_types']}")
        print("="*60)

        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return False

def run_business_only():
    """Run only the business data processing."""
    logger.info("Running business data processing only...")

    pipeline = SFBusinessPipeline()
    pipeline.setup_database()

    df = pipeline.load_and_clean_csv("data/Registered_Business_Locations_-_San_Francisco_20250916.csv")
    df_final = pipeline.save_businesses_data(df)

    stats = pipeline.get_business_stats(df_final)
    print("\n=== BUSINESS DATA RESULTS ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

def run_buildings_only():
    """Run only the building footprint extraction."""
    logger.info("Running building extraction only...")

    extractor = OSMBuildingExtractor()
    extractor.setup_output()

    elements = extractor.fetch_sf_buildings()
    buildings = extractor.process_osm_buildings(elements)
    df = extractor.save_buildings_data(buildings)

    if df is not None:
        stats = extractor.get_building_stats(df)
        print("\n=== BUILDING DATA RESULTS ===")
        for key, value in stats.items():
            print(f"{key}: {value}")

def run_spatial_only():
    """Run only the spatial joining."""
    logger.info("Running spatial joining only...")

    joiner = SpatialJoiner()
    businesses_df, buildings_df = joiner.load_data()
    joined_df = joiner.perform_spatial_join(businesses_df, buildings_df)
    joiner.save_spatial_join_results(joined_df)
    joiner.analyze_spatial_relationships(joined_df)

def main():
    """Main entry point with command line options."""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "business":
            run_business_only()
        elif command == "buildings":
            run_buildings_only()
        elif command == "spatial":
            run_spatial_only()
        elif command == "full":
            run_full_pipeline()
        else:
            print("Usage: python main.py [business|buildings|spatial|full]")
            print("  business  - Process business data only")
            print("  buildings - Extract building footprints only")
            print("  spatial   - Perform spatial joining only")
            print("  full      - Run complete pipeline (default)")
            sys.exit(1)
    else:
        # Default: run full pipeline
        run_full_pipeline()

if __name__ == "__main__":
    main()
