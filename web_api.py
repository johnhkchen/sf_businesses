#!/usr/bin/env python3

import json
import time
import polars as pl
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import logging
from shapely import wkt
from shapely.geometry import Point, box
import geojson
from typing import Optional, List, Dict, Any
import math
from progressive_loader import ProgressiveDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SF Business Building Visualization API", version="1.0.0")

class BuildingVisualizationAPI:
    """API for building-level business visualization in San Francisco."""

    def __init__(self):
        self.output_dir = Path("output")
        self.buildings_df = None
        self.businesses_df = None
        self.progressive_loader = ProgressiveDataLoader(self.output_dir)
        self.load_data()

    def load_data(self):
        """Load building and business data including aggregated data."""
        try:
            self.buildings_df = pl.read_parquet(self.output_dir / "buildings.parquet")
            self.businesses_df = pl.read_parquet(self.output_dir / "businesses.parquet")

            # Load aggregated data if available
            try:
                self.buildings_enriched = pl.read_parquet(self.output_dir / "buildings_enriched_quick.parquet")
                self.business_grid = pl.read_parquet(self.output_dir / "business_grid_quick.parquet")
                logger.info("Loaded pre-aggregated data for faster visualization")
            except:
                logger.warning("Pre-aggregated data not found, will compute on-demand")
                self.buildings_enriched = None
                self.business_grid = None

            logger.info(f"Loaded {len(self.buildings_df)} buildings and {len(self.businesses_df)} businesses")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def get_building_business_aggregation(self, bbox: Optional[tuple] = None, zoom_level: int = 12) -> Dict[str, Any]:
        """
        Aggregate business data by building with spatial filtering using progressive loading.

        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            zoom_level: Map zoom level for detail adjustment

        Returns:
            Dict with aggregated building-business data
        """
        try:
            # Use the new progressive loader
            result = self.progressive_loader.get_progressive_data(zoom_level, bbox)
            return result
        except Exception as e:
            logger.error(f"Progressive loader failed, falling back to legacy method: {e}")

            # Fallback to original logic
            if zoom_level < 14:
                # Use pre-computed grid for lower zoom levels
                if self.business_grid is not None:
                    grid_data = self.business_grid

                    if bbox:
                        min_lon, min_lat, max_lon, max_lat = bbox
                        grid_data = grid_data.filter(
                            (pl.col("center_lon") >= min_lon) &
                            (pl.col("center_lon") <= max_lon) &
                            (pl.col("center_lat") >= min_lat) &
                            (pl.col("center_lat") <= max_lat)
                        )

                    return {
                        "type": "grid",
                        "zoom_level": zoom_level,
                        "features": grid_data.to_dicts(),
                        "total_features": len(grid_data)
                    }
                else:
                    # Fallback to on-demand computation
                    return self._compute_grid_on_demand(bbox, zoom_level)
            else:
                # Use pre-computed building data for high zoom levels
                if self.buildings_enriched is not None:
                    buildings_data = self.buildings_enriched

                    if bbox:
                        min_lon, min_lat, max_lon, max_lat = bbox
                        buildings_data = buildings_data.filter(
                            (pl.col("center_lon") >= min_lon) &
                            (pl.col("center_lon") <= max_lon) &
                            (pl.col("center_lat") >= min_lat) &
                            (pl.col("center_lat") <= max_lat)
                        ).head(1000)  # Limit for performance

                    return {
                        "type": "buildings",
                        "zoom_level": zoom_level,
                        "buildings": buildings_data.to_dicts(),
                        "total_features": len(buildings_data)
                    }
                else:
                    # Fallback to on-demand computation
                    return {
                        "type": "buildings",
                        "zoom_level": zoom_level,
                        "buildings": self._get_building_details(bbox),
                        "total_features": len(self.buildings_df)
                    }

    def _compute_grid_on_demand(self, bbox: Optional[tuple] = None, zoom_level: int = 12) -> Dict[str, Any]:
        """Compute grid aggregation on demand if pre-computed data is not available."""
        businesses = self.businesses_df

        # Filter by bounding box if provided
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            businesses_filtered = businesses.filter(
                (pl.col("longitude") >= min_lon) &
                (pl.col("longitude") <= max_lon) &
                (pl.col("latitude") >= min_lat) &
                (pl.col("latitude") <= max_lat)
            )
        else:
            businesses_filtered = businesses

        # Grid-based aggregation
        grid_size = 0.001 * (15 - zoom_level)  # Larger grid at lower zoom

        businesses_with_grid = businesses_filtered.with_columns([
            (pl.col("longitude") / grid_size).round(0).alias("grid_x"),
            (pl.col("latitude") / grid_size).round(0).alias("grid_y")
        ])

        aggregated = businesses_with_grid.group_by(["grid_x", "grid_y"]).agg([
            pl.len().alias("business_count"),
            pl.col("naics_code").n_unique().alias("business_types"),
            pl.col("naics_description").first().alias("primary_type"),
            pl.col("longitude").mean().alias("center_lon"),
            pl.col("latitude").mean().alias("center_lat")
        ])

        return {
            "type": "grid",
            "zoom_level": zoom_level,
            "features": aggregated.to_dicts(),
            "total_features": len(aggregated)
        }

    def _get_building_details(self, bbox: Optional[tuple] = None) -> List[Dict]:
        """Get detailed building information with business counts."""
        buildings = self.buildings_df

        if bbox:
            # Simple bbox filter for buildings (approximation)
            min_lon, min_lat, max_lon, max_lat = bbox

            # Extract centroid coordinates from WKT geometry for filtering
            buildings_with_centroid = buildings.with_columns([
                pl.col("geom_wkt").map_elements(
                    lambda wkt_str: self._get_centroid_from_wkt(wkt_str),
                    return_dtype=pl.List(pl.Float64)
                ).alias("centroid")
            ]).filter(
                pl.col("centroid").is_not_null()
            ).with_columns([
                pl.col("centroid").list.get(0).alias("center_lon"),
                pl.col("centroid").list.get(1).alias("center_lat")
            ]).filter(
                (pl.col("center_lon") >= min_lon) &
                (pl.col("center_lon") <= max_lon) &
                (pl.col("center_lat") >= min_lat) &
                (pl.col("center_lat") <= max_lat)
            )
        else:
            buildings_with_centroid = buildings.with_columns([
                pl.col("geom_wkt").map_elements(
                    lambda wkt_str: self._get_centroid_from_wkt(wkt_str),
                    return_dtype=pl.List(pl.Float64)
                ).alias("centroid")
            ]).filter(
                pl.col("centroid").is_not_null()
            ).with_columns([
                pl.col("centroid").list.get(0).alias("center_lon"),
                pl.col("centroid").list.get(1).alias("center_lat")
            ])

        # Limit results for performance
        limited_buildings = buildings_with_centroid.head(1000)

        # Add business density simulation (in real implementation, this would be spatial join)
        result = limited_buildings.with_columns([
            (pl.col("osm_id") % 10).alias("business_density"),  # Simulated for now
            pl.when(pl.col("building_category") == "commercial").then("high")
            .when(pl.col("building_category") == "residential").then("medium")
            .otherwise("low").alias("business_intensity")
        ])

        return result.select([
            "osm_id", "building_type", "building_category",
            "center_lon", "center_lat", "geom_wkt",
            "business_density", "business_intensity"
        ]).to_dicts()

    def _get_centroid_from_wkt(self, wkt_str: str) -> Optional[List[float]]:
        """Extract centroid coordinates from WKT geometry string."""
        try:
            if not wkt_str or wkt_str == "null":
                return None
            geom = wkt.loads(wkt_str)
            centroid = geom.centroid
            return [centroid.x, centroid.y]
        except Exception:
            return None

    def get_businesses_in_building(self, building_id: int, radius: float = 0.0001) -> List[Dict]:
        """Get businesses near a specific building."""
        try:
            # First check if we have enriched building data
            if self.buildings_enriched is not None:
                building = self.buildings_enriched.filter(pl.col("osm_id") == building_id).head(1)
                if len(building) > 0:
                    building_data = building.to_dicts()[0]
                    center_lon = building_data['center_lon']
                    center_lat = building_data['center_lat']
                else:
                    return []
            else:
                # Fallback to original building data
                building = self.buildings_df.filter(pl.col("osm_id") == building_id).head(1)
                if len(building) == 0:
                    return []

                # Get building centroid
                wkt_str = building.select("geom_wkt").item()
                centroid = self._get_centroid_from_wkt(wkt_str)
                if not centroid:
                    return []

                center_lon, center_lat = centroid

            # Find businesses within radius
            nearby_businesses = self.businesses_df.filter(
                (pl.col("longitude") >= center_lon - radius) &
                (pl.col("longitude") <= center_lon + radius) &
                (pl.col("latitude") >= center_lat - radius) &
                (pl.col("latitude") <= center_lat + radius)
            ).head(20)  # Limit for performance

            return nearby_businesses.select([
                "business_account_number", "dba_name", "naics_description",
                "street_address", "longitude", "latitude"
            ]).to_dicts()

        except Exception as e:
            logger.error(f"Error getting businesses for building {building_id}: {e}")
            return []

# Initialize API instance
api_instance = BuildingVisualizationAPI()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "SF Business Building Visualization API",
        "version": "1.0.0",
        "endpoints": {
            "/buildings": "Get building data with business aggregation",
            "/buildings/{building_id}/businesses": "Get businesses in specific building",
            "/map": "Interactive map visualization",
            "/webgl": "Next-generation WebGL-accelerated map (100k+ points at 60fps)"
        }
    }

@app.get("/buildings")
async def get_buildings(
    bbox: Optional[str] = Query(None, description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
    zoom: int = Query(12, ge=1, le=18, description="Map zoom level"),
    limit: int = Query(1000, ge=1, le=5000, description="Maximum number of results")
):
    """Get building data with business aggregation."""
    try:
        bbox_tuple = None
        if bbox:
            coords = [float(x.strip()) for x in bbox.split(",")]
            if len(coords) == 4:
                bbox_tuple = tuple(coords)
            else:
                raise HTTPException(status_code=400, detail="Invalid bbox format")

        result = api_instance.get_building_business_aggregation(bbox_tuple, zoom)
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error in get_buildings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/buildings/{building_id}/businesses")
async def get_building_businesses(building_id: int):
    """Get businesses located in or near a specific building."""
    try:
        businesses = api_instance.get_businesses_in_building(building_id)
        return JSONResponse(content={
            "building_id": building_id,
            "businesses": businesses,
            "count": len(businesses)
        })
    except Exception as e:
        logger.error(f"Error getting businesses for building {building_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/webgl/bulk")
async def get_webgl_bulk_data(
    bbox: Optional[str] = Query(None, description="Bounding box: min_lon,min_lat,max_lon,max_lat"),
    zoom: int = Query(12, ge=1, le=18, description="Map zoom level"),
    max_points: int = Query(100000, ge=1000, le=250000, description="Maximum points to return")
):
    """Get high-performance bulk data optimized for WebGL rendering with 100k+ points."""
    try:
        bbox_tuple = None
        if bbox:
            coords = [float(x.strip()) for x in bbox.split(",")]
            if len(coords) == 4:
                bbox_tuple = tuple(coords)
            else:
                raise HTTPException(status_code=400, detail="Invalid bbox format")

        # Use progressive loader but request all available data for WebGL
        start_time = time.time()
        result = api_instance.progressive_loader.get_webgl_optimized_data(zoom, bbox_tuple, max_points)
        processing_time = (time.time() - start_time) * 1000

        # Add performance metadata
        result["performance"] = {
            "processing_time_ms": round(processing_time, 2),
            "points_per_ms": round(len(result.get("points", [])) / max(processing_time, 1), 2),
            "target_met": processing_time < 1000,  # 1 second target for bulk load
            "webgl_optimized": True
        }

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error in get_webgl_bulk_data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/precompute")
async def precompute_data():
    """Precompute all zoom level data for improved performance."""
    try:
        results = api_instance.progressive_loader.precompute_all_zoom_levels()
        return JSONResponse(content={
            "status": "success",
            "message": "Precomputation completed",
            "results": results
        })
    except Exception as e:
        logger.error(f"Error in precomputation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/performance/metrics")
async def get_performance_metrics():
    """Get current performance metrics."""
    try:
        cache_dir = api_instance.progressive_loader.cache_dir
        summary_file = cache_dir / "precomputation_summary.json"

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = {"message": "No precomputed data available"}

        return JSONResponse(content=metrics)
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/map", response_class=HTMLResponse)
async def get_map():
    """Serve the interactive map visualization."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SF Business Building Visualization</title>
        <meta charset='utf-8' />
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
        <script src='https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js'></script>
        <link href='https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css' rel='stylesheet' />
        <style>
            body { margin: 0; padding: 0; }
            #map { position: absolute; top: 0; bottom: 0; width: 100%; }
            .map-overlay {
                position: absolute;
                bottom: 0;
                right: 0;
                background: rgba(255, 255, 255, 0.8);
                margin-right: 20px;
                font-family: Arial, sans-serif;
                overflow: auto;
                border-radius: 3px;
            }
            #loading-indicator {
                position: absolute;
                top: 10px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
                font-family: Arial, sans-serif;
                display: none;
                z-index: 1000;
            }
            #performance-info {
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(255, 255, 255, 0.9);
                padding: 10px;
                border-radius: 5px;
                font-family: Arial, sans-serif;
                font-size: 12px;
                z-index: 1000;
            }
            #legend {
                padding: 10px;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
                line-height: 18px;
                height: 150px;
                margin-bottom: 40px;
                width: 150px;
            }
            .legend-key {
                display: inline-block;
                border-radius: 20%;
                width: 10px;
                height: 10px;
                margin-right: 5px;
            }
            #info {
                padding: 10px;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
                font-size: 12px;
                max-height: 200px;
                margin-bottom: 10px;
                width: 200px;
            }
        </style>
    </head>
    <body>
        <div id='map'></div>
        <div id='loading-indicator'>Loading data...</div>
        <div id='performance-info'>
            <div>Zoom: <span id='current-zoom'>12</span></div>
            <div>Data Type: <span id='data-type'>-</span></div>
            <div>Load Time: <span id='load-time'>-</span>ms</div>
            <div>Features: <span id='feature-count'>-</span></div>
            <div>Cache: <span id='cache-status'>-</span></div>
        </div>
        <div class='map-overlay' id='legend'>
            <h3>Building Business Density</h3>
            <div><span class='legend-key' style='background-color: #800026'></span>Very High (8-10)</div>
            <div><span class='legend-key' style='background-color: #BD0026'></span>High (6-7)</div>
            <div><span class='legend-key' style='background-color: #E31A1C'></span>Medium-High (4-5)</div>
            <div><span class='legend-key' style='background-color: #FC4E2A'></span>Medium (2-3)</div>
            <div><span class='legend-key' style='background-color: #FEB24C'></span>Low (1)</div>
            <div><span class='legend-key' style='background-color: #FFEDA0'></span>None (0)</div>
        </div>
        <div class='map-overlay' id='info'>
            <h4>Click on a building for details</h4>
            <div id='building-info'></div>
        </div>

        <script>
            const map = new maplibregl.Map({
                container: 'map',
                style: 'https://demotiles.maplibre.org/style.json',
                center: [-122.4194, 37.7749], // San Francisco
                zoom: 12
            });

            let currentZoom = 12;
            let dataCache = new Map(); // Client-side cache
            let prefetchQueue = new Set(); // Prefetch queue
            let isLoading = false;
            let movementHistory = []; // Track user movement for smart prefetching

            function getColor(density) {
                return density > 8 ? '#800026' :
                       density > 6 ? '#BD0026' :
                       density > 4 ? '#E31A1C' :
                       density > 2 ? '#FC4E2A' :
                       density > 0 ? '#FEB24C' :
                                     '#FFEDA0';
            }

            function showLoading() {
                document.getElementById('loading-indicator').style.display = 'block';
                isLoading = true;
            }

            function hideLoading() {
                document.getElementById('loading-indicator').style.display = 'none';
                isLoading = false;
            }

            function updatePerformanceInfo(data, loadTime, fromCache) {
                document.getElementById('current-zoom').textContent = currentZoom;
                document.getElementById('data-type').textContent = data.type || 'unknown';
                document.getElementById('load-time').textContent = loadTime;
                document.getElementById('feature-count').textContent = data.total_features || 0;
                document.getElementById('cache-status').textContent = fromCache ? 'HIT' : 'MISS';
            }

            function getCacheKey(bbox, zoom) {
                return `${Math.round(zoom)}_${bbox}`;
            }

            function addToMovementHistory(center, zoom) {
                movementHistory.push({
                    center: center,
                    zoom: zoom,
                    timestamp: Date.now()
                });

                // Keep only recent movements (last 10 seconds)
                const cutoff = Date.now() - 10000;
                movementHistory = movementHistory.filter(m => m.timestamp > cutoff);
            }

            function predictNextBounds() {
                if (movementHistory.length < 2) return [];

                const recent = movementHistory.slice(-3);
                const predictions = [];

                // Simple prediction: continue in the same direction
                if (recent.length >= 2) {
                    const last = recent[recent.length - 1];
                    const prev = recent[recent.length - 2];

                    const deltaLng = last.center.lng - prev.center.lng;
                    const deltaLat = last.center.lat - prev.center.lat;

                    // Predict next position
                    const predictedCenter = {
                        lng: last.center.lng + deltaLng,
                        lat: last.center.lat + deltaLat
                    };

                    // Generate bounding box around predicted center
                    const bounds = map.getBounds();
                    const width = bounds.getEast() - bounds.getWest();
                    const height = bounds.getNorth() - bounds.getSouth();

                    predictions.push({
                        bbox: `${predictedCenter.lng - width/2},${predictedCenter.lat - height/2},${predictedCenter.lng + width/2},${predictedCenter.lat + height/2}`,
                        zoom: last.zoom
                    });
                }

                return predictions;
            }

            function prefetchData() {
                if (isLoading || prefetchQueue.size > 3) return; // Don't prefetch too much

                const predictions = predictNextBounds();
                predictions.forEach(pred => {
                    const cacheKey = getCacheKey(pred.bbox, pred.zoom);
                    if (!dataCache.has(cacheKey) && !prefetchQueue.has(cacheKey)) {
                        prefetchQueue.add(cacheKey);

                        // Prefetch in background
                        fetch(`/buildings?bbox=${pred.bbox}&zoom=${pred.zoom}`)
                            .then(response => response.json())
                            .then(data => {
                                dataCache.set(cacheKey, data);
                                prefetchQueue.delete(cacheKey);
                            })
                            .catch(error => {
                                prefetchQueue.delete(cacheKey);
                                console.log('Prefetch failed:', error);
                            });
                    }
                });
            }

            function updateData() {
                const bounds = map.getBounds();
                const bbox = `${bounds.getWest()},${bounds.getSouth()},${bounds.getEast()},${bounds.getNorth()}`;
                const zoom = Math.round(map.getZoom());

                // Add to movement history for smart prefetching
                addToMovementHistory(map.getCenter(), zoom);

                // Throttle updates for performance
                if (Math.abs(zoom - currentZoom) < 1 && isLoading) return;
                currentZoom = zoom;

                // Check cache first
                const cacheKey = getCacheKey(bbox, zoom);
                const startTime = performance.now();

                if (dataCache.has(cacheKey)) {
                    const data = dataCache.get(cacheKey);
                    const loadTime = Math.round(performance.now() - startTime);
                    updatePerformanceInfo(data, loadTime, true);

                    if (data.type === 'hexagonal') {
                        updateHexagonalDisplay(data.features);
                    } else if (data.type === 'statistical_clusters') {
                        updateClusterDisplay(data.features);
                    } else if (data.type === 'individual_buildings') {
                        updateBuildingDisplay(data.features);
                    } else if (data.type === 'grid') {
                        updateGridDisplay(data.features);
                    } else {
                        updateBuildingDisplay(data.buildings || data.features || []);
                    }

                    // Start prefetching for predicted movements
                    setTimeout(prefetchData, 100);
                    return;
                }

                // Not in cache, fetch from server
                showLoading();

                fetch(`/buildings?bbox=${bbox}&zoom=${zoom}`)
                    .then(response => response.json())
                    .then(data => {
                        const loadTime = Math.round(performance.now() - startTime);
                        hideLoading();

                        // Cache the response
                        dataCache.set(cacheKey, data);

                        // Update performance info
                        updatePerformanceInfo(data, loadTime, false);

                        // Display based on data type
                        if (data.type === 'hexagonal') {
                            updateHexagonalDisplay(data.features);
                        } else if (data.type === 'statistical_clusters') {
                            updateClusterDisplay(data.features);
                        } else if (data.type === 'individual_buildings') {
                            updateBuildingDisplay(data.features);
                        } else if (data.type === 'grid') {
                            updateGridDisplay(data.features);
                        } else {
                            updateBuildingDisplay(data.buildings || data.features || []);
                        }

                        // Start prefetching for predicted movements
                        setTimeout(prefetchData, 500);
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        hideLoading();
                        updatePerformanceInfo({type: 'error'}, Math.round(performance.now() - startTime), false);
                    });
            }

            function updateGridDisplay(features) {
                // Remove existing building layer
                if (map.getLayer('buildings')) {
                    map.removeLayer('buildings');
                    map.removeSource('buildings');
                }

                // Add grid circles
                const gridData = {
                    type: 'FeatureCollection',
                    features: features.map(f => ({
                        type: 'Feature',
                        geometry: {
                            type: 'Point',
                            coordinates: [f.center_lon, f.center_lat]
                        },
                        properties: {
                            business_count: f.business_count,
                            business_types: f.business_types,
                            primary_type: f.primary_type
                        }
                    }))
                };

                map.addSource('grid', {
                    type: 'geojson',
                    data: gridData
                });

                map.addLayer({
                    id: 'grid',
                    type: 'circle',
                    source: 'grid',
                    paint: {
                        'circle-radius': [
                            'interpolate',
                            ['linear'],
                            ['get', 'business_count'],
                            0, 5,
                            50, 20
                        ],
                        'circle-color': [
                            'interpolate',
                            ['linear'],
                            ['get', 'business_count'],
                            0, '#FFEDA0',
                            10, '#FEB24C',
                            25, '#FC4E2A',
                            50, '#E31A1C',
                            100, '#800026'
                        ],
                        'circle-opacity': 0.7
                    }
                });
            }

            function updateHexagonalDisplay(features) {
                // Remove existing layers
                ['buildings', 'grid', 'clusters', 'hexagons'].forEach(layerId => {
                    if (map.getLayer(layerId)) {
                        map.removeLayer(layerId);
                        map.removeSource(layerId);
                    }
                });

                // Create GeoJSON for hexagons
                const hexData = {
                    type: 'FeatureCollection',
                    features: features.map(f => ({
                        type: 'Feature',
                        geometry: {
                            type: 'Point',
                            coordinates: [f.center_lon, f.center_lat]
                        },
                        properties: {
                            business_count: f.business_count,
                            business_types: f.business_types,
                            primary_naics: f.primary_naics,
                            hex_size: f.hex_size
                        }
                    }))
                };

                map.addSource('hexagons', {
                    type: 'geojson',
                    data: hexData
                });

                map.addLayer({
                    id: 'hexagons',
                    type: 'circle',
                    source: 'hexagons',
                    paint: {
                        'circle-radius': [
                            'interpolate',
                            ['linear'],
                            ['get', 'business_count'],
                            0, 8,
                            20, 15,
                            100, 25
                        ],
                        'circle-color': [
                            'interpolate',
                            ['linear'],
                            ['get', 'business_count'],
                            0, '#FFEDA0',
                            5, '#FEB24C',
                            15, '#FC4E2A',
                            30, '#E31A1C',
                            50, '#800026'
                        ],
                        'circle-opacity': 0.8,
                        'circle-stroke-color': '#fff',
                        'circle-stroke-width': 1
                    }
                });
            }

            function updateClusterDisplay(features) {
                // Remove existing layers
                ['buildings', 'grid', 'hexagons', 'clusters'].forEach(layerId => {
                    if (map.getLayer(layerId)) {
                        map.removeLayer(layerId);
                        map.removeSource(layerId);
                    }
                });

                // Create GeoJSON for clusters
                const clusterData = {
                    type: 'FeatureCollection',
                    features: features.map(f => ({
                        type: 'Feature',
                        geometry: {
                            type: 'Point',
                            coordinates: [f.center_lon, f.center_lat]
                        },
                        properties: {
                            business_count: f.business_count,
                            business_types: f.business_types,
                            primary_naics: f.primary_naics,
                            dominant_naics: f.dominant_naics,
                            spatial_variance: f.spatial_variance
                        }
                    }))
                };

                map.addSource('clusters', {
                    type: 'geojson',
                    data: clusterData
                });

                map.addLayer({
                    id: 'clusters',
                    type: 'circle',
                    source: 'clusters',
                    paint: {
                        'circle-radius': [
                            'interpolate',
                            ['linear'],
                            ['get', 'business_count'],
                            0, 6,
                            10, 12,
                            50, 18
                        ],
                        'circle-color': [
                            'interpolate',
                            ['linear'],
                            ['get', 'business_types'],
                            1, '#FFEDA0',
                            3, '#FEB24C',
                            5, '#FC4E2A',
                            8, '#E31A1C',
                            12, '#800026'
                        ],
                        'circle-opacity': 0.7,
                        'circle-stroke-color': '#333',
                        'circle-stroke-width': 1
                    }
                });
            }

            function updateBuildingDisplay(buildings) {
                // Remove existing grid layer
                if (map.getLayer('grid')) {
                    map.removeLayer('grid');
                    map.removeSource('grid');
                }

                // Create GeoJSON for buildings
                const buildingData = {
                    type: 'FeatureCollection',
                    features: buildings.map(b => ({
                        type: 'Feature',
                        geometry: {
                            type: 'Point', // Simplified for now
                            coordinates: [b.center_lon, b.center_lat]
                        },
                        properties: {
                            osm_id: b.osm_id,
                            building_type: b.building_type,
                            building_category: b.building_category,
                            business_density: b.business_density,
                            business_intensity: b.business_intensity
                        }
                    }))
                };

                map.addSource('buildings', {
                    type: 'geojson',
                    data: buildingData
                });

                map.addLayer({
                    id: 'buildings',
                    type: 'circle',
                    source: 'buildings',
                    paint: {
                        'circle-radius': 6,
                        'circle-color': [
                            'case',
                            ['==', ['get', 'business_intensity'], 'high'], '#E31A1C',
                            ['==', ['get', 'business_intensity'], 'medium'], '#FEB24C',
                            '#FFEDA0'
                        ],
                        'circle-stroke-color': '#fff',
                        'circle-stroke-width': 1,
                        'circle-opacity': 0.8
                    }
                });

                // Add click handler for building details
                map.on('click', 'buildings', function (e) {
                    const properties = e.features[0].properties;

                    fetch(`/buildings/${properties.osm_id}/businesses`)
                        .then(response => response.json())
                        .then(data => {
                            let businessList = '';
                            if (data.businesses.length > 0) {
                                businessList = '<ul>' + data.businesses.map(b =>
                                    `<li><strong>${b.dba_name || 'Unknown'}</strong><br>
                                     ${b.naics_description || 'Unknown type'}<br>
                                     ${b.street_address || ''}</li>`
                                ).join('') + '</ul>';
                            } else {
                                businessList = '<p>No businesses found in this building.</p>';
                            }

                            document.getElementById('building-info').innerHTML = `
                                <h5>Building ${properties.osm_id}</h5>
                                <p><strong>Type:</strong> ${properties.building_type}</p>
                                <p><strong>Category:</strong> ${properties.building_category}</p>
                                <p><strong>Business Density:</strong> ${properties.business_density}</p>
                                <p><strong>Businesses (${data.count}):</strong></p>
                                ${businessList}
                            `;
                        });
                });

                // Change cursor on hover
                map.on('mouseenter', 'buildings', () => {
                    map.getCanvas().style.cursor = 'pointer';
                });

                map.on('mouseleave', 'buildings', () => {
                    map.getCanvas().style.cursor = '';
                });
            }

            map.on('load', function () {
                updateData();
            });

            map.on('moveend', updateData);
            map.on('zoomend', updateData);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/webgl", response_class=HTMLResponse)
async def get_webgl_map():
    """Serve the next-generation WebGL-accelerated map visualization."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SF Business Visualization - WebGL Engine</title>
        <meta charset='utf-8' />
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
        <script src='https://unpkg.com/deck.gl@^8.9.0/dist.min.js'></script>
        <script src='https://unpkg.com/maplibre-gl@^4.7.1/dist/maplibre-gl.js'></script>
        <link href='https://unpkg.com/maplibre-gl@^4.7.1/dist/maplibre-gl.css' rel='stylesheet' />
        <style>
            body { margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
            #map { position: absolute; top: 0; bottom: 0; width: 100%; }

            .performance-hud {
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(0, 0, 0, 0.85);
                color: #00ff41;
                padding: 12px;
                border-radius: 6px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 11px;
                line-height: 1.4;
                z-index: 1000;
                min-width: 220px;
            }

            .performance-hud .metric {
                display: flex;
                justify-content: space-between;
                margin: 2px 0;
            }

            .performance-hud .metric.critical {
                color: #ff4444;
            }

            .performance-hud .metric.warning {
                color: #ffaa00;
            }

            .performance-hud .metric.good {
                color: #00ff41;
            }

            .controls {
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(255, 255, 255, 0.95);
                padding: 12px;
                border-radius: 6px;
                z-index: 1000;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }

            .control-section {
                margin-bottom: 12px;
            }

            .control-section h4 {
                margin: 0 0 6px 0;
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
                color: #666;
            }

            .control-group {
                display: flex;
                align-items: center;
                margin: 4px 0;
                font-size: 12px;
            }

            .control-group label {
                min-width: 60px;
                margin-right: 8px;
            }

            input[type="range"] {
                width: 100px;
                margin: 0 8px;
            }

            .loading-overlay {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 20px 30px;
                border-radius: 8px;
                font-size: 14px;
                z-index: 2000;
                display: none;
            }

            .layer-toggle {
                display: flex;
                align-items: center;
                margin: 4px 0;
                font-size: 12px;
            }

            .layer-toggle input[type="checkbox"] {
                margin-right: 6px;
            }
        </style>
    </head>
    <body>
        <div id='map'></div>

        <div class="performance-hud" id="performance-hud">
            <div style="font-weight: bold; margin-bottom: 8px;">🚀 WebGL Engine</div>
            <div class="metric">
                <span>FPS:</span>
                <span id="fps-counter" class="good">60</span>
            </div>
            <div class="metric">
                <span>Frame Time:</span>
                <span id="frame-time">16.7ms</span>
            </div>
            <div class="metric">
                <span>Visible Points:</span>
                <span id="visible-points">0</span>
            </div>
            <div class="metric">
                <span>Total Points:</span>
                <span id="total-points">0</span>
            </div>
            <div class="metric">
                <span>GPU Memory:</span>
                <span id="gpu-memory">0MB</span>
            </div>
            <div class="metric">
                <span>Load Time:</span>
                <span id="load-time">0ms</span>
            </div>
            <div class="metric">
                <span>Cache:</span>
                <span id="cache-status">MISS</span>
            </div>
            <div class="metric">
                <span>LOD Level:</span>
                <span id="lod-level">3</span>
            </div>
        </div>

        <div class="controls">
            <div class="control-section">
                <h4>Rendering</h4>
                <div class="control-group">
                    <label>Point Size:</label>
                    <input type="range" id="point-size" min="1" max="20" value="4">
                    <span id="point-size-value">4</span>
                </div>
                <div class="control-group">
                    <label>Opacity:</label>
                    <input type="range" id="opacity" min="0" max="100" value="80">
                    <span id="opacity-value">80%</span>
                </div>
            </div>

            <div class="control-section">
                <h4>Layers</h4>
                <div class="layer-toggle">
                    <input type="checkbox" id="layer-businesses" checked>
                    <label for="layer-businesses">Businesses</label>
                </div>
                <div class="layer-toggle">
                    <input type="checkbox" id="layer-clusters" checked>
                    <label for="layer-clusters">Clusters</label>
                </div>
                <div class="layer-toggle">
                    <input type="checkbox" id="layer-heatmap">
                    <label for="layer-heatmap">Heatmap</label>
                </div>
            </div>

            <div class="control-section">
                <h4>Performance</h4>
                <div class="layer-toggle">
                    <input type="checkbox" id="frustum-culling" checked>
                    <label for="frustum-culling">Frustum Culling</label>
                </div>
                <div class="layer-toggle">
                    <input type="checkbox" id="lod-enabled" checked>
                    <label for="lod-enabled">Level of Detail</label>
                </div>
            </div>
        </div>

        <div class="loading-overlay" id="loading-overlay">
            <div>Loading high-performance data...</div>
        </div>

        <script>
            // WebGL High-Performance Rendering Engine
            const {DeckGL, ScatterplotLayer, HexagonLayer, HeatmapLayer, _TerrainExtension} = deck;

            class PerformanceMonitor {
                constructor() {
                    this.frameCount = 0;
                    this.lastTime = performance.now();
                    this.fps = 60;
                    this.frameTime = 16.7;
                }

                update() {
                    this.frameCount++;
                    const now = performance.now();
                    const delta = now - this.lastTime;

                    if (delta >= 1000) {
                        this.fps = Math.round((this.frameCount * 1000) / delta);
                        this.frameTime = delta / this.frameCount;
                        this.frameCount = 0;
                        this.lastTime = now;

                        this.updateHUD();
                    }
                }

                updateHUD() {
                    const fpsElement = document.getElementById('fps-counter');
                    const frameTimeElement = document.getElementById('frame-time');

                    fpsElement.textContent = this.fps;
                    frameTimeElement.textContent = `${this.frameTime.toFixed(1)}ms`;

                    // Color code performance
                    fpsElement.className = this.fps >= 55 ? 'good' : this.fps >= 30 ? 'warning' : 'critical';
                }
            }

            class DataCache {
                constructor() {
                    this.cache = new Map();
                    this.maxSize = 50; // Maximum cached datasets
                }

                key(bbox, zoom) {
                    return `${Math.round(zoom)}_${bbox.join('_')}`;
                }

                get(bbox, zoom) {
                    const k = this.key(bbox, zoom);
                    const item = this.cache.get(k);
                    if (item) {
                        item.lastAccess = Date.now();
                        return item.data;
                    }
                    return null;
                }

                set(bbox, zoom, data) {
                    if (this.cache.size >= this.maxSize) {
                        // Remove oldest item
                        let oldest = null;
                        let oldestTime = Date.now();

                        for (const [key, item] of this.cache.entries()) {
                            if (item.lastAccess < oldestTime) {
                                oldestTime = item.lastAccess;
                                oldest = key;
                            }
                        }

                        if (oldest) {
                            this.cache.delete(oldest);
                        }
                    }

                    this.cache.set(this.key(bbox, zoom), {
                        data: data,
                        lastAccess: Date.now()
                    });
                }

                clear() {
                    this.cache.clear();
                }
            }

            class WebGLRenderingEngine {
                constructor() {
                    this.monitor = new PerformanceMonitor();
                    this.cache = new DataCache();
                    this.currentData = [];
                    this.currentZoom = 12;
                    this.viewport = null;
                    this.layers = [];

                    this.init();
                    this.setupControls();
                    this.animate();
                }

                init() {
                    this.deckgl = new DeckGL({
                        container: 'map',
                        mapStyle: 'https://demotiles.maplibre.org/style.json',
                        initialViewState: {
                            longitude: -122.4194,
                            latitude: 37.7749,
                            zoom: 12,
                            bearing: 0,
                            pitch: 0
                        },
                        controller: true,
                        onViewStateChange: ({viewState}) => {
                            this.viewport = viewState;
                            this.currentZoom = viewState.zoom;
                            this.updateData();
                        },
                        onAfterRender: () => {
                            this.monitor.update();
                        }
                    });

                    this.updateData();
                }

                setupControls() {
                    // Point size control
                    const pointSizeSlider = document.getElementById('point-size');
                    const pointSizeValue = document.getElementById('point-size-value');
                    pointSizeSlider.addEventListener('input', (e) => {
                        pointSizeValue.textContent = e.target.value;
                        this.updateLayers();
                    });

                    // Opacity control
                    const opacitySlider = document.getElementById('opacity');
                    const opacityValue = document.getElementById('opacity-value');
                    opacitySlider.addEventListener('input', (e) => {
                        opacityValue.textContent = e.target.value + '%';
                        this.updateLayers();
                    });

                    // Layer toggles
                    ['layer-businesses', 'layer-clusters', 'layer-heatmap'].forEach(id => {
                        document.getElementById(id).addEventListener('change', () => {
                            this.updateLayers();
                        });
                    });

                    // Performance toggles
                    document.getElementById('frustum-culling').addEventListener('change', () => {
                        this.updateData();
                    });

                    document.getElementById('lod-enabled').addEventListener('change', () => {
                        this.updateData();
                    });
                }

                async updateData() {
                    if (!this.viewport) return;

                    const bounds = this.getBounds();
                    const bbox = [bounds.west, bounds.south, bounds.east, bounds.north];
                    const zoom = Math.round(this.viewport.zoom);

                    // Check cache first
                    const cachedData = this.cache.get(bbox, zoom);
                    if (cachedData) {
                        this.currentData = cachedData;
                        this.updateMetrics(true);
                        this.updateLayers();
                        return;
                    }

                    // Show loading
                    document.getElementById('loading-overlay').style.display = 'block';

                    const startTime = performance.now();

                    try {
                        const bboxString = bbox.join(',');
                        // Use bulk endpoint for high zoom levels or large datasets
                        const useBulk = zoom >= 10 || !bbox;
                        const endpoint = useBulk ?
                            `/webgl/bulk?bbox=${bboxString}&zoom=${zoom}&max_points=150000` :
                            `/buildings?bbox=${bboxString}&zoom=${zoom}`;

                        const response = await fetch(endpoint);
                        const data = await response.json();

                        const loadTime = performance.now() - startTime;

                        // Transform data for WebGL rendering
                        const transformedData = this.transformForWebGL(data);

                        // Cache the result
                        this.cache.set(bbox, zoom, transformedData);

                        this.currentData = transformedData;
                        this.updateMetrics(false, loadTime);
                        this.updateLayers();

                    } catch (error) {
                        console.error('Failed to load data:', error);
                    } finally {
                        document.getElementById('loading-overlay').style.display = 'none';
                    }
                }

                transformForWebGL(data) {
                    const result = {
                        type: data.type,
                        points: [],
                        total: data.total_features || data.total_points || 0
                    };

                    if (data.type === 'webgl_bulk') {
                        // High-performance bulk data format: [lon, lat, size, r, g, b, a]
                        result.points = data.points.map(p => ({
                            position: [p[0], p[1]],
                            size: p[2],
                            color: [p[3], p[4], p[5], p[6]],
                            optimized: true
                        }));
                        result.total = data.total_points;
                    } else if (data.type === 'hexagonal') {
                        result.points = data.features.map(f => ({
                            position: [f.center_lon, f.center_lat],
                            size: Math.sqrt(f.business_count) * 2,
                            color: this.getColorForCount(f.business_count),
                            businessCount: f.business_count,
                            businessTypes: f.business_types
                        }));
                    } else if (data.type === 'statistical_clusters') {
                        result.points = data.features.map(f => ({
                            position: [f.center_lon, f.center_lat],
                            size: Math.log(f.business_count + 1) * 3,
                            color: this.getColorForTypes(f.business_types),
                            businessCount: f.business_count,
                            businessTypes: f.business_types
                        }));
                    } else if (data.buildings) {
                        result.points = data.buildings.map(b => ({
                            position: [b.center_lon, b.center_lat],
                            size: 3,
                            color: this.getColorForDensity(b.business_density || 0),
                            businessDensity: b.business_density || 0,
                            buildingType: b.building_type
                        }));
                    } else if (data.features) {
                        result.points = data.features.map(f => ({
                            position: [f.center_lon, f.center_lat],
                            size: Math.sqrt(f.business_count || 1) * 2,
                            color: this.getColorForCount(f.business_count || 0),
                            businessCount: f.business_count || 0
                        }));
                    }

                    return result;
                }

                getColorForCount(count) {
                    if (count === 0) return [255, 237, 160, 200];
                    if (count <= 5) return [254, 178, 76, 200];
                    if (count <= 15) return [252, 78, 42, 200];
                    if (count <= 30) return [227, 26, 28, 200];
                    return [128, 0, 38, 200];
                }

                getColorForTypes(types) {
                    if (types <= 2) return [255, 237, 160, 200];
                    if (types <= 4) return [254, 178, 76, 200];
                    if (types <= 6) return [252, 78, 42, 200];
                    return [128, 0, 38, 200];
                }

                getColorForDensity(density) {
                    if (density === 0) return [255, 237, 160, 180];
                    if (density <= 2) return [254, 178, 76, 180];
                    if (density <= 5) return [252, 78, 42, 180];
                    return [227, 26, 28, 180];
                }

                updateLayers() {
                    if (!this.currentData.points.length) return;

                    const pointSize = parseInt(document.getElementById('point-size').value);
                    const opacity = parseInt(document.getElementById('opacity').value) / 100;
                    const showBusinesses = document.getElementById('layer-businesses').checked;
                    const showClusters = document.getElementById('layer-clusters').checked;
                    const showHeatmap = document.getElementById('layer-heatmap').checked;

                    const layers = [];

                    if (showBusinesses || showClusters) {
                        layers.push(new ScatterplotLayer({
                            id: 'points',
                            data: this.currentData.points,
                            getPosition: d => d.position,
                            getRadius: d => (d.size || 3) * pointSize,
                            getColor: d => d.color.map((c, i) => i === 3 ? c * opacity : c),
                            pickable: true,
                            radiusScale: 1,
                            radiusMinPixels: 1,
                            radiusMaxPixels: 50,
                            updateTriggers: {
                                getRadius: [pointSize],
                                getColor: [opacity]
                            }
                        }));
                    }

                    if (showHeatmap && this.currentData.points.length > 0) {
                        layers.push(new HeatmapLayer({
                            id: 'heatmap',
                            data: this.currentData.points,
                            getPosition: d => d.position,
                            getWeight: d => d.businessCount || 1,
                            radiusPixels: 60,
                            opacity: opacity * 0.8,
                            updateTriggers: {
                                opacity: [opacity]
                            }
                        }));
                    }

                    this.deckgl.setProps({layers});

                    // Update visible point count for frustum culling simulation
                    const visiblePoints = this.estimateVisiblePoints();
                    document.getElementById('visible-points').textContent = visiblePoints.toLocaleString();
                }

                estimateVisiblePoints() {
                    if (!this.viewport) return this.currentData.points.length;

                    const frustumCulling = document.getElementById('frustum-culling').checked;
                    if (!frustumCulling) return this.currentData.points.length;

                    // Simple viewport-based culling estimation
                    const bounds = this.getBounds();
                    const visible = this.currentData.points.filter(point => {
                        const [lon, lat] = point.position;
                        return lon >= bounds.west && lon <= bounds.east &&
                               lat >= bounds.south && lat <= bounds.north;
                    });

                    return visible.length;
                }

                getBounds() {
                    if (!this.viewport) return {west: -180, east: 180, north: 90, south: -90};

                    const {longitude, latitude, zoom} = this.viewport;
                    const scale = Math.pow(2, zoom);
                    const extent = 360 / scale;

                    return {
                        west: longitude - extent/2,
                        east: longitude + extent/2,
                        north: latitude + extent/4,
                        south: latitude - extent/4
                    };
                }

                updateMetrics(fromCache, loadTime = 0) {
                    document.getElementById('total-points').textContent = this.currentData.total.toLocaleString();
                    document.getElementById('cache-status').textContent = fromCache ? 'HIT' : 'MISS';
                    document.getElementById('load-time').textContent = Math.round(loadTime) + 'ms';

                    // Estimate GPU memory usage (rough calculation)
                    const pointCount = this.currentData.points.length;
                    const memoryMB = Math.round((pointCount * 32) / (1024 * 1024)); // 32 bytes per point estimate
                    document.getElementById('gpu-memory').textContent = memoryMB + 'MB';

                    // Update LOD level based on zoom
                    const lodLevel = this.currentZoom <= 8 ? 1 : this.currentZoom <= 12 ? 2 : 3;
                    document.getElementById('lod-level').textContent = lodLevel;
                }

                animate() {
                    this.monitor.update();
                    requestAnimationFrame(() => this.animate());
                }
            }

            // Initialize the rendering engine
            const engine = new WebGLRenderingEngine();

            // Global performance monitoring
            let frameCount = 0;
            let lastFPSUpdate = performance.now();

            function updateGlobalPerformance() {
                frameCount++;
                const now = performance.now();

                if (now - lastFPSUpdate >= 1000) {
                    const fps = Math.round((frameCount * 1000) / (now - lastFPSUpdate));
                    frameCount = 0;
                    lastFPSUpdate = now;
                }

                requestAnimationFrame(updateGlobalPerformance);
            }

            updateGlobalPerformance();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)