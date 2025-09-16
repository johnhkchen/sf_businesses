#!/usr/bin/env python3

import json
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SF Business Building Visualization API", version="1.0.0")

class BuildingVisualizationAPI:
    """API for building-level business visualization in San Francisco."""

    def __init__(self):
        self.output_dir = Path("output")
        self.buildings_df = None
        self.businesses_df = None
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
        Aggregate business data by building with spatial filtering.

        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            zoom_level: Map zoom level for detail adjustment

        Returns:
            Dict with aggregated building-business data
        """
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
            "/map": "Interactive map visualization"
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

            function getColor(density) {
                return density > 8 ? '#800026' :
                       density > 6 ? '#BD0026' :
                       density > 4 ? '#E31A1C' :
                       density > 2 ? '#FC4E2A' :
                       density > 0 ? '#FEB24C' :
                                     '#FFEDA0';
            }

            function updateData() {
                const bounds = map.getBounds();
                const bbox = `${bounds.getWest()},${bounds.getSouth()},${bounds.getEast()},${bounds.getNorth()}`;
                const zoom = Math.round(map.getZoom());

                if (Math.abs(zoom - currentZoom) < 1) return; // Avoid excessive updates
                currentZoom = zoom;

                fetch(`/buildings?bbox=${bbox}&zoom=${zoom}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.type === 'grid') {
                            updateGridDisplay(data.features);
                        } else {
                            updateBuildingDisplay(data.buildings);
                        }
                    })
                    .catch(error => console.error('Error:', error));
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)