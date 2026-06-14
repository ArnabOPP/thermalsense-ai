"""
ThermalSense AI — Global Heatmap via MODIS
Fetches real-time LST for any city bbox from Google Earth Engine MODIS MOD11A1.
"""
from fastapi import APIRouter, Query, HTTPException
from loguru import logger
import os
import numpy as np

router = APIRouter()


def get_modis_lst(lat: float, lon: float, radius_deg: float = 0.15):
    """Fetch MODIS LST for a bbox around lat/lon using GEE."""
    import ee

    project = os.environ.get("EE_PROJECT_ID", "thermalsense")

    try:
        ee.Initialize(project=project)
    except Exception:
        try:
            ee.Authenticate(auth_mode="service_account")
            ee.Initialize(project=project)
        except Exception as auth_err:
            raise HTTPException(status_code=500, detail=f"GEE auth failed: {auth_err}")

    bbox = ee.Geometry.BBox(
        lon - radius_deg, lat - radius_deg,
        lon + radius_deg, lat + radius_deg
    )

    modis = (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterDate("2024-03-01", "2024-05-31")
        .filterBounds(bbox)
        .select("LST_Day_1km")
        .mean()
    )

    lst_celsius = modis.multiply(0.02).subtract(273.15)

    points = lst_celsius.sample(
        region=bbox,
        scale=1000,
        numPixels=2000,
        seed=42,
        geometries=True
    )

    features = points.getInfo()["features"]
    pixels = []
    for f in features:
        val = f["properties"].get("LST_Day_1km")
        coords = f["geometry"]["coordinates"]
        if val is not None and 10 < val < 65:
            pixels.append({
                "lat": round(coords[1], 5),
                "lon": round(coords[0], 5),
                "value": round(float(val), 2)
            })

    return pixels


@router.get("/heatmap/global", tags=["Global"])
def global_heatmap(
    lat:    float = Query(..., description="City center latitude"),
    lon:    float = Query(..., description="City center longitude"),
    radius: float = Query(0.15, description="Radius in degrees (~15km)"),
    name:   str   = Query("Unknown", description="City name for logging"),
):
    """Fetch real-time MODIS LST for any city on Earth."""
    logger.info(f"Global heatmap: {name} ({lat:.3f}, {lon:.3f})")

    try:
        pixels = get_modis_lst(lat, lon, radius)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MODIS fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not pixels:
        raise HTTPException(status_code=404, detail="No MODIS data found for this location")

    values = [p["value"] for p in pixels]

    return {
        "city": name,
        "source": "MODIS MOD11A1 (Terra) 1km LST",
        "period": "2024 pre-monsoon (Mar-May)",
        "lat": lat,
        "lon": lon,
        "n_pixels": len(pixels),
        "value_min": round(float(np.min(values)), 2),
        "value_max": round(float(np.max(values)), 2),
        "value_mean": round(float(np.mean(values)), 2),
        "unit": "°C",
        "pixels": pixels,
    }