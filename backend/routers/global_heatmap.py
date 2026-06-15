"""
ThermalSense AI — Global Heatmap via MODIS
Uses GEE sampleRectangle for exhaustive pixel extraction without rasterio.
"""
from fastapi import APIRouter, Query, HTTPException
from loguru import logger
import os
import numpy as np

router = APIRouter()


def init_gee():
    import ee
    project = os.environ.get("EE_PROJECT_ID", "thermalsense")
    sa_email = os.environ.get("EE_SERVICE_ACCOUNT")
    private_key = os.environ.get("EE_PRIVATE_KEY")
    if sa_email and private_key:
        private_key = private_key.replace('\\n', '\n')
        credentials = ee.ServiceAccountCredentials(sa_email, key_data=private_key)
        ee.Initialize(credentials, project=project)
    else:
        ee.Initialize(project=project)
    return ee


def get_modis_lst(lat: float, lon: float, radius_deg: float = 0.15):
    ee = init_gee()

    west  = lon - radius_deg
    south = lat - radius_deg
    east  = lon + radius_deg
    north = lat + radius_deg

    region = ee.Geometry.Rectangle([west, south, east, north])

    modis = (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterDate("2024-03-01", "2024-05-31")
        .filterBounds(region)
        .select("LST_Day_1km")
        .mean()
    )

    lst_celsius = modis.multiply(0.02).subtract(273.15).rename("lst")

    # sampleRectangle returns a 2D array of all pixels
    logger.info(f"Sampling rectangle radius={radius_deg:.3f} for ({lat:.2f},{lon:.2f})")
    
    # Adaptive scale — keep under GEE's 262144 pixel limit
    MAX_GEE_PIXELS = 250000
    area_deg = (radius_deg * 2) ** 2
    pixels_at_1km = area_deg / (0.009 ** 2)
    if pixels_at_1km > MAX_GEE_PIXELS:
        scale = int(1000 * (pixels_at_1km / MAX_GEE_PIXELS) ** 0.5) + 100
    else:
        scale = 1000
    logger.info(f"Using scale={scale}m for {pixels_at_1km:.0f} potential pixels")
    lst_unmasked = lst_celsius.unmask(0).reproject(crs='EPSG:4326', scale=scale)
    rect_data = lst_unmasked.sampleRectangle(region=region)
    arr_info = rect_data.getInfo()
    
    arr = np.array(arr_info["properties"]["lst"], dtype=np.float32)
    nrows, ncols = arr.shape
    logger.info(f"Got array: {nrows}x{ncols} = {nrows*ncols} pixels")

    lat_step = (north - south) / nrows
    lon_step = (east - west) / ncols

    pixels = []
    for row in range(nrows):
        for col in range(ncols):
            val = float(arr[row, col])
            if not (10 < val < 65):
                continue
            px_lat = north - (row + 0.5) * lat_step
            px_lon = west  + (col + 0.5) * lon_step
            pixels.append({
                "lat": round(px_lat, 5),
                "lon": round(px_lon, 5),
                "value": round(val, 2)
            })

    logger.info(f"Extracted {len(pixels)} valid pixels")

    # Random subsample for browser performance
    MAX_PIXELS = 5000
    if len(pixels) > MAX_PIXELS:
        import random
        pixels = random.sample(pixels, MAX_PIXELS)
        logger.info(f"Subsampled to {len(pixels)} pixels")

    return pixels


@router.get("/heatmap/global", tags=["Global"])
def global_heatmap(
    lat:    float = Query(...),
    lon:    float = Query(...),
    radius: float = Query(0.15),
    name:   str   = Query("Unknown"),
):
    """Fetch real-time MODIS LST for any city on Earth."""
    logger.info(f"Global heatmap: {name} ({lat:.3f},{lon:.3f}) r={radius:.3f}")

    try:
        pixels = get_modis_lst(lat, lon, radius)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MODIS fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not pixels:
        raise HTTPException(status_code=404, detail="No MODIS data found")

    values = [p["value"] for p in pixels]

    return {
        "city": name,
        "source": "MODIS MOD11A1 (Terra) 1km LST",
        "period": "2024 pre-monsoon (Mar-May)",
        "lat": lat, "lon": lon,
        "n_pixels": len(pixels),
        "value_min": round(float(np.min(values)), 2),
        "value_max": round(float(np.max(values)), 2),
        "value_mean": round(float(np.mean(values)), 2),
        "unit": "°C",
        "pixels": pixels,
    }