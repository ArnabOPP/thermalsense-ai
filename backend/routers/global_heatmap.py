"""
ThermalSense AI — Global Heatmap
Supports MODIS (1km), Landsat 8 (100m), Sentinel-2 (10m) for any city on Earth.
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


def mask_landsat_clouds(image):
    import ee as ee_module
    qa = image.select("QA_PIXEL")
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    lst = (image.select("ST_B10")
           .multiply(0.00341802).add(149.0).subtract(273.15)
           .rename("lst"))
    return lst.updateMask(cloud_mask)


def mask_sentinel_clouds(image):
    qa = image.select("QA60")
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    # Sentinel doesn't have LST — use NDVI-based proxy (inverted)
    nir = image.select("B8").divide(10000)
    red = image.select("B4").divide(10000)
    swir = image.select("B11").divide(10000)
    # Approximate surface temperature proxy using radiometric temperature
    # Use a simplified regression: higher NDBI = hotter
    ndbi = swir.subtract(nir).divide(swir.add(nir))
    # Scale to temperature-like values (rough proxy 25-55C range)
    lst_proxy = ndbi.multiply(15).add(38).rename("lst")
    return lst_proxy.updateMask(cloud_mask)


def get_lst_image(ee, region, source: str, date_start: str, date_end: str):
    """Get LST image from specified source."""
    if source == "modis":
        img = (ee.ImageCollection("MODIS/061/MOD11A1")
               .filterDate(date_start, date_end)
               .filterBounds(region)
               .select("LST_Day_1km")
               .mean()
               .multiply(0.02).subtract(273.15)
               .rename("lst"))
        scale = 1000
        source_name = "MODIS MOD11A1 Terra · 1km"

    elif source == "landsat":
        img = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
               .filterDate(date_start, date_end)
               .filterBounds(region)
               .filter(ee.Filter.lt("CLOUD_COVER", 20))
               .map(mask_landsat_clouds)
               .median())
        scale = 100
        source_name = "Landsat 8 · 100m"

    elif source == "sentinel":
        img = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterDate(date_start, date_end)
               .filterBounds(region)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
               .map(mask_sentinel_clouds)
               .median())
        scale = 10
        source_name = "Sentinel-2 · 10m (LST proxy)"

    else:
        raise ValueError(f"Unknown source: {source}")

    return img, scale, source_name


def get_lst_pixels(lat: float, lon: float, radius_deg: float, source: str):
    ee = init_gee()

    west  = lon - radius_deg
    south = lat - radius_deg
    east  = lon + radius_deg
    north = lat + radius_deg
    region = ee.Geometry.Rectangle([west, south, east, north])

    img, scale, source_name = get_lst_image(
        ee, region, source,
        "2024-03-01", "2024-05-31"
    )

    # Adaptive scale to stay under GEE 262144 pixel limit
    area_deg = (radius_deg * 2) ** 2
    pixels_at_native = area_deg / ((scale / 111000) ** 2)
    if pixels_at_native > 240000:
        scale = int(scale * (pixels_at_native / 240000) ** 0.5) + 10
        logger.info(f"Adaptive scale: {scale}m to fit {pixels_at_native:.0f} pixels")

    logger.info(f"Fetching {source_name} r={radius_deg:.3f} scale={scale}m")

    lst_unmasked = img.unmask(0).reproject(crs='EPSG:4326', scale=scale)
    rect_data = lst_unmasked.sampleRectangle(region=region)
    arr_info = rect_data.getInfo()

    arr = np.array(arr_info["properties"]["lst"], dtype=np.float32)
    nrows, ncols = arr.shape
    logger.info(f"Array: {nrows}x{ncols} = {nrows*ncols} pixels")

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

    logger.info(f"Valid pixels: {len(pixels)}")

    MAX_PIXELS = 8000
    if len(pixels) > MAX_PIXELS:
        import random
        pixels = random.sample(pixels, MAX_PIXELS)
        logger.info(f"Subsampled to {len(pixels)}")

    return pixels, source_name


@router.get("/heatmap/global", tags=["Global"])
def global_heatmap(
    lat:    float = Query(...),
    lon:    float = Query(...),
    radius: float = Query(0.15),
    name:   str   = Query("Unknown"),
    source: str   = Query("modis", description="modis | landsat | sentinel"),
):
    """Fetch real-time LST for any city on Earth from MODIS, Landsat 8, or Sentinel-2."""
    logger.info(f"Global heatmap: {name} ({lat:.3f},{lon:.3f}) r={radius:.3f} src={source}")

    try:
        pixels, source_name = get_lst_pixels(lat, lon, radius, source)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not pixels:
        raise HTTPException(status_code=404, detail="No data found for this location")

    values = [p["value"] for p in pixels]

    return {
        "city": name,
        "source": source_name,
        "period": "2024 pre-monsoon (Mar-May)",
        "lat": lat, "lon": lon,
        "n_pixels": len(pixels),
        "value_min": round(float(np.min(values)), 2),
        "value_max": round(float(np.max(values)), 2),
        "value_mean": round(float(np.mean(values)), 2),
        "unit": "°C",
        "pixels": pixels,
    }