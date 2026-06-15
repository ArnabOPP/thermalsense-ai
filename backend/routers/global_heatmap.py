"""
ThermalSense AI — Global Heatmap (FIXED date handling)
- Validates collection is non-empty before processing
- Auto-widens date window by ±8 days if no image found for exact date
- Returns actual acquisition date in response
"""
from fastapi import APIRouter, Query, HTTPException
from loguru import logger
import os
import numpy as np
from datetime import datetime, timedelta

router = APIRouter()


def init_gee():
    import ee
    project    = os.environ.get("EE_PROJECT_ID", "thermalsense")
    sa_email   = os.environ.get("EE_SERVICE_ACCOUNT")
    private_key = os.environ.get("EE_PRIVATE_KEY")
    if sa_email and private_key:
        private_key = private_key.replace('\\\\n', '\n')
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
    nir  = image.select("B8").divide(10000)
    swir = image.select("B11").divide(10000)
    ndbi = swir.subtract(nir).divide(swir.add(nir))
    lst_proxy = ndbi.multiply(15).add(38).rename("lst")
    return lst_proxy.updateMask(cloud_mask)


def date_window(date_str: str, days: int):
    """Return (start, end) strings expanding date_str by ±days."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    s = (d - timedelta(days=days)).strftime("%Y-%m-%d")
    e = (d + timedelta(days=days + 1)).strftime("%Y-%m-%d")
    return s, e


def get_lst_image(ee, region, source: str, date_str: str):
    """
    Get LST image for a SINGLE target date.
    Tries exact day first; if empty, expands window by ±8 days.
    Returns (image, scale, source_name, actual_period_used).
    """
    # For a single day, GEE filterDate needs [start, start+1day)
    d = datetime.strptime(date_str, "%Y-%m-%d")
    day_start = date_str
    day_end   = (d + timedelta(days=1)).strftime("%Y-%m-%d")

    if source == "modis":
        # MODIS is daily — try exact day, then ±8 days if cloudy/missing
        def build_modis(s, e):
            return (ee.ImageCollection("MODIS/061/MOD11A1")
                    .filterDate(s, e)
                    .filterBounds(region)
                    .select("LST_Day_1km")
                    .mean()
                    .multiply(0.02).subtract(273.15)
                    .rename("lst"))

        col = (ee.ImageCollection("MODIS/061/MOD11A1")
               .filterDate(day_start, day_end)
               .filterBounds(region))
        count = col.size().getInfo()
        if count > 0:
            img = build_modis(day_start, day_end)
            period_used = date_str
        else:
            # Widen to ±8 days
            ws, we = date_window(date_str, 8)
            logger.warning(f"No MODIS image on {date_str}, widening to {ws}→{we}")
            col2 = (ee.ImageCollection("MODIS/061/MOD11A1")
                    .filterDate(ws, we)
                    .filterBounds(region))
            count2 = col2.size().getInfo()
            if count2 == 0:
                raise HTTPException(status_code=404,
                    detail=f"No MODIS data found for {date_str} ±8 days. Try a different date.")
            img = build_modis(ws, we)
            period_used = f"{ws} to {we} (nearest available)"

        scale = 1000
        source_name = "MODIS MOD11A1 Terra · 1km"

    elif source == "landsat":
        # Landsat has 16-day repeat — widen to ±16 days to guarantee a pass
        ws, we = date_window(date_str, 16)
        col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
               .filterDate(ws, we)
               .filterBounds(region)
               .filter(ee.Filter.lt("CLOUD_COVER", 30)))
        count = col.size().getInfo()
        if count == 0:
            ws2, we2 = date_window(date_str, 32)
            col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                   .filterDate(ws2, we2)
                   .filterBounds(region)
                   .filter(ee.Filter.lt("CLOUD_COVER", 50)))
            count = col.size().getInfo()
            if count == 0:
                raise HTTPException(status_code=404,
                    detail=f"No Landsat 8 imagery near {date_str} (±32 days). Try a different date or MODIS.")
            ws, we = ws2, we2
        img = col.map(mask_landsat_clouds).median()
        period_used = f"nearest pass ±16 days of {date_str}"
        scale = 100
        source_name = "Landsat 8 · 100m"

    elif source == "sentinel":
        ws, we = date_window(date_str, 15)
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterDate(ws, we)
               .filterBounds(region)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)))
        count = col.size().getInfo()
        if count == 0:
            ws2, we2 = date_window(date_str, 30)
            col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                   .filterDate(ws2, we2)
                   .filterBounds(region)
                   .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50)))
            count = col.size().getInfo()
            if count == 0:
                raise HTTPException(status_code=404,
                    detail=f"No Sentinel-2 imagery near {date_str} (±30 days). Try MODIS instead.")
            ws, we = ws2, we2
        img = col.map(mask_sentinel_clouds).median()
        period_used = f"nearest pass ±15 days of {date_str}"
        scale = 10
        source_name = "Sentinel-2 · 10m (LST proxy)"

    else:
        raise ValueError(f"Unknown source: {source}")

    return img, scale, source_name, period_used


def get_lst_pixels(lat: float, lon: float, radius_deg: float,
                   source: str, date_str: str):
    ee = init_gee()

    west   = lon - radius_deg
    south  = lat - radius_deg
    east   = lon + radius_deg
    north  = lat + radius_deg
    region = ee.Geometry.Rectangle([west, south, east, north])

    img, scale, source_name, period_used = get_lst_image(ee, region, source, date_str)

    # Adaptive scale to stay under GEE 262144 pixel limit
    area_deg = (radius_deg * 2) ** 2
    pixels_at_native = area_deg / ((scale / 111000) ** 2)
    if pixels_at_native > 240000:
        scale = int(scale * (pixels_at_native / 240000) ** 0.5) + 10
        logger.info(f"Adaptive scale: {scale}m")

    logger.info(f"Fetching {source_name} | date={date_str} | r={radius_deg:.3f} | scale={scale}m")

    lst_unmasked = img.unmask(0).reproject(crs='EPSG:4326', scale=scale)
    rect_data    = lst_unmasked.sampleRectangle(region=region)
    arr_info     = rect_data.getInfo()

    arr = np.array(arr_info["properties"]["lst"], dtype=np.float32)
    nrows, ncols = arr.shape
    logger.info(f"Array: {nrows}x{ncols}")

    lat_step = (north - south) / nrows
    lon_step = (east  - west)  / ncols

    # Temperature validity filter — must be physically plausible
    # MODIS LST for India: 5°C (winter nights) to 60°C (summer desert surface)
    T_MIN, T_MAX = 5.0, 62.0

    pixels = []
    for row in range(nrows):
        for col in range(ncols):
            val = float(arr[row, col])
            if not (T_MIN < val < T_MAX):
                continue
            px_lat = north - (row + 0.5) * lat_step
            px_lon = west  + (col + 0.5) * lon_step
            pixels.append({
                "lat":   round(px_lat, 5),
                "lon":   round(px_lon, 5),
                "value": round(val, 2)
            })

    logger.info(f"Valid pixels: {len(pixels)}")

    if not pixels:
        raise HTTPException(status_code=404,
            detail=f"No valid LST pixels for {date_str}. The area may be cloud-covered or the date has no data.")

    MAX_PIXELS = 8000
    if len(pixels) > MAX_PIXELS:
        import random
        pixels = random.sample(pixels, MAX_PIXELS)

    return pixels, source_name, period_used


@router.get("/heatmap/global", tags=["Global"])
def global_heatmap(
    lat:    float = Query(...),
    lon:    float = Query(...),
    radius: float = Query(0.15),
    name:   str   = Query("Unknown"),
    source: str   = Query("modis", description="modis | landsat | sentinel"),
    date_start: str = Query("2024-03-01", description="Target date YYYY-MM-DD (date_end ignored, single-day logic)"),
    date_end:   str = Query("2024-05-31", description="Ignored — kept for backward compat"),
):
    """Fetch real LST for any city · MODIS / Landsat 8 / Sentinel-2"""
    # Use date_start as the single target date (date_end is legacy)
    target_date = date_start
    logger.info(f"Global heatmap: {name} ({lat:.3f},{lon:.3f}) src={source} date={target_date}")

    try:
        pixels, source_name, period_used = get_lst_pixels(lat, lon, radius, source, target_date)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    values = [p["value"] for p in pixels]

    return {
        "city":        name,
        "source":      source_name,
        "period":      period_used,
        "target_date": target_date,
        "lat": lat, "lon": lon,
        "n_pixels":    len(pixels),
        "value_min":   round(float(np.min(values)),  2),
        "value_max":   round(float(np.max(values)),  2),
        "value_mean":  round(float(np.mean(values)), 2),
        "unit":        "°C",
        "pixels":      pixels,
    }