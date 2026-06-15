"""
ThermalSense AI — Global Heatmap (fixed GEE auth + single-date logic)
"""
from fastapi import APIRouter, Query, HTTPException
from loguru import logger
import os
import json
import numpy as np
from datetime import datetime, timedelta

router = APIRouter()


def init_gee():
    import ee

    project     = os.environ.get("EE_PROJECT_ID", "thermalsense")
    sa_email    = os.environ.get("EE_SERVICE_ACCOUNT")
    private_key = os.environ.get("EE_PRIVATE_KEY")

    if sa_email and private_key:
        key = private_key

        # Fix escaped newlines — Railway often stores \n as literal backslash-n
        if '\\n' in key:
            key = key.replace('\\n', '\n')

        # Validate PEM header present
        if '-----BEGIN' not in key:
            raise RuntimeError(
                "EE_PRIVATE_KEY missing PEM header. "
                "Ensure the Railway env var contains the raw JSON from your service account key file."
            )

        # Support: full JSON blob OR bare PEM private key string
        stripped = key.strip()
        if stripped.startswith('{'):
            # Full service-account JSON
            try:
                key_data = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"EE_PRIVATE_KEY JSON parse failed: {e}")
            credentials = ee.ServiceAccountCredentials(
                sa_email, key_data=json.dumps(key_data)
            )
        else:
            # Bare PEM — wrap in minimal JSON GEE expects
            key_json = json.dumps({
                "type": "service_account",
                "client_email": sa_email,
                "private_key": key,
                "private_key_id": "",
                "token_uri": "https://oauth2.googleapis.com/token",
            })
            credentials = ee.ServiceAccountCredentials(sa_email, key_data=key_json)

        ee.Initialize(credentials, project=project)
        logger.info(f"GEE initialized with service account: {sa_email}")
    else:
        ee.Initialize(project=project)
        logger.info("GEE initialized with application default credentials")

    return ee


def mask_landsat_clouds(image):
    qa = image.select("QA_PIXEL")
    cloud_mask = (qa.bitwiseAnd(1 << 3).eq(0)
                    .And(qa.bitwiseAnd(1 << 4).eq(0)))
    lst = (image.select("ST_B10")
               .multiply(0.00341802).add(149.0).subtract(273.15)
               .rename("lst"))
    return lst.updateMask(cloud_mask)


def mask_sentinel_clouds(image):
    qa = image.select("QA60")
    cloud_mask = (qa.bitwiseAnd(1 << 10).eq(0)
                    .And(qa.bitwiseAnd(1 << 11).eq(0)))
    nir  = image.select("B8").divide(10000)
    swir = image.select("B11").divide(10000)
    ndbi = swir.subtract(nir).divide(swir.add(nir))
    return ndbi.multiply(15).add(38).rename("lst").updateMask(cloud_mask)


def expand_window(date_str: str, days: int):
    """Return (start_str, end_str) symmetric around date_str by ±days."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    s = (d - timedelta(days=days)).strftime("%Y-%m-%d")
    e = (d + timedelta(days=days + 1)).strftime("%Y-%m-%d")
    return s, e


def get_lst_image(ee, region, source: str, date_str: str):
    """
    Returns (image, scale_m, source_name, period_label).
    For single target date — widens window automatically if no data found.
    """
    d = datetime.strptime(date_str, "%Y-%m-%d")
    d1 = date_str
    d2 = (d + timedelta(days=1)).strftime("%Y-%m-%d")

    if source == "modis":
        def make(s, e):
            return (ee.ImageCollection("MODIS/061/MOD11A1")
                    .filterDate(s, e)
                    .filterBounds(region)
                    .select("LST_Day_1km")
                    .mean()
                    .multiply(0.02).subtract(273.15)
                    .rename("lst"))

        n = (ee.ImageCollection("MODIS/061/MOD11A1")
             .filterDate(d1, d2).filterBounds(region).size().getInfo())

        if n > 0:
            img, label = make(d1, d2), date_str
        else:
            ws, we = expand_window(date_str, 8)
            n2 = (ee.ImageCollection("MODIS/061/MOD11A1")
                  .filterDate(ws, we).filterBounds(region).size().getInfo())
            if n2 == 0:
                raise HTTPException(404,
                    detail=f"No MODIS data for {date_str} ±8 days. Try a different date.")
            img, label = make(ws, we), f"~{date_str} (±8 day window)"
        scale, name = 1000, "MODIS MOD11A1 Terra · 1km"

    elif source == "landsat":
        ws, we = expand_window(date_str, 16)
        col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
               .filterDate(ws, we).filterBounds(region)
               .filter(ee.Filter.lt("CLOUD_COVER", 30)))
        if col.size().getInfo() == 0:
            ws, we = expand_window(date_str, 32)
            col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
                   .filterDate(ws, we).filterBounds(region)
                   .filter(ee.Filter.lt("CLOUD_COVER", 50)))
            if col.size().getInfo() == 0:
                raise HTTPException(404,
                    detail=f"No Landsat 8 data near {date_str} (±32 days). Try MODIS.")
        img   = col.map(mask_landsat_clouds).median()
        label = f"nearest pass ±16 days of {date_str}"
        scale, name = 100, "Landsat 8 · 100m"

    elif source == "sentinel":
        ws, we = expand_window(date_str, 15)
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
               .filterDate(ws, we).filterBounds(region)
               .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)))
        if col.size().getInfo() == 0:
            ws, we = expand_window(date_str, 30)
            col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                   .filterDate(ws, we).filterBounds(region)
                   .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50)))
            if col.size().getInfo() == 0:
                raise HTTPException(404,
                    detail=f"No Sentinel-2 data near {date_str} (±30 days). Try MODIS.")
        img   = col.map(mask_sentinel_clouds).median()
        label = f"nearest pass ±15 days of {date_str}"
        scale, name = 10, "Sentinel-2 · 10m (LST proxy)"

    else:
        raise ValueError(f"Unknown source: {source}")

    return img, scale, name, label


def get_lst_pixels(lat, lon, radius_deg, source, date_str):
    ee = init_gee()

    w, s, e, n = lon-radius_deg, lat-radius_deg, lon+radius_deg, lat+radius_deg
    region = ee.Geometry.Rectangle([w, s, e, n])

    img, scale, source_name, period = get_lst_image(ee, region, source, date_str)

    # Adaptive scale — stay under GEE 262144 pixel limit
    area_deg2       = (radius_deg * 2) ** 2
    pixels_native   = area_deg2 / ((scale / 111000) ** 2)
    if pixels_native > 240000:
        scale = int(scale * (pixels_native / 240000) ** 0.5) + 10
        logger.info(f"Adaptive scale → {scale}m")

    logger.info(f"{source_name} | {date_str} | r={radius_deg:.3f} | scale={scale}m")

    arr_info = (img.unmask(0)
                   .reproject(crs='EPSG:4326', scale=scale)
                   .sampleRectangle(region=region)
                   .getInfo())

    arr = np.array(arr_info["properties"]["lst"], dtype=np.float32)
    nrows, ncols = arr.shape

    lat_step = (n - s) / nrows
    lon_step = (e - w) / ncols

    pixels = []
    for r in range(nrows):
        for c in range(ncols):
            val = float(arr[r, c])
            if not (5.0 < val < 65.0):
                continue
            pixels.append({
                "lat":   round(n - (r + 0.5) * lat_step, 5),
                "lon":   round(w + (c + 0.5) * lon_step, 5),
                "value": round(val, 2),
            })

    logger.info(f"Valid pixels: {len(pixels)}")

    if not pixels:
        raise HTTPException(404,
            detail=f"No valid LST pixels for {date_str}. Area may be cloud-covered.")

    if len(pixels) > 8000:
        import random
        pixels = random.sample(pixels, 8000)

    return pixels, source_name, period


@router.get("/heatmap/global", tags=["Global"])
def global_heatmap(
    lat:        float = Query(...),
    lon:        float = Query(...),
    radius:     float = Query(0.15),
    name:       str   = Query("Unknown"),
    source:     str   = Query("modis"),
    date_start: str   = Query("2024-03-01"),
    date_end:   str   = Query("2024-05-31"),   # kept for backward compat, ignored
):
    """Real LST for any city · MODIS / Landsat 8 / Sentinel-2"""
    target = date_start   # single-date mode
    logger.info(f"Heatmap: {name} ({lat:.3f},{lon:.3f}) src={source} date={target}")

    try:
        pixels, source_name, period = get_lst_pixels(lat, lon, radius, source, target)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        raise HTTPException(500, detail=str(e))

    values = [p["value"] for p in pixels]
    return {
        "city":        name,
        "source":      source_name,
        "period":      period,
        "target_date": target,
        "lat": lat, "lon": lon,
        "n_pixels":    len(pixels),
        "value_min":   round(float(np.min(values)),  2),
        "value_max":   round(float(np.max(values)),  2),
        "value_mean":  round(float(np.mean(values)), 2),
        "unit":        "°C",
        "pixels":      pixels,
    }