"""
ThermalSense AI — Global Heatmap via MODIS
Downloads full MODIS LST raster for any bbox and extracts all pixels.
"""
from fastapi import APIRouter, Query, HTTPException
from loguru import logger
import os, io, tempfile
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
    import requests
    ee = init_gee()

    bbox_coords = [
        lon - radius_deg, lat - radius_deg,
        lon + radius_deg, lat + radius_deg
    ]
    bbox = ee.Geometry.BBox(*bbox_coords)

    modis = (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterDate("2024-03-01", "2024-05-31")
        .filterBounds(bbox)
        .select("LST_Day_1km")
        .mean()
    )

    lst_celsius = modis.multiply(0.02).subtract(273.15)

    # Download full raster
    url = lst_celsius.getDownloadURL({
        "scale": 1000,
        "region": bbox,
        "format": "GEO_TIFF",
        "filePerBand": False,
    })

    logger.info(f"Downloading MODIS raster for {lat:.2f},{lon:.2f} r={radius_deg:.2f}...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    # Parse GeoTIFF
    import rasterio
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        f.write(r.content)
        tif_path = f.name

    pixels = []
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        transform = src.transform
        nrows, ncols = arr.shape

        for row in range(nrows):
            for col in range(ncols):
                val = arr[row, col]
                if nodata is not None and val == nodata:
                    continue
                if not (10 < val < 65):
                    continue
                lon_px, lat_px = transform * (col + 0.5, row + 0.5)
                pixels.append({
                    "lat": round(float(lat_px), 5),
                    "lon": round(float(lon_px), 5),
                    "value": round(float(val), 2)
                })

    os.unlink(tif_path)
    logger.info(f"Extracted {len(pixels)} pixels")
    # Subsample for browser performance using random sampling
    MAX_PIXELS = 5000
    if len(pixels) > MAX_PIXELS:
        import random
        pixels = random.sample(pixels, MAX_PIXELS)
        logger.info(f"Subsampled to {len(pixels)} pixels for browser")
    return pixels


@router.get("/heatmap/global", tags=["Global"])
def global_heatmap(
    lat:    float = Query(...),
    lon:    float = Query(...),
    radius: float = Query(0.15),
    name:   str   = Query("Unknown"),
):
    """Fetch real-time MODIS LST for any city on Earth — full raster coverage."""
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