"""
ThermalSense AI — Script 02
Pull Sentinel-2 spectral indices (NDVI, NDWI, NDBI) + albedo for any city.

What this produces:
  outputs/processed/{city}/sentinel2/s2_indices_{year}_{season}_utm{epsg}.tif
  — 4-band GeoTIFF: band1=NDVI, band2=NDWI, band3=NDBI, band4=albedo

Usage:
  python scripts/02_pull_sentinel2.py --city kolkata
  python scripts/02_pull_sentinel2.py --city delhi --year 2024

Author: Person A
"""

import argparse
import sys
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio.enums import Resampling
import requests
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils import (
    setup_logger, load_config, get_city_config, print_banner,
    retry, ensure_dirs, save_metadata, bbox_to_ee_geometry,
    log_raster_stats
)


# ─── Cloud masking ─────────────────────────────────────────────────────────────

def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """
    Mask clouds in Sentinel-2 using the QA60 band.
    Bit 10 = opaque clouds, Bit 11 = cirrus clouds.
    """
    qa = image.select("QA60")
    cloud_mask = (
        qa.bitwiseAnd(1 << 10).eq(0)
        .And(qa.bitwiseAnd(1 << 11).eq(0))
    )
    return image.updateMask(cloud_mask).divide(10000)  # scale to [0,1]


# ─── Index computation ─────────────────────────────────────────────────────────

def compute_indices(image: ee.Image) -> ee.Image:
    """
    Compute spectral indices from Sentinel-2 SR bands.

    NDVI  = (NIR - Red) / (NIR + Red)          — vegetation density
    NDWI  = (Green - NIR) / (Green + NIR)       — water bodies
    NDBI  = (SWIR1 - NIR) / (SWIR1 + NIR)      — built-up index
    Albedo = simplified Liang (2001) formula for broadband surface albedo
    """
    nir   = image.select("B8")
    red   = image.select("B4")
    green = image.select("B3")
    blue  = image.select("B2")
    swir1 = image.select("B11")
    swir2 = image.select("B12")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi")
    ndwi = green.subtract(nir).divide(green.add(nir)).rename("ndwi")
    ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename("ndbi")

    # Liang 2001 simplified broadband albedo
    albedo = (
        blue.multiply(0.356)
        .add(red.multiply(0.130))
        .add(nir.multiply(0.373))
        .add(swir1.multiply(0.085))
        .add(swir2.multiply(0.072))
        .subtract(0.0018)
        .rename("albedo")
    )

    return ee.Image.cat([ndvi, ndwi, ndbi, albedo])


# ─── Download ──────────────────────────────────────────────────────────────────

@retry(max_attempts=3, wait_seconds=15)
def download_multiband_image(
    image: ee.Image,
    bands: list[str],
    bbox: list[float],
    scale: int,
    output_path: Path,
) -> None:
    """Download a multi-band GEE image as a single GeoTIFF."""
    west, south, east, north = bbox
    west, south, east, north = bbox
    region = ee.Geometry.Rectangle([west, south, east, north])

    url = image.select(bands).getDownloadURL({
        "scale": scale,
        "region": region,
        "format": "GEO_TIFF",
        "filePerBand": False,
    })

    logger.info("  Downloading Sentinel-2 composite... (30-120 seconds)")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"  Downloaded: {output_path.name} ({size_mb:.1f} MB)")


def reproject_multiband(
    input_path: Path,
    output_path: Path,
    utm_epsg: int,
    resolution: int,
    band_names: list[str],
) -> dict[str, np.ndarray]:
    """Reproject multiband GeoTIFF to UTM, return dict of band arrays."""
    import subprocess

    cmd = [
        "gdalwarp",
        "-t_srs", f"EPSG:{utm_epsg}",
        "-tr", str(resolution), str(resolution),
        "-r", "bilinear",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        "-overwrite",
        str(input_path),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdalwarp failed:\n{result.stderr}")

    arrays = {}
    with rasterio.open(output_path) as src:
        for i, name in enumerate(band_names, start=1):
            arr = src.read(i).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            arrays[name] = arr

    return arrays


# ─── LULC classification ───────────────────────────────────────────────────────

def classify_lulc(ndvi: np.ndarray, ndwi: np.ndarray, ndbi: np.ndarray) -> np.ndarray:
    """
    Simple rule-based LULC classification from spectral indices.
    Classes:
      1 = water          (NDWI > 0.2)
      2 = dense veg      (NDVI > 0.4)
      3 = sparse veg     (NDVI 0.1–0.4)
      4 = built-up       (NDBI > 0.0 and NDVI < 0.1)
      5 = bare soil      (everything else)

    Note: for the proposal we use this quick classification.
    For production, Bhuvan LULC or a trained classifier replaces this.
    """
    lulc = np.full(ndvi.shape, 5, dtype=np.uint8)  # default: bare soil
    lulc[ndbi > 0.0] = 4                             # built-up
    lulc[(ndvi > 0.1) & (ndvi <= 0.4)] = 3          # sparse veg
    lulc[ndvi > 0.4] = 2                              # dense veg
    lulc[ndwi > 0.2] = 1                              # water (highest priority)
    return lulc


# ─── Main ──────────────────────────────────────────────────────────────────────

def run(city: str, years: list[int], seasons: list[str]) -> None:
    setup_logger("02_pull_sentinel2")
    print_banner(
        "ThermalSense AI — Script 02",
        f"Pulling Sentinel-2 indices | City: {city.upper()} | Years: {years}",
    )

    cfg_all = load_config()
    city_cfg = get_city_config(city)
    pipe_cfg = cfg_all["pipeline"]

    import os
    project_id = os.environ.get("EE_PROJECT_ID", "")
    if not project_id:
        logger.error("EE_PROJECT_ID not set in .env")
        sys.exit(1)

    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)

    bbox = city_cfg["bbox"]
    utm_epsg = city_cfg["utm_epsg"]
    resolution = pipe_cfg["target_resolution"]
    cloud_max = pipe_cfg["sentinel2"]["cloud_cover_max"]
    season_defs = pipe_cfg["landsat"]["seasons"]  # same season def as Landsat

    paths = cfg_all["paths"]
    raw_dir = ROOT / paths["raw_dir"] / city / "sentinel2"
    proc_dir = ROOT / paths["processed_dir"] / city / "sentinel2"
    ensure_dirs(raw_dir, proc_dir)

    bbox_geom = bbox_to_ee_geometry(bbox)
    band_names = ["ndvi", "ndwi", "ndbi", "albedo"]

    for year in years:
        for season in seasons:
            if season not in season_defs:
                continue

            sm = season_defs[season]
            start_date = f"{year}-{sm[0]:02d}-01"
            end_month = sm[1]
            end_year = year if end_month < 12 else year + 1
            end_month_adj = (end_month % 12) + 1
            end_date = f"{end_year}-{end_month_adj:02d}-01"
            tag = f"{year}_{season}"

            raw_path = raw_dir / f"s2_indices_{tag}_wgs84.tif"
            proc_path = proc_dir / f"s2_indices_{tag}_utm{utm_epsg}.tif"
            lulc_path = proc_dir / f"s2_lulc_{tag}_utm{utm_epsg}.tif"

            if proc_path.exists() and lulc_path.exists():
                logger.info(f"  {tag}: already processed, skipping")
                continue

            logger.info(f"\n{'─'*60}")
            logger.info(f"Processing: Sentinel-2 | {city.upper()} | {tag}")

            # Build composite
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(bbox_geom)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
                .map(mask_s2_clouds)
            )

            count = collection.size().getInfo()
            logger.info(f"  Sentinel-2 scenes: {count}")
            if count == 0:
                logger.warning(f"  No scenes for {tag}, trying with cloud_max=30%")
                collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(bbox_geom)
                    .filterDate(start_date, end_date)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                    .map(mask_s2_clouds)
                )
                count = collection.size().getInfo()
                if count == 0:
                    logger.error(f"  No scenes at all for {tag} — skipping")
                    continue

            composite = collection.map(compute_indices).median()

            # Download
            download_multiband_image(composite, band_names, bbox, resolution, raw_path)

            # Reproject
            logger.info(f"  Reprojecting to EPSG:{utm_epsg} at {resolution}m...")
            arrays = reproject_multiband(raw_path, proc_path, utm_epsg, resolution, band_names)

            # Log stats
            for name, arr in arrays.items():
                log_raster_stats(arr, f"S2 {name} {tag}")

            # NDVI sanity check
            ndvi_mean = float(np.nanmean(arrays["ndvi"]))
            if ndvi_mean < 0.05 or ndvi_mean > 0.8:
                logger.warning(f"Unexpected NDVI mean: {ndvi_mean:.3f} — check imagery")

            # Quick LULC classification
            lulc = classify_lulc(arrays["ndvi"], arrays["ndwi"], arrays["ndbi"])

            # Save LULC as separate GeoTIFF
            with rasterio.open(proc_path) as src:
                meta = src.meta.copy()
            meta.update(dtype=rasterio.uint8, count=1, nodata=0)
            with rasterio.open(lulc_path, "w", **meta) as dst:
                dst.write(lulc, 1)
                dst.update_tags(
                    classes="1=water 2=dense_veg 3=sparse_veg 4=built_up 5=bare_soil"
                )
            logger.info(f"  LULC saved: {lulc_path.name}")

            # Class distribution
            total = lulc.size
            for cls_id, cls_name in [(1,"water"),(2,"dense_veg"),(3,"sparse_veg"),(4,"built_up"),(5,"bare_soil")]:
                pct = (lulc == cls_id).sum() / total * 100
                logger.info(f"    {cls_name}: {pct:.1f}%")

            save_metadata(proc_path, {
                "city": city,
                "year": year,
                "season": season,
                "scene_count": count,
                "bands": band_names,
                "resolution_m": resolution,
                "utm_epsg": utm_epsg,
                "bbox_wgs84": bbox,
                "ndvi_mean": float(np.nanmean(arrays["ndvi"])),
                "albedo_mean": float(np.nanmean(arrays["albedo"])),
            })

            logger.success(f"  ✓ {tag} complete → {proc_path.name}")

    logger.success("Script 02 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull Sentinel-2 spectral indices via GEE")
    parser.add_argument("--city", default="kolkata")
    parser.add_argument("--year", type=int, nargs="+")
    parser.add_argument("--season", nargs="+")
    args = parser.parse_args()

    cfg = load_config()
    years = args.year or cfg["pipeline"]["landsat"]["years"]
    seasons = args.season or list(cfg["pipeline"]["landsat"]["seasons"].keys())

    run(city=args.city, years=years, seasons=seasons)
