"""
ThermalSense AI — Script 01
Pull Landsat 8/9 Land Surface Temperature for any city via Google Earth Engine.

What this produces:
  outputs/raw/{city}/landsat_lst_{year}_{season}.tif   — LST in °C, 100m, UTM
  outputs/raw/{city}/landsat_lst_{year}_{season}.meta.json

Usage:
  python scripts/01_pull_landsat_lst.py --city kolkata
  python scripts/01_pull_landsat_lst.py --city delhi
  python scripts/01_pull_landsat_lst.py --city kolkata --year 2024 --season pre_monsoon

Author: Person A
"""

import argparse
import sys
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import requests
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils import (
    setup_logger, load_config, get_city_config, print_banner,
    make_progress, retry, ensure_dirs, save_metadata,
    bbox_to_ee_geometry, kelvin_to_celsius, log_raster_stats
)

from rich.console import Console
console = Console()

# ─── GEE Authentication ────────────────────────────────────────────────────────

def authenticate_gee(project_id: str) -> None:
    """
    Authenticate with Google Earth Engine.
    On first run this opens a browser window — follow the instructions.
    After first run the credentials are cached and this is silent.
    """
    try:
        ee.Initialize(project=project_id)
        logger.info(f"GEE authenticated — project: {project_id}")
    except ee.EEException:
        logger.warning("GEE not authenticated. Opening browser for login...")
        ee.Authenticate()
        ee.Initialize(project=project_id)
        logger.info("GEE authentication successful and cached.")


# ─── Cloud masking ─────────────────────────────────────────────────────────────

def mask_landsat_clouds(image: ee.Image) -> ee.Image:
    """
    Mask clouds and cloud shadows using Landsat Collection 2 QA_PIXEL band.
    Bit 3 = cloud shadow, Bit 4 = cloud.
    Also applies the official USGS scaling factors for Surface Temperature.
    """
    qa = image.select("QA_PIXEL")
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0)   # cloud shadow
    cloud_mask = cloud_mask.And(qa.bitwiseAnd(1 << 4).eq(0))  # cloud

    # Apply scale factor to thermal band
    thermal = (
        image.select("ST_B10")
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)   # → °C directly
        .rename("lst_celsius")
    )
    return image.addBands(thermal).updateMask(cloud_mask)


# ─── LST computation ───────────────────────────────────────────────────────────

def compute_lst_composite(
    bbox_geom: ee.Geometry,
    year: int,
    season_months: list[int],
    cloud_max: int = 15,
) -> ee.Image:
    """
    Build a median cloud-free LST composite for a given year and season.

    Args:
        bbox_geom: GEE geometry for the city
        year: e.g. 2024
        season_months: e.g. [3, 4, 5] for pre-monsoon
        cloud_max: maximum cloud cover % to include a scene

    Returns:
        Single-band ee.Image 'lst_celsius' — median composite
    """
    start_month, end_month = season_months[0], season_months[-1]
    start_date = f"{year}-{start_month:02d}-01"
    # End date: first day of the month after end_month
    end_year = year if end_month < 12 else year + 1
    end_month_adj = (end_month % 12) + 1
    end_date = f"{end_year}-{end_month_adj:02d}-01"

    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(bbox_geom)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", cloud_max))
        .map(mask_landsat_clouds)
        .select("lst_celsius")
    )

    count = collection.size().getInfo()
    logger.info(f"  Landsat scenes found: {count} (year={year}, months={start_month}–{end_month}, cloud<{cloud_max}%)")

    if count == 0:
        logger.error(f"No scenes found! Try increasing cloud_max (currently {cloud_max}%)")
        raise ValueError(f"No Landsat scenes for {year} months {start_month}-{end_month}")

    # Median composite — robust to residual clouds
    composite = collection.median().rename("lst_celsius")
    return composite, count


# ─── Download to local GeoTIFF ─────────────────────────────────────────────────

@retry(max_attempts=3, wait_seconds=10)
def download_ee_image(
    image: ee.Image,
    bbox: list[float],
    scale: int,
    output_path: Path,
) -> None:
    """
    Download a GEE image as a GeoTIFF via getDownloadURL.
    For production-scale use, prefer Export.image.toCloudStorage().
    For hackathon / local use this is fine up to ~50km² at 100m.
    """

    west, south, east, north = bbox
    region = ee.Geometry.Rectangle([west, south, east, north])

    url = image.getDownloadURL({
        "scale": scale,
        "region": region,
        "format": "GEO_TIFF",
        "filePerBand": False,
    })

    logger.info(f"  Downloading from GEE... (this may take 30-120 seconds)")
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"  Downloaded: {output_path.name} ({size_mb:.1f} MB)")


# ─── Reproject to UTM ──────────────────────────────────────────────────────────

def reproject_to_utm(
    input_path: Path,
    output_path: Path,
    utm_epsg: int,
    resolution: int = 100,
) -> np.ndarray:
    """
    Reproject a GeoTIFF from WGS84 to UTM at the target resolution.
    Returns the reprojected array for immediate stats logging.
    """
    import subprocess

    cmd = [
        "gdalwarp",
        "-t_srs", f"EPSG:{utm_epsg}",
        "-tr", str(resolution), str(resolution),
        "-r", "bilinear",
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
        "-overwrite",
        str(input_path),
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gdalwarp failed:\n{result.stderr}")

    with rasterio.open(output_path) as src:
        arr = src.read(1).astype(np.float32)
        arr[arr == src.nodata] = np.nan

    return arr


# ─── Main ──────────────────────────────────────────────────────────────────────

def run(city: str, years: list[int], seasons: list[str]) -> None:
    setup_logger("01_pull_landsat_lst")
    print_banner(
        "ThermalSense AI — Script 01",
        f"Pulling Landsat 8 LST | City: {city.upper()} | Years: {years} | Seasons: {seasons}",
    )

    cfg_all = load_config()
    city_cfg = get_city_config(city)
    pipe_cfg = cfg_all["pipeline"]

    project_id = __import__("os").environ.get("EE_PROJECT_ID", "")
    if not project_id:
        logger.error("EE_PROJECT_ID not set in .env — see .env.example")
        sys.exit(1)

    authenticate_gee(project_id)

    bbox = city_cfg["bbox"]
    utm_epsg = city_cfg["utm_epsg"]
    resolution = pipe_cfg["target_resolution"]
    cloud_max = pipe_cfg["landsat"]["cloud_cover_max"]
    season_defs = pipe_cfg["landsat"]["seasons"]

    paths = cfg_all["paths"]
    raw_dir = ROOT / paths["raw_dir"] / city / "landsat"
    proc_dir = ROOT / paths["processed_dir"] / city / "landsat"
    ensure_dirs(raw_dir, proc_dir)

    bbox_geom = bbox_to_ee_geometry(bbox)
    results = []

    for year in years:
        for season in seasons:
            if season not in season_defs:
                logger.warning(f"Unknown season '{season}', skipping. Valid: {list(season_defs.keys())}")
                continue

            season_months = list(range(
                season_defs[season][0],
                season_defs[season][1] + 1
            ))
            tag = f"{year}_{season}"
            raw_path = raw_dir / f"landsat_lst_{tag}_wgs84.tif"
            proc_path = proc_dir / f"landsat_lst_{tag}_utm{utm_epsg}.tif"

            if proc_path.exists():
                logger.info(f"  {tag}: already processed, skipping (delete to re-run)")
                results.append({"year": year, "season": season, "path": str(proc_path), "status": "cached"})
                continue

            logger.info(f"\n{'─'*60}")
            logger.info(f"Processing: {city.upper()} | {year} | {season}")

            # 1. Build GEE composite
            composite, scene_count = compute_lst_composite(
                bbox_geom, year, season_months, cloud_max
            )

            # 2. Download raw GeoTIFF (WGS84)
            download_ee_image(composite, bbox, resolution, raw_path)

            # 3. Reproject to UTM at target resolution
            logger.info(f"  Reprojecting to EPSG:{utm_epsg} at {resolution}m...")
            arr = reproject_to_utm(raw_path, proc_path, utm_epsg, resolution)

            # 4. Log stats + sanity check
            log_raster_stats(arr, f"LST {tag}", "°C")

            # Sanity: Kolkata LST should be 20–55°C. Alert if outside range.
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                if valid.min() < 10 or valid.max() > 65:
                    logger.warning(
                        f"LST values outside expected range (10–65°C)! "
                        f"Got {valid.min():.1f}–{valid.max():.1f}°C. "
                        f"Check cloud masking or scaling."
                    )

            # 5. Save metadata sidecar
            save_metadata(proc_path, {
                "city": city,
                "year": year,
                "season": season,
                "season_months": season_months,
                "scene_count": scene_count,
                "cloud_max_pct": cloud_max,
                "source": "LANDSAT/LC08/C02/T1_L2",
                "band": "ST_B10",
                "unit": "celsius",
                "resolution_m": resolution,
                "utm_epsg": utm_epsg,
                "bbox_wgs84": bbox,
                "lst_min": float(np.nanmin(arr)) if len(valid) > 0 else None,
                "lst_max": float(np.nanmax(arr)) if len(valid) > 0 else None,
                "lst_mean": float(np.nanmean(arr)) if len(valid) > 0 else None,
                "nan_pct": float(np.isnan(arr).sum() / arr.size * 100),
            })

            results.append({
                "year": year,
                "season": season,
                "path": str(proc_path),
                "status": "done",
                "lst_mean": float(np.nanmean(arr)) if len(valid) > 0 else None,
            })

            logger.success(f"  ✓ {tag} complete → {proc_path.name}")

    # Final summary
    console.print("\n[bold green]Script 01 complete[/bold green]")
    for r in results:
        status_icon = "✓" if r["status"] in ("done", "cached") else "✗"
        mean_str = f" | LST mean: {r['lst_mean']:.1f}°C" if r.get("lst_mean") else ""
        console.print(f"  {status_icon} {r['year']} {r['season']}{mean_str}")
        console.print(f"    {r['path']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pull Landsat 8 LST via GEE")
    parser.add_argument("--city", default="kolkata", help="City name from config (default: kolkata)")
    parser.add_argument("--year", type=int, nargs="+", help="Years to pull (default: all config years)")
    parser.add_argument("--season", nargs="+", help="Seasons to pull: pre_monsoon post_monsoon (default: both)")
    args = parser.parse_args()

    cfg = load_config()
    years = args.year or cfg["pipeline"]["landsat"]["years"]
    seasons = args.season or list(cfg["pipeline"]["landsat"]["seasons"].keys())

    run(city=args.city, years=years, seasons=seasons)
