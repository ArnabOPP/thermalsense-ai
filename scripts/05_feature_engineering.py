"""
ThermalSense AI — Script 05
Feature engineering: align all raster sources to the 100m UTM grid,
compute 14 features per pixel, export as a flat Parquet feature matrix.

This is the core input to Person B's PINN model.

Input  (from scripts 01–04):
  outputs/processed/{city}/landsat/     → LST per year/season
  outputs/processed/{city}/sentinel2/   → NDVI, NDWI, NDBI, albedo
  outputs/processed/{city}/morphology/  → building_density, SVF, ISA, etc.
  outputs/processed/{city}/insat3d/     → Tatm

Output:
  outputs/exports/{city}/feature_matrix_{city}_{year}_{season}.parquet
  outputs/exports/{city}/feature_matrix_{city}_ALL.parquet   ← all years combined

Feature matrix schema (one row = one 100m pixel × date):
  city, year, season, pixel_row, pixel_col, centroid_lat, centroid_lon,
  lst_celsius (TARGET),
  ndvi, ndwi, ndbi, albedo,
  isa_pct, svf, building_height, building_density, canyon_ratio, dist_water_m,
  tatm, era5_humidity, era5_wind_speed, doy_sin

Usage:
  python scripts/05_feature_engineering.py --city kolkata
  python scripts/05_feature_engineering.py --city kolkata --year 2024

Author: Person A
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils import (
    setup_logger, load_config, get_city_config, print_banner,
    ensure_dirs, save_metadata, log_raster_stats
)


# ─── Raster alignment ──────────────────────────────────────────────────────────

def get_reference_profile(city: str, cfg: dict) -> dict:
    pipe_cfg = cfg["pipeline"]
    city_cfg = cfg["cities"][city]
    utm_epsg = city_cfg["utm_epsg"]
    resolution = pipe_cfg["target_resolution"]

    lst_dir = ROOT / cfg["paths"]["processed_dir"] / city / "landsat"
    lst_files = sorted(lst_dir.glob("*.tif"))
    if not lst_files:
        raise FileNotFoundError(
            f"No processed Landsat files found in {lst_dir}\n"
            f"Run script 01 first: python scripts/01_pull_landsat_lst.py --city {city}"
        )

    with rasterio.open(lst_files[0]) as src:
        ref_profile = src.profile.copy()
        ref_profile.update(count=1, dtype="float32", nodata=np.nan)

    logger.info(f"Reference grid: {ref_profile['width']}×{ref_profile['height']} px @ {resolution}m")
    logger.info(f"CRS: EPSG:{utm_epsg}")
    return ref_profile


def align_raster_to_reference(
    src_path: Path,
    ref_profile: dict,
    band: int = 1,
) -> np.ndarray:
    """
    Reproject and resample any raster to exactly match the reference grid.
    Returns a float32 array with NaN for nodata.
    """
    out_arr = np.full(
        (ref_profile["height"], ref_profile["width"]),
        np.nan,
        dtype=np.float32
    )

    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, band),
            destination=out_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )

    return out_arr


# ─── Coordinate grid ───────────────────────────────────────────────────────────

def make_coordinate_arrays(ref_profile: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate lat/lon arrays for each pixel centroid in the reference grid.
    Returns (lat_arr, lon_arr) in WGS84.
    """
    from pyproj import Transformer

    height = ref_profile["height"]
    width  = ref_profile["width"]
    transform = ref_profile["transform"]
    crs_str = str(ref_profile["crs"])
    utm_epsg = int(crs_str.split(":")[-1]) if ":" in crs_str else 32645

    # Pixel centroid coordinates in UTM
    cols = np.arange(width)
    rows = np.arange(height)
    col_grid, row_grid = np.meshgrid(cols, rows)

    # Pixel center: transform * (col + 0.5, row + 0.5)
    x_utm = transform.c + transform.a * (col_grid + 0.5)
    y_utm = transform.f + transform.e * (row_grid + 0.5)

    # Convert to WGS84
    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    lon_arr, lat_arr = transformer.transform(x_utm.ravel(), y_utm.ravel())

    return (
        lat_arr.reshape(height, width).astype(np.float32),
        lon_arr.reshape(height, width).astype(np.float32),
    )


# ─── DOY cyclical encoding ─────────────────────────────────────────────────────

def doy_sin(year: int, season: str, season_defs: dict) -> float:
    """
    Encode the middle day of the season as sin(2π * DOY / 365).
    This allows the model to understand seasonal cycles continuously.
    """
    sm = season_defs[season]
    mid_month = (sm[0] + sm[1]) // 2
    mid_date = pd.Timestamp(year=year, month=mid_month, day=15)
    doy = mid_date.day_of_year
    return float(np.sin(2 * np.pi * doy / 365.0))


# ─── ERA5 humidity + wind (if not already from script 04) ──────────────────────

def try_load_era5_band(era5_path: Path, band: int) -> np.ndarray | None:
    """Load a specific band from ERA5 3-band GeoTIFF if it exists."""
    if not era5_path.exists():
        return None
    try:
        with rasterio.open(era5_path) as src:
            if src.count < band:
                return None
            arr = src.read(band).astype(np.float32)
            arr[arr < -999] = np.nan
        return arr
    except Exception as e:
        logger.debug(f"  ERA5 band {band} load failed: {e}")
        return None


# ─── Main feature assembly ────────────────────────────────────────────────────

def build_feature_matrix(
    city: str,
    year: int,
    season: str,
    ref_profile: dict,
    lat_arr: np.ndarray,
    lon_arr: np.ndarray,
    cfg: dict,
) -> pd.DataFrame | None:
    """
    Assemble all features for one city × year × season into a flat DataFrame.
    Returns None if the LST file for this year/season is missing.
    """
    pipe_cfg   = cfg["pipeline"]
    city_cfg   = cfg["cities"][city]
    utm_epsg   = city_cfg["utm_epsg"]
    season_defs = pipe_cfg["landsat"]["seasons"]

    proc_dir  = ROOT / cfg["paths"]["processed_dir"] / city
    tag       = f"{year}_{season}"

    # ── LST (target variable) ─────────────────────────────────────────────────
    lst_path = proc_dir / "landsat" / f"landsat_lst_{tag}_utm{utm_epsg}.tif"
    if not lst_path.exists():
        logger.debug(f"  LST not found: {lst_path.name} — skipping")
        return None

    logger.info(f"  Loading LST: {lst_path.name}")
    lst = align_raster_to_reference(lst_path, ref_profile, band=1)
    log_raster_stats(lst, f"LST {tag}", "°C")

    # Skip if too many NaNs (cloud-heavy season)
    nan_pct = np.isnan(lst).mean()
    if nan_pct > 0.6:
        logger.warning(f"  {tag}: {nan_pct*100:.0f}% NaN in LST — skipping (too cloudy)")
        return None

    # ── Sentinel-2 indices ────────────────────────────────────────────────────
    s2_path = proc_dir / "sentinel2" / f"s2_indices_{tag}_utm{utm_epsg}.tif"
    if s2_path.exists():
        ndvi   = align_raster_to_reference(s2_path, ref_profile, band=1)
        ndwi   = align_raster_to_reference(s2_path, ref_profile, band=2)
        ndbi   = align_raster_to_reference(s2_path, ref_profile, band=3)
        albedo = align_raster_to_reference(s2_path, ref_profile, band=4)
    else:
        logger.warning(f"  S2 indices not found: {s2_path.name} — filling with NaN")
        ndvi = ndwi = ndbi = albedo = np.full_like(lst, np.nan)

    # ── Morphology (static — same for all years) ──────────────────────────────
    morph_path = proc_dir / "morphology" / f"morphology_{city}_utm{utm_epsg}.tif"
    if morph_path.exists():
        building_density = align_raster_to_reference(morph_path, ref_profile, band=1)
        building_height  = align_raster_to_reference(morph_path, ref_profile, band=2)
        canyon_ratio     = align_raster_to_reference(morph_path, ref_profile, band=3)
        svf              = align_raster_to_reference(morph_path, ref_profile, band=4)
        dist_water       = align_raster_to_reference(morph_path, ref_profile, band=5)
        isa_pct          = align_raster_to_reference(morph_path, ref_profile, band=6)
    else:
        logger.warning(f"  Morphology not found — run script 03 first")
        n = ref_profile["height"] * ref_profile["width"]
        building_density = building_height = canyon_ratio = svf = np.full_like(lst, np.nan)
        dist_water = isa_pct = np.full_like(lst, np.nan)

    # ── INSAT-3D / ERA5 atmospheric temperature ───────────────────────────────
    insat_path = proc_dir / "insat3d" / f"insat3d_tatm_{tag}_utm{utm_epsg}.tif"
    if insat_path.exists():
        tatm = align_raster_to_reference(insat_path, ref_profile, band=1)
    else:
        logger.warning(f"  INSAT-3D not found: {insat_path.name} — filling with NaN")
        tatm = np.full_like(lst, np.nan)

    # ERA5 humidity + wind (band 2 and 3 of the era5 file from script 04)
    era5_path = proc_dir / "insat3d" / f"insat3d_tatm_{tag}_utm{utm_epsg}.tif"
    humidity   = try_load_era5_band(era5_path, 2)
    wind_speed = try_load_era5_band(era5_path, 3)
    if humidity is None:
        humidity = np.full_like(lst, np.nan)
    else:
        humidity = align_raster_to_reference(era5_path, ref_profile, band=2)
    if wind_speed is None:
        wind_speed = np.full_like(lst, np.nan)
    else:
        wind_speed = align_raster_to_reference(era5_path, ref_profile, band=3)

    # ── DOY encoding ──────────────────────────────────────────────────────────
    doy_sin_val = doy_sin(year, season, season_defs)

    # ── Flatten to DataFrame ──────────────────────────────────────────────────
    h, w = lst.shape
    n_pixels = h * w

    row_idx, col_idx = np.indices((h, w))

    df = pd.DataFrame({
        # Identifiers
        "city":       city,
        "year":       np.int16(year),
        "season":     season,
        "pixel_row":  row_idx.ravel().astype(np.int16),
        "pixel_col":  col_idx.ravel().astype(np.int16),
        "centroid_lat": lat_arr.ravel(),
        "centroid_lon": lon_arr.ravel(),

        # Target
        "lst_celsius": lst.ravel(),

        # Vegetation + water + built-up indices
        "ndvi":   ndvi.ravel(),
        "ndwi":   ndwi.ravel(),
        "ndbi":   ndbi.ravel(),
        "albedo": albedo.ravel(),

        # Urban morphology
        "isa_pct":          isa_pct.ravel(),
        "svf":              svf.ravel(),
        "building_height":  building_height.ravel(),
        "building_density": building_density.ravel(),
        "canyon_ratio":     canyon_ratio.ravel(),
        "dist_water_m":     dist_water.ravel(),

        # Atmospheric
        "tatm":           tatm.ravel(),
        "era5_humidity":  humidity.ravel(),
        "era5_wind_speed": wind_speed.ravel(),

        # Temporal
        "doy_sin": np.float32(doy_sin_val),
    })

    # Drop pixels where LST is NaN (no target = useless for training)
    df = df.dropna(subset=["lst_celsius"]).reset_index(drop=True)

    logger.info(f"  Feature matrix: {len(df):,} valid pixels ({len(df)/n_pixels*100:.1f}% of grid)")

    # Log feature-level NaN rates
    for col in ["ndvi", "albedo", "svf", "tatm", "isa_pct"]:
        nan_pct = df[col].isna().mean()
        if nan_pct > 0.3:
            logger.warning(f"  {col}: {nan_pct*100:.0f}% NaN — consider imputation")

    return df


# ─── Main ──────────────────────────────────────────────────────────────────────

def run(city: str, years: list[int]) -> None:
    setup_logger("05_feature_engineering")
    print_banner(
        "ThermalSense AI — Script 05",
        f"Feature engineering | City: {city.upper()} | Years: {years}",
    )

    cfg     = load_config()
    cfg         = cfg
    pipe_cfg    = cfg["pipeline"]
    season_defs = pipe_cfg["landsat"]["seasons"]

    export_dir = ROOT / cfg["paths"]["exports_dir"] / city
    ensure_dirs(export_dir)

    ref_profile = get_reference_profile(city, cfg)
    logger.info("Computing pixel coordinate grid...")
    lat_arr, lon_arr = make_coordinate_arrays(ref_profile)

    all_dfs = []
    results = []

    for year in years:
        for season in season_defs.keys():
            tag = f"{year}_{season}"
            out_path = export_dir / f"feature_matrix_{city}_{tag}.parquet"

            if out_path.exists():
                logger.info(f"  {tag}: already exported, loading for combined file")
                df = pd.read_parquet(out_path)
                all_dfs.append(df)
                results.append({"tag": tag, "rows": len(df), "status": "cached"})
                continue

            logger.info(f"\n{'─'*60}")
            logger.info(f"Building features: {city.upper()} | {tag}")

            df = build_feature_matrix(city, year, season, ref_profile, lat_arr, lon_arr, cfg)

            if df is None:
                results.append({"tag": tag, "rows": 0, "status": "skipped"})
                continue

            # Save per-season Parquet
            df.to_parquet(out_path, index=False, compression="zstd")
            size_mb = out_path.stat().st_size / 1024 / 1024
            logger.success(f"  ✓ Saved: {out_path.name} ({len(df):,} rows, {size_mb:.1f} MB)")

            save_metadata(out_path, {
                "city": city,
                "year": year,
                "season": season,
                "n_pixels": len(df),
                "n_features": len(df.columns),
                "features": list(df.columns),
                "lst_mean": float(df["lst_celsius"].mean()),
                "lst_std":  float(df["lst_celsius"].std()),
            })

            all_dfs.append(df)
            results.append({"tag": tag, "rows": len(df), "status": "done"})

    # ── Combined file for model training ──────────────────────────────────────
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = export_dir / f"feature_matrix_{city}_ALL.parquet"
        combined.to_parquet(combined_path, index=False, compression="zstd")
        size_mb = combined_path.stat().st_size / 1024 / 1024

        logger.info(f"\n{'='*60}")
        logger.success(f"Combined feature matrix: {combined_path.name}")
        logger.info(f"  Total rows:    {len(combined):,}")
        logger.info(f"  Total columns: {len(combined.columns)}")
        logger.info(f"  File size:     {size_mb:.1f} MB")
        logger.info(f"  LST range:     {combined['lst_celsius'].min():.1f}–{combined['lst_celsius'].max():.1f}°C")
        logger.info(f"  LST mean:      {combined['lst_celsius'].mean():.2f}°C ± {combined['lst_celsius'].std():.2f}°C")
        logger.info(f"  City bbox:     Kolkata region")
        logger.info(f"\n  NaN rates per feature:")
        for col in combined.columns:
            if col not in ["city", "season", "year", "pixel_row", "pixel_col"]:
                pct = combined[col].isna().mean() * 100
                flag = " ← HIGH" if pct > 20 else ""
                logger.info(f"    {col:<25} {pct:.1f}%{flag}")

        save_metadata(combined_path, {
            "city": city,
            "years": years,
            "seasons": list(season_defs.keys()),
            "n_rows": len(combined),
            "n_features": len(combined.columns),
            "features": list(combined.columns),
            "lst_mean": float(combined["lst_celsius"].mean()),
            "lst_std":  float(combined["lst_celsius"].std()),
            "lst_min":  float(combined["lst_celsius"].min()),
            "lst_max":  float(combined["lst_celsius"].max()),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Script 05 summary:")
    for r in results:
        icon = "✓" if r["status"] in ("done", "cached") else "–"
        logger.info(f"  {icon} {r['tag']}: {r['rows']:,} pixels ({r['status']})")

    logger.success("Script 05 complete — feature matrix ready for Person B's PINN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature matrix from all raster sources")
    parser.add_argument("--city", default="kolkata")
    parser.add_argument("--year", type=int, nargs="+")
    args = parser.parse_args()

    cfg = load_config()
    years = args.year or cfg["pipeline"]["landsat"]["years"]
    run(city=args.city, years=years)
