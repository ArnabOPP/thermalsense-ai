"""
ThermalSense AI — Script 03
Compute urban morphology features from OpenStreetMap + GHSL.

Features computed per 100m grid cell:
  - building_density   : building footprint area / cell area
  - building_height    : mean building height (from GHSL GHS_BUILT_H)
  - canyon_ratio       : mean H/W ratio of street canyons
  - svf                : sky view factor (approximated from building height + density)
  - dist_water_m       : distance to nearest OSM water body (metres)
  - isa_pct            : impervious surface area % (proxy from built-up + roads)

What this produces:
  outputs/processed/{city}/morphology/morphology_{city}_utm{epsg}.tif
  — 6-band GeoTIFF, one band per feature above

Usage:
  python scripts/03_osm_morphology.py --city kolkata

Author: Person A
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import osmnx as ox
ox.settings.overpass_url = "https://overpass.kumi.systems/api/interpreter"
ox.settings.log_console = False
ox.settings.use_cache = True
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.crs import CRS
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils import (
    setup_logger, load_config, get_city_config, print_banner,
    ensure_dirs, save_metadata, log_raster_stats, retry
)


# ─── Grid setup ────────────────────────────────────────────────────────────────

def make_grid_raster_profile(
    bbox_utm: tuple[float, float, float, float],
    resolution: int,
    utm_epsg: int,
) -> tuple[dict, int, int]:
    """
    Build rasterio profile for the output grid.
    Returns (profile, n_rows, n_cols).
    """
    west, south, east, north = bbox_utm
    n_cols = int((east - west) / resolution)
    n_rows = int((north - south) / resolution)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": n_cols,
        "height": n_rows,
        "count": 6,         # 6 morphology bands
        "crs": CRS.from_epsg(utm_epsg),
        "transform": from_bounds(west, south, east, north, n_cols, n_rows),
        "compress": "lzw",
        "tiled": True,
        "nodata": np.nan,
    }
    return profile, n_rows, n_cols


def bbox_wgs84_to_utm(bbox_wgs84: list, utm_epsg: int) -> tuple:
    """Convert WGS84 bbox to UTM bbox using geopandas."""
    west, south, east, north = bbox_wgs84
    poly = gpd.GeoDataFrame(geometry=[box(west, south, east, north)], crs="EPSG:4326")
    poly_utm = poly.to_crs(epsg=utm_epsg)
    b = poly_utm.total_bounds  # [minx, miny, maxx, maxy]
    return tuple(b)


# ─── OSM data fetching ─────────────────────────────────────────────────────────

@retry(max_attempts=3, wait_seconds=30)
def fetch_osm_buildings(bbox_wgs84: list, buffer_m: int = 500) -> gpd.GeoDataFrame:
    """Fetch OSM buildings using alternative Overpass mirrors."""
    west, south, east, north = bbox_wgs84
    buf = 0.005

    # Try multiple Overpass mirrors in sequence
    mirrors = [
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.ru/api/interpreter",
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    ]

    tags = {"building": True}
    last_error = None

    for mirror in mirrors:
        try:
            logger.info(f"  Trying OSM mirror: {mirror}")
            ox.settings.overpass_url = mirror
            gdf = ox.features_from_bbox(
                bbox=(north + buf, south - buf, east + buf, west - buf),
                tags=tags,
            )
            gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
            gdf = gdf[["geometry"]].reset_index(drop=True)
            logger.info(f"  OSM buildings fetched: {len(gdf):,}")
            return gdf
        except Exception as e:
            logger.warning(f"  Mirror {mirror} failed: {e}")
            last_error = e
            continue

    raise last_error

@retry(max_attempts=3, wait_seconds=20)
def fetch_osm_water(bbox_wgs84: list) -> gpd.GeoDataFrame:
    """Fetch OSM water bodies (rivers, lakes, ponds, wetlands)."""
    west, south, east, north = bbox_wgs84
    buf = 0.01
    logger.info("  Fetching OSM water bodies...")

    tags = {
        "natural": ["water", "wetland"],
        "waterway": ["river", "canal", "stream"],
        "landuse": "reservoir",
    }
    try:
        gdf = ox.features_from_bbox(
            bbox=(north + buf, south - buf, east + buf, west - buf),
            tags=tags,
        )
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon", "LineString", "MultiLineString"])].copy()
        gdf = gdf[["geometry"]].reset_index(drop=True)
        logger.info(f"  OSM water features: {len(gdf):,}")
    except Exception as e:
        logger.warning(f"  OSM water fetch failed: {e} — dist_water will be empty")
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gdf


@retry(max_attempts=3, wait_seconds=20)
def fetch_osm_roads(bbox_wgs84: list) -> gpd.GeoDataFrame:
    """Fetch OSM road network for canyon ratio computation."""
    west, south, east, north = bbox_wgs84
    buf = 0.005
    logger.info("  Fetching OSM road network...")
    try:
        G = ox.graph_from_bbox(
            bbox=(north + buf, south - buf, east + buf, west - buf),
            network_type="drive",
        )
        _, edges = ox.graph_to_gdfs(G)
        edges = edges[["geometry", "length"]].reset_index(drop=True)
        logger.info(f"  OSM road segments: {len(edges):,}")
        return edges
    except Exception as e:
        logger.warning(f"  OSM road fetch failed: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


# ─── GHSL building height ──────────────────────────────────────────────────────

def fetch_ghsl_height(bbox_wgs84: list, utm_epsg: int, output_path: Path) -> np.ndarray:
    """
    Pull GHSL building height layer from GEE and save as GeoTIFF.
    Falls back to zeros if GEE not available.
    """
    try:
        import ee, os, requests
        project_id = os.environ.get("EE_PROJECT_ID", "")
        if not project_id:
            raise ValueError("EE_PROJECT_ID not set")

        try:
            ee.Initialize(project=project_id)
        except Exception:
            ee.Authenticate()
            ee.Initialize(project=project_id)

        west, south, east, north = bbox_wgs84
        region = ee.Geometry.Rectangle([west, south, east, north])

        image = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018").select("b1")
        url = image.getDownloadURL({
            "scale": 100,
            "region": region,
            "format": "GEO_TIFF",
        })

        logger.info("  Downloading GHSL building height from GEE...")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)

        import subprocess
        reproj_path = output_path.with_name(output_path.stem + f"_utm{utm_epsg}.tif")
        subprocess.run([
            "gdalwarp", "-t_srs", f"EPSG:{utm_epsg}",
            "-tr", "100", "100", "-r", "bilinear", "-overwrite",
            str(output_path), str(reproj_path)
        ], check=True, capture_output=True)

        with rasterio.open(reproj_path) as src:
            arr = src.read(1).astype(np.float32)
            arr[arr < 0] = 0
        logger.info(f"  GHSL height: mean={np.nanmean(arr):.1f}m max={np.nanmax(arr):.1f}m")
        return arr

    except Exception as e:
        logger.warning(f"  GHSL height not available ({e}) — using OSM-based estimate")
        return None


# ─── Morphology computation ────────────────────────────────────────────────────

def compute_building_density(
    buildings_utm: gpd.GeoDataFrame,
    profile: dict,
    n_rows: int,
    n_cols: int,
) -> np.ndarray:
    """
    Rasterize building footprints and compute fractional coverage per 100m cell.
    """
    if len(buildings_utm) == 0:
        logger.warning("  No buildings — building_density will be zero")
        return np.zeros((n_rows, n_cols), dtype=np.float32)

    # Rasterize at fine resolution (10m), then aggregate to 100m
    fine_res = 10
    fine_transform = from_bounds(
        *rasterio.transform.array_bounds(n_rows, n_cols, profile["transform"]),
        n_cols * 10, n_rows * 10
    )

    shapes = [(geom, 1) for geom in buildings_utm.geometry if geom is not None]
    fine_mask = rasterize(
        shapes,
        out_shape=(n_rows * 10, n_cols * 10),
        transform=fine_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )

    # Block-average to 100m
    density = fine_mask.reshape(n_rows, 10, n_cols, 10).mean(axis=(1, 3)).astype(np.float32)
    logger.info(f"  Building density: mean={density.mean():.3f} max={density.max():.3f}")
    return density


def compute_svf_from_height_density(
    building_height: np.ndarray,
    building_density: np.ndarray,
    resolution: int = 100,
) -> np.ndarray:
    """
    Approximate Sky View Factor from building height and density.
    Uses Grimmond & Oke (1999) parameterisation:
      SVF ≈ 1 - (building_density * sin(arctan(H/W)))
    where W = effective street width = resolution * (1 - sqrt(density))

    SVF = 1 means fully open sky, SVF = 0 means completely blocked.
    """
    h = np.clip(building_height, 0, None)
    d = np.clip(building_density, 0, 1)

    # Effective street width from density
    w = resolution * np.maximum(1 - np.sqrt(d), 0.05)

    # Canyon ratio H/W
    hw = np.where(w > 0, h / w, 0)

    # SVF approximation
    svf = 1.0 - d * np.sin(np.arctan(hw))
    svf = np.clip(svf, 0.05, 1.0).astype(np.float32)

    logger.info(f"  SVF: mean={svf.mean():.3f} min={svf.min():.3f}")
    return svf, hw.astype(np.float32)


def compute_dist_water(
    water_utm: gpd.GeoDataFrame,
    profile: dict,
    n_rows: int,
    n_cols: int,
    resolution: int = 100,
) -> np.ndarray:
    """
    Compute Euclidean distance to nearest water body in metres.
    Uses rasterio rasterize + scipy distance_transform_edt.
    """
    bounds = rasterio.transform.array_bounds(n_rows, n_cols, profile["transform"])

    if len(water_utm) == 0:
        logger.warning("  No water features — dist_water will be max distance")
        max_dist = np.sqrt(n_rows**2 + n_cols**2) * resolution
        return np.full((n_rows, n_cols), max_dist, dtype=np.float32)

    shapes = [(geom, 1) for geom in water_utm.geometry if geom is not None]
    water_mask = rasterize(
        shapes,
        out_shape=(n_rows, n_cols),
        transform=profile["transform"],
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )

    # distance_transform_edt gives distance in pixels → multiply by resolution
    dist_pixels = distance_transform_edt(water_mask == 0)
    dist_metres = (dist_pixels * resolution).astype(np.float32)

    logger.info(f"  Dist water: mean={dist_metres.mean():.0f}m min={dist_metres.min():.0f}m max={dist_metres.max():.0f}m")
    return dist_metres


def estimate_isa(
    building_density: np.ndarray,
    roads_utm: gpd.GeoDataFrame,
    profile: dict,
    n_rows: int,
    n_cols: int,
    resolution: int = 100,
) -> np.ndarray:
    """
    Estimate Impervious Surface Area % from buildings + roads.
    ISA = building_density + road_density (capped at 1.0)
    """
    # Road density: rasterize road lines, compute fraction of cell covered
    if len(roads_utm) > 0:
        road_shapes = [(geom.buffer(5), 1) for geom in roads_utm.geometry if geom is not None]
        road_mask = rasterize(
            road_shapes,
            out_shape=(n_rows, n_cols),
            transform=profile["transform"],
            fill=0,
            dtype=np.float32,
            all_touched=True,
        )
    else:
        road_mask = np.zeros((n_rows, n_cols), dtype=np.float32)

    isa = np.clip(building_density + road_mask * 0.3, 0.0, 1.0).astype(np.float32)
    isa_pct = isa * 100

    logger.info(f"  ISA: mean={isa_pct.mean():.1f}% max={isa_pct.max():.1f}%")
    return isa_pct


# ─── Main ──────────────────────────────────────────────────────────────────────

def run(city: str, skip_osm: bool = False) -> None:
    setup_logger("03_osm_morphology")
    print_banner(
        "ThermalSense AI — Script 03",
        f"Computing urban morphology | City: {city.upper()}",
    )

    cfg_all = load_config()
    city_cfg = get_city_config(city)
    pipe_cfg = cfg_all["pipeline"]

    bbox_wgs84 = city_cfg["bbox"]
    utm_epsg   = city_cfg["utm_epsg"]
    resolution = pipe_cfg["target_resolution"]

    morph_dir = ROOT / cfg_all["paths"]["processed_dir"] / city / "morphology"
    ensure_dirs(morph_dir)

    output_path = morph_dir / f"morphology_{city}_utm{utm_epsg}.tif"
    if output_path.exists():
        logger.info(f"Morphology already exists: {output_path} — delete to recompute")
        return

    # ── 1. Convert bbox to UTM ────────────────────────────────────────────────
    logger.info("Converting bbox to UTM...")
    bbox_utm = bbox_wgs84_to_utm(bbox_wgs84, utm_epsg)
    logger.info(f"  UTM bbox: {[round(x) for x in bbox_utm]}")

    profile, n_rows, n_cols = make_grid_raster_profile(bbox_utm, resolution, utm_epsg)
    logger.info(f"  Output grid: {n_rows} rows × {n_cols} cols ({n_rows*n_cols:,} cells)")

    # ── 2. Fetch OSM data ─────────────────────────────────────────────────────
    if skip_osm:
        logger.warning("--skip-osm active: using empty OSM data, ISA derived from NDBI")
        buildings_wgs84 = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        water_wgs84     = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        roads_wgs84     = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    else:
        logger.info("\nFetching OSM data...")
        buildings_wgs84 = fetch_osm_buildings(bbox_wgs84)
        water_wgs84     = fetch_osm_water(bbox_wgs84)
        roads_wgs84     = fetch_osm_roads(bbox_wgs84)

    # ── 3. Reproject OSM to UTM ───────────────────────────────────────────────
    logger.info(f"\nReprojecting to EPSG:{utm_epsg}...")
    crs_utm = f"EPSG:{utm_epsg}"
    buildings_utm = buildings_wgs84.set_crs("EPSG:4326", allow_override=True).to_crs(crs_utm) if len(buildings_wgs84) > 0 else buildings_wgs84
    water_utm     = water_wgs84.set_crs("EPSG:4326", allow_override=True).to_crs(crs_utm)     if len(water_wgs84) > 0 else water_wgs84
    roads_utm     = roads_wgs84.set_crs("EPSG:4326", allow_override=True).to_crs(crs_utm)     if len(roads_wgs84) > 0 else roads_wgs84

    # ── 4. GHSL building height ───────────────────────────────────────────────
    logger.info("\nFetching GHSL building heights...")
    ghsl_raw = morph_dir / "ghsl_height_wgs84.tif"
    building_height = fetch_ghsl_height(bbox_wgs84, utm_epsg, ghsl_raw)
    if building_height is None:
        # Fallback: estimate height from OSM building levels if available
        logger.info("  Using fallback height estimate (5m default)")
        building_height = np.full((n_rows, n_cols), 5.0, dtype=np.float32)

    # Ensure height array matches output grid shape
    if building_height.shape != (n_rows, n_cols):
        from scipy.ndimage import zoom
        zoom_factors = (n_rows / building_height.shape[0], n_cols / building_height.shape[1])
        building_height = zoom(building_height, zoom_factors, order=1).astype(np.float32)

    # ── 5. Compute morphology features ────────────────────────────────────────
    logger.info("\nComputing morphology features...")

    building_density = compute_building_density(buildings_utm, profile, n_rows, n_cols)
    svf, canyon_ratio = compute_svf_from_height_density(building_height, building_density, resolution)
    dist_water = compute_dist_water(water_utm, profile, n_rows, n_cols, resolution)
    isa_pct = estimate_isa(building_density, roads_utm, profile, n_rows, n_cols, resolution)

    # ── 6. Write 6-band GeoTIFF ───────────────────────────────────────────────
    logger.info(f"\nWriting output: {output_path.name}")
    bands = {
        1: ("building_density", building_density),
        2: ("building_height_m", building_height),
        3: ("canyon_ratio_hw", canyon_ratio),
        4: ("svf", svf),
        5: ("dist_water_m", dist_water),
        6: ("isa_pct", isa_pct),
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        for band_idx, (band_name, arr) in bands.items():
            # Resize to match profile if needed
            if arr.shape != (n_rows, n_cols):
                from scipy.ndimage import zoom as spzoom
                arr = spzoom(arr, (n_rows/arr.shape[0], n_cols/arr.shape[1]), order=1).astype(np.float32)
            dst.write(arr, band_idx)
            dst.update_tags(band_idx, name=band_name)

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.success(f"Morphology GeoTIFF: {output_path.name} ({size_mb:.1f} MB)")

    save_metadata(output_path, {
        "city": city,
        "bbox_wgs84": bbox_wgs84,
        "bbox_utm": list(bbox_utm),
        "utm_epsg": utm_epsg,
        "resolution_m": resolution,
        "grid_shape": [n_rows, n_cols],
        "n_buildings": len(buildings_utm),
        "n_water_features": len(water_utm),
        "n_road_segments": len(roads_utm),
        "bands": [v[0] for v in bands.values()],
        "building_density_mean": float(building_density.mean()),
        "svf_mean": float(svf.mean()),
        "isa_pct_mean": float(isa_pct.mean()),
    })

    logger.success("Script 03 complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute urban morphology from OSM + GHSL")
    parser.add_argument("--city", default="kolkata")
    parser.add_argument("--skip-osm", action="store_true",
                        help="Skip OSM, derive ISA from Sentinel-2 NDBI")
    args = parser.parse_args()
    run(city=args.city, skip_osm=args.skip_osm)
