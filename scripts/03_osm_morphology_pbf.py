"""
ThermalSense AI — Script 03b
Compute urban morphology from a locally downloaded Geofabrik PBF file.
No Overpass API — no timeouts, no rate limits.

Usage:
  python scripts/03_osm_morphology_pbf.py --city kolkata

Author: Person A
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from rasterio.crs import CRS
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box
from loguru import logger
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
load_dotenv(ROOT / ".env")

from utils import (
    setup_logger, load_config, get_city_config, print_banner,
    ensure_dirs, save_metadata, log_raster_stats
)

PBF_PATH = ROOT / "outputs" / "raw" / "kolkata" / "west-bengal.osm.pbf"


# ─── Grid helpers ──────────────────────────────────────────────────────────────

def make_profile(bbox_utm, resolution, utm_epsg):
    west, south, east, north = bbox_utm
    n_cols = int((east - west) / resolution)
    n_rows = int((north - south) / resolution)
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": n_cols, "height": n_rows, "count": 6,
        "crs": CRS.from_epsg(utm_epsg),
        "transform": from_bounds(west, south, east, north, n_cols, n_rows),
        "compress": "lzw", "tiled": True, "nodata": np.nan,
    }
    return profile, n_rows, n_cols


def bbox_to_utm(bbox_wgs84, utm_epsg):
    west, south, east, north = bbox_wgs84
    poly = gpd.GeoDataFrame(geometry=[box(west, south, east, north)], crs="EPSG:4326")
    b = poly.to_crs(epsg=utm_epsg).total_bounds
    return tuple(b)


# ─── PBF reading ───────────────────────────────────────────────────────────────

def read_buildings_from_pbf(pbf_path: Path, bbox_wgs84: list) -> gpd.GeoDataFrame:
    """Read building footprints from local PBF file, clipped to bbox."""
    try:
        from pyrosm import OSM
        west, south, east, north = bbox_wgs84
        buf = 0.01
        logger.info(f"  Reading buildings from PBF: {pbf_path.name}")
        osm = OSM(str(pbf_path), bounding_box=[west-buf, south-buf, east+buf, north+buf])
        buildings = osm.get_buildings()
        if buildings is None or len(buildings) == 0:
            logger.warning("  No buildings found in PBF for this bbox")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        buildings = buildings[buildings.geometry.geom_type.isin(["Polygon","MultiPolygon"])].copy()
        buildings = buildings[["geometry"]].reset_index(drop=True)
        if buildings.crs is None:
            buildings = buildings.set_crs("EPSG:4326")
        logger.info(f"  Buildings found: {len(buildings):,}")
        return buildings
    except Exception as e:
        logger.error(f"  PBF buildings read failed: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def read_water_from_pbf(pbf_path: Path, bbox_wgs84: list) -> gpd.GeoDataFrame:
    """Read water bodies from local PBF file."""
    try:
        from pyrosm import OSM
        west, south, east, north = bbox_wgs84
        buf = 0.02
        logger.info("  Reading water bodies from PBF...")
        osm = OSM(str(pbf_path), bounding_box=[west-buf, south-buf, east+buf, north+buf])
        natural = osm.get_natural()
        water_gdfs = []
        if natural is not None and len(natural) > 0:
            water = natural[natural.get("natural", "").isin(["water","wetland","riverbank"])]
            if len(water) > 0:
                water_gdfs.append(water[["geometry"]])
        landuse = osm.get_landuse()
        if landuse is not None and len(landuse) > 0:
            water_lu = landuse[landuse.get("landuse","").isin(["reservoir","basin"])]
            if len(water_lu) > 0:
                water_gdfs.append(water_lu[["geometry"]])
        if not water_gdfs:
            logger.warning("  No water features found")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        gdf = gpd.pd.concat(water_gdfs, ignore_index=True)
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon","MultiPolygon","LineString","MultiLineString"])].copy()
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        logger.info(f"  Water features found: {len(gdf):,}")
        return gdf
    except Exception as e:
        logger.warning(f"  PBF water read failed: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def read_roads_from_pbf(pbf_path: Path, bbox_wgs84: list) -> gpd.GeoDataFrame:
    """Read road network from local PBF file."""
    try:
        from pyrosm import OSM
        west, south, east, north = bbox_wgs84
        buf = 0.01
        logger.info("  Reading roads from PBF...")
        osm = OSM(str(pbf_path), bounding_box=[west-buf, south-buf, east+buf, north+buf])
        roads = osm.get_network(network_type="driving")
        if roads is None or len(roads) == 0:
            logger.warning("  No roads found")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        roads = roads[["geometry"]].copy()
        if roads.crs is None:
            roads = roads.set_crs("EPSG:4326")
        logger.info(f"  Road segments found: {len(roads):,}")
        return roads
    except Exception as e:
        logger.warning(f"  PBF roads read failed: {e}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


# ─── Feature computation ───────────────────────────────────────────────────────

def compute_building_density(buildings_utm, profile, n_rows, n_cols):
    if len(buildings_utm) == 0:
        logger.warning("  No buildings — density = 0")
        return np.zeros((n_rows, n_cols), dtype=np.float32)
    fine_transform = from_bounds(
        *rasterio.transform.array_bounds(n_rows, n_cols, profile["transform"]),
        n_cols * 10, n_rows * 10
    )
    shapes = [(g, 1) for g in buildings_utm.geometry if g is not None]
    fine = rasterize(shapes, out_shape=(n_rows*10, n_cols*10),
                     transform=fine_transform, fill=0, dtype=np.uint8, all_touched=True)
    density = fine.reshape(n_rows, 10, n_cols, 10).mean(axis=(1,3)).astype(np.float32)
    logger.info(f"  Building density: mean={density.mean():.3f} max={density.max():.3f}")
    return density


def compute_svf(building_height, building_density, resolution=100):
    h = np.clip(building_height, 0, None)
    d = np.clip(building_density, 0, 1)
    w = resolution * np.maximum(1 - np.sqrt(d), 0.05)
    hw = np.where(w > 0, h / w, 0)
    svf = np.clip(1.0 - d * np.sin(np.arctan(hw)), 0.05, 1.0).astype(np.float32)
    logger.info(f"  SVF: mean={svf.mean():.3f}")
    return svf, hw.astype(np.float32)


def compute_dist_water(water_utm, profile, n_rows, n_cols, resolution=100):
    if len(water_utm) == 0:
        max_dist = np.sqrt(n_rows**2 + n_cols**2) * resolution
        return np.full((n_rows, n_cols), max_dist, dtype=np.float32)
    shapes = [(g, 1) for g in water_utm.geometry if g is not None]
    mask = rasterize(shapes, out_shape=(n_rows, n_cols),
                     transform=profile["transform"], fill=0, dtype=np.uint8, all_touched=True)
    dist = (distance_transform_edt(mask == 0) * resolution).astype(np.float32)
    logger.info(f"  Dist water: mean={dist.mean():.0f}m min={dist.min():.0f}m")
    return dist


def compute_isa(building_density, roads_utm, profile, n_rows, n_cols):
    if len(roads_utm) > 0:
        road_shapes = [(g.buffer(5), 1) for g in roads_utm.geometry if g is not None]
        road_mask = rasterize(road_shapes, out_shape=(n_rows, n_cols),
                              transform=profile["transform"], fill=0,
                              dtype=np.float32, all_touched=True)
    else:
        road_mask = np.zeros((n_rows, n_cols), dtype=np.float32)
    isa_pct = np.clip(building_density + road_mask * 0.3, 0, 1) * 100
    logger.info(f"  ISA: mean={isa_pct.mean():.1f}% max={isa_pct.max():.1f}%")
    return isa_pct.astype(np.float32)


def fetch_ghsl_height(bbox_wgs84, utm_epsg, output_path, n_rows, n_cols):
    """Pull GHSL building height from GEE."""
    try:
        import ee, os, requests, subprocess
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
        image = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018").select("built_height")
        url = image.getDownloadURL({"scale": 100, "region": region, "format": "GEO_TIFF"})
        logger.info("  Downloading GHSL building height from GEE...")
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        reproj = output_path.with_name(output_path.stem + f"_utm{utm_epsg}.tif")
        subprocess.run(["gdalwarp", "-t_srs", f"EPSG:{utm_epsg}",
                        "-tr", "100", "100", "-r", "bilinear", "-overwrite",
                        str(output_path), str(reproj)], check=True, capture_output=True)
        with rasterio.open(reproj) as src:
            arr = src.read(1).astype(np.float32)
            arr[arr < 0] = 0
        logger.info(f"  GHSL height: mean={np.nanmean(arr):.1f}m max={np.nanmax(arr):.1f}m")
        return arr
    except Exception as e:
        logger.warning(f"  GHSL height failed ({e}) — using 5m default")
        return np.full((n_rows, n_cols), 5.0, dtype=np.float32)


# ─── Main ──────────────────────────────────────────────────────────────────────

def run(city: str) -> None:
    setup_logger("03_osm_morphology_pbf")
    print_banner("ThermalSense AI — Script 03b",
                 f"Urban morphology from PBF | City: {city.upper()}")

    if not PBF_PATH.exists():
        logger.error(f"PBF file not found: {PBF_PATH}")
        logger.error("Download from: https://download.geofabrik.de/asia/india.html")
        sys.exit(1)

    logger.info(f"PBF file: {PBF_PATH} ({PBF_PATH.stat().st_size/1024/1024:.0f} MB)")

    cfg = load_config()
    city_cfg = get_city_config(city)
    pipe_cfg = cfg["pipeline"]

    bbox_wgs84 = city_cfg["bbox"]
    utm_epsg = city_cfg["utm_epsg"]
    resolution = pipe_cfg["target_resolution"]

    morph_dir = ROOT / cfg["paths"]["processed_dir"] / city / "morphology"
    ensure_dirs(morph_dir)

    output_path = morph_dir / f"morphology_{city}_utm{utm_epsg}.tif"
    if output_path.exists():
        logger.info(f"Morphology exists — delete to recompute: {output_path}")
        return

    # 1. Convert bbox
    logger.info("Converting bbox to UTM...")
    bbox_utm = bbox_to_utm(bbox_wgs84, utm_epsg)
    profile, n_rows, n_cols = make_profile(bbox_utm, resolution, utm_epsg)
    logger.info(f"  Grid: {n_rows}×{n_cols} = {n_rows*n_cols:,} cells")

    # 2. Read from PBF
    crs_utm = f"EPSG:{utm_epsg}"
    buildings_wgs84 = read_buildings_from_pbf(PBF_PATH, bbox_wgs84)
    water_wgs84 = read_water_from_pbf(PBF_PATH, bbox_wgs84)
    roads_wgs84 = read_roads_from_pbf(PBF_PATH, bbox_wgs84)

    # 3. Reproject
    logger.info(f"Reprojecting to {crs_utm}...")
    def reproject(gdf):
        if len(gdf) == 0:
            return gdf
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        return gdf.to_crs(crs_utm)

    buildings_utm = reproject(buildings_wgs84)
    water_utm = reproject(water_wgs84)
    roads_utm = reproject(roads_wgs84)

    # 4. GHSL height
    logger.info("Fetching GHSL building heights...")
    ghsl_raw = morph_dir / "ghsl_height_wgs84.tif"
    building_height = fetch_ghsl_height(bbox_wgs84, utm_epsg, ghsl_raw, n_rows, n_cols)
    if building_height.shape != (n_rows, n_cols):
        from scipy.ndimage import zoom
        building_height = zoom(building_height,
                               (n_rows/building_height.shape[0], n_cols/building_height.shape[1]),
                               order=1).astype(np.float32)

    # 5. Compute features
    logger.info("Computing morphology features...")
    building_density = compute_building_density(buildings_utm, profile, n_rows, n_cols)
    svf, canyon_ratio = compute_svf(building_height, building_density, resolution)
    dist_water = compute_dist_water(water_utm, profile, n_rows, n_cols, resolution)
    isa_pct = compute_isa(building_density, roads_utm, profile, n_rows, n_cols)

    # 6. Write GeoTIFF
    logger.info(f"Writing: {output_path.name}")
    bands = {
        1: ("building_density", building_density),
        2: ("building_height_m", building_height),
        3: ("canyon_ratio_hw", canyon_ratio),
        4: ("svf", svf),
        5: ("dist_water_m", dist_water),
        6: ("isa_pct", isa_pct),
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        for idx, (name, arr) in bands.items():
            if arr.shape != (n_rows, n_cols):
                from scipy.ndimage import zoom as spzoom
                arr = spzoom(arr, (n_rows/arr.shape[0], n_cols/arr.shape[1]), order=1).astype(np.float32)
            dst.write(arr, idx)
            dst.update_tags(idx, name=name)

    size_mb = output_path.stat().st_size / 1024 / 1024
    logger.success(f"Morphology GeoTIFF: {output_path.name} ({size_mb:.1f} MB)")

    save_metadata(output_path, {
        "city": city, "bbox_wgs84": bbox_wgs84,
        "utm_epsg": utm_epsg, "resolution_m": resolution,
        "grid_shape": [n_rows, n_cols],
        "pbf_source": str(PBF_PATH),
        "n_buildings": int(len(buildings_utm)),
        "n_water": int(len(water_utm)),
        "n_roads": int(len(roads_utm)),
        "building_density_mean": float(building_density.mean()),
        "svf_mean": float(svf.mean()),
        "isa_pct_mean": float(isa_pct.mean()),
    })
    logger.success("Script 03b complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="kolkata")
    args = parser.parse_args()
    run(city=args.city)
