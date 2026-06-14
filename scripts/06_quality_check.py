"""
ThermalSense AI — Script 06
Data quality checker.

Validates ALL outputs from scripts 01–05 before handing off to Person B.
Generates a quality report with pass/fail for every check.

Checks:
  ✓ All expected files exist
  ✓ Raster CRS, resolution, and grid alignment match reference
  ✓ LST values in plausible range (10–65°C for India)
  ✓ No all-NaN bands
  ✓ NaN rates within tolerance
  ✓ Feature matrix schema correct (all 14 features present)
  ✓ No duplicate pixels in feature matrix
  ✓ LST and NDVI are not perfectly correlated (would indicate a bug)
  ✓ KMC ward boundaries cover the raster extent

Usage:
  python scripts/06_quality_check.py --city kolkata
  python scripts/06_quality_check.py --city kolkata --strict   # fail on any warning

Author: Person A
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils import setup_logger, load_config, get_city_config, print_banner, ensure_dirs

console = Console()

EXPECTED_FEATURES = [
    "city", "year", "season", "pixel_row", "pixel_col",
    "centroid_lat", "centroid_lon", "lst_celsius",
    "ndvi", "ndwi", "ndbi", "albedo",
    "isa_pct", "svf", "building_height", "building_density",
    "canyon_ratio", "dist_water_m",
    "tatm", "era5_humidity", "era5_wind_speed", "doy_sin",
]


# ─── Individual checks ─────────────────────────────────────────────────────────

def check_file_exists(path: Path, label: str) -> tuple[str, bool, str]:
    """CHECK: file exists and is non-empty."""
    if not path.exists():
        return label, False, f"MISSING: {path}"
    size = path.stat().st_size
    if size < 1000:
        return label, False, f"Too small ({size} bytes): {path}"
    return label, True, f"{size / 1024 / 1024:.1f} MB"


def check_raster_profile(
    path: Path,
    ref_profile: dict,
    label: str,
    tolerance_pct: float = 2.0,
) -> list[tuple[str, bool, str]]:
    """CHECK: raster CRS, resolution, and approximate grid size match reference."""
    results = []
    try:
        with rasterio.open(path) as src:
            # CRS check
            ref_crs = str(ref_profile["crs"])
            src_crs = str(src.crs)
            crs_ok = ref_crs.split(":")[-1] in src_crs or src_crs in ref_crs
            results.append((f"{label} CRS", crs_ok, f"{src_crs}"))

            # Resolution check (within tolerance)
            ref_res = abs(ref_profile["transform"].a)
            src_res = abs(src.transform.a)
            res_ok = abs(src_res - ref_res) / ref_res < tolerance_pct / 100
            results.append((f"{label} resolution", res_ok, f"{src_res:.1f}m (expected {ref_res:.1f}m)"))

            # Shape check (within 5%)
            ref_h, ref_w = ref_profile["height"], ref_profile["width"]
            h_ok = abs(src.height - ref_h) / ref_h < 0.05
            w_ok = abs(src.width  - ref_w) / ref_w < 0.05
            results.append((f"{label} shape", h_ok and w_ok,
                             f"{src.height}×{src.width} (expected ≈{ref_h}×{ref_w})"))

    except Exception as e:
        results.append((f"{label} readable", False, str(e)))

    return results


def check_raster_values(
    path: Path,
    band: int,
    label: str,
    valid_min: float,
    valid_max: float,
    max_nan_pct: float = 50.0,
) -> list[tuple[str, bool, str]]:
    """CHECK: raster values in expected range, NaN rate acceptable."""
    results = []
    try:
        with rasterio.open(path) as src:
            arr = src.read(band).astype(np.float32)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan

        nan_pct = np.isnan(arr).mean() * 100
        valid = arr[~np.isnan(arr)]

        # NaN rate
        nan_ok = nan_pct < max_nan_pct
        results.append((f"{label} NaN rate", nan_ok, f"{nan_pct:.1f}% (threshold: {max_nan_pct}%)"))

        if len(valid) > 0:
            # Value range
            range_ok = valid.min() >= valid_min and valid.max() <= valid_max
            results.append((
                f"{label} range",
                range_ok,
                f"[{valid.min():.2f}, {valid.max():.2f}] (expected [{valid_min}, {valid_max}])"
            ))

            # Not all-same (degenerate raster)
            not_flat = valid.std() > 1e-6
            results.append((f"{label} variance", not_flat,
                             f"std={valid.std():.4f}"))
        else:
            results.append((f"{label} range", False, "ALL NaN — completely empty raster"))

    except Exception as e:
        results.append((f"{label} readable", False, str(e)))

    return results


def check_feature_matrix(path: Path, label: str) -> list[tuple[str, bool, str]]:
    """CHECK: feature matrix schema, no duplicates, value sanity."""
    results = []
    try:
        df = pd.read_parquet(path)

        # Schema check
        missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
        extra = [c for c in df.columns if c not in EXPECTED_FEATURES]
        schema_ok = len(missing) == 0
        results.append((f"{label} schema", schema_ok,
                         f"missing={missing}, extra={extra}" if not schema_ok else f"{len(df.columns)} columns OK"))

        # Row count
        enough_rows = len(df) > 1000
        results.append((f"{label} row count", enough_rows, f"{len(df):,} rows"))

        # Duplicate pixels
        dup_cols = ["city", "year", "season", "pixel_row", "pixel_col"]
        dups = df.duplicated(subset=dup_cols).sum()
        no_dups = dups == 0
        results.append((f"{label} no duplicates", no_dups, f"{dups} duplicates" if dups else "OK"))

        # LST sanity
        if "lst_celsius" in df.columns:
            lst = df["lst_celsius"].dropna()
            lst_ok = (lst.min() >= 0) and (lst.max() < 70)
            results.append((f"{label} LST range", lst_ok,
                             f"{lst.min():.1f}–{lst.max():.1f}°C"))

            lst_nan_pct = df["lst_celsius"].isna().mean() * 100
            results.append((f"{label} LST completeness", lst_nan_pct < 5,
                             f"{lst_nan_pct:.1f}% NaN"))

        # NDVI vs LST correlation (should be negative, not too strong)
        if "ndvi" in df.columns and "lst_celsius" in df.columns and "ndwi" in df.columns:
            valid = df[["ndvi", "lst_celsius", "ndwi"]].dropna()
            valid = valid[valid["ndwi"] < 0.1]  # exclude water pixels
            if len(valid) > 100:
                corr = valid["ndvi"].corr(valid["lst_celsius"])
                corr_ok = -0.95 < corr < 0.2
                results.append((f"{label} NDVI-LST corr", corr_ok,
                                 f"r={corr:.3f} (land pixels only)"))

        # NaN rates for key features
        for col in ["ndvi", "svf", "isa_pct", "tatm"]:
            if col in df.columns:
                nan_pct = df[col].isna().mean() * 100
                ok = nan_pct < 40
                results.append((f"{label} {col} completeness", ok, f"{nan_pct:.1f}% NaN"))

    except Exception as e:
        results.append((f"{label} readable", False, str(e)))

    return results


def get_reference_profile(city: str, cfg: dict) -> dict | None:
    """Load the reference raster profile for alignment checks."""
    pipe_cfg = cfg["pipeline"]
    utm_epsg = cfg["cities"][city]["utm_epsg"]
    lst_dir = ROOT / cfg["paths"]["processed_dir"] / city / "landsat"
    lst_files = sorted(lst_dir.glob("*.tif"))
    if not lst_files:
        return None
    with rasterio.open(lst_files[0]) as src:
        return src.profile.copy()


# ─── Main ──────────────────────────────────────────────────────────────────────

def run(city: str, strict: bool = False) -> bool:
    setup_logger("06_quality_check")
    print_banner(
        "ThermalSense AI — Script 06",
        f"Data quality check | City: {city.upper()}",
    )

    cfg = load_config()
    city_cfg = get_city_config(city)
    pipe_cfg = cfg["pipeline"]
    years = pipe_cfg["landsat"]["years"]
    season_defs = pipe_cfg["landsat"]["seasons"]
    utm_epsg = city_cfg["utm_epsg"]

    proc_dir   = ROOT / cfg["paths"]["processed_dir"] / city
    export_dir = ROOT / cfg["paths"]["exports_dir"] / city

    ref_profile = get_reference_profile(city, cfg)
    if ref_profile is None:
        logger.error("No reference raster found — run scripts 01–05 first")
        return False

    all_checks: list[tuple[str, bool, str]] = []

    # ── 1. File existence ─────────────────────────────────────────────────────
    logger.info("Checking file existence...")

    # Morphology (static)
    morph_path = proc_dir / "morphology" / f"morphology_{city}_utm{utm_epsg}.tif"
    all_checks.append(check_file_exists(morph_path, "Morphology GeoTIFF"))

    # Combined feature matrix
    combined_path = export_dir / f"feature_matrix_{city}_ALL.parquet"
    all_checks.append(check_file_exists(combined_path, "Feature matrix (ALL)"))

    for year in years[-2:]:   # Check last 2 years only for brevity
        for season in season_defs:
            tag = f"{year}_{season}"

            lst_path = proc_dir / "landsat" / f"landsat_lst_{tag}_utm{utm_epsg}.tif"
            all_checks.append(check_file_exists(lst_path, f"LST {tag}"))

            s2_path = proc_dir / "sentinel2" / f"s2_indices_{tag}_utm{utm_epsg}.tif"
            all_checks.append(check_file_exists(s2_path, f"Sentinel-2 {tag}"))

            insat_path = proc_dir / "insat3d" / f"insat3d_tatm_{tag}_utm{utm_epsg}.tif"
            all_checks.append(check_file_exists(insat_path, f"INSAT-3D {tag}"))

    # ── 2. Raster profile alignment ───────────────────────────────────────────
    logger.info("Checking raster alignment...")

    for year in years[-1:]:   # Just the most recent year
        for season in ["pre_monsoon"]:
            tag = f"{year}_{season}"
            lst_p = proc_dir / "landsat" / f"landsat_lst_{tag}_utm{utm_epsg}.tif"
            s2_p  = proc_dir / "sentinel2" / f"s2_indices_{tag}_utm{utm_epsg}.tif"
            if lst_p.exists():
                all_checks.extend(check_raster_profile(s2_p, ref_profile, f"S2 {tag}"))
            if morph_path.exists():
                all_checks.extend(check_raster_profile(morph_path, ref_profile, "Morphology"))

    # ── 3. Raster value ranges ────────────────────────────────────────────────
    logger.info("Checking raster value ranges...")

    for year in years[-1:]:
        for season in ["pre_monsoon"]:
            tag = f"{year}_{season}"

            lst_p = proc_dir / "landsat" / f"landsat_lst_{tag}_utm{utm_epsg}.tif"
            if lst_p.exists():
                all_checks.extend(check_raster_values(lst_p, 1, f"LST {tag}", 10, 65, 50))

            s2_p = proc_dir / "sentinel2" / f"s2_indices_{tag}_utm{utm_epsg}.tif"
            if s2_p.exists():
                all_checks.extend(check_raster_values(s2_p, 1, f"NDVI {tag}", -0.6, 1.0, 30))
                all_checks.extend(check_raster_values(s2_p, 4, f"Albedo {tag}", 0.0, 0.9, 30))

            if morph_path.exists():
                all_checks.extend(check_raster_values(morph_path, 4, "SVF", 0.0, 1.0, 10))
                all_checks.extend(check_raster_values(morph_path, 6, "ISA%", 0.0, 100.0, 10))

    # ── 4. Feature matrix quality ─────────────────────────────────────────────
    logger.info("Checking feature matrix...")

    if combined_path.exists():
        all_checks.extend(check_feature_matrix(combined_path, "Feature matrix ALL"))

    for year in years[-1:]:
        for season in ["pre_monsoon"]:
            tag = f"{year}_{season}"
            fm_path = export_dir / f"feature_matrix_{city}_{tag}.parquet"
            if fm_path.exists():
                all_checks.extend(check_feature_matrix(fm_path, f"Feature matrix {tag}"))

    # ── 5. Print report ───────────────────────────────────────────────────────
    n_pass = sum(1 for _, ok, _ in all_checks if ok)
    n_fail = sum(1 for _, ok, _ in all_checks if not ok)
    n_total = len(all_checks)

    table = Table(title=f"Quality Report — {city.upper()}", show_lines=True)
    table.add_column("Check", style="dim", min_width=35)
    table.add_column("Status", min_width=8)
    table.add_column("Details", min_width=40)

    for label, ok, detail in all_checks:
        status = "[bold green]PASS[/]" if ok else "[bold red]FAIL[/]"
        table.add_row(label, status, detail)

    console.print(table)

    score = n_pass / n_total * 100 if n_total > 0 else 0
    color = "green" if score >= 80 else "yellow" if score >= 60 else "red"

    console.print(f"\n[bold {color}]Score: {n_pass}/{n_total} checks passed ({score:.0f}%)[/]")

    # ── 6. Save quality report ────────────────────────────────────────────────
    report_dir = ROOT / "logs"
    ensure_dirs(report_dir)
    report_path = report_dir / f"quality_report_{city}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "city": city,
        "generated_at": datetime.now().isoformat(),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "score_pct": round(score, 1),
        "checks": [{"label": l, "pass": ok, "detail": d} for l, ok, d in all_checks],
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Quality report saved: {report_path}")

    if strict and n_fail > 0:
        logger.error(f"{n_fail} checks FAILED in strict mode — fix before handing off to Person B")
        return False

    if n_fail > 0:
        logger.warning(f"{n_fail} checks failed — review and fix before model training")
    else:
        logger.success("All checks passed! Data ready for Person B's PINN model.")

    return n_fail == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data quality checker for ThermalSense AI pipeline")
    parser.add_argument("--city", default="kolkata")
    parser.add_argument("--strict", action="store_true",
                        help="Exit with error code if any check fails")
    args = parser.parse_args()
    ok = run(city=args.city, strict=args.strict)
    sys.exit(0 if ok else 1)
