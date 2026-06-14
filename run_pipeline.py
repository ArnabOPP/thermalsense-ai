"""
ThermalSense AI — Master pipeline runner
Runs scripts 01 → 06 in sequence for any city.

This is the entry point for Person A. Run this once per city.
For Kolkata, the full pipeline takes approximately:
  Script 01 (Landsat):      20–40 min (12 year×season combos)
  Script 02 (Sentinel-2):   20–40 min
  Script 03 (Morphology):    5–15 min (OSM fetch + compute)
  Script 04 (INSAT-3D):      5–15 min (or ERA5 fallback)
  Script 05 (Features):      5–10 min (alignment + Parquet export)
  Script 06 (QC):            1–2  min
  TOTAL:                   ~60–120 min

Usage:
  python run_pipeline.py --city kolkata
  python run_pipeline.py --city delhi
  python run_pipeline.py --city kolkata --year 2024 --season pre_monsoon --skip-qc

Author: Person A
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils import setup_logger, load_config, print_banner

console = Console()


def run_step(step_name: str, fn, *args, **kwargs) -> tuple[bool, float]:
    """Run a pipeline step, return (success, elapsed_seconds)."""
    console.print(f"\n[bold blue]{'─'*60}[/]")
    console.print(f"[bold white]Starting: {step_name}[/]")
    console.print(f"[dim]{datetime.now().strftime('%H:%M:%S')}[/]")

    t0 = time.time()
    try:
        fn(*args, **kwargs)
        elapsed = time.time() - t0
        console.print(f"[bold green]✓ {step_name} complete ({elapsed:.0f}s)[/]")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        console.print(f"[bold red]✗ {step_name} FAILED after {elapsed:.0f}s[/]")
        console.print(f"[red]{traceback.format_exc()}[/]")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="ThermalSense AI — full data pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --city kolkata
  python run_pipeline.py --city delhi
  python run_pipeline.py --city kolkata --year 2024 --season pre_monsoon
  python run_pipeline.py --city kolkata --skip 01 02  --only 05 06
  python run_pipeline.py --city kolkata --use-era5    (skip MOSDAC)
        """
    )
    parser.add_argument("--city",       default="kolkata",  help="City name (must be in config)")
    parser.add_argument("--year",       type=int, nargs="+", help="Specific years (default: all)")
    parser.add_argument("--season",     nargs="+",           help="Specific seasons (default: both)")
    parser.add_argument("--skip",       nargs="+", default=[], help="Step numbers to skip e.g. --skip 01 03")
    parser.add_argument("--only",       nargs="+", default=[], help="Only run these steps e.g. --only 05 06")
    parser.add_argument("--use-era5",   action="store_true",  help="Use ERA5 instead of MOSDAC for atmospheric temp")
    parser.add_argument("--skip-qc",    action="store_true",  help="Skip quality check (script 06)")
    parser.add_argument("--strict-qc",  action="store_true",  help="Fail pipeline if quality check fails")
    args = parser.parse_args()

    setup_logger("run_pipeline")

    print_banner(
        "ThermalSense AI — Data Pipeline",
        f"City: {args.city.upper()} | Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    )

    cfg = load_config()
    if args.city not in cfg.get("cities", {}):
        logger.error(f"City '{args.city}' not in config/config.yaml")
        logger.info(f"Available cities: {list(cfg['cities'].keys())}")
        sys.exit(1)

    years   = args.year   or cfg["pipeline"]["landsat"]["years"]
    seasons = args.season or list(cfg["pipeline"]["landsat"]["seasons"].keys())

    logger.info(f"City:    {args.city}")
    logger.info(f"Years:   {years}")
    logger.info(f"Seasons: {seasons}")

    # Import step modules
    import importlib.util

    def load_script(num: str):
        scripts = {
            "01": "01_pull_landsat_lst",
            "02": "02_pull_sentinel2",
            "03": "03_osm_morphology",
            "04": "04_mosdac_insat3d",
            "05": "05_feature_engineering",
            "06": "06_quality_check",
        }
        name = scripts[num]
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def should_run(step_num: str) -> bool:
        if args.only:
            return step_num in args.only
        return step_num not in args.skip

    # ── Execute steps ─────────────────────────────────────────────────────────
    results = {}
    t_pipeline_start = time.time()

    if should_run("01"):
        mod = load_script("01")
        ok, elapsed = run_step(
            "Script 01 — Landsat 8 LST",
            mod.run,
            city=args.city, years=years, seasons=seasons,
        )
        results["01"] = (ok, elapsed)
        if not ok:
            logger.error("Script 01 failed — cannot continue without LST data")
            sys.exit(1)

    if should_run("02"):
        mod = load_script("02")
        ok, elapsed = run_step(
            "Script 02 — Sentinel-2 NDVI/LULC",
            mod.run,
            city=args.city, years=years, seasons=seasons,
        )
        results["02"] = (ok, elapsed)

    if should_run("03"):
        mod = load_script("03")
        ok, elapsed = run_step(
            "Script 03 — OSM Urban Morphology",
            mod.run,
            city=args.city,
        )
        results["03"] = (ok, elapsed)

    if should_run("04"):
        mod = load_script("04")
        ok, elapsed = run_step(
            "Script 04 — MOSDAC INSAT-3D",
            mod.run,
            city=args.city,
            use_era5_fallback=args.use_era5,
        )
        results["04"] = (ok, elapsed)

    if should_run("05"):
        mod = load_script("05")
        ok, elapsed = run_step(
            "Script 05 — Feature Engineering",
            mod.run,
            city=args.city, years=years,
        )
        results["05"] = (ok, elapsed)
        if not ok:
            logger.error("Script 05 failed — feature matrix not produced")
            sys.exit(1)

    if should_run("06") and not args.skip_qc:
        mod = load_script("06")
        ok, elapsed = run_step(
            "Script 06 — Quality Check",
            mod.run,
            city=args.city,
            strict=args.strict_qc,
        )
        results["06"] = (ok, elapsed)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_pipeline_start
    n_steps = len(results)
    n_pass  = sum(1 for ok, _ in results.values() if ok)

    console.print(f"\n{'='*60}")
    console.print(Panel(
        f"[bold]Pipeline complete: {n_pass}/{n_steps} steps passed[/]\n"
        f"City: [cyan]{args.city.upper()}[/] | "
        f"Time: [yellow]{total_elapsed/60:.1f} min[/]",
        style="green" if n_pass == n_steps else "yellow",
    ))

    for step, (ok, elapsed) in results.items():
        icon = "[green]✓[/]" if ok else "[red]✗[/]"
        console.print(f"  {icon} Script {step}: {elapsed:.0f}s")

    if n_pass == n_steps:
        cfg_paths = cfg["pipeline"]["paths"]
        exports = ROOT / cfg_paths["exports_dir"] / args.city
        feature_file = exports / f"feature_matrix_{args.city}_ALL.parquet"

        console.print(f"\n[bold green]✓ All done! Hand off to Person B:[/]")
        console.print(f"  Feature matrix: {feature_file}")
        console.print(f"  Run: python model/xgb_baseline.py --city {args.city}")
    else:
        console.print(f"\n[yellow]Some steps failed — review logs/ directory[/]")


if __name__ == "__main__":
    main()
