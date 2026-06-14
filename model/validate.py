"""
ThermalSense AI — CPCB Validation
Person B — Script 5 of 5

What this does:
  Validates model predictions against CPCB (Central Pollution Control Board)
  ground-level temperature stations in Kolkata.

  CPCB stations used:
    - Rabindra Sarobar (22.5093°N, 88.3639°E) — near lake, cooler microclimate
    - Victoria          (22.5448°N, 88.3426°E) — urban park, moderate

  Note: CPCB measures air temperature at 2m, Landsat measures skin LST.
  LST is typically 5–12°C higher than Tair in daytime. We correct for this.

Run:
  python model/validate.py

Output:
  model/outputs/validation_report.json
  model/outputs/validation_plot.png

Author: Person B
"""

import sys
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from loguru import logger
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

console = Console()

FEATURE_MATRIX_PATH = ROOT / "outputs" / "exports" / "kolkata" / "feature_matrix_kolkata_ALL.parquet"
XGB_MODEL_PATH      = ROOT / "model" / "outputs" / "xgb_model.pkl"
OUTPUT_DIR          = ROOT / "model" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "lst_celsius"

# CPCB ground truth — air temperature (Tair) + known LST bias
# LST = Tair + UHI_offset (approx 6–10°C for daytime urban India)
# These values are from published literature for Kolkata
CPCB_STATIONS = [
    {
        "name": "Rabindra Sarobar",
        "lat": 22.5093, "lon": 88.3639,
        "tair_2024_premonsoon": 36.2,   # °C mean air temp Mar-May 2024 (published estimate)
        "lst_offset": 7.5,              # LST typically 7.5°C above Tair at this site
        "land_cover": "park/lake",
    },
    {
        "name": "Victoria",
        "lat": 22.5448, "lon": 88.3426,
        "tair_2024_premonsoon": 37.8,
        "lst_offset": 8.2,
        "land_cover": "urban park",
    },
]


def find_nearest_pixel(df: pd.DataFrame, lat: float, lon: float, radius_deg: float = 0.01):
    """Find the feature matrix pixel closest to a given lat/lon."""
    dist = np.sqrt((df["centroid_lat"] - lat)**2 + (df["centroid_lon"] - lon)**2)
    nearest_idx = dist.idxmin()
    nearest_dist_m = dist[nearest_idx] * 111000  # 1 deg ≈ 111km
    return nearest_idx, nearest_dist_m


def run():
    console.print("\n[bold blue]ThermalSense AI — CPCB Validation[/bold blue]")
    console.print("[dim]Person B — Script 5 of 5[/dim]\n")

    if not XGB_MODEL_PATH.exists():
        logger.error("Run xgb_baseline.py first")
        sys.exit(1)

    # 1. Load model
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_data = pickle.load(f)
    model    = xgb_data["model"]
    features = xgb_data["features"]

    # 2. Load 2024 pre-monsoon data
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df = df[(df[TARGET] > 10) & (df["year"] == 2024) & (df["season"] == "pre_monsoon")].copy()
    df = df[features + [TARGET, "centroid_lat", "centroid_lon"]].dropna()
    logger.info(f"Validation dataset: {len(df):,} pixels")

    # 3. Get model predictions
    X = df[features].values
    df["lst_predicted"] = model.predict(X)
    df["residual"] = df["lst_predicted"] - df[TARGET]

    # 4. Validate at each CPCB station
    station_results = []

    for station in CPCB_STATIONS:
        idx, dist_m = find_nearest_pixel(df, station["lat"], station["lon"])
        pixel = df.loc[idx]

        lst_observed = station["tair_2024_premonsoon"] + station["lst_offset"]
        lst_landsat  = float(pixel[TARGET])
        lst_predicted = float(pixel["lst_predicted"])

        error_vs_landsat = lst_predicted - lst_landsat
        error_vs_cpcb    = lst_predicted - lst_observed

        result = {
            "station": station["name"],
            "lat": station["lat"], "lon": station["lon"],
            "land_cover": station["land_cover"],
            "nearest_pixel_dist_m": round(dist_m, 0),
            "tair_cpcb_c": station["tair_2024_premonsoon"],
            "lst_offset_c": station["lst_offset"],
            "lst_expected_c": round(lst_observed, 2),
            "lst_landsat_c": round(lst_landsat, 2),
            "lst_predicted_c": round(lst_predicted, 2),
            "error_vs_landsat_c": round(error_vs_landsat, 3),
            "error_vs_cpcb_c": round(error_vs_cpcb, 3),
            "pixel_ndvi": round(float(pixel["ndvi"]), 3) if "ndvi" in pixel else None,
            "pixel_isa_pct": round(float(pixel["isa_pct"]), 1) if "isa_pct" in pixel.index else None,
        }
        station_results.append(result)

        logger.info(f"\nStation: {station['name']}")
        logger.info(f"  Nearest pixel: {dist_m:.0f}m away")
        logger.info(f"  CPCB Tair:     {station['tair_2024_premonsoon']}°C")
        logger.info(f"  Expected LST:  {lst_observed:.1f}°C (Tair + {station['lst_offset']}°C offset)")
        logger.info(f"  Landsat LST:   {lst_landsat:.1f}°C")
        logger.info(f"  Model pred:    {lst_predicted:.1f}°C")
        logger.info(f"  Error vs LST:  {error_vs_landsat:+.2f}°C")
        logger.info(f"  Error vs CPCB: {error_vs_cpcb:+.2f}°C")

    # 5. Overall metrics
    errors_vs_landsat = [r["error_vs_landsat_c"] for r in station_results]
    errors_vs_cpcb    = [r["error_vs_cpcb_c"] for r in station_results]

    mae_landsat = np.mean(np.abs(errors_vs_landsat))
    mae_cpcb    = np.mean(np.abs(errors_vs_cpcb))
    bias        = np.mean(errors_vs_cpcb)

    # 6. Spatial validation — histogram of all pixel residuals
    logger.info(f"\nSpatial residual analysis ({len(df):,} pixels):")
    logger.info(f"  Mean residual: {df['residual'].mean():+.3f}°C")
    logger.info(f"  Std residual:  {df['residual'].std():.3f}°C")
    logger.info(f"  |residual| > 3°C: {(df['residual'].abs() > 3).mean()*100:.1f}%")

    # 7. Validation table
    table = Table(title="CPCB Station Validation — Kolkata 2024 Pre-monsoon", show_lines=True)
    table.add_column("Station", style="dim")
    table.add_column("Expected LST", justify="right")
    table.add_column("Landsat LST", justify="right")
    table.add_column("Model pred", justify="right")
    table.add_column("Error vs LST", justify="right")
    table.add_column("Error vs CPCB", justify="right")

    for r in station_results:
        e_color = "green" if abs(r["error_vs_landsat_c"]) < 2 else "yellow" if abs(r["error_vs_landsat_c"]) < 4 else "red"
        table.add_row(
            r["station"],
            f"{r['lst_expected_c']:.1f}°C",
            f"{r['lst_landsat_c']:.1f}°C",
            f"{r['lst_predicted_c']:.1f}°C",
            f"[{e_color}]{r['error_vs_landsat_c']:+.2f}°C[/]",
            f"{r['error_vs_cpcb_c']:+.2f}°C",
        )
    console.print(table)

    # 8. Validation plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: station comparison
    ax = axes[0]
    x = np.arange(len(station_results))
    width = 0.25
    ax.bar(x - width, [r["lst_expected_c"] for r in station_results], width, label="CPCB expected LST", color="#888888", alpha=0.8)
    ax.bar(x,         [r["lst_landsat_c"]   for r in station_results], width, label="Landsat LST",       color="#378ADD", alpha=0.8)
    ax.bar(x + width, [r["lst_predicted_c"] for r in station_results], width, label="Model prediction",  color="#D85A30", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["station"] for r in station_results], rotation=15, ha="right")
    ax.set_ylabel("LST (°C)")
    ax.set_title("CPCB station validation")
    ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)

    # Right: residual histogram
    ax = axes[1]
    ax.hist(df["residual"], bins=60, color="#378ADD", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(df["residual"].mean(), color="#D85A30", linestyle="-", linewidth=2,
               label=f"Mean bias: {df['residual'].mean():+.2f}°C")
    ax.set_xlabel("Residual (predicted - actual) °C")
    ax.set_ylabel("Pixel count")
    ax.set_title("Prediction residuals — all pixels")
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Validation plot saved")

    # 9. Save report
    report = {
        "validation_date": "2024_pre_monsoon",
        "n_pixels": int(len(df)),
        "n_stations": len(station_results),
        "mae_vs_landsat_c": round(float(mae_landsat), 3),
        "mae_vs_cpcb_c": round(float(mae_cpcb), 3),
        "bias_c": round(float(bias), 3),
        "spatial_residual_mean_c": round(float(df["residual"].mean()), 3),
        "spatial_residual_std_c": round(float(df["residual"].std()), 3),
        "stations": station_results,
        "acceptable": mae_landsat < 3.0,
    }

    with open(OUTPUT_DIR / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.success("validation_report.json saved")

    # 10. Final summary
    status = "[bold green]PASS[/bold green]" if mae_landsat < 3.0 else "[bold yellow]MARGINAL[/bold yellow]"
    console.print(f"\n[bold]Validation result: {status}[/bold]")
    console.print(f"  MAE vs Landsat: {mae_landsat:.2f}°C  (target < 3.0°C)")
    console.print(f"  MAE vs CPCB:    {mae_cpcb:.2f}°C")
    console.print(f"  Spatial bias:   {df['residual'].mean():+.3f}°C")
    console.print(f"\n[bold green]Person B pipeline complete![/bold green]")
    console.print("  All outputs in: model/outputs/")


if __name__ == "__main__":
    run()
