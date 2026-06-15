"""
ThermalSense AI — Intervention Scenarios
Person B — Script 4 of 5

What this does:
  Simulates 4 urban cooling interventions by modifying input features
  and measuring predicted LST change using the trained XGBoost model.

  Interventions:
    1. Urban greening    — NDVI += 0.25 in low-vegetation zones
    2. Cool roofs        — albedo 0.12 → 0.65 in dense residential zones
    3. EKW restoration   — expand East Kolkata Wetlands (water pixels)
    4. Green corridors   — NDVI boost along major roads

  Each scenario outputs:
    - Mean ΔT across Kolkata (°C)
    - Spatial map of where cooling is strongest
    - Cost-benefit estimate (₹ per °C cooling per km²)

Run:
  python model/scenarios.py

Output:
  model/outputs/scenario_results.json
  model/outputs/scenario_maps.png

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

# Cost estimates (rough, for proposal narrative)
SCENARIO_COSTS = {
    "urban_greening":   {"cost_cr_per_km2": 2.5,  "name": "Urban greening (+15% tree cover)"},
    "cool_roofs":       {"cost_cr_per_km2": 1.8,  "name": "Cool roofs (albedo 0.12→0.65)"},
    "ekw_restoration":  {"cost_cr_per_km2": 0.8,  "name": "EKW wetland restoration"},
    "green_corridors":  {"cost_cr_per_km2": 1.2,  "name": "Green corridors (AJC Bose + EM Bypass)"},
}


def load_baseline():
    """Load model and 2024 pre-monsoon data as the scenario baseline."""
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_data = pickle.load(f)
    model    = xgb_data["model"]
    features = xgb_data["features"]

    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df = df[(df[TARGET] > 10) & (df["year"] == 2024) & (df["season"] == "pre_monsoon")].copy()
    df = df[features + [TARGET, "centroid_lat", "centroid_lon"]].dropna()

    X_baseline = df[features].values
    lst_baseline = model.predict(X_baseline)
    df["lst_baseline"] = lst_baseline

    logger.info(f"Baseline loaded: {len(df):,} pixels")
    logger.info(f"  Baseline LST: mean={lst_baseline.mean():.2f}°C  max={lst_baseline.max():.2f}°C")

    return model, features, df


def scenario_urban_greening(df: pd.DataFrame, features: list, model) -> pd.Series:
    """
    Scenario 1: Urban greening — increase tree cover by 15%.
    Target: pixels with NDVI < 0.3 (sparse vegetation or built-up).
    Intervention: NDVI += 0.25 (simulates planting trees on 15% of land).
    """
    X_mod = df[features].copy()
    low_veg_mask = X_mod["ndvi"] < 0.3

    X_mod.loc[low_veg_mask, "ndvi"]  += 0.25
    X_mod.loc[low_veg_mask, "ndwi"]  += 0.05   # more vegetation → slightly higher moisture
    X_mod.loc[low_veg_mask, "albedo"] -= 0.02   # canopy slightly darker than bare soil

    # Clip to valid ranges
    X_mod["ndvi"]   = X_mod["ndvi"].clip(-1, 1)
    X_mod["ndwi"]   = X_mod["ndwi"].clip(-1, 1)
    X_mod["albedo"] = X_mod["albedo"].clip(0, 0.9)

    return pd.Series(model.predict(X_mod[features].values), index=df.index)


def scenario_cool_roofs(df: pd.DataFrame, features: list, model) -> pd.Series:
    """
    Scenario 2: Cool roofs — increase roof albedo in dense urban areas.
    Target: pixels with ISA% > 60% (dense built-up).
    Intervention: albedo 0.12 → 0.65 (white/reflective coating).
    """
    X_mod = df[features].copy()
    if "isa_pct" in features:
        dense_mask = X_mod["isa_pct"] > 60
    else:
        dense_mask = X_mod["ndbi"] > 0.0   # fallback: NDBI > 0 = built-up

    # Cool roofs increase albedo significantly
    new_albedo = (X_mod.loc[dense_mask, "albedo"].clip(lower=0) + 0.40).clip(upper=0.65)
    X_mod.loc[dense_mask, "albedo"] = new_albedo.astype(np.float32)

    return pd.Series(model.predict(X_mod[features].values), index=df.index)


def scenario_ekw_restoration(df: pd.DataFrame, features: list, model) -> pd.Series:
    """
    Scenario 3: East Kolkata Wetlands restoration.
    Target: pixels in the eastern zone (lon > 88.37°) with low NDWI.
    Intervention: convert degraded land to water/wetland.
    """
    X_mod = df[features].copy()

    # EKW is in the eastern part of the bbox
    # Use relative position — eastern 30% of the city's bbox
    lon_threshold = df["centroid_lon"].quantile(0.70)
    eastern_mask = df["centroid_lon"] > lon_threshold
    degraded_mask = X_mod["ndwi"] < 0.0

    target_mask = eastern_mask & degraded_mask
    target_pct = target_mask.mean() * 100
    logger.info(f"  EKW restoration target: {target_mask.sum():,} pixels ({target_pct:.1f}% of city)")

    # Simulate wetland: increase NDWI, increase NDVI (wetland veg), decrease NDBI
    X_mod.loc[target_mask, "ndwi"]  = X_mod.loc[target_mask, "ndwi"].clip(lower=0) + 0.15
    X_mod.loc[target_mask, "ndvi"]  = X_mod.loc[target_mask, "ndvi"].clip(lower=0) + 0.10
    X_mod.loc[target_mask, "ndbi"]  = X_mod.loc[target_mask, "ndbi"] - 0.10
    X_mod.loc[target_mask, "albedo"] -= 0.03  # water is dark

    X_mod["ndwi"]  = X_mod["ndwi"].clip(-1, 1)
    X_mod["ndvi"]  = X_mod["ndvi"].clip(-1, 1)
    X_mod["ndbi"]  = X_mod["ndbi"].clip(-1, 1)
    X_mod["albedo"] = X_mod["albedo"].clip(0, 0.9)

    return pd.Series(model.predict(X_mod[features].values), index=df.index)


def scenario_green_corridors(df: pd.DataFrame, features: list, model) -> pd.Series:
    """
    Scenario 4: Green corridors along AJC Bose Road and EM Bypass.
    Target: pixels within 300m of major road corridors.
    Intervention: NDVI boost from street trees.
    """
    X_mod = df[features].copy()

    # AJC Bose Road roughly: lat 22.54–22.56, lon 88.34–88.36
    # EM Bypass roughly: lat 22.50–22.58, lon 88.37–88.39
    # Use relative positions — central corridor of the city
    lat_mid = df["centroid_lat"].mean()
    lon_mid = df["centroid_lon"].mean()
    lat_span = df["centroid_lat"].std()
    lon_span = df["centroid_lon"].std()

    # Main north-south corridor
    ajc_mask = (
        df["centroid_lat"].between(lat_mid - lat_span*0.3, lat_mid + lat_span*0.3) &
        df["centroid_lon"].between(lon_mid - lon_span*0.2, lon_mid + lon_span*0.2)
    )
    # Eastern bypass
    em_bypass_mask = (
        df["centroid_lat"].between(lat_mid - lat_span*0.5, lat_mid + lat_span*0.5) &
        df["centroid_lon"].between(lon_mid + lon_span*0.1, lon_mid + lon_span*0.5)
    )
    corridor_mask = ajc_mask | em_bypass_mask

    logger.info(f"  Corridor pixels: {corridor_mask.sum():,} ({corridor_mask.mean()*100:.1f}% of city)")

    # Plant street trees: NDVI increases, albedo slightly decreases (canopy shade)
    X_mod.loc[corridor_mask, "ndvi"]   += 0.20
    X_mod.loc[corridor_mask, "albedo"] -= 0.02
    X_mod["ndvi"]   = X_mod["ndvi"].clip(-1, 1)
    X_mod["albedo"] = X_mod["albedo"].clip(0, 0.9)

    return pd.Series(model.predict(X_mod[features].values), index=df.index)


def run():
    console.print("\n[bold blue]ThermalSense AI — Intervention Scenarios[/bold blue]")
    console.print("[dim]Person B — Script 4 of 5[/dim]\n")

    if not XGB_MODEL_PATH.exists():
        logger.error("Run xgb_baseline.py first")
        sys.exit(1)

    # 1. Load
    model, features, df = load_baseline()

    # 2. Run all 4 scenarios
    scenarios = {
        "urban_greening":  scenario_urban_greening,
        "cool_roofs":      scenario_cool_roofs,
        "ekw_restoration": scenario_ekw_restoration,
        "green_corridors": scenario_green_corridors,
    }

    results = {}
    scenario_predictions = {}

    for key, fn in scenarios.items():
        logger.info(f"\nRunning: {SCENARIO_COSTS[key]['name']}")
        lst_modified = fn(df, features, model)
        delta_t = lst_modified - df["lst_baseline"]

        mean_delta = float(delta_t.mean())
        max_cooling = float(delta_t.min())
        pct_pixels_cooled = float((delta_t < -0.5).mean() * 100)

        cost_cr = SCENARIO_COSTS[key]["cost_cr_per_km2"]
        area_km2 = (df["centroid_lat"].nunique() * 0.1) * (df["centroid_lon"].nunique() * 0.1)
        total_cost = cost_cr * area_km2
        cost_per_degree = total_cost / abs(mean_delta) if mean_delta < 0 else float("inf")

        results[key] = {
            "name": SCENARIO_COSTS[key]["name"],
            "mean_delta_t_c": round(mean_delta, 3),
            "max_cooling_c": round(max_cooling, 3),
            "pct_pixels_cooled": round(pct_pixels_cooled, 1),
            "cost_cr_per_km2": cost_cr,
            "cost_efficiency_cr_per_degree": round(cost_per_degree, 2),
        }
        scenario_predictions[key] = lst_modified

        logger.info(f"  ΔT mean:      {mean_delta:+.3f}°C")
        logger.info(f"  ΔT max cool:  {max_cooling:+.3f}°C")
        logger.info(f"  Pixels cooled >0.5°C: {pct_pixels_cooled:.1f}%")

    # 3. Rich results table
    table = Table(title="Intervention Scenario Results — Kolkata 2024", show_lines=True)
    table.add_column("Scenario", style="dim", min_width=30)
    table.add_column("Mean ΔT", justify="right")
    table.add_column("Max cooling", justify="right")
    table.add_column("% pixels cooled", justify="right")
    table.add_column("Cost (₹Cr/km²)", justify="right")

    for key, r in results.items():
        color = "green" if r["mean_delta_t_c"] < -1 else "yellow"
        table.add_row(
            r["name"],
            f"[{color}]{r['mean_delta_t_c']:+.2f}°C[/]",
            f"{r['max_cooling_c']:+.2f}°C",
            f"{r['pct_pixels_cooled']:.1f}%",
            f"₹{r['cost_cr_per_km2']:.1f}Cr",
        )
    console.print(table)

    # 4. Spatial maps
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (key, lst_mod) in zip(axes, scenario_predictions.items()):
        delta = lst_mod - df["lst_baseline"]
        sc = ax.scatter(df["centroid_lon"], df["centroid_lat"],
                        c=delta, cmap="RdBu", s=3, alpha=0.8,
                        vmin=-5, vmax=2)
        plt.colorbar(sc, ax=ax, label="ΔT (°C)")
        ax.set_title(f"{SCENARIO_COSTS[key]['name']}\nMean ΔT = {results[key]['mean_delta_t_c']:+.2f}°C",
                     fontsize=9)
        ax.set_xlabel("Longitude", fontsize=8)
        ax.set_ylabel("Latitude", fontsize=8)
        ax.spines[["top","right"]].set_visible(False)

    plt.suptitle("Spatial cooling impact — Kolkata intervention scenarios", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "scenario_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Spatial maps saved: scenario_maps.png")

    # 5. Save JSON
    with open(OUTPUT_DIR / "scenario_results.json", "w") as f:
        json.dump({"baseline_lst_mean_c": round(float(df["lst_baseline"].mean()), 2),
                   "scenarios": results}, f, indent=2)
    logger.success("scenario_results.json saved")

    # 6. Best scenario
    best = min(results, key=lambda k: results[k]["mean_delta_t_c"])
    console.print(f"\n[bold green]Best scenario: {results[best]['name']}[/bold green]")
    console.print(f"  Mean cooling: {results[best]['mean_delta_t_c']:+.2f}°C")
    console.print(f"\nNext step: [cyan]python model/validate.py[/cyan]")


if __name__ == "__main__":
    run()
