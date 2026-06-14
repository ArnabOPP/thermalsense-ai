"""
ThermalSense AI — SHAP Analysis
Person B — Script 3 of 5

What this does:
  Uses SHAP (SHapley Additive exPlanations) to explain which features
  drive LST in each part of Kolkata. Produces:
    - Global feature importance bar chart
    - SHAP beeswarm summary plot
    - Spatial SHAP maps (one per feature) showing WHERE each feature
      contributes most to heat stress

Run:
  python model/shap_analysis.py

Requires:
  model/outputs/xgb_model.pkl

Output:
  model/outputs/shap_summary.png
  model/outputs/shap_beeswarm.png
  model/outputs/shap_values.parquet

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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

console = Console()

FEATURE_MATRIX_PATH = ROOT / "outputs" / "exports" / "kolkata" / "feature_matrix_kolkata_ALL.parquet"
XGB_MODEL_PATH      = ROOT / "model" / "outputs" / "xgb_model.pkl"
OUTPUT_DIR          = ROOT / "model" / "outputs"
SHAP_DIR            = OUTPUT_DIR / "shap_maps"
SHAP_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "lst_celsius"


def run():
    console.print("\n[bold blue]ThermalSense AI — SHAP Analysis[/bold blue]")
    console.print("[dim]Person B — Script 3 of 5[/dim]\n")

    import shap

    # 1. Load XGBoost model
    if not XGB_MODEL_PATH.exists():
        logger.error("Run xgb_baseline.py first")
        sys.exit(1)

    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_data = pickle.load(f)
    model    = xgb_data["model"]
    features = xgb_data["features"]
    logger.info(f"Model loaded, features: {features}")

    # 2. Load data — use 2024 pre-monsoon for SHAP (most recent, highest LST)
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df = df[(df[TARGET] > 10) & (df["year"] == 2024) & (df["season"] == "pre_monsoon")].copy()
    df = df[features + [TARGET, "centroid_lat", "centroid_lon"]].dropna()
    logger.info(f"SHAP dataset: {len(df):,} pixels (2024 pre-monsoon)")

    X = df[features].values

    # 3. Compute SHAP values using TreeExplainer (fast for XGBoost)
    logger.info("Computing SHAP values (TreeExplainer)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    logger.success(f"SHAP values computed: {shap_values.shape}")

    # 4. Global summary — mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx    = np.argsort(mean_abs_shap)[::-1]

    logger.info("\nGlobal feature importance (mean |SHAP|):")
    for i in sorted_idx:
        logger.info(f"  {features[i]:<25} {mean_abs_shap[i]:.4f}°C")

    # 5. Bar chart — global importance
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#D85A30" if "isa" in features[i] or "ndbi" in features[i]
              else "#378ADD" if "ndvi" in features[i] or "dist_water" in features[i]
              else "#888888"
              for i in sorted_idx]
    ax.barh([features[i] for i in sorted_idx[::-1]],
            mean_abs_shap[sorted_idx[::-1]],
            color=colors[::-1], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Mean |SHAP value| (°C impact on LST prediction)", fontsize=11)
    ax.set_title("Feature importance — LST drivers in Kolkata 2024", fontsize=12)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: shap_summary.png")

    # 6. Beeswarm plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X, feature_names=features,
                      show=False, plot_size=None)
    plt.title("SHAP beeswarm — Kolkata 2024 pre-monsoon", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: shap_beeswarm.png")

    # 7. Spatial SHAP maps — scatter plot of SHAP contribution by location
    logger.info("\nGenerating spatial SHAP maps...")
    top_features = [features[i] for i in sorted_idx[:4]]  # top 4 features

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, feat in zip(axes, top_features):
        feat_idx  = features.index(feat)
        shap_feat = shap_values[:, feat_idx]

        sc = ax.scatter(df["centroid_lon"], df["centroid_lat"],
                        c=shap_feat, cmap="RdBu_r", s=3, alpha=0.7,
                        vmin=-np.percentile(np.abs(shap_feat), 95),
                        vmax= np.percentile(np.abs(shap_feat), 95))
        plt.colorbar(sc, ax=ax, label="SHAP value (°C)")
        ax.set_title(f"SHAP contribution: {feat}", fontsize=10)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.spines[["top","right"]].set_visible(False)

    plt.suptitle("Spatial SHAP maps — Kolkata 2024 pre-monsoon LST drivers", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_spatial_maps.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved: shap_spatial_maps.png")

    # 8. Save SHAP values as parquet (for the API to serve)
    shap_df = df[["centroid_lat", "centroid_lon"]].copy()
    for i, feat in enumerate(features):
        shap_df[f"shap_{feat}"] = shap_values[:, i]
    shap_df["shap_total"] = shap_values.sum(axis=1)
    shap_df[TARGET] = df[TARGET].values

    shap_path = OUTPUT_DIR / "shap_values.parquet"
    shap_df.to_parquet(shap_path, index=False)
    logger.success(f"SHAP values saved: {shap_path}")

    # 9. Save summary JSON
    summary = {
        "year": 2024, "season": "pre_monsoon",
        "n_pixels": int(len(df)),
        "feature_importance": {
            features[i]: round(float(mean_abs_shap[i]), 4)
            for i in sorted_idx
        },
        "top_heat_driver": features[sorted_idx[0]],
        "top_cooling_driver": features[sorted_idx[-1]],
    }
    with open(OUTPUT_DIR / "shap_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\n[bold green]SHAP analysis complete[/bold green]")
    console.print(f"  Top heat driver:    [red]{summary['top_heat_driver']}[/red]")
    console.print(f"  Top cooling driver: [green]{summary['top_cooling_driver']}[/green]")
    console.print(f"\nNext step: [cyan]python model/scenarios.py[/cyan]")


if __name__ == "__main__":
    run()
