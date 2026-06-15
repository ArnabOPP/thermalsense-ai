"""
ThermalSense AI — XGBoost Baseline Model
Person B — Script 1 of 5

What this does:
  Loads the feature matrix from Person A, trains an XGBoost model to predict
  Land Surface Temperature (LST) from 8 geospatial + atmospheric features,
  evaluates performance, and saves the trained model.

This is the baseline. The PINN (Script 2) adds physics constraints on top.

Run:
  python model/xgb_baseline.py

Output:
  model/outputs/xgb_model.pkl
  model/outputs/xgb_feature_importance.png
  model/outputs/xgb_predictions.parquet

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
matplotlib.use("Agg")  # non-interactive backend for Windows
from loguru import logger
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

console = Console()

# ─── Config ───────────────────────────────────────────────────────────────────

# Set dynamically based on --city arg
OUTPUT_DIR = ROOT / "model" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Features used for training
# Morphology columns (building_density, svf, etc.) included if not NaN
CORE_FEATURES = [
    "ndvi",
    "ndwi",
    "ndbi",
    "albedo",
    "tatm",
    "era5_humidity",
    "era5_wind_speed",
    "doy_sin",
]

MORPHOLOGY_FEATURES = [
    "isa_pct",
    "svf",
    "building_height",
    "building_density",
    "canyon_ratio",
    "dist_water_m",
]

TARGET = "lst_celsius"

# ─── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Load feature matrix, filter bad pixels, select features.
    Returns (clean_df, feature_columns).
    """
    logger.info(f"Loading feature matrix: {FEATURE_MATRIX_PATH}")
    if not FEATURE_MATRIX_PATH.exists():
        logger.error(f"Feature matrix not found: {FEATURE_MATRIX_PATH}")
        logger.error("Run Person A's pipeline first: python run_pipeline.py --city kolkata")
        sys.exit(1)

    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    logger.info(f"  Loaded: {len(df):,} rows × {len(df.columns)} columns")

    # Filter water pixels (LST near 0 = water body, not urban surface)
    before = len(df)
    df = df[df[TARGET] > 10].copy()
    logger.info(f"  After filtering LST > 10°C: {len(df):,} rows (removed {before-len(df):,} water pixels)")

    # Decide which features to use based on NaN rates
    features = CORE_FEATURES.copy()
    morph_nan_rates = {col: df[col].isna().mean() for col in MORPHOLOGY_FEATURES if col in df.columns}
    for col, nan_rate in morph_nan_rates.items():
        if nan_rate < 0.05:  # less than 5% NaN → include
            features.append(col)
            logger.info(f"  Including morphology feature: {col} (NaN: {nan_rate*100:.1f}%)")
        else:
            logger.warning(f"  Skipping {col} ({nan_rate*100:.0f}% NaN)")

    # Drop any remaining NaN rows in selected features + target
    df = df[features + [TARGET, "year", "season", "centroid_lat", "centroid_lon"]].dropna()
    logger.info(f"  After dropna: {len(df):,} rows, {len(features)} features")

    # Log feature stats
    logger.info("\nFeature statistics:")
    for f in features:
        logger.info(f"  {f:<25} mean={df[f].mean():.3f} std={df[f].std():.3f}")

    return df, features


# ─── Train/test split ──────────────────────────────────────────────────────────

def spatial_train_test_split(df: pd.DataFrame, test_fraction: float = 0.2):
    """
    Spatial split: use northern half of Kolkata for testing,
    southern half for training. This prevents spatial autocorrelation
    leaking between train and test sets (more rigorous than random split).
    """
    lat_threshold = df["centroid_lat"].quantile(0.8)
    train_mask = df["centroid_lat"] <= lat_threshold
    test_mask = df["centroid_lat"] > lat_threshold

    train_df = df[train_mask].copy()
    test_df  = df[test_mask].copy()

    logger.info(f"\nSpatial train/test split (threshold lat={lat_threshold:.4f}°N):")
    logger.info(f"  Train: {len(train_df):,} rows ({len(train_df)/len(df)*100:.0f}%)")
    logger.info(f"  Test:  {len(test_df):,} rows ({len(test_df)/len(df)*100:.0f}%)")
    logger.info(f"  Train LST: {train_df[TARGET].mean():.1f}°C ± {train_df[TARGET].std():.1f}°C")
    logger.info(f"  Test LST:  {test_df[TARGET].mean():.1f}°C ± {test_df[TARGET].std():.1f}°C")

    return train_df, test_df


# ─── XGBoost training ──────────────────────────────────────────────────────────

def train_xgboost(train_df: pd.DataFrame, features: list[str]):
    """
    Train XGBoost with reasonable defaults.
    No hyperparameter tuning here — Optuna tuning is in pinn_model.py.
    """
    import xgboost as xgb

    X_train = train_df[features].values
    y_train = train_df[TARGET].values

    logger.info("\nTraining XGBoost...")
    logger.info(f"  X shape: {X_train.shape}")
    logger.info(f"  y shape: {y_train.shape}")
    logger.info(f"  y mean: {y_train.mean():.2f}°C  std: {y_train.std():.2f}°C")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,       # L1 regularization
        reg_lambda=1.0,      # L2 regularization
        random_state=42,
        n_jobs=-1,           # use all CPU cores
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )

    logger.success("XGBoost training complete")
    return model


# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, df: pd.DataFrame, features: list[str], split_name: str) -> dict:
    """Compute regression metrics for a dataset split."""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    X = df[features].values
    y_true = df[TARGET].values
    y_pred = model.predict(X)

    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = (y_pred - y_true).mean()

    logger.info(f"\n{split_name} metrics:")
    logger.info(f"  R²   = {r2:.4f}")
    logger.info(f"  MAE  = {mae:.3f}°C")
    logger.info(f"  RMSE = {rmse:.3f}°C")
    logger.info(f"  Bias = {bias:+.3f}°C")

    return {
        "split": split_name,
        "r2": round(r2, 4),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "bias": round(float(bias), 3),
        "n": int(len(y_true)),
        "y_pred": y_pred,
        "y_true": y_true,
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_feature_importance(model, features: list[str]) -> None:
    """Bar chart of XGBoost feature importance — saved to outputs/."""
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(features)),
                  importance[sorted_idx],
                  color="#378ADD", alpha=0.85, edgecolor="white")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([features[i] for i in sorted_idx], rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Feature importance (gain)", fontsize=11)
    ax.set_title("XGBoost — Feature importance for LST prediction", fontsize=12)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    path = OUTPUT_DIR / "xgb_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Feature importance plot saved: {path.name}")


def plot_scatter(train_metrics: dict, test_metrics: dict) -> None:
    """Predicted vs actual scatter plot for train and test."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, m, color in [(ax1, train_metrics, "#378ADD"), (ax2, test_metrics, "#D85A30")]:
        ax.scatter(m["y_true"], m["y_pred"], alpha=0.15, s=3, color=color)
        mn = min(m["y_true"].min(), m["y_pred"].min())
        mx = max(m["y_true"].max(), m["y_pred"].max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.5, label="Perfect fit")
        ax.set_xlabel("Actual LST (°C)", fontsize=11)
        ax.set_ylabel("Predicted LST (°C)", fontsize=11)
        ax.set_title(f"{m['split']}  R²={m['r2']:.3f}  RMSE={m['rmse']:.2f}°C", fontsize=11)
        ax.spines[["top","right"]].set_visible(False)

    plt.suptitle("XGBoost — Predicted vs Actual LST", fontsize=13, y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / "xgb_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Scatter plot saved: {path.name}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", default="kolkata")
    args, _ = parser.parse_known_args()
    city = args.city

    FEATURE_MATRIX_PATH = ROOT / "outputs" / "exports" / city / f"feature_matrix_{city}_ALL.parquet"
    console.print("\n[bold blue]ThermalSense AI — XGBoost Baseline[/bold blue]")
    console.print("[dim]Person B — Script 1 of 5[/dim]\n")

    # 1. Load data
    df, features = load_data()

    # 2. Split
    train_df, test_df = spatial_train_test_split(df)

    # 3. Train
    model = train_xgboost(train_df, features)

    # 4. Evaluate
    train_metrics = evaluate(model, train_df, features, "Train")
    test_metrics  = evaluate(model, test_df,  features, "Test")

    # 5. Rich summary table
    table = Table(title="XGBoost Performance", show_lines=True)
    table.add_column("Metric", style="dim")
    table.add_column("Train", justify="right")
    table.add_column("Test",  justify="right")
    table.add_column("Target", justify="right", style="green")

    table.add_row("R²",   f"{train_metrics['r2']:.4f}",  f"{test_metrics['r2']:.4f}",  "> 0.75")
    table.add_row("MAE",  f"{train_metrics['mae']:.3f}°C", f"{test_metrics['mae']:.3f}°C", "< 2.0°C")
    table.add_row("RMSE", f"{train_metrics['rmse']:.3f}°C", f"{test_metrics['rmse']:.3f}°C", "< 3.0°C")
    table.add_row("Bias", f"{train_metrics['bias']:+.3f}°C", f"{test_metrics['bias']:+.3f}°C", "≈ 0°C")
    table.add_row("N",    f"{train_metrics['n']:,}", f"{test_metrics['n']:,}", "—")
    console.print(table)

    # 6. Check target
    r2_test = test_metrics["r2"]
    if r2_test >= 0.75:
        console.print(f"\n[bold green]✓ R² = {r2_test:.4f} — target achieved (> 0.75)[/bold green]")
    elif r2_test >= 0.65:
        console.print(f"\n[bold yellow]⚠ R² = {r2_test:.4f} — close but below target. PINN may improve it.[/bold yellow]")
    else:
        console.print(f"\n[bold red]✗ R² = {r2_test:.4f} — below target. Check feature quality.[/bold red]")

    # 7. Save model
    model_path = OUTPUT_DIR / "xgb_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": features, "metrics": {
            "train": {k:v for k,v in train_metrics.items() if k not in ("y_pred","y_true")},
            "test":  {k:v for k,v in test_metrics.items()  if k not in ("y_pred","y_true")},
        }}, f)
    logger.success(f"Model saved: {model_path}")

    # 8. Save predictions
    test_df = test_df.copy()
    test_df["lst_predicted"] = test_metrics["y_pred"]
    test_df["residual"] = test_df["lst_predicted"] - test_df[TARGET]
    pred_path = OUTPUT_DIR / "xgb_predictions.parquet"
    test_df.to_parquet(pred_path, index=False)
    logger.success(f"Predictions saved: {pred_path}")

    # 9. Save metrics JSON
    metrics_path = OUTPUT_DIR / "xgb_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "model": "XGBoost",
            "features": features,
            "train": {k:v for k,v in train_metrics.items() if k not in ("y_pred","y_true")},
            "test":  {k:v for k,v in test_metrics.items()  if k not in ("y_pred","y_true")},
        }, f, indent=2)

    # 10. Plots
    logger.info("\nGenerating plots...")
    plot_feature_importance(model, features)
    plot_scatter(train_metrics, test_metrics)

    console.print(f"\n[bold green]XGBoost baseline complete.[/bold green]")
    console.print(f"Next step: [cyan]python model/pinn_model.py[/cyan]")

    return model, features, test_df


if __name__ == "__main__":
    run()
