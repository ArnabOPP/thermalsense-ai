"""
ThermalSense AI — Physics-Informed Neural Network (PINN)
Person B — Script 2 of 5

Architecture:
  XGBoost baseline (already trained) predicts LST from features.
  PINN learns to predict the RESIDUAL that XGBoost misses.
  Final prediction = XGBoost prediction + PINN residual correction.

  This hybrid approach is more stable than a pure PINN:
  - XGBoost handles the non-linear feature relationships
  - PINN corrects systematic errors while respecting physics
  - Physics loss penalises predictions that violate energy balance

Run:
  python model/pinn_model.py

Requires:
  model/outputs/xgb_model.pkl  (run xgb_baseline.py first)

Output:
  model/outputs/pinn_model.pt
  model/outputs/pinn_metrics.json
  model/outputs/pinn_scatter.png

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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "model"))

from pinn_loss import PhysicsLoss

console = Console()

FEATURE_MATRIX_PATH = ROOT / "outputs" / "exports" / "kolkata" / "feature_matrix_kolkata_ALL.parquet"
XGB_MODEL_PATH      = ROOT / "model" / "outputs" / "xgb_model.pkl"
OUTPUT_DIR          = ROOT / "model" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "lst_celsius"


# ─── PINN architecture ────────────────────────────────────────────────────────

class ResidualPINN(nn.Module):
    """
    Small MLP that learns the residual correction on top of XGBoost.

    Input:  n_features (same as XGBoost)
    Output: scalar residual correction (°C)

    The final LST prediction is:
      LST = XGBoost(features) + PINN(features)
    """

    def __init__(self, n_features: int, hidden_dim: int = 128, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()

        layers = []
        in_dim = n_features

        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else hidden_dim // 2
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

        # Initialize weights small so PINN starts as near-zero correction
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─── Data prep ────────────────────────────────────────────────────────────────

def load_and_prepare():
    """Load feature matrix and XGBoost model, compute XGB predictions."""
    logger.info("Loading data and XGBoost model...")

    # Load XGBoost
    if not XGB_MODEL_PATH.exists():
        logger.error(f"XGBoost model not found: {XGB_MODEL_PATH}")
        logger.error("Run: python model/xgb_baseline.py first")
        sys.exit(1)

    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_data = pickle.load(f)
    xgb_model = xgb_data["model"]
    features  = xgb_data["features"]

    logger.info(f"  XGBoost loaded, features: {features}")

    # Load feature matrix
    df = pd.read_parquet(FEATURE_MATRIX_PATH)
    df = df[df[TARGET] > 10].copy()
    df = df[features + [TARGET, "centroid_lat", "centroid_lon", "year", "season"]].dropna()

    logger.info(f"  Dataset: {len(df):,} rows")

    # XGBoost predictions
    X = df[features].values
    xgb_pred = xgb_model.predict(X)
    df["xgb_pred"] = xgb_pred
    df["residual"] = df[TARGET] - xgb_pred  # what PINN needs to learn

    logger.info(f"  XGBoost residuals: mean={df['residual'].mean():.3f}°C  std={df['residual'].std():.3f}°C")
    logger.info(f"  XGBoost test R² (from saved metrics): {xgb_data['metrics']['test']['r2']:.4f}")

    # Spatial split (same as XGBoost)
    lat_threshold = df["centroid_lat"].quantile(0.8)
    train_df = df[df["centroid_lat"] <= lat_threshold].copy()
    test_df  = df[df["centroid_lat"] >  lat_threshold].copy()

    # Normalize features (important for neural networks)
    X_train = train_df[features].values.astype(np.float32)
    X_test  = test_df[features].values.astype(np.float32)

    # Fit scaler on train, apply to both
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm  = (X_test  - mean) / std

    y_train_resid = train_df["residual"].values.astype(np.float32)
    y_test_resid  = test_df["residual"].values.astype(np.float32)
    y_train_lst   = train_df[TARGET].values.astype(np.float32)
    y_test_lst    = test_df[TARGET].values.astype(np.float32)
    xgb_train     = train_df["xgb_pred"].values.astype(np.float32)
    xgb_test      = test_df["xgb_pred"].values.astype(np.float32)

    scaler = {"mean": mean, "std": std}

    return (X_train_norm, X_test_norm,
            y_train_resid, y_test_resid,
            y_train_lst, y_test_lst,
            xgb_train, xgb_test,
            features, scaler)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_pinn(
    X_train, y_train_resid, y_train_lst, xgb_train,
    X_test, y_test_lst, xgb_test,
    features: list[str],
    n_epochs: int = 100,
    batch_size: int = 2048,
    lr: float = 1e-3,
    lambda_phys: float = 0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nTraining PINN on: {device}")
    logger.info(f"  Epochs: {n_epochs}  Batch: {batch_size}  LR: {lr}  λ_phys: {lambda_phys}")

    # Tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train_resid, dtype=torch.float32, device=device)
    y_tr_lst = torch.tensor(y_train_lst, dtype=torch.float32, device=device)
    xgb_tr   = torch.tensor(xgb_train, dtype=torch.float32, device=device)

    dataset = TensorDataset(X_tr, y_tr, y_tr_lst, xgb_tr)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ResidualPINN(n_features=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    loss_fn = PhysicsLoss(lambda_phys=lambda_phys)

    history = {"epoch": [], "L_total": [], "L_data": [], "L_physics": []}

    with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
                  BarColumn(), TextColumn("{task.percentage:>3.0f}%"),
                  TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("Training PINN...", total=n_epochs)

        for epoch in range(n_epochs):
            model.train()
            epoch_losses = {"L_total": [], "L_data": [], "L_physics": []}

            for X_batch, y_resid_batch, y_lst_batch, xgb_batch in loader:
                optimizer.zero_grad()

                # PINN predicts residual correction
                resid_pred = model(X_batch)

                # Final LST prediction = XGBoost + PINN correction
                lst_pred = xgb_batch + resid_pred

                # Compute loss with physics constraints
                loss, components = loss_fn(lst_pred, y_lst_batch, X_batch, features)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                for k, v in components.items():
                    if k in epoch_losses:
                        epoch_losses[k].append(v)

            scheduler.step()

            avg_total = np.mean(epoch_losses["L_total"])
            avg_data  = np.mean(epoch_losses["L_data"])
            avg_phys  = np.mean(epoch_losses["L_physics"])

            history["epoch"].append(epoch + 1)
            history["L_total"].append(avg_total)
            history["L_data"].append(avg_data)
            history["L_physics"].append(avg_phys)

            progress.advance(task)

            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1:3d}: L_total={avg_total:.4f}  L_data={avg_data:.4f}  L_phys={avg_phys:.4f}")

    return model, history, device


# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_pinn(model, X_test, xgb_test, y_test_lst, features, device):
    """Evaluate PINN on test set — compare to XGBoost alone."""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    model.eval()
    X_te = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        resid_pred = model(X_te).cpu().numpy()

    xgb_only = xgb_test
    pinn_pred = xgb_test + resid_pred  # XGBoost + PINN correction

    xgb_r2   = r2_score(y_test_lst, xgb_only)
    pinn_r2  = r2_score(y_test_lst, pinn_pred)
    xgb_mae  = mean_absolute_error(y_test_lst, xgb_only)
    pinn_mae = mean_absolute_error(y_test_lst, pinn_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test_lst, xgb_only))
    pinn_rmse= np.sqrt(mean_squared_error(y_test_lst, pinn_pred))

    logger.info("\nTest set comparison:")
    logger.info(f"  {'Metric':<10} {'XGBoost':>10} {'PINN':>10} {'Change':>10}")
    logger.info(f"  {'R²':<10} {xgb_r2:>10.4f} {pinn_r2:>10.4f} {pinn_r2-xgb_r2:>+10.4f}")
    logger.info(f"  {'MAE (°C)':<10} {xgb_mae:>10.3f} {pinn_mae:>10.3f} {pinn_mae-xgb_mae:>+10.3f}")
    logger.info(f"  {'RMSE (°C)':<10} {xgb_rmse:>10.3f} {pinn_rmse:>10.3f} {pinn_rmse-xgb_rmse:>+10.3f}")

    # Physics violation rate
    X_te_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    pinn_pred_tensor = torch.tensor(pinn_pred, dtype=torch.float32, device=device)
    loss_fn = PhysicsLoss()
    phys_viol = loss_fn.physics_violation_rate(pinn_pred_tensor, X_te_tensor, features)
    logger.info(f"  Physics violation rate: {phys_viol*100:.1f}%")

    return {
        "xgb":  {"r2": round(xgb_r2,4),  "mae": round(xgb_mae,3),  "rmse": round(xgb_rmse,3)},
        "pinn": {"r2": round(pinn_r2,4),  "mae": round(pinn_mae,3), "rmse": round(pinn_rmse,3),
                 "physics_violation_pct": round(phys_viol*100, 1)},
        "improvement": {"r2": round(pinn_r2-xgb_r2,4), "mae": round(pinn_mae-xgb_mae,3)},
        "y_pred": pinn_pred,
        "y_true": y_test_lst,
        "xgb_pred": xgb_only,
    }


def plot_training_history(history: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["epoch"], history["L_total"], label="Total", color="#378ADD")
    ax1.plot(history["epoch"], history["L_data"],  label="Data (MSE)", color="#D85A30", linestyle="--")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training loss")
    ax1.legend(); ax1.spines[["top","right"]].set_visible(False)

    ax2.plot(history["epoch"], history["L_physics"], color="#639922")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Physics loss"); ax2.set_title("Physics constraint loss")
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pinn_training_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Training history plot saved")


def plot_pinn_scatter(metrics: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    y_true = metrics["y_true"]

    for ax, preds, label, color, r2 in [
        (ax1, metrics["xgb_pred"], "XGBoost only", "#378ADD", metrics["xgb"]["r2"]),
        (ax2, metrics["y_pred"],   "XGBoost + PINN", "#D85A30", metrics["pinn"]["r2"]),
    ]:
        ax.scatter(y_true, preds, alpha=0.1, s=2, color=color)
        mn, mx = min(y_true.min(), preds.min()), max(y_true.max(), preds.max())
        ax.plot([mn,mx],[mn,mx],"k--",lw=1,alpha=0.5)
        ax.set_xlabel("Actual LST (°C)"); ax.set_ylabel("Predicted LST (°C)")
        ax.set_title(f"{label}  R²={r2:.4f}")
        ax.spines[["top","right"]].set_visible(False)

    plt.suptitle("PINN vs XGBoost baseline", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pinn_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  PINN scatter plot saved")


# ─── Main ──────────────────────────────────────────────────────────────────────

def run():
    console.print("\n[bold blue]ThermalSense AI — Physics-Informed Neural Network[/bold blue]")
    console.print("[dim]Person B — Script 2 of 5[/dim]\n")

    # 1. Load data
    (X_train, X_test, y_train_resid, y_test_resid,
     y_train_lst, y_test_lst, xgb_train, xgb_test,
     features, scaler) = load_and_prepare()

    # 2. Train PINN
    model, history, device = train_pinn(
        X_train, y_train_resid, y_train_lst, xgb_train,
        X_test, y_test_lst, xgb_test,
        features,
        n_epochs=100,
        batch_size=2048,
        lr=1e-3,
        lambda_phys=0.1,
    )

    # 3. Evaluate
    metrics = evaluate_pinn(model, X_test, xgb_test, y_test_lst, features, device)

    # 4. Save model + scaler
    pinn_path = OUTPUT_DIR / "pinn_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": X_train.shape[1],
        "features": features,
        "scaler_mean": scaler["mean"].tolist(),
        "scaler_std":  scaler["std"].tolist(),
        "metrics": {k:v for k,v in metrics.items() if k not in ("y_pred","y_true","xgb_pred")},
    }, pinn_path)
    logger.success(f"PINN model saved: {pinn_path}")

    # 5. Save metrics
    metrics_path = OUTPUT_DIR / "pinn_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k:v for k,v in metrics.items() if k not in ("y_pred","y_true","xgb_pred")}, f, indent=2)

    # 6. Plots
    plot_training_history(history)
    plot_pinn_scatter(metrics)

    # 7. Summary
    console.print(f"\n[bold green]PINN training complete[/bold green]")
    console.print(f"  XGBoost R²: {metrics['xgb']['r2']:.4f}")
    console.print(f"  PINN R²:    {metrics['pinn']['r2']:.4f}  (+{metrics['improvement']['r2']:+.4f})")
    console.print(f"  Physics violations: {metrics['pinn']['physics_violation_pct']:.1f}%")
    console.print(f"\nNext step: [cyan]python model/shap_analysis.py[/cyan]")

    return model, features, scaler, device


if __name__ == "__main__":
    run()
