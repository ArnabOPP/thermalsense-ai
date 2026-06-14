# ThermalSense AI — Model (Person B)

## What this folder contains

| File | What it does |
|------|-------------|
| `xgb_baseline.py` | XGBoost baseline — first model to run |
| `pinn_loss.py` | Physics loss function (energy balance) |
| `pinn_model.py` | PyTorch PINN — the god-level model |
| `shap_analysis.py` | SHAP explainability per ward |
| `scenarios.py` | 4 cooling intervention simulations |
| `validate.py` | Validate against CPCB ground truth |

## Setup (Person B — same machine as Person A)

The conda environment is already installed. Just activate it:

```powershell
conda activate thermalsense
pip install xgboost shap optuna
```

## Run order

```powershell
# 1. Train XGBoost baseline first (fast, ~2 min)
python model/xgb_baseline.py

# 2. Train PINN (slower, ~10 min)
python model/pinn_model.py

# 3. SHAP analysis
python model/shap_analysis.py

# 4. Run intervention scenarios
python model/scenarios.py

# 5. Validate against CPCB
python model/validate.py
```

## Input

Feature matrix from Person A:
```
outputs/exports/kolkata/feature_matrix_kolkata_ALL.parquet
```

## Output

```
model/outputs/
  xgb_model.pkl          ← trained XGBoost
  pinn_model.pt          ← trained PINN weights
  shap_ward_maps/        ← per-ward SHAP GeoTIFFs
  scenario_results.json  ← ΔT per ward per intervention
  validation_report.json ← CPCB comparison metrics
```
