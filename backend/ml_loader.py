"""
ThermalSense AI — ML Model Loader (Multi-city)
Loads models and caches data for all supported cities.
"""

import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent

XGB_PATH       = ROOT / "model" / "outputs" / "xgb_model.pkl"
CITY_XGB_PATH  = ROOT / "model" / "outputs" / "xgb_city_model.pkl"
PINN_PATH      = ROOT / "model" / "outputs" / "pinn_model.pt"
SHAP_PATH      = ROOT / "model" / "outputs" / "shap_values.parquet"
SCENARIOS_PATH = ROOT / "model" / "outputs" / "scenario_results.json"

SUPPORTED_CITIES = {
    "kolkata": {
        "name": "Kolkata",
        "center": [22.55, 88.37],
        "zoom": 12,
        "feature_matrix": ROOT / "outputs" / "exports" / "kolkata" / "feature_matrix_kolkata_clipped.parquet",
    },
    "delhi": {
        "name": "Delhi",
        "center": [28.65, 77.20],
        "zoom": 11,
        "feature_matrix": ROOT / "outputs" / "exports" / "delhi" / "feature_matrix_delhi_ALL.parquet",
    },
}

CORE_FEATURES = ["ndvi", "ndwi", "ndbi", "albedo", "tatm", "era5_humidity", "era5_wind_speed", "doy_sin"]


class ModelStore:
    xgb_model    = None
    city_model   = None
    features: list = []
    city_features: list = CORE_FEATURES
    scaler_mean  = None
    scaler_std   = None
    pinn_model   = None
    city_cache: dict = {}
    shap_df      = None
    scenario_results = None
    is_loaded: bool = False


store = ModelStore()


def load_xgboost():
    if not XGB_PATH.exists():
        raise FileNotFoundError(f"XGBoost model not found: {XGB_PATH}")
    with open(XGB_PATH, "rb") as f:
        data = pickle.load(f)
    store.xgb_model = data["model"]
    store.features  = data["features"]
    logger.info(f"XGBoost loaded — {len(store.features)} features")

    if CITY_XGB_PATH.exists():
        with open(CITY_XGB_PATH, "rb") as f:
            city_data = pickle.load(f)
        store.city_model    = city_data["model"]
        store.city_features = city_data["features"]
        logger.info(f"City model loaded — {len(store.city_features)} features (multi-city)")


def load_pinn():
    try:
        import torch
        sys.path.insert(0, str(ROOT / "model"))
        from pinn_model import ResidualPINN
        if not PINN_PATH.exists():
            return
        checkpoint = torch.load(PINN_PATH, map_location="cpu", weights_only=False)
        model = ResidualPINN(n_features=len(checkpoint["features"]))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        store.pinn_model  = model
        store.scaler_mean = np.array(checkpoint["scaler_mean"])
        store.scaler_std  = np.array(checkpoint["scaler_std"])
        logger.info("PINN model loaded")
    except Exception as e:
        logger.warning(f"PINN load failed ({e}) — XGBoost only")


def load_city(city: str):
    if city in store.city_cache:
        return
    city_cfg = SUPPORTED_CITIES.get(city)
    if not city_cfg:
        raise ValueError(f"City not supported: {city}")

    path = city_cfg["feature_matrix"]
    if not path.exists():
        raise FileNotFoundError(f"Feature matrix not found for {city}: {path}")

    df = pd.read_parquet(path)
    df = df[df["lst_celsius"] > 10].copy()

    use_features = store.city_features if store.city_model else store.features
    available = [f for f in use_features if f in df.columns and df[f].notna().mean() > 0.9]
    df = df[available + ["lst_celsius", "centroid_lat", "centroid_lon", "year", "season"]].dropna()

    model = store.city_model if store.city_model else store.xgb_model
    df["lst_predicted"] = model.predict(df[available].values.astype(np.float32))

    store.city_cache[city] = df
    logger.info(f"City '{city}' loaded: {len(df):,} pixels")


def load_all():
    logger.info("Loading ThermalSense AI models...")
    load_xgboost()
    load_pinn()
    for city in SUPPORTED_CITIES:
        try:
            load_city(city)
        except Exception as e:
            logger.warning(f"Could not load {city}: {e}")
    if SHAP_PATH.exists():
        store.shap_df = pd.read_parquet(SHAP_PATH)
        logger.info(f"SHAP values loaded: {len(store.shap_df):,} pixels")
    if SCENARIOS_PATH.exists():
        import json
        with open(SCENARIOS_PATH) as f:
            store.scenario_results = json.load(f)
    store.is_loaded = True
    logger.success("All models loaded — API ready")


def get_city_df(city: str) -> pd.DataFrame:
    if city not in store.city_cache:
        load_city(city)
    return store.city_cache[city]


def predict_lst(feature_dict: dict, city: str = "kolkata") -> tuple:
    model = store.city_model if store.city_model else store.xgb_model
    features = store.city_features if store.city_model else store.features
    x = np.array([[feature_dict.get(f, 0.0) for f in features]], dtype=np.float32)
    pred = float(model.predict(x)[0])

    if store.pinn_model is not None and city == "kolkata":
        try:
            import torch
            x_norm = (x - store.scaler_mean) / store.scaler_std
            with torch.no_grad():
                residual = float(store.pinn_model(torch.tensor(x_norm, dtype=torch.float32)).item())
            return pred + residual, "XGBoost+PINN"
        except Exception:
            pass

    return pred, "XGBoost"


def run_scenario_live(scenario: str, city: str = "kolkata") -> pd.DataFrame:
    df = get_city_df(city).copy()
    model = store.city_model if store.city_model else store.xgb_model
    features = store.city_features if store.city_model else store.features
    available = [f for f in features if f in df.columns]

    X_mod = df[available].copy().astype(np.float32)

    if scenario == "urban_greening":
        mask = X_mod["ndvi"] < 0.3
        X_mod.loc[mask, "ndvi"]   = (X_mod.loc[mask, "ndvi"] + 0.25).clip(-1, 1)
        X_mod.loc[mask, "albedo"] = (X_mod.loc[mask, "albedo"] - 0.02).clip(0, 0.9)
    elif scenario == "cool_roofs":
        mask = X_mod["ndbi"] > 0.0
        X_mod.loc[mask, "albedo"] = (X_mod.loc[mask, "albedo"] + 0.40).clip(upper=0.65)
    elif scenario == "ekw_restoration":
        eastern = df["centroid_lon"] > (df["centroid_lon"].mean() + 0.03)
        mask = eastern & (X_mod["ndwi"] < 0.0)
        X_mod.loc[mask, "ndwi"]   = (X_mod.loc[mask, "ndwi"] + 0.15).clip(-1, 1)
        X_mod.loc[mask, "ndvi"]   = (X_mod.loc[mask, "ndvi"] + 0.10).clip(-1, 1)
        X_mod.loc[mask, "albedo"] = (X_mod.loc[mask, "albedo"] - 0.03).clip(0, 0.9)
    elif scenario == "green_corridors":
        lat_mid = df["centroid_lat"].mean()
        lon_mid = df["centroid_lon"].mean()
        mask = (df["centroid_lat"].between(lat_mid - 0.05, lat_mid + 0.05) |
                df["centroid_lon"].between(lon_mid - 0.02, lon_mid + 0.02))
        X_mod.loc[mask, "ndvi"]   = (X_mod.loc[mask, "ndvi"] + 0.20).clip(-1, 1)
        X_mod.loc[mask, "albedo"] = (X_mod.loc[mask, "albedo"] - 0.02).clip(0, 0.9)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    df["lst_modified"] = model.predict(X_mod[available].values.astype(np.float32))
    df["delta_t"] = df["lst_modified"] - df["lst_predicted"]
    return df