"""
ThermalSense AI — /heatmap endpoint
Returns spatial LST or SHAP data for map rendering.
"""

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
import numpy as np

from backend.models import HeatmapResponse
from backend.ml_loader import store

router = APIRouter(prefix="/heatmap", tags=["Heatmap"])

VALID_VARIABLES = ["lst", "ndvi", "ndbi", "isa_pct", "dist_water_m", "shap_ndvi", "shap_isa_pct", "shap_era5_humidity"]


@router.get("", response_model=HeatmapResponse)
def get_heatmap(
    variable:   str = Query(default="lst"),
    city:       str = Query(default="kolkata"),
    year:       int = Query(default=2024),
    season:     str = Query(default="pre_monsoon"),
    max_pixels: int = Query(default=5000, le=10000),
):
    """
    Get spatial heatmap data for Kolkata.

    Returns a list of {lat, lon, value} points for rendering on a map.
    Default variable is LST (land surface temperature in °C).
    """
    if not store.is_loaded:
        raise HTTPException(503, "Models not loaded")

    if variable not in VALID_VARIABLES:
        raise HTTPException(400, f"Invalid variable. Choose from: {VALID_VARIABLES}")

    # Use cached 2024 pre-monsoon data
    from backend.ml_loader import get_city_df
    df = get_city_df(city) if variable != "shap_..." else store.shap_df

    # Determine which column to use
    if variable == "lst":
        col = "lst_celsius"
        unit = "°C"
    elif variable.startswith("shap_"):
        if store.shap_df is None:
            raise HTTPException(404, "SHAP values not loaded — run model/shap_analysis.py")
        shap_col = variable  # e.g. "shap_ndvi"
        if shap_col not in store.shap_df.columns:
            raise HTTPException(404, f"SHAP column not found: {shap_col}")
        df = store.shap_df
        col = shap_col
        unit = "°C (SHAP)"
    elif variable in df.columns:
        col = variable
        unit = "%" if "pct" in variable else ("m" if "_m" in variable else "")
    else:
        raise HTTPException(404, f"Variable '{variable}' not in feature matrix")

    # Downsample if needed
    if len(df) > max_pixels:
        df = df.sample(max_pixels, random_state=42)

    values = df[col].values
    valid  = values[~np.isnan(values)]

    pixels = [
        {"lat": round(float(r["centroid_lat"]), 5),
         "lon": round(float(r["centroid_lon"]), 5),
         "value": round(float(r[col]), 3) if not np.isnan(r[col]) else None}
        for _, r in df.iterrows()
    ]

    return HeatmapResponse(
        city="kolkata",
        year=year,
        season=season,
        variable=variable,
        n_pixels=len(pixels),
        pixels=pixels,
        value_min=round(float(valid.min()), 2) if len(valid) > 0 else 0,
        value_max=round(float(valid.max()), 2) if len(valid) > 0 else 0,
        value_mean=round(float(valid.mean()), 2) if len(valid) > 0 else 0,
        unit=unit,
    )
