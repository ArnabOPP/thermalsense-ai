"""
ThermalSense AI — /shap endpoint
SHAP explainability — what's driving heat in each part of Kolkata.
"""

from fastapi import APIRouter, HTTPException, Query
import numpy as np

from backend.models import ShapResponse
from backend.ml_loader import store

router = APIRouter(prefix="/shap", tags=["Explainability"])


@router.get("", response_model=ShapResponse)
def get_shap(
    max_pixels: int = Query(default=3000, le=10000),
):
    """
    Get SHAP explainability values for Kolkata 2024 pre-monsoon.

    Returns per-pixel SHAP contributions for each feature,
    showing WHERE each driver (humidity, NDBI, NDVI etc.) causes heat stress.
    """
    if not store.is_loaded:
        raise HTTPException(503, "Models not loaded")

    if store.shap_df is None:
        raise HTTPException(404, "SHAP values not available — run model/shap_analysis.py")

    df = store.shap_df
    if len(df) > max_pixels:
        df = df.sample(max_pixels, random_state=42)

    # Global feature importance
    shap_cols = [c for c in df.columns if c.startswith("shap_") and c != "shap_total"]
    importance = {
        col.replace("shap_", ""): round(float(df[col].abs().mean()), 4)
        for col in shap_cols
    }
    sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    top_heat    = max(importance, key=importance.get)
    top_cooling = min(importance, key=importance.get)

    pixels = []
    for _, row in df.iterrows():
        pixel = {
            "lat": round(float(row["centroid_lat"]), 5),
            "lon": round(float(row["centroid_lon"]), 5),
            "lst": round(float(row["lst_celsius"]), 2),
        }
        for col in shap_cols:
            feat = col.replace("shap_", "")
            pixel[f"shap_{feat}"] = round(float(row[col]), 4)
        pixels.append(pixel)

    return ShapResponse(
        year=2024,
        season="pre_monsoon",
        n_pixels=len(pixels),
        feature_importance=sorted_importance,
        top_heat_driver=top_heat,
        pixels=pixels,
    )
