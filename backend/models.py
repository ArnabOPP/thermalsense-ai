"""
ThermalSense AI — Pydantic schemas
Request and response models for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ─── Prediction ───────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Input features for a single pixel LST prediction."""
    ndvi:             float = Field(..., ge=-1, le=1,   description="NDVI (-1 to 1)")
    ndwi:             float = Field(..., ge=-1, le=1,   description="NDWI (-1 to 1)")
    ndbi:             float = Field(..., ge=-1, le=1,   description="NDBI (-1 to 1)")
    albedo:           float = Field(..., ge=0,  le=1,   description="Surface albedo (0 to 1)")
    tatm:             float = Field(..., ge=0,  le=50,  description="Atmospheric temp (°C)")
    era5_humidity:    float = Field(..., ge=0,  le=100, description="Relative humidity (%)")
    era5_wind_speed:  float = Field(..., ge=0,  le=20,  description="Wind speed (m/s)")
    doy_sin:          float = Field(..., ge=-1, le=1,   description="Day of year sine encoding")
    isa_pct:          Optional[float] = Field(None, ge=0, le=100, description="Impervious surface area (%)")
    svf:              Optional[float] = Field(None, ge=0, le=1,   description="Sky view factor")
    building_density: Optional[float] = Field(None, ge=0, le=1,   description="Building footprint fraction")
    canyon_ratio:     Optional[float] = Field(None, ge=0,          description="H/W canyon ratio")
    dist_water_m:     Optional[float] = Field(None, ge=0,          description="Distance to water (m)")

    class Config:
        json_schema_extra = {
            "example": {
                "ndvi": 0.32, "ndwi": -0.33, "ndbi": -0.03,
                "albedo": 0.13, "tatm": 31.0, "era5_humidity": 68.0,
                "era5_wind_speed": 1.2, "doy_sin": 0.5,
                "isa_pct": 57.8, "svf": 0.885,
                "building_density": 0.35, "canyon_ratio": 0.21,
                "dist_water_m": 240.0,
            }
        }


class PredictResponse(BaseModel):
    lst_predicted_c:  float = Field(..., description="Predicted LST (°C)")
    model_used:       str   = Field(..., description="Model used for prediction")
    confidence_band:  float = Field(..., description="±°C confidence band")


# ─── Heatmap ──────────────────────────────────────────────────────────────────

class HeatmapResponse(BaseModel):
    """Spatial heatmap of LST or SHAP values across Kolkata."""
    city:       str
    year:       int
    season:     str
    variable:   str
    n_pixels:   int
    pixels:     list[dict]  # [{lat, lon, value}]
    value_min:  float
    value_max:  float
    value_mean: float
    unit:       str


# ─── Scenarios ────────────────────────────────────────────────────────────────

class ScenarioRequest(BaseModel):
    city: str = Field("kolkata", description="kolkata or delhi")
    scenario: str = Field(..., description="One of: urban_greening, cool_roofs, ekw_restoration, green_corridors")
    year:     int = Field(2024, description="Year to simulate")
    season:   str = Field("pre_monsoon", description="pre_monsoon or post_monsoon")

    class Config:
        json_schema_extra = {
            "example": {"scenario": "ekw_restoration", "year": 2024, "season": "pre_monsoon"}
        }


class ScenarioResponse(BaseModel):
    scenario:          str
    scenario_name:     str
    mean_delta_t_c:    float
    max_cooling_c:     float
    pct_pixels_cooled: float
    cost_cr_per_km2:   float
    baseline_lst_c:    float
    pixels:            list[dict]  # [{lat, lon, baseline_lst, modified_lst, delta_t}]


# ─── SHAP ─────────────────────────────────────────────────────────────────────

class ShapResponse(BaseModel):
    year:              int
    season:            str
    n_pixels:          int
    feature_importance: dict   # {feature_name: mean_abs_shap}
    top_heat_driver:   str
    pixels:            list[dict]  # [{lat, lon, shap_ndvi, shap_isa_pct, ...}]


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    city:         str
    n_pixels:     int
    version:      str = "1.0.0"
