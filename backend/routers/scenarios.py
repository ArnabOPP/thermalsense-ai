"""
ThermalSense AI — /scenarios endpoint
Run cooling intervention simulations.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
import numpy as np

from backend.models import ScenarioRequest, ScenarioResponse
from backend.ml_loader import store, run_scenario_live

router = APIRouter(prefix="/scenarios", tags=["Scenarios"])

VALID_SCENARIOS = ["urban_greening", "cool_roofs", "ekw_restoration", "green_corridors"]

SCENARIO_NAMES = {
    "urban_greening":  "Urban greening (+15% tree cover)",
    "cool_roofs":      "Cool roofs (albedo 0.12→0.65)",
    "ekw_restoration": "EKW wetland restoration",
    "green_corridors": "Green corridors (AJC Bose + EM Bypass)",
}

SCENARIO_COSTS = {
    "urban_greening":  2.5,
    "cool_roofs":      1.8,
    "ekw_restoration": 0.8,
    "green_corridors": 1.2,
}


@router.get("")
def list_scenarios():
    """List all available intervention scenarios."""
    return {
        "scenarios": [
            {
                "id": k,
                "name": SCENARIO_NAMES[k],
                "cost_cr_per_km2": SCENARIO_COSTS[k],
            }
            for k in VALID_SCENARIOS
        ]
    }


@router.post("", response_model=ScenarioResponse)
def run_scenario(req: ScenarioRequest):
    """
    Simulate a cooling intervention scenario.

    Modifies input features to reflect the intervention,
    runs the model, and returns per-pixel ΔT values.
    """
    if not store.is_loaded:
        raise HTTPException(503, "Models not loaded")

    if req.scenario not in VALID_SCENARIOS:
        raise HTTPException(400, f"Invalid scenario. Choose from: {VALID_SCENARIOS}")

    try:
        df = run_scenario_live(req.scenario, req.year, req.season, req.city)
    except Exception as e:
        logger.error(f"Scenario error: {e}")
        raise HTTPException(500, str(e))

    mean_delta = float(df["delta_t"].mean())
    max_cool   = float(df["delta_t"].min())
    pct_cooled = float((df["delta_t"] < -0.5).mean() * 100)

    pixels = [
        {
            "lat":          round(float(r["centroid_lat"]), 5),
            "lon":          round(float(r["centroid_lon"]), 5),
            "baseline_lst": round(float(r["lst_predicted"]), 2),
            "modified_lst": round(float(r["lst_modified"]), 2),
            "delta_t":      round(float(r["delta_t"]), 3),
        }
        for _, r in df.iterrows()
    ]

    return ScenarioResponse(
        scenario=req.scenario,
        scenario_name=SCENARIO_NAMES[req.scenario],
        mean_delta_t_c=round(mean_delta, 3),
        max_cooling_c=round(max_cool, 3),
        pct_pixels_cooled=round(pct_cooled, 1),
        cost_cr_per_km2=SCENARIO_COSTS[req.scenario],
        baseline_lst_c=round(float(df["lst_predicted"].mean()), 2),
        pixels=pixels,
    )


@router.get("/summary")
def scenario_summary():
    """Return pre-computed scenario results from last full run."""
    if store.scenario_results is None:
        raise HTTPException(404, "Scenario results not found — run model/scenarios.py")
    return store.scenario_results
