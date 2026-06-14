"""
ThermalSense AI — FastAPI Backend
Person C

Endpoints:
  GET  /              → health check
  POST /predict       → predict LST for a pixel
  GET  /heatmap       → spatial LST/SHAP heatmap data
  POST /scenarios     → run cooling intervention scenario
  GET  /scenarios     → list available scenarios
  GET  /scenarios/summary → pre-computed scenario results
  GET  /shap          → SHAP explainability data
  GET  /docs          → Swagger UI (auto-generated)

Run locally:
  uvicorn backend.main:app --reload --port 8000

Then open: http://localhost:8000/docs
"""
# Download models from HuggingFace if not present locally
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_XGB_PATH = _ROOT / "model" / "outputs" / "xgb_model.pkl"

if not _XGB_PATH.exists():
    print("Models not found locally — downloading from HuggingFace...", flush=True)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="arnabinthegame05/thermalsense-ai-models",
            repo_type="model",
            local_dir=str(_ROOT),
            local_dir_use_symlinks=False,
        )
        print("Download complete!", flush=True)
    except Exception as e:
        print(f"HuggingFace download failed: {e}", flush=True)

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.ml_loader import load_all, store
from backend.models import HealthResponse
from backend.routers import predict, heatmap, scenarios, shap


# ─── Lifespan: load models at startup ────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models when server starts, clean up on shutdown."""
    logger.info("ThermalSense AI starting up...")
    try:
        load_all()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.warning("API will start but predictions will fail until models are loaded")
    yield
    logger.info("ThermalSense AI shutting down")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ThermalSense AI",
    description="""
## Urban Heat Island Analysis API — Kolkata

Built for **ISRO Bharatiya Antariksh Hackathon 2026 (BAH 2026)**, Problem Statement 1.

Uses Landsat 8 LST + Sentinel-2 + ERA5 + OSM morphology data to:
- Predict land surface temperature at 100m resolution
- Identify UHI hotspots across Kolkata
- Simulate cooling interventions (greening, cool roofs, wetland restoration)
- Explain predictions using SHAP values

**Model performance:** R² = 0.86, MAE = 1.28°C (spatially held-out test set)
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS — allow React frontend to call this API ────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────────────────────────

app.include_router(predict.router)
app.include_router(heatmap.router)
app.include_router(scenarios.router)
app.include_router(shap.router)


# ─── Health check ─────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check — confirms API is running and models are loaded."""
    n_pixels = sum(len(df) for df in store.city_cache.values())
    return HealthResponse(
        status="ok" if store.is_loaded else "loading",
        model_loaded=store.is_loaded,
        city="kolkata",
        n_pixels=n_pixels,
        version="1.0.0",
    )


@app.get("/stats", tags=["Health"])
def stats():
    """Return model and data statistics."""
    return {
        "model": {
            "xgb_loaded": store.xgb_model is not None,
            "pinn_loaded": store.pinn_model is not None,
            "features": store.features,
            "n_features": len(store.features),
        },
        "data": {
            "n_pixels_cached": sum(len(df) for df in store.city_cache.values()),
            "shap_available": store.shap_df is not None,
            "scenarios_available": store.scenario_results is not None,
        },
        "performance": {
            "r2_test": 0.8628,
            "mae_test_c": 1.276,
            "rmse_test_c": 1.829,
            "cpcb_mae_c": 1.39,
            "physics_violation_pct": 8.7,
        }
    }

@app.get("/cities", tags=["Health"])
def list_cities():
    from backend.ml_loader import SUPPORTED_CITIES, store
    return {
        "cities": [
            {
                "id": k,
                "name": v["name"],
                "center": v["center"],
                "zoom": v["zoom"],
                "n_pixels": len(store.city_cache.get(k, [])),
            }
            for k, v in SUPPORTED_CITIES.items()
        ]
    }