# ThermalSense AI — Backend (Person C)

## What this folder contains

```
backend/
  main.py              ← FastAPI app — start here
  ml_loader.py         ← loads models at startup
  models.py            ← request/response schemas
  routers/
    predict.py         ← POST /predict
    heatmap.py         ← GET  /heatmap
    scenarios.py       ← GET/POST /scenarios
    shap.py            ← GET  /shap
requirements-backend.txt
Procfile               ← for Render deployment
```

## Setup (same machine as Person A/B)

```powershell
conda activate thermalsense
pip install fastapi uvicorn[standard] python-multipart
```

## Run locally

```powershell
cd D:\ISROPS1\data-pipeline
uvicorn backend.main:app --reload --port 8000
```

Then open in browser: **http://localhost:8000/docs**

You'll see the full Swagger UI with all endpoints — try them directly in the browser.

## Test all endpoints

```powershell
# Health check
curl http://localhost:8000/

# Predict LST for a pixel
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"ndvi\":0.32,\"ndwi\":-0.33,\"ndbi\":-0.03,\"albedo\":0.13,\"tatm\":31.0,\"era5_humidity\":68.0,\"era5_wind_speed\":1.2,\"doy_sin\":0.5,\"isa_pct\":57.8,\"svf\":0.885,\"building_density\":0.35,\"canyon_ratio\":0.21,\"dist_water_m\":240.0}"

# Get heatmap
curl http://localhost:8000/heatmap?variable=lst&max_pixels=100

# List scenarios
curl http://localhost:8000/scenarios

# Run EKW scenario
curl -X POST http://localhost:8000/scenarios \
  -H "Content-Type: application/json" \
  -d "{\"scenario\":\"ekw_restoration\"}"

# SHAP values
curl http://localhost:8000/shap?max_pixels=100
```

## Deploy to Render

1. Push code to GitHub
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Set:
   - Build command: `pip install -r backend/requirements-backend.txt`
   - Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `PYTHONPATH=.`
6. Deploy

## API endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/` | Health check |
| GET | `/stats` | Model stats |
| POST | `/predict` | Predict LST for one pixel |
| GET | `/heatmap` | Spatial heatmap data |
| GET | `/scenarios` | List scenarios |
| POST | `/scenarios` | Run scenario |
| GET | `/scenarios/summary` | Pre-computed results |
| GET | `/shap` | SHAP explainability |
| GET | `/docs` | Swagger UI |
