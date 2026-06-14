"""
ThermalSense AI — /predict endpoint
POST a feature vector, get predicted LST back.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.models import PredictRequest, PredictResponse
from backend.ml_loader import store, predict_lst

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict Land Surface Temperature for a single pixel.

    Send feature values, get back predicted LST in °C.
    Uses XGBoost + PINN ensemble if PINN is loaded, XGBoost only otherwise.
    """
    if not store.is_loaded:
        raise HTTPException(503, "Models not loaded yet — try again in a moment")

    feature_dict = req.model_dump(exclude_none=False)

    try:
        lst, model_used = predict_lst(feature_dict)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

    return PredictResponse(
        lst_predicted_c=round(float(lst), 2),
        model_used=model_used,
        confidence_band=1.83,  # RMSE from validation
    )
