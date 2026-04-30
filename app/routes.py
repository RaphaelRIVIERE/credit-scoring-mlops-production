import hashlib
import os
import time
import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from app.schemas import ClientFeatures, PredictionResponse
from app import model as model_state
from app.logger import log_prediction
from src.preprocessing import feature_engineering

router = APIRouter()

THRESHOLD = 0.5
_API_KEY_HASH = hashlib.sha256(os.getenv("API_KEY", "").encode()).hexdigest()
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _verify_api_key(api_key: str = Security(_api_key_header)):
    if not api_key or hashlib.sha256(api_key.encode()).hexdigest() != _API_KEY_HASH:
        raise HTTPException(status_code=403, detail="Non autorisé")


@router.get(
    "/health",
    tags=["Health"],
    summary="Vérification de l'état du service",
)
def health():
    return {"status": "ok"}


@router.post(
    "/predict",
    tags=["Predictions"],
    summary="Prédiction du risque de crédit",
    description=(
        "Soumet les features d'un client au modèle et retourne un score de défaut de paiement "
        "ainsi qu'une décision d'octroi de crédit."
    ),
    response_description="Score de probabilité de défaut et décision d'octroi.",
    responses={
        403: {"description": "Clé API manquante ou invalide."},
        503: {"description": "Modèle non chargé — service indisponible."},
    },
    response_model=PredictionResponse,
)
def predict(request: Request, features: ClientFeatures, _: None = Security(_verify_api_key)):
    if model_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = pd.DataFrame([features.model_dump()])
    data = feature_engineering(data)
    t0 = time.perf_counter()
    score = float(model_state.model.predict_proba(data)[0][1])
    inference_time_ms = (time.perf_counter() - t0) * 1000
    decision = "rejected" if score >= THRESHOLD else "approved"
    request.state.prediction_id = log_prediction(features, score, decision, inference_time_ms)
    return PredictionResponse(score=score, decision=decision)
