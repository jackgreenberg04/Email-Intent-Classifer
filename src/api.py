"""FastAPI application exposing the email intent classifier."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .model import get_default_model


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: str
    confidence: float


app = FastAPI(title="Email Intent Classifier")


@app.on_event("startup")
def load_model() -> None:
    # Preload the model so the first request is fast and predictable.
    get_default_model()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        prediction = get_default_model().predict(request.text)
    except FileNotFoundError as exc:  # pragma: no cover - runtime guard
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return PredictionResponse(**prediction.__dict__)
