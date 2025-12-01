"""Model loading and inference utilities for the email intent classifier."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
from sklearn.pipeline import Pipeline

from .train import MODEL_PATH


@dataclass
class PredictionResult:
    label: str
    confidence: float


class IntentModel:
    """A lightweight wrapper around the trained scikit-learn pipeline."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self._pipeline: Optional[Pipeline] = None
        self.load()

    def load(self) -> None:
        """Load the model from disk.

        Raises
        ------
        FileNotFoundError
            If the serialized model is missing.
        """

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Run `python -m src.train` first."
            )
        self._pipeline = joblib.load(self.model_path)

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self.load()
        return self._pipeline  # type: ignore[return-value]

    def predict(self, text: str) -> PredictionResult:
        """Predict the intent label and confidence for a given email."""

        probabilities = self.pipeline.predict_proba([text])[0]
        labels = self.pipeline.classes_
        best_index = probabilities.argmax()
        return PredictionResult(label=str(labels[best_index]), confidence=float(probabilities[best_index]))


_default_model: Optional[IntentModel] = None


def get_default_model() -> IntentModel:
    """Return a singleton instance of :class:`IntentModel`."""

    global _default_model
    if _default_model is None:
        _default_model = IntentModel()
    return _default_model


def predict_intent(text: str) -> PredictionResult:
    """Convenience function to make a prediction using the default model."""

    model = get_default_model()
    return model.predict(text)
