"""Training script for the email intent classifier.

This module trains a simple text classifier using a toy dataset and
saves the resulting model to ``data/processed/model.joblib``.
"""
from pathlib import Path
from typing import Iterable, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "data" / "processed" / "model.joblib"

TOY_DATA: Tuple[Tuple[str, str], ...] = (
    ("Your payment is due tomorrow. Please process the invoice.", "billing"),
    ("Invoice #42 is attached. Kindly complete the payment.", "billing"),
    ("Let's reschedule our meeting to next Monday.", "meeting"),
    ("Can we move the project sync to 3pm?", "meeting"),
    ("My account is locked. Can you help me regain access?", "support"),
    ("I'm having trouble logging in to the dashboard.", "support"),
    ("Package shipment is delayed, please advise.", "logistics"),
    ("Where is the tracking number for the latest order?", "logistics"),
    ("Please review the attached contract and sign.", "operations"),
    ("We need approval on the updated statement of work.", "operations"),
)


def build_pipeline() -> Pipeline:
    """Create a text classification pipeline.

    Returns
    -------
    Pipeline
        A scikit-learn pipeline combining TF-IDF vectorization and
        a logistic regression classifier.
    """

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    classifier = LogisticRegression(max_iter=1000, multi_class="auto")
    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])


def load_training_data() -> Tuple[Iterable[str], Iterable[str]]:
    """Return the toy dataset texts and labels."""

    texts, labels = zip(*TOY_DATA)
    return texts, labels


def train_and_save_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """Train the classifier and persist it to disk.

    Parameters
    ----------
    model_path:
        Destination for the serialized model.

    Returns
    -------
    Pipeline
        The trained scikit-learn pipeline.
    """

    texts, labels = load_training_data()
    pipeline = build_pipeline()
    pipeline.fit(texts, labels)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    return pipeline


if __name__ == "__main__":
    trained = train_and_save_model()
    print(f"Model trained and saved to {MODEL_PATH}")
