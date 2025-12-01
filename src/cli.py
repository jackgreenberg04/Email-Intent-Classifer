"""Command-line interface for the email intent classifier."""
import argparse

from .model import get_default_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify an email's intent")
    parser.add_argument("text", help="Email text to classify")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = get_default_model()
    prediction = model.predict(args.text)
    print(f"Prediction: {prediction.label} (confidence: {prediction.confidence:.2f})")


if __name__ == "__main__":
    main()
