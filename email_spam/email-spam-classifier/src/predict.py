"""Predict whether a message is spam using a saved model."""

import argparse
from pathlib import Path

import joblib

from src.preprocess import clean_text


ROOT = Path(__file__).resolve().parents[1]


def predict_message(message: str, model_path: str | Path = ROOT / "models" / "spam_classifier.pkl"):
    model = joblib.load(model_path)
    cleaned = clean_text(message)
    label = model.predict([cleaned])[0]
    probabilities = model.predict_proba([cleaned])[0]
    spam_probability = float(probabilities[list(model.classes_).index("spam")])
    return label, spam_probability


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a message as spam or ham.")
    parser.add_argument("text", nargs="?", help="Message text to classify")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "spam_classifier.pkl")
    args = parser.parse_args()
    message = args.text or input("Message: ")
    label, probability = predict_message(message, args.model)
    print(f"Prediction: {label} (spam probability: {probability:.1%})")


if __name__ == "__main__":
    main()
