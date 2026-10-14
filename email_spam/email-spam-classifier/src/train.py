"""Train and evaluate the SMS spam classifier."""

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.preprocess import clean_text, load_dataset
from src.visualize import plot_top_spam_words


ROOT = Path(__file__).resolve().parents[1]


def train(data_path: Path, model_path: Path, output_dir: Path) -> None:
    dataset = load_dataset(data_path)
    texts = dataset["message"].map(clean_text)
    labels = dataset["label"]
    label_counts = labels.value_counts()
    stratify = labels if label_counts.min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
            ("classifier", MultinomialNB()),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    figure, axis = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, labels=["ham", "spam"], display_labels=["Ham", "Spam"],
        cmap="Blues", colorbar=False, ax=axis,
    )
    axis.set_title("Spam classifier confusion matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    plot_top_spam_words(model, output_dir / "top_spam_words.png")

    print(classification_report(y_test, predictions, labels=["ham", "spam"], zero_division=0))
    print(f"Saved model to {model_path}")
    print(f"Saved plots to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SMS spam classifier.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "scam.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "spam_classifier.pkl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    train(args.data, args.model, args.output_dir)


if __name__ == "__main__":
    main()
