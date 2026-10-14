"""Generate model interpretation plots."""

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def plot_top_spam_words(model, output_path: str | Path, limit: int = 20) -> None:
    """Plot terms whose learned Naive Bayes weights favor spam most strongly."""
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    classes = list(classifier.classes_)
    if "spam" not in classes or "ham" not in classes:
        raise ValueError("The model must be trained with 'ham' and 'spam' labels.")

    spam_index = classes.index("spam")
    ham_index = classes.index("ham")
    terms = vectorizer.get_feature_names_out()
    spam_scores = classifier.feature_log_prob_[spam_index] - classifier.feature_log_prob_[ham_index]
    selected = spam_scores.argsort()[-limit:]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(terms[selected], spam_scores[selected], color="#d65a4a")
    axis.set(title="Terms most associated with spam", xlabel="Spam vs. ham log-probability")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot top spam-associated terms.")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "spam_classifier.pkl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "top_spam_words.png")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    plot_top_spam_words(joblib.load(args.model), args.output, args.limit)
    print(f"Saved keyword plot to {args.output}")


if __name__ == "__main__":
    main()
