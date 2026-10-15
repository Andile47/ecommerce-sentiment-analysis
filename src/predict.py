import argparse
from pathlib import Path

import joblib

from load_data import PROJECT_ROOT


MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"
EXAMPLE_REVIEWS = [
    "The product is excellent and I love using it.",
    "The item broke after one day and the quality is terrible.",
    "The product is average and works as described.",
]


def predict_sentiment(review: str, model_path: Path = MODEL_PATH) -> str:
    """Load the saved pipeline and predict one non-empty review."""
    if not isinstance(review, str) or not review.strip():
        raise ValueError("Please provide a non-empty review.")

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. "
            "Train it first with: python src/train_model.py"
        )

    model = joblib.load(model_path)
    return str(model.predict([review.strip()])[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a product review sentiment.")
    parser.add_argument("review", nargs="?", help="Review text to classify")
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Predict sentiment for three example reviews",
    )
    arguments = parser.parse_args()

    if arguments.examples:
        reviews = EXAMPLE_REVIEWS
    else:
        review = arguments.review
        if review is None:
            review = input("Enter a customer review: ")
        if not review.strip():
            print("Please enter a non-empty review.")
            return
        reviews = [review]

    try:
        for review in reviews:
            sentiment = predict_sentiment(review)
            print(f"Review: {review}")
            print(f"Predicted sentiment: {sentiment}")
            print("-" * 40)
    except FileNotFoundError as error:
        print(error)


if __name__ == "__main__":
    main()