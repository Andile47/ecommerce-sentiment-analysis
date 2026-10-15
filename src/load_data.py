
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "customer_reviews.csv"
REQUIRED_COLUMNS = {"review", "sentiment"}
VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def load_reviews(csv_path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load reviews and check that the required columns are present."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Review dataset not found: {csv_path}")

    reviews = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS.difference(reviews.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {missing}")

    return reviews


def data_quality_report(reviews: pd.DataFrame) -> dict[str, int]:
    """Count missing data, duplicate review-label pairs, and invalid labels."""
    review_text = reviews["review"].astype("string").str.strip()
    labels = reviews["sentiment"].astype("string").str.strip().str.lower()
    missing_reviews = review_text.isna() | review_text.eq("")
    missing_labels = labels.isna() | labels.eq("")
    invalid_labels = ~missing_labels & ~labels.isin(VALID_SENTIMENTS)

    return {
        "rows": len(reviews),
        "missing_reviews": int(missing_reviews.sum()),
        "missing_labels": int(missing_labels.sum()),
        "duplicate_review_label_pairs": int(
            reviews.duplicated(subset=["review", "sentiment"]).sum()
        ),
        "invalid_labels": int(invalid_labels.sum()),
    }


def main() -> None:
    reviews = load_reviews()
    print(f"Loaded {len(reviews)} reviews from {DEFAULT_DATA_PATH}")
    print("Data quality:")
    for check, count in data_quality_report(reviews).items():
        print(f"  {check.replace('_', ' ').title()}: {count}")
    print("\nFirst five reviews:")
    print(reviews.head())


if __name__ == "__main__":
    main()


