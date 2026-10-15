
import re

import pandas as pd

from load_data import (
    PROJECT_ROOT,
    VALID_SENTIMENTS,
    data_quality_report,
    load_reviews,
)


CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "cleaned_customer_reviews.csv"


def clean_review_text(review: object) -> str:
    """Normalize text while leaving the original review column unchanged."""
    if pd.isna(review):
        return ""

    text = str(review).lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return " ".join(text.split())


def prepare_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Clean text and exclude rows without usable reviews or sentiment labels."""
    prepared = reviews.copy()
    labels = prepared["sentiment"].astype("string").str.strip().str.lower()
    review_text = prepared["review"].astype("string").str.strip()
    usable = (
        review_text.notna()
        & review_text.ne("")
        & labels.isin(VALID_SENTIMENTS)
    )

    prepared = prepared.loc[usable].copy()
    prepared["sentiment"] = labels.loc[usable]
    prepared = prepared.drop_duplicates(subset=["review", "sentiment"])
    prepared["cleaned_review"] = prepared["review"].map(clean_review_text)
    return prepared


def main() -> None:
    reviews = load_reviews()
    print("Input data quality:")
    for check, count in data_quality_report(reviews).items():
        print(f"  {check.replace('_', ' ').title()}: {count}")

    prepared = prepare_reviews(reviews)
    print(f"\nUsable reviews: {len(prepared)} of {len(reviews)}")
    print("Original and cleaned reviews:")
    print(prepared[["review", "cleaned_review"]].head(10))

    prepared.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"\nCleaned dataset saved to {CLEANED_DATA_PATH}")


if __name__ == "__main__":
    main()
