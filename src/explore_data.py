
import pandas as pd

from load_data import (
    VALID_SENTIMENTS,
    data_quality_report,
    load_reviews,
)


def main() -> None:
    reviews = load_reviews()
    print("Data quality:")
    for check, count in data_quality_report(reviews).items():
        print(f"  {check.replace('_', ' ').title()}: {count}")

    labels = reviews["sentiment"].astype("string").str.strip().str.lower()
    valid_labels = labels[labels.isin(VALID_SENTIMENTS)]
    counts = valid_labels.value_counts().reindex(sorted(VALID_SENTIMENTS), fill_value=0)

    print(f"\nTotal reviews: {len(reviews)}")
    print("\nSentiment distribution:")
    if valid_labels.empty:
        print("No valid sentiment labels are available to summarize.")
    else:
        distribution = pd.DataFrame(
            {
                "count": counts,
                "percentage": (counts / len(valid_labels) * 100).round(1),
            }
        )
        print(distribution)

    print("\nExample review for each sentiment:")
    for sentiment in sorted(VALID_SENTIMENTS):
        examples = reviews.loc[labels.eq(sentiment), "review"]
        if not examples.empty:
            print(f"{sentiment}: {examples.iloc[0]}")


if __name__ == "__main__":
    main()

