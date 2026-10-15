
import pandas as pd
# Create sample e-commerce customer reviews
data = {
    "review": [
        "The product is amazing and works perfectly",
        "I love the quality and fast delivery",
        "The product is okay, nothing special",
        "Delivery was slow and disappointing",
        "The item arrived damaged and I am unhappy",
        "Excellent product, I would buy it again",
        "The quality is poor for the price",
        "The product meets my expectations",
        "Very happy with my purchase",
        "Terrible experience, I want a refund"
    ],
    "sentiment": [
        "positive",
        "positive",
        "neutral",
        "negative",
        "negative",
        "positive",
        "negative",
        "neutral",
        "positive",
        "negative"
    ]
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)


# Display the first five reviews
print("\nFirst five reviews:")
print(df.head())

# Show the number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Show column names and data types
print("\nDataset information:")
df.info()

# Count reviews in each sentiment category
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())


# Count reviews in each sentiment category
sentiment_counts = df["sentiment"].value_counts()

# Display the counts
print("\nSentiment Distribution:")
print(sentiment_counts)

# Calculate the percentage of each sentiment
sentiment_percentages = df["sentiment"].value_counts(normalize=True) * 100

print("\nSentiment Percentages:")
print(sentiment_percentages.round(1))



