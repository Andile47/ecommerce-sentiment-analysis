
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample customer reviews
reviews = [
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
]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Convert the reviews into numerical features
tfidf_matrix = vectorizer.fit_transform(reviews)

# Display the results
print("Number of reviews:", len(reviews))
print("Number of unique words:", len(vectorizer.get_feature_names_out()))
print("TF-IDF matrix shape:", tfidf_matrix.shape)

# Display some of the words the model can use
print("\nFirst 10 vocabulary terms:")
print(vectorizer.get_feature_names_out()[:10])
