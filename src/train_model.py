
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from load_data import PROJECT_ROOT, VALID_SENTIMENTS, load_reviews
from prepare_text import prepare_reviews


MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_pipeline.joblib"
LABEL_ORDER = ["negative", "neutral", "positive"]


def build_model_pipeline() -> Pipeline:
    """Keep TF-IDF fitting and classification together to prevent leakage."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def evaluate_model(texts: pd.Series, labels: pd.Series) -> None:
    """Print stratified cross-validation metrics when each class permits it."""
    class_counts = labels.value_counts()
    if set(class_counts.index) != VALID_SENTIMENTS:
        missing = sorted(VALID_SENTIMENTS.difference(class_counts.index))
        print(f"Evaluation skipped because these sentiment classes are missing: {missing}")
        return

    smallest_class = int(class_counts.min())
    if smallest_class < 2:
        print(
            "Evaluation skipped: each sentiment needs at least two examples "
            "for stratified cross-validation."
        )
        return

    fold_count = min(5, smallest_class)
    cross_validation = StratifiedKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=42,
    )
    predictions = cross_val_predict(
        build_model_pipeline(),
        texts,
        labels,
        cv=cross_validation,
    )

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )
    print(f"\nExploratory stratified {fold_count}-fold cross-validation:")
    print(f"Accuracy:  {accuracy_score(labels, predictions):.3f}")
    print(f"Precision: {precision:.3f} (macro average)")
    print(f"Recall:    {recall:.3f} (macro average)")
    print(f"F1-score:  {f1_score:.3f} (macro average)")
    print("\nPer-class report:")
    print(
        classification_report(
            labels,
            predictions,
            labels=LABEL_ORDER,
            zero_division=0,
        )
    )
    matrix = pd.DataFrame(
        confusion_matrix(labels, predictions, labels=LABEL_ORDER),
        index=LABEL_ORDER,
        columns=LABEL_ORDER,
    )
    print("Confusion matrix (rows: actual, columns: predicted):")
    print(matrix)


def main() -> None:
    prepared = prepare_reviews(load_reviews())
    if prepared.empty:
        raise ValueError("No usable reviews remain after data validation.")

    texts = prepared["cleaned_review"]
    labels = prepared["sentiment"]
    if set(labels.unique()) != VALID_SENTIMENTS:
        raise ValueError(
            "Training requires at least one review for each label: "
            "positive, negative, and neutral."
        )

    print(f"Training examples: {len(prepared)}")
    print("Training data distribution:")
    print(labels.value_counts().reindex(LABEL_ORDER, fill_value=0))

    if len(prepared) < 100 or int(labels.value_counts().min()) < 20:
        print(
            "\nWarning: this dataset is too small to estimate real-world "
            "performance reliably. Treat the following metrics as educational "
            "examples and collect more representative labeled reviews."
        )
    evaluate_model(texts, labels)

    final_pipeline = build_model_pipeline()
    final_pipeline.fit(texts, labels)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"\nFinal model trained on all {len(prepared)} usable reviews.")
    print(f"Saved model pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
