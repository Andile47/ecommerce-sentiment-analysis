import sys
import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import predict_sentiment


class PredictSentimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temporary_directory.name) / "test_model.joblib"
        model = Pipeline(
            [
                ("tfidf", TfidfVectorizer()),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )
        reviews = [
            "excellent great product",
            "excellent product quality",
            "broken terrible product",
            "awful terrible quality",
            "average ordinary product",
            "ordinary average quality",
        ]
        labels = [
            "positive",
            "positive",
            "negative",
            "negative",
            "neutral",
            "neutral",
        ]
        model.fit(reviews, labels)
        joblib.dump(model, self.model_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_reloads_saved_pipeline_and_predicts(self) -> None:
        prediction = predict_sentiment("excellent great product", self.model_path)
        self.assertEqual(prediction, "positive")

    def test_rejects_empty_review(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty review"):
            predict_sentiment("  ", self.model_path)

    def test_missing_model_has_training_instruction(self) -> None:
        missing_path = Path(self.temporary_directory.name) / "missing.joblib"
        with self.assertRaisesRegex(FileNotFoundError, "python src/train_model.py"):
            predict_sentiment("a review", missing_path)


if __name__ == "__main__":
    unittest.main()