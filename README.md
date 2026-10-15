# E-commerce Sentiment Analysis

A beginner-friendly Python project that explores product reviews and predicts whether each review is positive, negative, or neutral. It includes a command-line workflow and a small Streamlit interface.

## Problem and Objectives

Customer reviews contain useful feedback, but reading them one by one makes it difficult to summarize sentiment at scale. This project demonstrates a basic text-classification workflow that can:

- Load and validate a CSV of reviews.
- Report missing values, duplicate review-label pairs, invalid labels, and sentiment distribution.
- Preserve original review text while producing a cleaned text column.
- Train a TF-IDF and Logistic Regression classifier.
- Evaluate with stratified cross-validation and predict sentiment for new text.

## Technologies

- Python, Pandas, and NumPy
- Scikit-learn for TF-IDF, Logistic Regression, and evaluation
- Joblib for saving and loading the fitted model pipeline
- Streamlit for the review form
- `unittest` for lightweight automated tests

The project does not directly import or use Matplotlib or Pillow. Streamlit may install Pillow as a transitive dependency, but this app does not use image features.

## Project Structure

```text
.
|-- data/                         # Local CSV files; not committed
|-- models/                       # Generated model pipeline; not committed
|-- src/
|   |-- app.py                    # Streamlit interface
|   |-- explore_data.py           # Counts and percentages by sentiment
|   |-- load_data.py              # CSV loading and quality checks
|   |-- predict.py                # Load saved pipeline and predict
|   |-- prepare_text.py           # Clean reviews, preserving originals
|   `-- train_model.py            # Cross-validation and model training
|-- tests/
|   `-- test_predict.py           # Prediction and persistence tests
|-- requirements.txt
`-- README.md
```

## Setup on Windows

Use PowerShell from the repository root. Python 3.10 or newer is recommended. Keep the virtual environment outside deeply nested OneDrive folders: in this workspace, PyArrow's compiled extension failed to load from the project-local `.venv`, while the same dependencies worked from a short path under the user profile.

```powershell
py -m venv "$env:USERPROFILE\.venvs\ecommerce-sentiment-analysis"
& "$env:USERPROFILE\.venvs\ecommerce-sentiment-analysis\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, allow it for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& "$env:USERPROFILE\.venvs\ecommerce-sentiment-analysis\Scripts\Activate.ps1"
```

Place a CSV at `data/customer_reviews.csv`. It must have `review` and `sentiment` columns. Sentiment values must be `positive`, `negative`, or `neutral` (capitalization and surrounding whitespace are normalized). Keep personal or otherwise private review data out of Git. CSV files in `data/` and generated model files in `models/` are ignored.

## Run the Project

Run these commands from the repository root after activating `.venv`:

```powershell
python src/load_data.py
python src/explore_data.py
python src/prepare_text.py
python src/train_model.py
```

`prepare_text.py` writes `data/cleaned_customer_reviews.csv`, retaining the original `review` and adding `cleaned_review`. Training reads the validated source CSV and uses the cleaned text in memory.

Predict one review interactively:

```powershell
python src/predict.py
```

Or provide text directly, or run the sample predictions:

```powershell
python src/predict.py "The product broke after one day and the quality is terrible."
python src/predict.py --examples
```

Train the model before predicting. The fitted TF-IDF vectorizer and classifier are saved together as `models/sentiment_pipeline.joblib`. Joblib reloads both as one pipeline, ensuring new reviews use the same fitted vocabulary. The prediction command gives a clear error if the model file is missing.

Run the interface after training:

```powershell
python -m streamlit run src/app.py
```

Streamlit prints a local URL in the terminal, usually `http://localhost:8501`.

Run the tests:

```powershell
python -m unittest discover -s tests -v
```

## How the Model Works

TF-IDF converts review words into numeric features. It gives more weight to words that are informative within a review and less weight to words that are common across many reviews. Logistic Regression learns patterns between those features and the supplied sentiment labels.

The two steps are kept in a scikit-learn `Pipeline`. During evaluation, cross-validation fits TF-IDF only on each training fold, which avoids using held-out review text to build the vocabulary. The final pipeline is then fitted on all usable reviews for prediction.

## Evaluation and Limitations

The current local CSV has only 10 reviews: 4 positive, 4 negative, and 2 neutral. The current 2-fold cross-validation run reported 0.30 accuracy, 0.20 macro precision, 0.25 macro recall, and 0.22 macro F1. It did not identify either neutral review in that run. These metrics are included as a transparent snapshot, not a claim of model quality; with so few examples, each review has an outsized effect and results are unreliable.

For a useful evaluation, collect a much larger and more representative set of independently labeled reviews for all three classes. Include varied products, writing styles, and mixed or ambiguous opinions. Keep a final test set separate until model choices are complete. A successful training run or a plausible individual prediction does not prove the model is accurate.

Example interaction:

```text
Review: The product broke after one day and the quality is terrible.
Predicted sentiment: negative
```

Predictions are estimates based on the training examples, not guaranteed judgments. The model may misclassify a review, especially when its wording or context differs from the small training dataset.

## Future Improvements

- Add a larger, representative, appropriately shareable labeled review dataset.
- Compare the baseline with other classifiers and tune settings using validation data.
- Add more robust tests for data validation and training behavior.
- Track model versions and evaluation results as the dataset grows.
