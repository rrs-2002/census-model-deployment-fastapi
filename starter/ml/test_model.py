"""
Unit tests for the ML model functions.
Run with: pytest starter/ml/test_model.py -v
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference


@pytest.fixture(scope="module")
def data():
    """Load and return the cleaned census data."""
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    df = pd.read_csv(os.path.join(base_dir, 'data', 'census.csv'))
    return df


@pytest.fixture(scope="module")
def cat_features():
    """Return the list of categorical features."""
    return [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]


@pytest.fixture(scope="module")
def trained_artifacts(data, cat_features):
    """Train a model and return model, encoder, lb, and test data."""
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb
    )
    model = train_model(X_train, y_train)
    return model, encoder, lb, X_test, y_test


def test_train_model_returns_correct_type(trained_artifacts):
    """Test that train_model returns a RandomForestClassifier."""
    model, _, _, _, _ = trained_artifacts
    assert isinstance(model, RandomForestClassifier), \
        "train_model should return a RandomForestClassifier"


def test_inference_returns_correct_shape(trained_artifacts):
    """Test that inference returns predictions with the right length."""
    model, _, _, X_test, y_test = trained_artifacts
    preds = inference(model, X_test)
    assert len(preds) == len(y_test), \
        "Predictions length should match test labels length"
    assert isinstance(preds, np.ndarray), \
        "Predictions should be a numpy array"


def test_compute_model_metrics_returns_three_floats(trained_artifacts):
    """Test that compute_model_metrics returns 3 float values."""
    model, _, _, X_test, y_test = trained_artifacts
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert isinstance(precision, float), "Precision should be a float"
    assert isinstance(recall, float), "Recall should be a float"
    assert isinstance(fbeta, float), "F-beta should be a float"
    # Metrics should be between 0 and 1
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0