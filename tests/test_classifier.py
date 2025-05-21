"""
Tests for the Auto-Sklearn2 Classifier
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from auto_sklearn2 import AutoSklearnClassifier


@pytest.fixture
def classification_data():
    """Generate a simple classification dataset"""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def test_classifier_initialization():
    """Test that the classifier can be initialized with default parameters"""
    clf = AutoSklearnClassifier()
    assert clf.time_limit == 120
    assert clf.n_jobs == -1
    assert clf.random_state is None
    assert clf.models is None
    assert clf.best_model is None
    assert clf.best_score == 0
    assert clf.best_params is None


def test_classifier_custom_params():
    """Test that the classifier can be initialized with custom parameters"""
    clf = AutoSklearnClassifier(time_limit=60, n_jobs=2, random_state=42)
    assert clf.time_limit == 60
    assert clf.n_jobs == 2
    assert clf.random_state == 42


def test_classifier_fit_predict(classification_data):
    """Test that the classifier can fit and predict"""
    X_train, X_test, y_train, y_test = classification_data
    
    # Use a short time limit for testing
    clf = AutoSklearnClassifier(time_limit=10, random_state=42)
    clf.fit(X_train, y_train)
    
    # Check that models were trained
    assert clf.models is not None
    assert len(clf.models) > 0
    
    # Check that a best model was selected
    assert clf.best_model is not None
    assert clf.best_score > 0
    assert clf.best_params is not None
    
    # Check that predictions can be made
    y_pred = clf.predict(X_test)
    assert y_pred.shape == y_test.shape
    
    # Check that probabilities can be obtained
    y_proba = clf.predict_proba(X_test)
    assert y_proba.shape == (y_test.shape[0], 2)  # Binary classification
    
    # Check that the score method works
    score = clf.score(X_test, y_test)
    assert 0 <= score <= 1
    
    # Check that model performance can be retrieved
    performance = clf.get_models_performance()
    assert len(performance) > 0
    assert all(0 <= score <= 1 for score in performance.values())


def test_classifier_not_fitted():
    """Test that appropriate errors are raised when the classifier is not fitted"""
    clf = AutoSklearnClassifier()
    X = np.random.rand(10, 5)
    
    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        clf.predict(X)
    
    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        clf.predict_proba(X)
    
    with pytest.raises(ValueError, match="Models have not been fitted yet"):
        clf.get_models_performance()
