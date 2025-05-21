"""
Tests for the Auto-Sklearn2 Regressor
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from auto_sklearn2 import AutoSklearnRegressor


@pytest.fixture
def regression_data():
    """Generate a simple regression dataset"""
    X, y = make_regression(
        n_samples=100,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def test_regressor_initialization():
    """Test that the regressor can be initialized with default parameters"""
    reg = AutoSklearnRegressor()
    assert reg.time_limit == 120
    assert reg.n_jobs == -1
    assert reg.random_state is None
    assert reg.scoring == 'r2'
    assert reg.models is None
    assert reg.best_model is None
    assert reg.best_score == -np.inf
    assert reg.best_params is None


def test_regressor_custom_params():
    """Test that the regressor can be initialized with custom parameters"""
    reg = AutoSklearnRegressor(time_limit=60, n_jobs=2, random_state=42, scoring='neg_mean_squared_error')
    assert reg.time_limit == 60
    assert reg.n_jobs == 2
    assert reg.random_state == 42
    assert reg.scoring == 'neg_mean_squared_error'


def test_regressor_fit_predict(regression_data):
    """Test that the regressor can fit and predict"""
    X_train, X_test, y_train, y_test = regression_data
    
    # Use a short time limit for testing
    reg = AutoSklearnRegressor(time_limit=10, random_state=42)
    reg.fit(X_train, y_train)
    
    # Check that models were trained
    assert reg.models is not None
    assert len(reg.models) > 0
    
    # Check that a best model was selected
    assert reg.best_model is not None
    assert reg.best_score > -np.inf
    assert reg.best_params is not None
    
    # Check that predictions can be made
    y_pred = reg.predict(X_test)
    assert y_pred.shape == y_test.shape
    
    # Check that the score method works
    score = reg.score(X_test, y_test)
    assert -np.inf < score <= 1  # R^2 can be negative for bad models
    
    # Check that model performance can be retrieved
    performance = reg.get_models_performance()
    assert len(performance) > 0


def test_regressor_not_fitted():
    """Test that appropriate errors are raised when the regressor is not fitted"""
    reg = AutoSklearnRegressor()
    X = np.random.rand(10, 5)
    
    with pytest.raises(ValueError, match="Model has not been fitted yet"):
        reg.predict(X)
    
    with pytest.raises(ValueError, match="Models have not been fitted yet"):
        reg.get_models_performance()
