"""
Simple regression example using Auto-Sklearn2
"""

from auto_sklearn2 import AutoSklearnRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# Load data
print("Loading diabetes dataset...")
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset shape: {X.shape}")
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create and fit the auto-sklearn regressor
print("\nTraining AutoSklearnRegressor...")
auto_sklearn = AutoSklearnRegressor(time_limit=60, random_state=42)
auto_sklearn.fit(X_train, y_train)

# Make predictions
print("\nMaking predictions...")
y_pred = auto_sklearn.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\nBest model: {auto_sklearn.best_params}")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")

# Show all models performance
print("\nAll models performance:")
for model_name, score in auto_sklearn.get_models_performance().items():
    print(f"{model_name}: {score:.4f}")
