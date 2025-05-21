"""
Simple classification example using Auto-Sklearn2
"""

from auto_sklearn2 import AutoSklearnClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
print("Loading breast cancer dataset...")
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dataset shape: {X.shape}")
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create and fit the auto-sklearn classifier
print("\nTraining AutoSklearnClassifier...")
auto_sklearn = AutoSklearnClassifier(time_limit=60, random_state=42)
auto_sklearn.fit(X_train, y_train)

# Make predictions
print("\nMaking predictions...")
y_pred = auto_sklearn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nBest model: {auto_sklearn.best_params}")
print(f"Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Show all models performance
print("\nAll models performance:")
for model_name, score in auto_sklearn.get_models_performance().items():
    print(f"{model_name}: {score:.4f}")
