# Auto-Sklearn2

A Python 3.13 compatible version of auto-sklearn for automated machine learning.

## Overview

Auto-Sklearn2 is a lightweight, Python 3.13 compatible alternative to the popular [auto-sklearn](https://github.com/automl/auto-sklearn) package. It provides automated machine learning capabilities without the dependency on ConfigSpace, which currently has compatibility issues with Python 3.13 and NumPy 2.0.

## Features

- **Python 3.13 Compatible**: Works with the latest Python version
- **Automated Machine Learning**: Automatically selects the best model and preprocessing pipeline
- **Classification and Regression**: Supports both classification and regression tasks
- **Time-Limited Optimization**: Set a time budget for model selection
- **Multiple Models**: Includes RandomForest, GradientBoosting, LogisticRegression, SVC, KNN, MLP, and more
- **Multiple Preprocessors**: Includes StandardScaler, MinMaxScaler, and RobustScaler
- **Cross-Validation**: Uses cross-validation for model evaluation

## Installation

```bash
pip install auto-sklearn2
```

## Quick Start for Classification

```python
from auto_sklearn2 import AutoSklearnClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the auto-sklearn classifier
auto_sklearn = AutoSklearnClassifier(time_limit=120, random_state=42)
auto_sklearn.fit(X_train, y_train)

# Make predictions
y_pred = auto_sklearn.predict(X_test)

# Get the best model details
print(f"Best model: {auto_sklearn.best_params}")
print(f"Accuracy: {auto_sklearn.score(X_test, y_test):.4f}")

# Show all models performance
for model_name, score in auto_sklearn.get_models_performance().items():
    print(f"{model_name}: {score:.4f}")
```

## Quick Start for Regression

```python
from auto_sklearn2 import AutoSklearnRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the auto-sklearn regressor
auto_sklearn = AutoSklearnRegressor(time_limit=120, random_state=42)
auto_sklearn.fit(X_train, y_train)

# Make predictions
y_pred = auto_sklearn.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Best model: {auto_sklearn.best_params}")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")

# Show all models performance
for model_name, score in auto_sklearn.get_models_performance().items():
    print(f"{model_name}: {score:.4f}")
```

## Differences from auto-sklearn

Auto-Sklearn2 is a simplified version of auto-sklearn with the following differences:

1. **No ConfigSpace Dependency**: Uses scikit-learn's built-in models and preprocessing methods
2. **Fewer Models and Preprocessors**: Includes only the most common models and preprocessors
3. **No Meta-Learning**: Does not use meta-learning to warm-start the optimization
4. **No Ensemble Building**: Does not build ensembles of models
5. **Simpler Hyperparameter Optimization**: Uses cross-validation instead of Bayesian optimization

## License

BSD 3-Clause License (same as auto-sklearn)

## Citation

If you use Auto-Sklearn2 in a scientific publication, please cite the original auto-sklearn paper:

```
@inproceedings{feurer-neurips15a,
    title     = {Efficient and Robust Automated Machine Learning},
    author    = {Feurer, Matthias and Klein, Aaron and Eggensperger, Katharina and
                 Springenberg, Jost and Blum, Manuel and Hutter, Frank},
    booktitle = {Advances in Neural Information Processing Systems 28},
    pages     = {2962--2970},
    year      = {2015}
}
```

## Acknowledgements

This package is inspired by and based on the original [auto-sklearn](https://github.com/automl/auto-sklearn) package.
