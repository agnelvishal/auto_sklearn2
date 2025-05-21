"""
Auto-Sklearn2 Classifier - A Python 3.13 compatible version of auto-sklearn
"""

import numpy as np
import os
import time
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline

# Configure logging
logger = logging.getLogger('auto_sklearn2')

class AutoSklearnClassifier(BaseEstimator, ClassifierMixin):
    """
    A simplified version of auto-sklearn that works with Python 3.13
    
    Parameters
    ----------
    time_limit : int, default=120
        Time limit in seconds for the search of appropriate models.
    n_jobs : int, default=-1
        Number of jobs to run in parallel. -1 means using all processors.
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        
    Attributes
    ----------
    models : dict
        Dictionary with all trained models and their performance.
    best_model : sklearn.pipeline.Pipeline
        The best model found.
    best_score : float
        The score of the best model.
    best_params : dict
        The parameters of the best model.
    """
    
    def __init__(self, time_limit=120, n_jobs=-1, random_state=None):
        self.time_limit = time_limit
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.models = None
        self.best_model = None
        self.best_score = 0
        self.best_params = None
        
    def _get_models(self):
        """
        Define a set of models to try
        
        Returns
        -------
        dict
            Dictionary with model name as key and model instance as value.
        """
        models = {
            'random_forest': RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=self.n_jobs),
            'svc': SVC(random_state=self.random_state, probability=True),
            'knn': KNeighborsClassifier(n_jobs=self.n_jobs),
            'mlp': MLPClassifier(random_state=self.random_state, max_iter=300)
        }
        return models
    
    def _get_preprocessors(self):
        """
        Define a set of preprocessors to try
        
        Returns
        -------
        dict
            Dictionary with preprocessor name as key and preprocessor instance as value.
        """
        preprocessors = {
            'standard_scaler': StandardScaler(),
            'minmax_scaler': MinMaxScaler(),
            'robust_scaler': RobustScaler()
        }
        return preprocessors
    
    def fit(self, X, y):
        """
        Fit the model to the data
        
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Target values
            
        Returns
        -------
        self : object
            Returns self.
        """
        logger.info(f"Starting AutoSklearnClassifier with time limit: {self.time_limit} seconds")
        
        start_time = time.time()
        models = self._get_models()
        preprocessors = self._get_preprocessors()
        
        # Store results
        self.models = {}
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for preprocessor_name, preprocessor in preprocessors.items():
            for model_name, model in models.items():
                # Check if we've exceeded the time limit
                if time.time() - start_time > self.time_limit:
                    logger.info(f"Time limit reached. Stopping search.")
                    break
                
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                
                # Evaluate model
                try:
                    logger.info(f"Evaluating {preprocessor_name} + {model_name}")
                    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=self.n_jobs)
                    mean_score = np.mean(scores)
                    
                    # Store model and score
                    self.models[f"{preprocessor_name}_{model_name}"] = {
                        'pipeline': pipeline,
                        'score': mean_score
                    }
                    
                    logger.info(f"{preprocessor_name} + {model_name}: {mean_score:.4f}")
                    
                    # Update best model
                    if mean_score > self.best_score:
                        self.best_score = mean_score
                        self.best_model = pipeline
                        self.best_params = {
                            'preprocessor': preprocessor_name,
                            'classifier': model_name
                        }
                except Exception as e:
                    logger.error(f"Error evaluating {preprocessor_name} + {model_name}: {str(e)}")
        
        # Fit the best model on the entire dataset
        if self.best_model is not None:
            logger.info(f"Best model: {self.best_params['preprocessor']} + {self.best_params['classifier']} with score: {self.best_score:.4f}")
            self.best_model.fit(X, y)
        else:
            logger.warning("No models were successfully trained.")
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the best model
        
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        C : array, shape (n_samples,)
            Predicted class labels.
        """
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.best_model.predict(X)
    
    def predict_proba(self, X):
        """
        Get probability estimates
        
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        P : array, shape (n_samples, n_classes)
            Probability of the sample for each class in the model.
        """
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.best_model.predict_proba(X)
    
    def score(self, X, y):
        """
        Return the accuracy score
        
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.
            
        Returns
        -------
        score : float
            Accuracy score.
        """
        return accuracy_score(y, self.predict(X))
    
    def get_models_performance(self):
        """
        Return the performance of all models
        
        Returns
        -------
        dict
            Dictionary with model name as key and score as value.
        """
        if self.models is None:
            raise ValueError("Models have not been fitted yet.")
        
        return {name: info['score'] for name, info in self.models.items()}
