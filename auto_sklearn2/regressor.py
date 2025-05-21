"""
Auto-Sklearn2 Regressor - A Python 3.13 compatible version of auto-sklearn
"""

import numpy as np
import os
import time
import logging
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline

# Configure logging
logger = logging.getLogger('auto_sklearn2')

class AutoSklearnRegressor(BaseEstimator, RegressorMixin):
    """
    A simplified version of auto-sklearn for regression that works with Python 3.13
    
    Parameters
    ----------
    time_limit : int, default=120
        Time limit in seconds for the search of appropriate models.
    n_jobs : int, default=-1
        Number of jobs to run in parallel. -1 means using all processors.
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
    scoring : str, default='r2'
        Scoring metric to use for model evaluation. Options are 'r2', 'neg_mean_squared_error',
        'neg_mean_absolute_error'.
        
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
    
    def __init__(self, time_limit=120, n_jobs=-1, random_state=None, scoring='r2'):
        self.time_limit = time_limit
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.scoring = scoring
        self.models = None
        self.best_model = None
        self.best_score = -np.inf if scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'] else np.inf
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
            'random_forest': RandomForestRegressor(random_state=self.random_state, n_jobs=self.n_jobs),
            'gradient_boosting': GradientBoostingRegressor(random_state=self.random_state),
            'linear_regression': LinearRegression(n_jobs=self.n_jobs),
            'ridge': Ridge(random_state=self.random_state),
            'lasso': Lasso(random_state=self.random_state),
            'elastic_net': ElasticNet(random_state=self.random_state),
            'svr': SVR(),
            'knn': KNeighborsRegressor(n_jobs=self.n_jobs),
            'mlp': MLPRegressor(random_state=self.random_state, max_iter=300)
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
        logger.info(f"Starting AutoSklearnRegressor with time limit: {self.time_limit} seconds")
        
        start_time = time.time()
        models = self._get_models()
        preprocessors = self._get_preprocessors()
        
        # Store results
        self.models = {}
        
        # Cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for preprocessor_name, preprocessor in preprocessors.items():
            for model_name, model in models.items():
                # Check if we've exceeded the time limit
                if time.time() - start_time > self.time_limit:
                    logger.info(f"Time limit reached. Stopping search.")
                    break
                
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', model)
                ])
                
                # Evaluate model
                try:
                    logger.info(f"Evaluating {preprocessor_name} + {model_name}")
                    scores = cross_val_score(pipeline, X, y, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs)
                    mean_score = np.mean(scores)
                    
                    # Store model and score
                    self.models[f"{preprocessor_name}_{model_name}"] = {
                        'pipeline': pipeline,
                        'score': mean_score
                    }
                    
                    logger.info(f"{preprocessor_name} + {model_name}: {mean_score:.4f}")
                    
                    # Update best model
                    if self.scoring in ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']:
                        if mean_score > self.best_score:
                            self.best_score = mean_score
                            self.best_model = pipeline
                            self.best_params = {
                                'preprocessor': preprocessor_name,
                                'regressor': model_name
                            }
                    else:
                        if mean_score < self.best_score:
                            self.best_score = mean_score
                            self.best_model = pipeline
                            self.best_params = {
                                'preprocessor': preprocessor_name,
                                'regressor': model_name
                            }
                except Exception as e:
                    logger.error(f"Error evaluating {preprocessor_name} + {model_name}: {str(e)}")
        
        # Fit the best model on the entire dataset
        if self.best_model is not None:
            logger.info(f"Best model: {self.best_params['preprocessor']} + {self.best_params['regressor']} with score: {self.best_score:.4f}")
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
            Predicted values.
        """
        if self.best_model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.best_model.predict(X)
    
    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction
        
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Test samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            True values for X.
            
        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        return r2_score(y, self.predict(X))
    
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
