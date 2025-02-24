# stdlib
from abc import ABC, abstractmethod
from typing import Optional

# third party
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mapie.quantile_regression import MapieQuantileRegressor
from scipy.stats import randint, uniform
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split


class ConformalPredictor(ABC):
    """Abstract base class for conformal prediction methods."""

    def __init__(
        self,
        alpha: float = 0.1,
        scale: bool = False,
        seed: int = 42,
        cal_size: float = 0.2,
    ) -> None:
        """
        Initialize the conformal predictor.

        Args:
            alpha: Significance level for the prediction intervals
            scale: Whether to scale the features
            seed: Random seed for reproducibility
            cal_size: Proportion of data to use for calibration
        """
        self.alpha = alpha
        self.scale = scale
        self.seed = seed
        self.cal_size = cal_size

        # These will be set during fitting
        self.range_max = None
        self.range_min = None
        self.cp_model = None

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the conformal predictor.

        Args:
            x_train: Training features
            y_train: Training targets
        """

    @abstractmethod
    def predict(self, x_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Generate predictions and prediction intervals.

        Args:
            x_test: Test features
            y_test: Test targets

        Returns:
            DataFrame containing predictions and intervals
        """


class CQR(ConformalPredictor):
    """Conformalized Quantile Regression implementation."""

    def __init__(
        self,
        alpha: float = 0.1,
        scale: bool = False,
        seed: int = 42,
        cal_size: float = 0.2,
        n_search: int = 100,
        param_bounds: Optional[dict] = None,
        n_splits: int = 5,
        seed_datasplit: int = 0,
    ) -> None:
        """
        Initialize the CQR predictor.

        Args:
            alpha: Significance level for the prediction intervals
            scale: Whether to scale the features
            seed: Random seed for reproducibility
            cal_size: Proportion of data to use for calibration
            n_search: Number of iterations for hyperparameter search
            param_bounds: Dictionary of parameter bounds for search space.
                        Expected format:
                        {
                            "num_leaves": {"low": 10, "high": 50},
                            "max_depth": {"low": 3, "high": 20},
                            "n_estimators": {"low": 50, "high": 300},
                            "learning_rate": {"low": 0, "high": 1}
                        }
                        If None, uses default bounds.
        """
        super().__init__(alpha, scale, seed, cal_size)
        self.n_search = n_search
        self.n_splits = n_splits
        self.seed_datasplit = seed_datasplit

        # Default parameter bounds
        self.param_bounds = param_bounds or {
            "num_leaves": {"low": 10, "high": 50},
            "max_depth": {"low": 3, "high": 20},
            "n_estimators": {"low": 50, "high": 300},
            "learning_rate": {"low": 0, "high": 1},
        }

    def _get_param_distributions(self) -> dict:
        """Convert parameter bounds to scipy distribution objects."""
        return {
            "num_leaves": randint(
                low=self.param_bounds["num_leaves"]["low"],
                high=self.param_bounds["num_leaves"]["high"],
            ),
            "max_depth": randint(
                low=self.param_bounds["max_depth"]["low"],
                high=self.param_bounds["max_depth"]["high"],
            ),
            "n_estimators": randint(
                low=self.param_bounds["n_estimators"]["low"],
                high=self.param_bounds["n_estimators"]["high"],
            ),
            "learning_rate": uniform(
                loc=self.param_bounds["learning_rate"]["low"],
                scale=self.param_bounds["learning_rate"]["high"]
                - self.param_bounds["learning_rate"]["low"],
            ),
        }

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the CQR model.

        Args:
            x_train: Training features
            y_train: Training targets
        """
        x_train = np.array(x_train).astype(np.float64)
        y_train = np.array(y_train).astype(np.float64)

        # Split into training and calibration sets
        X_train, X_cal, y_train, y_cal = train_test_split(
            x_train, y_train, test_size=self.cal_size, random_state=self.seed_datasplit
        )

        self.range_max = np.max(y_train)
        self.range_min = np.min(y_train)

        # Initialize and optimize base estimator
        estimator = LGBMRegressor(
            objective="quantile", alpha=self.alpha / 2, random_state=self.seed, n_jobs=1
        )

        optim_model = RandomizedSearchCV(
            estimator,
            param_distributions=self._get_param_distributions(),
            n_jobs=-1,
            n_iter=self.n_search,
            cv=KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed),
            verbose=-1,
            random_state=self.seed,
        )

        optim_model.fit(X_train, y_train)
        estimator = optim_model.best_estimator_

        # Fit MAPIE model
        params = {"method": "quantile", "cv": "split", "alpha": self.alpha}
        quantile_regressor = MapieQuantileRegressor(estimator, **params)
        quantile_regressor.fit(X_train, y_train, X_calib=X_cal, y_calib=y_cal)

        self.cp_model = quantile_regressor

    def predict(self, x_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Generate predictions with confidence intervals.

        Args:
            x_test: Test features
            y_test: Test targets

        Returns:
            DataFrame with predictions and confidence intervals
        """
        if self.cp_model is None:
            raise ValueError("Model must be fitted before making predictions")

        _, y_pis = self.cp_model.predict(x_test)

        lower_bound = y_pis[:, 0, 0]
        upper_bound = y_pis[:, 1, 0]

        header = ["min", "max", "true_val", "conf_interval"]
        size = upper_bound - lower_bound

        table = np.vstack([lower_bound.T, upper_bound.T, y_test, size.T]).T
        df = pd.DataFrame(table, columns=header)

        feature_range = self.range_max - self.range_min
        df["norm_interval"] = df["conf_interval"] / feature_range
        return df
