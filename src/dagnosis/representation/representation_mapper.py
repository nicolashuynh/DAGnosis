# stdlib
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

# third party
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# dagnosis absolute
from dagnosis.utils.dag import find_markov_boundary


class FeatureRepresentationMapper(ABC):
    """
    Abstract base class for mapping input features to a different representation space.
    This can be used for dimensionality reduction, feature selection, or structure-based mapping.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the base mapper."""
        self.logger = logger or logging.getLogger(__name__)
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> "FeatureRepresentationMapper":
        """
        Fit the mapper to the input data.

        """

    @abstractmethod
    def transform_feature(
        self, X: np.ndarray, feature_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data for a specific feature prediction.
        """


class DAGRepresentationMapper(FeatureRepresentationMapper):
    """Maps features using DAG structure and Markov boundaries."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.dag: Optional[np.ndarray] = None
        self.markov_boundaries: Optional[list] = None

    def set_dag(self, dag: np.ndarray) -> "DAGRepresentationMapper":
        """Set the DAG structure."""
        self.dag = dag
        return self

    def fit(self, X: np.ndarray) -> "DAGRepresentationMapper":
        """
        Compute Markov boundaries from the DAG.

        """
        if self.dag is None:
            raise ValueError("DAG not set. Call set_dag first.")

        if self.dag.shape != (X.shape[1], X.shape[1]):
            raise ValueError(
                f"DAG shape does not match input dimensions. Got {self.dag.shape}, expected {(X.shape[1], X.shape[1])}"
            )

        self.markov_boundaries = find_markov_boundary(self.dag)
        self._is_fitted = True
        return self

    def transform_feature(
        self, X: np.ndarray, feature_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using Markov boundary for the target feature.
        """
        if not self._is_fitted:
            raise ValueError("Mapper not fitted")
        assert self.markov_boundaries is not None, "Markov boundaries not set."
        conditioning_vars = self.markov_boundaries[feature_idx]
        X_transformed = (
            np.zeros((X.shape[0], 1))
            if len(conditioning_vars) == 0
            else X[:, conditioning_vars]
        )
        y = X[:, feature_idx]

        return X_transformed, y


class PCARepresentationMapper(FeatureRepresentationMapper):
    """Maps features using PCA-based dimensionality reduction."""

    def __init__(
        self,
        random_state: int = 0,
        compression_factor: float = 0.5,
        n_components: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize PCA mapper.
        """
        super().__init__(logger)
        self.random_state = random_state
        self.compression_factor = compression_factor
        self.scaler = StandardScaler()
        self.pca = None
        self.pca_transform = None
        self.n_components = n_components

    def fit(self, X: np.ndarray) -> "PCARepresentationMapper":
        """
        Fit PCA transformation.
        """
        if self.n_components is None:
            self.n_components = int(np.ceil(X.shape[1] * self.compression_factor))

        X_scaled = self.scaler.fit_transform(X)
        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)

        assert self.pca is not None, "PCA not fitted."
        self.pca_transform = self.pca.fit_transform(X_scaled)
        self._is_fitted = True
        return self

    def transform_feature(
        self, X: np.ndarray, feature_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data using PCA representation.
        """

        if not self._is_fitted:
            raise ValueError("Mapper not fitted")
        X_scaled = self.scaler.transform(X)
        assert self.pca is not None, "PCA not fitted."
        X_transformed = self.pca.transform(X_scaled)
        y = X[:, feature_idx]

        return X_transformed, y
