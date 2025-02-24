# stdlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

# third party
import numpy as np

# dagnosis absolute
from dagnosis.conformal.conformal_evaluator import ConformalEvaluator
from dagnosis.conformal.significance_correction import MultipleTestingCorrection
from dagnosis.dag_learner.extractor import DAGExtractor
from dagnosis.data.datamodule import Data
from dagnosis.representation.representation_mapper import (
    DAGRepresentationMapper,
    FeatureRepresentationMapper,
)


class BaseConformalPipeline(ABC):
    """
    Abstract base class for conformal pipeline (e.g. extract DAG and train conformal estimators).
    """

    def __init__(
        self,
        significance_correction: MultipleTestingCorrection,
        conformal_evaluator: ConformalEvaluator,
        representation_mapper: FeatureRepresentationMapper,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the base Evaluator.
        """
        self.significance_correction = significance_correction
        self.conformal_evaluator = conformal_evaluator
        self.representation_mapper = representation_mapper
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    def predict(self, X_test: np.ndarray, list_features: np.ndarray) -> Dict[int, Any]:
        """Make predictions using trained conformal predictors."""
        return self.conformal_evaluator.predict(X_test, list_features)


class DAGBasedConformalPipeline(BaseConformalPipeline):
    """
    A class for training feature-wise conformal predictors using different DAG structures.
    Supports multiple DAG extraction methods: ground truth, autoregressive, and NOTEARS.
    """

    representation_mapper: DAGRepresentationMapper

    def __init__(
        self,
        significance_correction: MultipleTestingCorrection,
        conformal_evaluator: ConformalEvaluator,
        representation_mapper: DAGRepresentationMapper,
        dag_extractor: DAGExtractor,
        logger: Optional[logging.Logger] = None,
    ):

        super().__init__(
            significance_correction, conformal_evaluator, representation_mapper, logger
        )
        self.dag_extractor = dag_extractor

    def get_dag(self, D: Data) -> np.ndarray:
        """Extract DAG from data."""
        dag = self.dag_extractor.extract(D)
        return dag

    def train(
        self,
        D: Data,
        list_features: np.ndarray,
        alpha: float = 0.1,
    ) -> "DAGBasedConformalPipeline":

        X_train = D.get_train()
        dag = self.get_dag(D)
        self.representation_mapper.set_dag(dag)

        alpha_adjusted = self.significance_correction.compute_correction(
            alpha, list_features
        )

        self.conformal_evaluator.train(
            X_train, self.representation_mapper, list_features, alpha_adjusted
        )

        return self


class PCABasedConformalPipeline(BaseConformalPipeline):
    """
    A class for training feature-wise conformal predictors using PCA-based dimensionality reduction.
    Uses PCA representations instead of DAG structure.
    """

    def __init__(
        self,
        significance_correction: MultipleTestingCorrection,
        conformal_evaluator: ConformalEvaluator,
        representation_mapper: FeatureRepresentationMapper,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the PCA-based Evaluator.

        Args:
            conf_predictor_cfg: Hydra config for conformal predictor
            rep_dim: Dimension of the PCA representation
            logger: Optional logger instance
        """
        super().__init__(
            significance_correction, conformal_evaluator, representation_mapper, logger
        )

    def train(
        self, D: Data, list_features: np.ndarray, alpha: float = 0.1
    ) -> "PCABasedConformalPipeline":
        """Train feature-wise conformal predictors using PCA representation."""
        X_train = D.get_train()

        alpha_adjusted = self.significance_correction.compute_correction(
            alpha, list_features
        )
        self.conformal_evaluator.train(
            X_train, self.representation_mapper, list_features, alpha_adjusted
        )

        return self
