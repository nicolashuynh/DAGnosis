# stdlib
import logging
from typing import Any, Dict, Optional

# third party
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

# dagnosis absolute
from dagnosis.representation.representation_mapper import FeatureRepresentationMapper


class ConformalEvaluator:
    """
    A unified class for conformal prediction using different feature representation approaches.
    """

    def __init__(
        self,
        conf_predictor_cfg: DictConfig,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the evaluator
        """
        self.mapper: Optional[FeatureRepresentationMapper] = None
        self.conf_predictor_cfg = conf_predictor_cfg
        self.logger = logger or logging.getLogger(__name__)
        self.list_conf: Optional[list] = None

    def _create_conf_predictor(self, alpha: float) -> Any:
        """Create a new instance of the conformal predictor using the config."""
        return instantiate(self.conf_predictor_cfg, alpha=alpha)

    def _train_single_feature(
        self,
        X_train: np.ndarray,
        feature: int,
        alpha: float = 0.1,
    ) -> Any:
        """Train conformal predictor for a single feature."""
        self.logger.info(f"Processing feature {feature}")
        assert self.mapper is not None, "Mapper not set."
        X_transformed, y = self.mapper.transform_feature(X_train, feature)

        conf = self._create_conf_predictor(alpha)
        conf.fit(X_transformed, y)
        return conf

    def train(
        self,
        X_train: np.ndarray,
        representation_mapper: FeatureRepresentationMapper,
        list_features: Optional[np.ndarray] = None,
        alphas_adjusted: Optional[Dict[int, float]] = None,
    ) -> "ConformalEvaluator":
        """
        Train feature-wise conformal predictors.
        """
        if list_features is None:
            list_features = np.arange(X_train.shape[1])

        assert alphas_adjusted is not None, "Alphas not set."

        # Fit the representation mapper
        representation_mapper.fit(X_train)

        # Set the mapper
        self.mapper = representation_mapper

        self.list_conf = [None for _ in range(X_train.shape[1])]

        assert list_features is not None, "List of features not set."

        for feature in tqdm(list_features):
            assert feature in alphas_adjusted, f"Alpha not set for feature {feature}"
            alpha = alphas_adjusted[feature]

            self.list_conf[feature] = self._train_single_feature(
                X_train, feature, alpha
            )

        return self

    def predict(self, X_test: np.ndarray, list_features: np.ndarray) -> Dict[int, Any]:
        """Make predictions using trained conformal predictors."""
        if self.list_conf is None:
            raise ValueError("Model not trained. Call train first.")

        conformal_dict: Dict[int, pd.DataFrame] = {}
        assert self.mapper is not None, "Mapper not set."
        for feature in list_features:
            conf = self.list_conf[feature]
            X_transformed, y = self.mapper.transform_feature(X_test, feature)

            conformal_dict[feature] = conf.predict(x_test=X_transformed, y_test=y)

        return conformal_dict
