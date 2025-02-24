# third party
import numpy as np
import pytest
from omegaconf import DictConfig

# dagnosis absolute
from dagnosis.conformal.conformal_evaluator import ConformalEvaluator
from dagnosis.data.datamodule import SyntheticData
from dagnosis.representation.representation_mapper import (
    DAGRepresentationMapper,
    PCARepresentationMapper,
)
from dagnosis.utils.seed import set_random_seeds


@pytest.fixture(scope="module")
def conf_predictor_cfg():
    """Create a config for the real CQR predictor"""
    return DictConfig(
        {
            "_target_": "dagnosis.conformal.conformal_predictor.CQR",
            "alpha": 0.1,
            "scale": False,
            "seed": 42,
            "cal_size": 0.2,
            "n_search": 1,
            "n_splits": 2,
        }
    )


@pytest.fixture(scope="module")
def dataset_and_arrays():
    """Create a small dataset and return both datamodule and numpy arrays"""
    set_random_seeds(42)
    data = SyntheticData(
        dim=5,
        s0=6,
        n_train=100,
        n_test=50,
        batch_size=16,
        sem_type="mlp",
        dag_type="ER",
    )
    data.setup()

    # Get train and test arrays
    X_train = data.train.dataset[data.train.indices][0].numpy()
    X_test = data.test.dataset[data.test.indices][0].numpy()

    return data, X_train, X_test


class TestConformalEvaluator:
    """Test the unified conformal evaluator with different mappers"""

    def test_initialization(self, conf_predictor_cfg):
        """Test evaluator initialization with different mappers"""
        # Test with DAG mapper
        evaluator = ConformalEvaluator(conf_predictor_cfg=conf_predictor_cfg)
        assert evaluator.list_conf is None

        # Test with PCA mapper
        evaluator = ConformalEvaluator(conf_predictor_cfg=conf_predictor_cfg)
        assert evaluator.list_conf is None

    def test_dag_based_workflow(self, conf_predictor_cfg, dataset_and_arrays):
        """Test the complete workflow with DAG-based representation"""
        data, X_train, X_test = dataset_and_arrays

        # Initialize mapper and evaluator
        dag_mapper = DAGRepresentationMapper()
        dag_mapper.set_dag(data.DAG)  # Set DAG before training

        evaluator = ConformalEvaluator(conf_predictor_cfg=conf_predictor_cfg)

        # Train
        alphas = {i: 0.1 for i in range(data.dim)}
        evaluator.train(
            X_train=X_train,
            representation_mapper=dag_mapper,
            list_features=np.arange(data.dim),
            alphas_adjusted=alphas,
        )

        assert evaluator.list_conf is not None
        assert len(evaluator.list_conf) == data.dim

        # Predict
        predictions = evaluator.predict(
            X_test=X_test, list_features=np.arange(data.dim)
        )

        assert isinstance(predictions, dict)
        assert len(predictions) == data.dim
        for feature in predictions:
            pred_dict = predictions[feature]
            assert all(
                k in pred_dict for k in ["min", "max", "true_val", "conf_interval"]
            )

    def test_pca_based_workflow(self, conf_predictor_cfg, dataset_and_arrays):
        """Test the complete workflow with PCA-based representation"""
        data, X_train, X_test = dataset_and_arrays

        # Initialize mapper and evaluator
        pca_mapper = PCARepresentationMapper(random_state=42)
        evaluator = ConformalEvaluator(conf_predictor_cfg=conf_predictor_cfg)

        # Train
        alphas = {i: 0.1 for i in range(data.dim)}
        evaluator.train(
            X_train=X_train,
            representation_mapper=pca_mapper,
            list_features=np.arange(data.dim),
            alphas_adjusted=alphas,
        )

        assert evaluator.list_conf is not None
        assert len(evaluator.list_conf) == data.dim

        # Predict
        predictions = evaluator.predict(
            X_test=X_test, list_features=np.arange(data.dim)
        )

        assert isinstance(predictions, dict)
        assert len(predictions) == data.dim
        for feature in predictions:
            pred_dict = predictions[feature]
            assert all(
                k in pred_dict for k in ["min", "max", "true_val", "conf_interval"]
            )

    def test_untrained_prediction(self, conf_predictor_cfg, dataset_and_arrays):
        """Test that prediction without training raises error"""
        _, _, X_test = dataset_and_arrays

        evaluator = ConformalEvaluator(conf_predictor_cfg=conf_predictor_cfg)

        with pytest.raises(ValueError, match="Model not trained"):
            evaluator.predict(X_test=X_test, list_features=np.arange(5))

    def test_invalid_alphas(self, conf_predictor_cfg, dataset_and_arrays):
        """Test training with missing alphas"""
        data, X_train, _ = dataset_and_arrays

        pca_mapper = PCARepresentationMapper(random_state=42)
        evaluator = ConformalEvaluator(conf_predictor_cfg=conf_predictor_cfg)

        # Missing alphas
        with pytest.raises(AssertionError, match="Alphas not set"):
            evaluator.train(
                X_train=X_train,
                representation_mapper=pca_mapper,
                list_features=np.arange(data.dim),
                alphas_adjusted=None,
            )

        # Incomplete alphas
        incomplete_alphas = {0: 0.1}  # Missing some features
        with pytest.raises(AssertionError, match="Alpha not set for feature"):
            evaluator.train(
                X_train=X_train,
                representation_mapper=pca_mapper,
                list_features=np.arange(data.dim),
                alphas_adjusted=incomplete_alphas,
            )
