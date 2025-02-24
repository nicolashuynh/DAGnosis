# stdlib
import logging
from typing import Any, Dict, Optional

# third party
import numpy as np
import pytest

# dagnosis absolute
from dagnosis.conformal.significance_correction import NoCorrection
from dagnosis.data.datamodule import SyntheticData
from dagnosis.dcai.conformal_pipeline import (
    DAGBasedConformalPipeline,
    PCABasedConformalPipeline,
)


class MockRepresentationMapper:
    def __init__(self):
        self.dag = None
        self.fit_called = False

    def set_dag(self, dag):
        self.dag = dag
        return self

    def fit(self, X):
        self.fit_called = True
        return self


class MockConformalEvaluator:
    """Mock of the unified ConformalEvaluator"""

    def __init__(self):
        self.train_called = False
        self.predict_called = False

    def train(
        self,
        X_train: np.ndarray,
        representation_mapper: MockRepresentationMapper,
        list_features: Optional[np.ndarray] = None,
        alphas_adjusted: Optional[Dict[int, float]] = None,
    ):
        self.train_called = True
        representation_mapper.fit(X_train)
        self.X_train = X_train
        self.list_features = list_features
        self.alphas_adjusted = alphas_adjusted
        return self

    def predict(self, X_test: np.ndarray, list_features: np.ndarray) -> Dict[int, Any]:
        self.predict_called = True
        return {0: {"min": 1, "max": 3, "true_val": 2, "conf_interval": [1, 3]}}


class MockDAGExtractor:
    def extract(self, D) -> np.ndarray:
        return np.eye(D._get_dim())


@pytest.fixture
def mock_representation_mapper():
    return MockRepresentationMapper()


@pytest.fixture
def mock_conformal_evaluator():
    return MockConformalEvaluator()


@pytest.fixture
def mock_dag_extractor():
    return MockDAGExtractor()


@pytest.fixture
def significance_correction():
    return NoCorrection()


@pytest.fixture
def sample_data():
    # Create sample training data
    dim = 5

    D = SyntheticData(
        dim=dim,  # Small dimension
        s0=2,  # Few edges
        n_train=100,  # Small sample size
        n_test=50,
        batch_size=16,
        sem_type="mlp",
        dag_type="ER",
    )

    D.setup()

    list_features = np.arange(D._get_dim())

    X_test = D.get_test()
    return D, list_features, X_test


def test_dag_based_pipeline_initialization(
    mock_conformal_evaluator,
    mock_dag_extractor,
    mock_representation_mapper,
    significance_correction,
):
    pipeline = DAGBasedConformalPipeline(
        significance_correction=significance_correction,
        conformal_evaluator=mock_conformal_evaluator,
        representation_mapper=mock_representation_mapper,
        dag_extractor=mock_dag_extractor,
    )

    assert pipeline.conformal_evaluator == mock_conformal_evaluator
    assert pipeline.dag_extractor == mock_dag_extractor
    assert isinstance(pipeline.logger, logging.Logger)


def test_dag_based_pipeline_get_dag(
    mock_conformal_evaluator,
    mock_dag_extractor,
    mock_representation_mapper,
    sample_data,
    significance_correction,
):
    D, _, _ = sample_data
    pipeline = DAGBasedConformalPipeline(
        significance_correction=significance_correction,
        conformal_evaluator=mock_conformal_evaluator,
        representation_mapper=mock_representation_mapper,
        dag_extractor=mock_dag_extractor,
    )

    dag = pipeline.get_dag(D)
    assert len(dag) == D._get_dim()
    assert np.isclose(dag, np.eye(D._get_dim())).all()


def test_dag_based_pipeline_train(
    mock_conformal_evaluator,
    mock_dag_extractor,
    mock_representation_mapper,
    sample_data,
    significance_correction,
):
    D, list_features, _ = sample_data
    pipeline = DAGBasedConformalPipeline(
        significance_correction=significance_correction,
        conformal_evaluator=mock_conformal_evaluator,
        representation_mapper=mock_representation_mapper,
        dag_extractor=mock_dag_extractor,
    )

    result = pipeline.train(D, list_features)

    # Test that DAG was set in the mapper
    assert mock_representation_mapper.dag is not None
    assert np.array_equal(
        mock_representation_mapper.dag,
        mock_dag_extractor.extract(D),
    )

    # Test that evaluator was trained
    assert mock_conformal_evaluator.train_called
    assert mock_conformal_evaluator.X_train.shape == (100, 5)
    assert np.array_equal(mock_conformal_evaluator.list_features, list_features)

    # Test that significance correction was applied
    assert isinstance(mock_conformal_evaluator.alphas_adjusted, dict)

    assert result == pipeline


def test_dag_based_pipeline_predict(
    mock_conformal_evaluator,
    mock_dag_extractor,
    mock_representation_mapper,
    sample_data,
    significance_correction,
):
    D, list_features, X_test = sample_data
    pipeline = DAGBasedConformalPipeline(
        significance_correction=significance_correction,
        conformal_evaluator=mock_conformal_evaluator,
        representation_mapper=mock_representation_mapper,
        dag_extractor=mock_dag_extractor,
    )

    pipeline.train(D, list_features)
    predictions = pipeline.predict(X_test, list_features)

    assert mock_conformal_evaluator.predict_called
    assert isinstance(predictions, dict)
    assert 0 in predictions
    pred_dict = predictions[0]
    assert all(k in pred_dict for k in ["min", "max", "true_val", "conf_interval"])


def test_pca_based_pipeline_initialization(
    mock_conformal_evaluator, mock_representation_mapper, significance_correction
):
    pipeline = PCABasedConformalPipeline(
        significance_correction=significance_correction,
        conformal_evaluator=mock_conformal_evaluator,
        representation_mapper=mock_representation_mapper,
    )

    assert pipeline.conformal_evaluator == mock_conformal_evaluator
    assert isinstance(pipeline.logger, logging.Logger)


def test_pca_based_pipeline_train(
    mock_conformal_evaluator,
    mock_representation_mapper,
    sample_data,
    significance_correction,
):
    D, list_features, _ = sample_data
    pipeline = PCABasedConformalPipeline(
        significance_correction=significance_correction,
        conformal_evaluator=mock_conformal_evaluator,
        representation_mapper=mock_representation_mapper,
    )

    result = pipeline.train(D, list_features)

    # Test that evaluator was trained
    assert mock_conformal_evaluator.train_called
    assert mock_conformal_evaluator.X_train.shape == (100, 5)
    assert np.array_equal(mock_conformal_evaluator.list_features, list_features)

    # Test that significance correction was applied
    assert isinstance(mock_conformal_evaluator.alphas_adjusted, dict)

    assert result == pipeline


def test_pca_based_pipeline_predict(
    mock_conformal_evaluator,
    mock_representation_mapper,
    sample_data,
    significance_correction,
):
    D, list_features, X_test = sample_data
    pipeline = PCABasedConformalPipeline(
        significance_correction=significance_correction,
        representation_mapper=mock_representation_mapper,
        conformal_evaluator=mock_conformal_evaluator,
    )

    pipeline.train(D, list_features)
    predictions = pipeline.predict(X_test, list_features)

    assert mock_conformal_evaluator.predict_called
    assert isinstance(predictions, dict)
    assert 0 in predictions
    pred_dict = predictions[0]
    assert all(k in pred_dict for k in ["min", "max", "true_val", "conf_interval"])
