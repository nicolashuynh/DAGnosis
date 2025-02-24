# stdlib
import logging

# third party
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

# dagnosis absolute
from dagnosis.representation.representation_mapper import (
    DAGRepresentationMapper,
    PCARepresentationMapper,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    X = np.random.randn(100, 5)  # 100 samples, 5 features
    return X


@pytest.fixture
def sample_dag():
    """Create a sample DAG for testing"""
    # Simple chain DAG: 0 -> 1 -> 2 -> 3 -> 4
    dag = np.zeros((5, 5))
    for i in range(4):
        dag[i, i + 1] = 1
    return dag


@pytest.fixture
def logger():
    """Create a logger for testing"""
    return logging.getLogger("test_logger")


class TestDAGRepresentationMapper:
    def test_initialization(self, logger):
        mapper = DAGRepresentationMapper(logger=logger)
        assert mapper.dag is None
        assert mapper.markov_boundaries is None
        assert not mapper._is_fitted
        assert mapper.logger == logger

    def test_set_dag(self, sample_dag):
        mapper = DAGRepresentationMapper()
        mapper.set_dag(sample_dag)
        assert_array_equal(mapper.dag, sample_dag)

    def test_fit_without_dag(self, sample_data):
        mapper = DAGRepresentationMapper()
        with pytest.raises(ValueError, match="DAG not set"):
            mapper.fit(sample_data)

    def test_fit_mismatched_dimensions(self, sample_data):
        mapper = DAGRepresentationMapper()
        wrong_size_dag = np.zeros((3, 3))
        mapper.set_dag(wrong_size_dag)
        with pytest.raises(ValueError, match="DAG shape does not match"):
            mapper.fit(sample_data)

    def test_fit_success(self, sample_data, sample_dag):
        mapper = DAGRepresentationMapper()
        mapper.set_dag(sample_dag)
        mapper.fit(sample_data)
        assert mapper._is_fitted
        assert mapper.markov_boundaries is not None
        assert len(mapper.markov_boundaries) == sample_data.shape[1]

    def test_transform_feature_not_fitted(self, sample_data):
        mapper = DAGRepresentationMapper()
        mapper.set_dag(np.zeros((5, 5)))
        with pytest.raises(ValueError, match="Mapper not fitted"):
            mapper.transform_feature(sample_data, 0)

    def test_transform_feature(self, sample_data, sample_dag):
        mapper = DAGRepresentationMapper()
        mapper.set_dag(sample_dag)
        mapper.fit(sample_data)

        # Test transform for middle node (should have two neighbors)
        X_transformed, y = mapper.transform_feature(sample_data, 2)
        assert y.shape == (sample_data.shape[0],)
        assert_array_equal(y, sample_data[:, 2])
        # No need to assert specific number of neighbors - let find_markov_boundary determine it
        assert X_transformed.shape[0] == sample_data.shape[0]


class TestPCARepresentationMapper:
    def test_initialization(self, sample_data):
        mapper = PCARepresentationMapper(random_state=42, compression_factor=0.6)
        assert mapper.random_state == 42
        assert mapper.compression_factor == 0.6
        assert not mapper._is_fitted
        assert mapper.n_components is None

    def test_fit(self, sample_data):
        mapper = PCARepresentationMapper(random_state=42)
        mapper.fit(sample_data)
        assert mapper._is_fitted
        assert mapper.pca is not None
        assert mapper.pca_transform is not None
        assert mapper.pca_transform.shape[0] == sample_data.shape[0]
        assert mapper.pca_transform.shape[1] == mapper.n_components
        assert mapper.n_components == int(
            np.ceil(sample_data.shape[1] * mapper.compression_factor)
        )

    def test_transform_feature_not_fitted(self, sample_data):
        mapper = PCARepresentationMapper()
        with pytest.raises(ValueError, match="Mapper not fitted"):
            mapper.transform_feature(sample_data, 0)

    def test_transform_feature(self, sample_data):
        mapper = PCARepresentationMapper(random_state=42)
        mapper.fit(sample_data)

        X_transformed, y = mapper.transform_feature(sample_data, 2)
        assert y.shape == (sample_data.shape[0],)
        assert_array_equal(y, sample_data[:, 2])
        assert X_transformed.shape == (sample_data.shape[0], mapper.n_components)

    def test_reproducibility(self, sample_data):
        """Test that same random state produces same results"""
        mapper1 = PCARepresentationMapper(random_state=42)
        mapper2 = PCARepresentationMapper(random_state=42)

        mapper1.fit(sample_data)
        mapper2.fit(sample_data)

        X_transformed1, _ = mapper1.transform_feature(sample_data, 0)
        X_transformed2, _ = mapper2.transform_feature(sample_data, 0)

        assert_array_almost_equal(X_transformed1, X_transformed2)

    def test_different_compression_factors(self, sample_data):
        mapper1 = PCARepresentationMapper(compression_factor=0.4)
        mapper2 = PCARepresentationMapper(compression_factor=0.8)

        mapper1.fit(sample_data)
        mapper2.fit(sample_data)

        X_transformed1, _ = mapper1.transform_feature(sample_data, 0)
        X_transformed2, _ = mapper2.transform_feature(sample_data, 0)

        assert X_transformed1.shape[1] < X_transformed2.shape[1]
