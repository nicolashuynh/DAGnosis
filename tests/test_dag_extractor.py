# third party
import numpy as np
import pytest
from pytorch_lightning import seed_everything

# dagnosis absolute
# Import your modules
from dagnosis.dag_learner.extractor import (
    AutoregressiveDAG,
    DAGExtractor,
    GroundTruthDAG,
    NotearsLinearDAG,
)
from dagnosis.data.datamodule import SyntheticData


@pytest.fixture(scope="module")
def small_data():
    """Create a small dataset for quick tests"""
    seed_everything(42)
    data = SyntheticData(
        dim=5,  # Small dimension
        s0=6,  # Few edges
        n_train=100,  # Small sample size
        n_test=50,
        batch_size=16,
        sem_type="mlp",
        dag_type="ER",
    )

    data.setup()
    return data


def test_data_initialization():
    """Test Data class initialization and basic properties"""
    data = SyntheticData(dim=5, s0=6, n_train=100, n_test=50)

    # Check basic attributes
    assert data.dim == 5
    assert data.s0 == 6
    assert data.n_train == 100
    assert data.n_test == 50
    assert data.N == 150

    # Check DAG properties
    assert data.DAG is not None
    assert data.DAG.shape == (5, 5)
    assert np.all(np.diag(data.DAG) == 0)  # No self-loops

    # Check data generation
    assert data.X is not None
    assert data.X.shape == (150, 5)


def test_ground_truth_dag(small_data):
    """Test GroundTruthDAG extractor"""
    extractor = GroundTruthDAG()
    assert extractor.name == "gt"

    result = extractor.extract(small_data)
    assert isinstance(result, np.ndarray)
    assert result.shape == (small_data.dim, small_data.dim)
    assert np.array_equal(result, small_data.DAG)


def test_autoregressive_dag(small_data):
    """Test AutoregressiveDAG extractor"""
    extractor = AutoregressiveDAG()
    assert extractor.name == "autoregressive"

    result = extractor.extract(small_data)
    assert isinstance(result, np.ndarray)
    assert result.shape == (small_data.dim, small_data.dim)
    assert np.allclose(result, np.tril(np.ones((small_data.dim, small_data.dim)), -1))


def test_notears_linear_structural_properties(small_data):
    """Test structural properties of NotearsLinearDAG output"""
    extractor = NotearsLinearDAG(lambda1=0.1, max_iter=10, w_threshold=0.3)
    result = extractor.extract(small_data)

    # Test basic properties
    assert isinstance(result, np.ndarray)
    assert result.shape == (small_data.dim, small_data.dim)
    assert np.all(np.diag(result) == 0)  # No self-loops
    assert np.all(np.abs(result) <= 1.0)  # Bounded weights
    assert not np.any(np.isnan(result))  # No NaN values
    assert not np.any(np.isinf(result))  # No infinite values


def test_error_cases():
    """Test various error cases"""
    with pytest.raises(TypeError):
        DAGExtractor()  # Should raise error as it's abstract

    # Test with invalid parameters
    with pytest.raises(Exception):
        SyntheticData(dim=-1)  # Invalid dimension


@pytest.mark.parametrize("sem_type", ["mim", "mlp"])
def test_different_sem_types(sem_type):
    """Test deterministic extractors with different SEM types"""
    data = SyntheticData(dim=5, s0=6, sem_type=sem_type, n_train=100, n_test=50)

    # Test only deterministic extractors
    extractors = [GroundTruthDAG(), AutoregressiveDAG()]

    for extractor in extractors:
        result = extractor.extract(data)
        assert result.shape == (5, 5)
