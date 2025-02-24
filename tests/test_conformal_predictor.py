# third party
import numpy as np
import pandas as pd
import pytest

# dagnosis absolute
from dagnosis.conformal.conformal_predictor import CQR, ConformalPredictor


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 3)  # 100 samples, 3 features
    y = 2 * X[:, 0] + X[:, 1] - 0.5 * X[:, 2] + np.random.randn(n_samples) * 0.1
    return X, y


@pytest.fixture
def cqr_predictor():
    """Create a CQR predictor instance with test parameters."""
    return CQR(
        alpha=0.1,
        scale=False,
        seed=42,
        cal_size=0.2,
        n_search=1,  # Reduced for testing
        n_splits=2,
        param_bounds={
            "num_leaves": {"low": 5, "high": 10},
            "max_depth": {"low": 2, "high": 5},
            "n_estimators": {"low": 10, "high": 20},
            "learning_rate": {"low": 0.1, "high": 0.3},
        },
    )


def test_cqr_initialization():
    """Test CQR initialization with default parameters."""
    cqr = CQR()
    assert cqr.alpha == 0.1
    assert cqr.scale is False
    assert cqr.seed == 42
    assert cqr.cal_size == 0.2
    assert cqr.n_search == 100
    assert cqr.n_splits == 5
    assert isinstance(cqr.param_bounds, dict)


def test_cqr_initialization_custom_params():
    """Test CQR initialization with custom parameters."""
    custom_bounds = {
        "num_leaves": {"low": 5, "high": 15},
        "max_depth": {"low": 3, "high": 10},
        "n_estimators": {"low": 20, "high": 50},
        "learning_rate": {"low": 0.01, "high": 0.5},
    }
    cqr = CQR(
        alpha=0.05, scale=True, seed=123, cal_size=0.3, param_bounds=custom_bounds
    )
    assert cqr.alpha == 0.05
    assert cqr.scale is True
    assert cqr.seed == 123
    assert cqr.cal_size == 0.3
    assert cqr.param_bounds == custom_bounds


def test_get_param_distributions(cqr_predictor):
    """Test parameter distribution generation."""
    distributions = cqr_predictor._get_param_distributions()
    assert len(distributions) == 4
    assert "num_leaves" in distributions
    assert "max_depth" in distributions
    assert "n_estimators" in distributions
    assert "learning_rate" in distributions


def test_fit_predict_workflow(cqr_predictor, sample_data):
    """Test the complete workflow of fitting and predicting."""
    X, y = sample_data
    X_test = X[:10]
    y_test = y[:10]

    # Fit the model
    cqr_predictor.fit(X, y)

    # Check if model attributes are set
    assert cqr_predictor.range_max is not None
    assert cqr_predictor.range_min is not None
    assert cqr_predictor.cp_model is not None

    # Make predictions
    predictions = cqr_predictor.predict(X_test, y_test)

    # Check prediction DataFrame structure
    assert isinstance(predictions, pd.DataFrame)
    assert all(
        col in predictions.columns
        for col in ["min", "max", "true_val", "conf_interval", "norm_interval"]
    )
    assert len(predictions) == len(X_test)

    # Check if predictions make sense
    assert (predictions["max"] >= predictions["min"]).all()
    assert (predictions["conf_interval"] >= 0).all()
    assert (predictions["norm_interval"] >= 0).all()
    assert (predictions["norm_interval"] <= 1).all()


def test_predict_without_fit(cqr_predictor, sample_data):
    """Test that prediction without fitting raises an error."""
    X, y = sample_data
    X_test = X[:10]
    y_test = y[:10]

    with pytest.raises(
        ValueError, match="Model must be fitted before making predictions"
    ):
        cqr_predictor.predict(X_test, y_test)


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_different_alpha_levels(alpha, sample_data):
    """Test model behavior with different alpha levels."""
    X, y = sample_data

    X_test = X[:10]
    y_test = y[:10]

    cqr = CQR(alpha=alpha, n_search=1, n_splits=2)
    cqr.fit(X, y)
    predictions = cqr.predict(X_test, y_test)

    # Higher alpha should generally lead to narrower intervals
    assert all(predictions["conf_interval"] >= 0)


def test_abstract_class():
    """Test that ConformalPredictor cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ConformalPredictor()
