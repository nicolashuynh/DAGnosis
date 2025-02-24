# third party
import pytest

# dagnosis absolute
from dagnosis.conformal.significance_correction import BonferroniCorrection


@pytest.fixture
def valid_dimension():
    return 10


@pytest.fixture
def valid_feature_indices():
    return [0, 2, 4, 6, 8]


class TestBonferroniCorrection:
    @pytest.mark.parametrize(
        "alpha,dimension,expected",
        [
            (0.05, 5, 0.01),  # typical case: 0.05/5
            (0.05, 10, 0.005),  # more features: 0.05/10
            (0.01, 5, 0.002),  # stricter alpha: 0.01/5
            (0.1, 20, 0.005),  # lenient alpha with many features: 0.1/20
        ],
    )
    def test_correction_values(self, alpha, dimension, expected):
        """Test that Bonferroni correction computes correct values."""
        indices = list(range(dimension))
        correction = BonferroniCorrection()
        result = correction.compute_correction(alpha, indices)

        # Check all values are equal to expected (within floating point precision)
        assert all(abs(val - expected) < 1e-10 for val in result.values())
        # Check all indices are present
        assert set(result.keys()) == set(indices)

    @pytest.mark.parametrize("alpha", [0, 1, -0.05, 1.5])
    def test_invalid_alpha(self, alpha, valid_dimension, valid_feature_indices):
        """Test that compute_correction fails with invalid alpha values."""
        correction = BonferroniCorrection()
        with pytest.raises(ValueError, match="significance must be between 0 and 1"):
            correction.compute_correction(alpha, valid_feature_indices)

    def test_subset_correction(self):
        """Test that correction works correctly with a subset of features."""
        feature_indices = [2, 5, 7]  # Only correct these features
        alpha = 0.05

        correction = BonferroniCorrection()
        result = correction.compute_correction(alpha, feature_indices)

        # Check only specified indices are present
        assert set(result.keys()) == set(feature_indices)
        # Check correction value
        expected = alpha / len(feature_indices)
        assert all(abs(val - expected) < 1e-10 for val in result.values())

    def test_edge_cases(self):
        """Test edge cases with different alpha values."""
        indices = [0, 1, 2, 3, 4]
        correction = BonferroniCorrection()

        # Test with common alpha values
        alpha_cases = {
            0.1: 0.02,  # lenient threshold
            0.05: 0.01,  # standard threshold
            0.01: 0.002,  # conservative threshold
            0.001: 0.0002,  # very conservative threshold
        }

        for alpha, expected in alpha_cases.items():
            result = correction.compute_correction(alpha, indices)
            assert all(abs(val - expected) < 1e-10 for val in result.values())
