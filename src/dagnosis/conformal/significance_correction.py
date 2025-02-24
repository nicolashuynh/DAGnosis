# stdlib
from abc import ABC, abstractmethod
from typing import Dict

# third party
import numpy as np


class MultipleTestingCorrection(ABC):
    """Abstract base class for multiple testing correction methods."""

    def check(self, significance: float, list_of_features: np.ndarray):
        if not 0 < significance < 1:
            raise ValueError("significance must be between 0 and 1")

        if list_of_features is None:
            raise ValueError("list_of_features cannot be empty")

    @abstractmethod
    def compute_correction(
        self, significance: float, list_of_features: np.ndarray
    ) -> Dict[int, float]:
        """
        Compute the corrected significance levels for each feature.
        """


class BonferroniCorrection(MultipleTestingCorrection):
    """
    Bonferroni correction for multiple testing.
    Controls the familywise error rate (FWER) at level alpha.
    """

    def compute_correction(
        self, significance: float, list_of_features: np.ndarray
    ) -> Dict[int, float]:
        """
        Compute Bonferroni-corrected significance levels.
        alpha_corrected = alpha / m, where m is the number of tests.
        """
        self.check(significance, list_of_features)
        correction = significance / len(list_of_features)
        return {idx: correction for idx in list_of_features}


class NoCorrection(MultipleTestingCorrection):
    """
    No correction for multiple testing.
    Returns the original significance level for all features.
    """

    def compute_correction(
        self, significance: float, list_of_features: np.ndarray
    ) -> Dict[int, float]:
        """
        Return uncorrected significance levels (same as input significance).
        """
        self.check(significance, list_of_features)
        return {idx: significance for idx in list_of_features}
