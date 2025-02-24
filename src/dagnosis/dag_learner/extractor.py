# stdlib
from abc import ABC, abstractmethod

# third party
import numpy as np
from numpy.typing import NDArray

# dagnosis absolute
from dagnosis.dag_learner.linear import notears_linear
from dagnosis.data.datamodule import AdultIncome, SyntheticData


class DAGExtractor(ABC):
    """Abstract base class for DAG extraction methods."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the DAG extraction method."""

    @abstractmethod
    def extract(self, *args, **kwargs) -> np.ndarray:
        pass


class GroundTruthDAG(DAGExtractor):
    """Extracts ground truth DAG from data."""

    @property
    def name(self) -> str:
        return "gt"

    def extract(self, data: SyntheticData) -> np.ndarray:
        assert data.DAG is not None, "Ground truth DAG not provided in data"
        return data.DAG


class AutoregressiveDAG(DAGExtractor):
    """Creates an autoregressive (triangular) DAG."""

    @property
    def name(self) -> str:
        return "autoregressive"

    def extract(self, data: SyntheticData) -> NDArray[np.float64]:
        matrix: NDArray[np.float64] = np.tril(np.ones(data.dim, dtype=np.float64), -1)
        return matrix


class NotearsLinearDAG(DAGExtractor):
    """Extracts DAG using linear NOTEARS algorithm."""

    @property
    def name(self) -> str:
        return "notears_linear"

    def __init__(
        self,
        lambda1: float = 0.1,
        loss_type: str = "l2",
        max_iter: int = 100,
        h_tol: float = 1e-8,
        rho_max: float = 1e16,
        w_threshold: float = 0.3,
    ):

        # Checks on the input parameters
        assert lambda1 >= 0, "lambda1 must be non-negative"
        assert max_iter > 0, "max_iter must be positive"

        self.lambda1 = lambda1
        self.loss_type = loss_type
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max
        self.w_threshold = w_threshold

    def extract(self, data: SyntheticData) -> np.ndarray:
        X_train = data.get_train()

        return notears_linear(
            X_train,
            self.lambda1,
            self.loss_type,
            max_iter=self.max_iter,
            h_tol=self.h_tol,
            rho_max=self.rho_max,
            w_threshold=self.w_threshold,
        )


class AdultIncomeGtDAG(DAGExtractor):
    """Extracts DAG for UCI Adult Income dataset."""

    @property
    def name(self) -> str:
        return "adult_income_gt"

    def extract(self, D: AdultIncome) -> np.ndarray:
        A = D.get_dag(dag_type="gt")
        return A


class AdultIncomePcDAG(DAGExtractor):
    """Extracts DAG for UCI Adult Income dataset."""

    @property
    def name(self) -> str:
        return "adult_income_pc"

    def extract(self, D: AdultIncome) -> np.ndarray:
        A = D.get_dag(dag_type="pc")
        return A
