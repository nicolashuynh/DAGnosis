"""
Data generation script for synthetic experiments.

This module generates synthetic datasets using
various structural equation models (SEM) and graph types.
"""

# stdlib
from pathlib import Path
from typing import Any, Dict

# third party
import dill

# dagnosis absolute
from dagnosis.data.datamodule import SyntheticData


class DataGenerator:
    """Handles the generation and storage of synthetic DAG learning datasets."""

    def __init__(
        self,
        sem_type: str,
        graph_type: str,
        save_path: str,
    ):
        """Initialize the data generator with explicit parameters."""

        self.sem_type = sem_type
        self.graph_type = graph_type
        self.save_path = Path(save_path)

        self._setup_paths()

    def _setup_paths(self) -> None:
        """Setup necessary directories and paths."""
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _get_unique_file_id(
        self, sparsity: float, dimension: int, n_train: int, sem_type: str
    ) -> str:
        """Generate a unique file identifier."""
        MAX_ATTEMPTS = int(1e5)  # Reasonable upper limit
        base_name = f"d_{dimension}_s_{sparsity}_n_{n_train}_sem_{sem_type}"

        for k in range(MAX_ATTEMPTS):
            id_str = f"id_{k}_{base_name}"
            if not (self.save_path / id_str).exists():
                return id_str

        raise RuntimeError(
            f"Failed to generate unique file ID after {MAX_ATTEMPTS} attempts"
        )

    def _generate_dataset(
        self, sparsity: int, dimension: int, n_train: int, n_test: int
    ) -> SyntheticData:
        """Generate a single dataset with given parameters."""

        dataset = SyntheticData(
            dim=dimension,
            s0=sparsity,
            sem_type=self.sem_type,
            dag_type=self.graph_type,
            n_train=n_train,
            n_test=n_test,
        )
        dataset.setup()
        return dataset

    def _prepare_data_dict(self, dataset: SyntheticData) -> Dict[str, Any]:

        return {
            "D": dataset,
        }

    def generate_and_save(
        self, sparsity: int, dimension: int, n_train: int, n_test: int
    ) -> None:
        """Generate and save datasets for all configurations."""

        dataset = self._generate_dataset(sparsity, dimension, n_train, n_test)
        data_dict = self._prepare_data_dict(dataset)
        file_id = self._get_unique_file_id(
            sparsity, dimension, n_train, sem_type=self.sem_type
        )

        with open(self.save_path / file_id, "wb") as f:
            dill.dump(data_dict, f)
