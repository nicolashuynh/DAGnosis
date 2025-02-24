# stdlib
import copy
import os
from abc import abstractmethod
from typing import Optional, Tuple, Union

# third party
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset as TorchSubset
from torch.utils.data import TensorDataset, random_split

# dagnosis absolute
import dagnosis.dag_learner.simulate as sm
from dagnosis.utils.dag import get_adult_DAG
from dagnosis.utils.data_loader import load_adult, preprocess_adult


class Data(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_train(self):
        pass


class SyntheticData(Data):
    def __init__(
        self,
        dim: int = 20,  # Amount of vars
        s0: int = 40,  # Expected amount of edges
        sem_type: str = "mim",  # SEM-type (
        #   'mim' -> index model,
        #   'mlp' -> multi layer perceptrion,
        #   'gp' -> gaussian process,
        #   'gp-add' -> additive gp)
        dag_type: str = "ER",  # Random graph type (
        #   'ER' -> Erdos-Renyi,
        #   'SF' -> Scale Free,
        #   'BP' -> BiPartite)
        n_train: int = 1000,
        n_test: int = 10000,
        seed: int = 42,
        batch_size: int = 32,
    ):

        super().__init__()
        self.prepare_data_per_node = False
        self.dim = dim
        self.s0 = s0
        self.sem_type = sem_type
        self.dag_type = dag_type

        self.n_train = n_train
        self.n_test = n_test
        self.N = self.n_train + self.n_test

        self.batch_size = (
            batch_size  # Will be used for NOTEARS training (DAG discovery)
        )

        self.seed = seed
        self.DAG: Optional[np.ndarray] = None

        self.train: TorchSubset
        self.test: TorchSubset

        self._simulate()
        self._generate_SEM()
        self._sample()

    def _get_dim(self):
        return self.dim

    def _get_s0(self):
        return self.s0

    def _simulate(self) -> None:
        self.DAG = sm.simulate_dag(self.dim, self.s0, self.dag_type)
        assert self.DAG is not None, "DAG not simulated"
        self._id = hash(
            self.DAG.__repr__() + self.DAG.__array_interface__["data"][0].__repr__()
        )

    def _generate_SEM(self):
        self.list_SEMs, self.list_parameters = sm.generate_list_SEM(
            self.DAG, self.sem_type
        )
        self.list_corrupted_SEMs = copy.deepcopy(self.list_SEMs)

    def _sample(self) -> None:
        assert self.DAG is not None, "No DAG simulated yet"
        assert self.list_SEMs is not None, "SEMs not simulated yet"
        self.X = sm.simulate_sem_by_list(
            self.DAG, self.N, self.list_SEMs, noise_scale=None
        )

    def setup(self, stage: Optional[str] = None) -> None:
        assert self.DAG is not None, "No DAG simulated yet"
        assert self.X is not None, "No SEM simulated yet"

        DX = TensorDataset(torch.from_numpy(self.X).float())

        self.train, self.test = random_split(
            DX,
            [self.n_train, self.n_test],
            generator=torch.Generator().manual_seed(self.seed),
        )

    def resample(self) -> None:
        """
        Resamples a new DAG and SEM
        Resets the train and test sets

        """
        self._simulate()
        self._sample()
        self.setup()

    def train_dataloader(self) -> DataLoader:
        cpu_count = os.cpu_count()
        assert cpu_count is not None, "cpu_count is None"
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=cpu_count // 3
        )

    def test_dataloader(self) -> DataLoader:
        num_workers = os.cpu_count()
        assert num_workers is not None, "num_workers is None"
        return DataLoader(self.test, batch_size=len(self.test), num_workers=num_workers)

    def get_train(self):
        X_train = self.train.dataset[self.train.indices][0].numpy()
        return X_train

    def get_test(self):
        X_test = self.test.dataset[self.test.indices][0].numpy()
        return X_test


class Subset(pl.LightningDataModule):
    def __init__(
        self, X: np.ndarray, train_size_ratio: float = 0.5, batch_size: int = 256
    ) -> None:
        super().__init__()

        self.train_size_ratio = train_size_ratio
        self.batch_size = batch_size

        self.X = X
        self.N = self.X.shape[0]

        self.train: TorchSubset
        self.test: TorchSubset

    def setup(self):
        DX = TensorDataset(torch.from_numpy(self.X))

        _train_size = np.floor(self.N * self.train_size_ratio)
        self.train, self.test = random_split(
            DX, [int(_train_size), int(self.N - _train_size)]
        )

    def train_dataloader(self) -> DataLoader:
        num_workers = os.cpu_count()
        assert num_workers is not None, "num_workers is None"
        return DataLoader(
            self.train, batch_size=self.batch_size, num_workers=num_workers
        )

    def test_dataloader(self) -> DataLoader:
        num_workers = os.cpu_count()
        assert num_workers is not None, "num_workers is None"
        return DataLoader(
            self.test, batch_size=self.batch_size, num_workers=num_workers
        )


class AdultIncome(Data):
    def __init__(
        self,
        test_size: float = 0.2,
        seed: int = 42,
    ):

        super().__init__()

        self.test_size = test_size

        dataset = load_adult()
        dataset = preprocess_adult(dataset)
        self.dataset = dataset

        self.set_list_edges()

        self.seed = seed
        self.set_train_test()

    def _get_dim(self) -> int:
        return len(self.dataset.columns) - 1  # To remove the target variable

    def get_dag(self, dag_type: str) -> np.ndarray:
        name_features = self.dataset.columns.tolist()
        # Remove the target
        name_features.remove("income")
        if dag_type == "gt":
            A = get_adult_DAG(name_features, list_edges=None)

        elif dag_type == "pc":
            A = get_adult_DAG(name_features, list_edges=None)

        else:
            raise ValueError(f"Unknown DAG type {dag_type}")

        return A

    def set_list_edges(self):
        name_features = self.dataset.columns.tolist()
        feature_to_index = {}
        for i in range(len(name_features)):
            feature_to_index[name_features[i]] = i

        list_edges_name = [
            ["age", "workclass"],
            ["age", "marital-status"],
            ["age", "occupation"],
            ["age", "relationship"],
            ["age", "capital-gain"],
            ["age", "capital-loss"],
            ["age", "hours-per-week"],
            ["marital-status", "occupation"],
            ["marital-status", "relationship"],
            ["marital-status", "hours-per-week"],
            ["race", "workclass"],
            ["race", "fnlwgt"],
            ["race", "marital-status"],
            ["race", "occupation"],
            ["race", "relationship"],
            ["sex", "workclass"],
            ["sex", "marital-status"],
            ["sex", "occupation"],
            ["sex", "relationship"],
            ["sex", "capital-gain"],
            ["native-country", "marital-status"],
            ["native-country", "occupation"],
            ["native-country", "relationship"],
        ]
        list_edges = [
            [feature_to_index[feat1], feature_to_index[feat2]]
            for (feat1, feat2) in list_edges_name
        ]

        self.list_edges = list_edges

    def set_train_test(self):
        df_sex_1 = self.dataset.query("sex ==1")

        salary_1_idx = self.dataset.query("sex == 0 & income == 1")
        salary_0_idx = self.dataset.query("sex == 0 & income == 0")
        X = df_sex_1.drop(["income"], axis=1)
        y = df_sex_1["income"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=True, random_state=self.seed
        )

        X_train = np.vstack([X_train, salary_0_idx.drop(["income"], axis=1)]).astype(
            np.float64
        )

        y_train = np.hstack([y_train, salary_0_idx["income"]])

        self.X_train = X_train

        self.y_train = y_train

        self.X_test_sex_0 = X_test
        self.y_test_sex_0 = y_test

        self.salary_1_idx = salary_1_idx

    def get_train(
        self, return_labels: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        assert isinstance(self.X_train, np.ndarray), "X_train is not an array"
        if return_labels:
            return self.X_train, self.y_train

        return self.X_train

    def get_test(
        self, n_women: int, return_labels: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

        X_test = np.vstack(
            [self.X_test_sex_0, self.salary_1_idx.drop(["income"], axis=1)[:n_women]]
        ).astype(np.float64)

        if return_labels:
            y_test = np.hstack(
                [self.y_test_sex_0, self.salary_1_idx["income"][:n_women]]
            )

            return X_test, y_test
        return X_test
