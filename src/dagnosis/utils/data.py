# stdlib
import copy
from typing import Optional, Tuple

# third party
import numpy as np

# dagnosis absolute
import dagnosis.dag_learner.simulate as sm
from dagnosis.data.datamodule import SyntheticData


def sample_corrupted(
    D: SyntheticData,
    n_samples: int,
    list_feature: np.ndarray,
    list_corruption_type: list,
    noise_mean_list: Optional[np.ndarray] = None,
    mean_linear: float = 5.0,
    std_linear: float = 1.0,
    mean_mlp: float = 2.0,
    std_mlp: float = 1.0,
    sample_last_layer: bool = True,
) -> Tuple[np.ndarray, list, list]:
    """Sample a corrupted dataset where some SEMs have been corrupted."""

    list_corrupted_SEMs = copy.deepcopy(D.list_SEMs)
    list_corrupted_parameters = copy.deepcopy(D.list_parameters)

    assert D.DAG is not None, "No DAG simulated yet"
    for feature, corruption_type in zip(list_feature, list_corruption_type):
        pa_size = sm.find_parent_size(D.DAG, feature)
        list_corrupted_SEMs, list_corrupted_parameters = sm.modify_single_sem(
            pa_size,
            corruption_type,
            feature,
            list_corrupted_SEMs,
            list_corrupted_parameters,
            mean_linear=mean_linear,
            std_linear=std_linear,
            mean_mlp=mean_mlp,
            std_mlp=std_mlp,
            sample_last_layer=sample_last_layer,
        )
    assert D.DAG is not None, "No DAG simulated yet"
    X_test_corrupted = sm.simulate_sem_by_list(
        D.DAG,
        n_samples,
        list_corrupted_SEMs,
        noise_scale=None,
        noise_mean_list=noise_mean_list,
    )
    return X_test_corrupted, list_corrupted_SEMs, list_corrupted_parameters
