# stdlib
import glob
import os
import pickle
from collections import defaultdict
from typing import Any, Optional, TypeAlias

# third party
import dill
import hydra
import numpy as np
from omegaconf import DictConfig

# dagnosis absolute
from dagnosis.utils.dag import sample_nodes_to_corrupt
from dagnosis.utils.data import sample_corrupted
from dagnosis.utils.logger import setup_logger
from dagnosis.utils.metrics import MetricsComputer
from dagnosis.utils.seed import set_random_seeds

MetricsDict: TypeAlias = dict[str, dict[str, list[float]]]


def get_unique_run_id(base_path: str, filename: str, max_iter: int = 1000) -> int:
    """Get a unique run ID for saving results."""
    for k in range(max_iter):
        if str(os.path.join(base_path, f"run_{k}_{filename}")) not in glob.glob(
            base_path + "*"
        ):
            return k
    raise RuntimeError(
        f"Could not find unique run ID after {max_iter} attempts. "
        f"Check directory: {base_path}"
    )


def save_results(
    base_path: str, filename: str, data: Any, run_id: Optional[int] = None
) -> None:
    """Save results to a file with optional run ID."""
    if run_id is not None:
        filename = f"run_{run_id}_{filename}"

    filepath = os.path.join(base_path, filename)
    with open(filepath, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):

    logger = setup_logger("test_cp")

    PATH_LOAD = cfg.PATH_SAVE_CP
    PATH_SAVE_confdict = cfg.PATH_SAVE_CONFDICT
    PATH_SAVE_metric = cfg.PATH_SAVE_METRIC

    # Create folder if does not exist
    if not os.path.exists(PATH_SAVE_confdict):
        os.makedirs(PATH_SAVE_confdict)
    if not os.path.exists(PATH_SAVE_metric):
        os.makedirs(PATH_SAVE_metric)

    set_random_seeds(cfg.random_seed)

    n_runs = cfg.n_runs_test
    n_samples_test_corrupted = cfg.n_samples_test_corrupted
    n_nodes_corrupted = cfg.n_nodes_corrupted

    metrics_computer = MetricsComputer()

    for filename in glob.glob(PATH_LOAD + "*"):
        # Initialize result containers
        metrics_methods_results: MetricsDict = {}
        logger.info(filename)

        with open(filename, "rb") as f:
            training_artefacts = dill.load(f)

        # Load the conformal evaluators
        evaluators = training_artefacts["evaluators"]

        # Load the data module
        D = training_artefacts["D"]
        dimension = D._get_dim()
        sparsity = D._get_s0()

        ground_truth_dag = D.DAG
        X_test_clean = D.get_test()

        assert len(X_test_clean) == cfg.n_test

        result_filename = filename.split("/")[-1]

        corruption_type = cfg.corruption_type
        noise_mean_list = np.zeros(dimension)
        features_test = np.arange(dimension)

        for run in range(n_runs):
            list_features_corruption = sample_nodes_to_corrupt(
                ground_truth_dag, n_nodes_corrupted
            )
            logger.info(f"Corrupted nodes: {list_features_corruption}")

            list_corruption_type = [corruption_type] * len(list_features_corruption)

            X_test_corrupted, _, _ = sample_corrupted(
                D,
                n_samples_test_corrupted,
                list_features_corruption,
                list_corruption_type,
                noise_mean_list=noise_mean_list,
            )

            conformal_dicts = {}

            # Compute metrics for each method (e.g. GT, AUTO, NOTEARS, ...) using the evaluators
            for method, evaluator in evaluators.items():

                # Compute the predictions, and then compute the metrics
                conformal_dict_clean = evaluator.predict(X_test_clean, features_test)
                conformal_dict_corrupted = evaluator.predict(
                    X_test_corrupted, features_test
                )

                method_metrics, conf_dict = metrics_computer.compute_metrics(
                    conformal_dict_clean, conformal_dict_corrupted
                )

                # Store results
                for name_metric in method_metrics:
                    if name_metric not in metrics_methods_results:
                        metrics_methods_results[name_metric] = defaultdict(list)
                    metrics_methods_results[name_metric][method].append(
                        method_metrics[name_metric]
                    )

                conformal_dicts[method] = conf_dict

            # Save conformal dictionaries for this run, this progressively overwrites
            try:
                run_id = get_unique_run_id(
                    PATH_SAVE_confdict, result_filename, max_iter=1000
                )
                save_results(
                    PATH_SAVE_confdict, result_filename, conformal_dicts, run_id=run_id
                )
            except RuntimeError as e:
                logger.error(f"Failed to save results: {e}")
                continue

        # Check that the length of the metrics is the same as the number of runs for each metrics and method combination
        for name_metric in metrics_methods_results:
            for method in metrics_methods_results[name_metric]:
                assert (
                    len(metrics_methods_results[name_metric][method]) == n_runs
                ), f"Got {len(metrics_methods_results[name_metric][method])} runs for {name_metric} and {method}, expected {n_runs}"

        additional_info = {"sparsity": sparsity}
        metrics_results = {
            "metrics": metrics_methods_results,
            "additional_info": additional_info,
        }
        # Save aggregated metrics for all runs
        save_results(PATH_SAVE_metric, result_filename, metrics_results)
        logger.info(f"Results saved for {result_filename}")


if __name__ == "__main__":
    main()
