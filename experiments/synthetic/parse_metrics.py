# stdlib
import glob
import itertools
import os
import pickle

# third party
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    PATH_LOAD_METRICS = cfg.PATH_SAVE_METRIC

    rows_results = []

    for filename in sorted(glob.glob(PATH_LOAD_METRICS + "*")):
        with open(filename, "rb") as f:
            metrics_and_additional_info = pickle.load(f)

        sparsity = metrics_and_additional_info["additional_info"]["sparsity"]
        metrics_methods = metrics_and_additional_info["metrics"]

        # Iterate over the different methods and metrics
        # First extract the list of methods
        methods = list(
            set().union(
                *(metric_dict.keys() for metric_dict in metrics_methods.values())
            )
        )

        n_runs = cfg.n_runs_test
        for run, method in itertools.product(range(n_runs), methods):
            row_sparsity_metrics_method = {}
            row_sparsity_metrics_method["number of edges"] = sparsity
            row_sparsity_metrics_method["method"] = method
            for metric in metrics_methods:
                metric_value = metrics_methods[metric][method][run]
                row_sparsity_metrics_method[metric] = metric_value

            rows_results.append(row_sparsity_metrics_method)

    df_results = pd.DataFrame(rows_results)

    # Save the results

    # Create folder if does not exist
    if not os.path.exists(cfg.PATH_SAVE_PARSE_METRIC):
        os.makedirs(cfg.PATH_SAVE_PARSE_METRIC)

    df_results.to_csv(cfg.PATH_SAVE_PARSE_METRIC + "results.csv", index=False)

    for method in methods:
        print(method)
        for sparsity in cfg.sparsity_levels:
            print("#" * 42)
            print(f"s: {sparsity}")
            x = df_results[df_results["number of edges"] == sparsity]
            x_gt = x[x["method"] == method]

            for metric in metrics_methods:
                print(
                    metric,
                    round(np.mean(x_gt[metric]), 2),
                    round(1.96 / np.sqrt(len(x_gt)) * np.std(x_gt[metric]), 2),
                )


if __name__ == "__main__":
    main()
