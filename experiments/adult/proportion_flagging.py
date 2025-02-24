# third party
import dill
import numpy as np
import pandas as pd

# dagnosis absolute
from dagnosis.data.datamodule import AdultIncome


def load_artifact_data(seed: int):
    """Load artifact data for a specific seed."""
    with open(f"artifacts_adult/artifacts_final_seed_{seed}", "rb") as f:
        return dill.load(f)


def collect_accuracy_data(seeds: list):
    """Collect accuracy data for all seeds and methods."""
    accuracy_records = []

    for seed in seeds:
        results = load_artifact_data(seed)
        accuracy_dict = results["accuracy"]

        # Get all methods from the dictionary directly
        methods = [m for m in accuracy_dict.keys()]

        for method in methods:
            accuracies = accuracy_dict[method]

            for k, accuracy in enumerate(accuracies):
                accuracy_records.append(
                    {"seed": seed, "method": method, "k": k, "accuracy": accuracy}
                )

    return pd.DataFrame(accuracy_records)


def collect_proportion_data(seeds: list, k: int = 5, n_test: int = 15264 + 1000 * 5):
    """Collect proportion data for flagged samples."""
    proportion_records = []

    for seed in seeds:
        results = load_artifact_data(seed)
        detection_all_dict = results["detection_all"]
        detection_dict = results["detection"]

        # Get methods directly from the dictionaries
        methods = [m for m in detection_dict.keys()]

        for method in methods:
            # Extract detection counts for the specific k
            detected_women = np.array(detection_all_dict[method])[k]
            detected_all = np.array(detection_dict[method])[k]

            # Calculate men by subtraction
            detected_men = detected_all - detected_women

            # Create records
            proportion_records.append(
                {
                    "seed": seed,
                    "method": method,
                    "k": k,
                    "f": "women",
                    "proportion": detected_women / n_test,
                }
            )
            proportion_records.append(
                {
                    "seed": seed,
                    "method": method,
                    "k": k,
                    "f": "men",
                    "proportion": detected_men / n_test,
                }
            )

    return pd.DataFrame(proportion_records)


def print_accuracy_stats(df_accuracy):
    """Print accuracy statistics for TikZ plots."""
    methods = df_accuracy["method"].unique()

    for method in methods:
        print(f"Method: {method}")

        # Get number of k values from the data
        k_values = df_accuracy["k"].unique()

        for j in sorted(k_values):
            df_subset = df_accuracy[
                (df_accuracy["method"] == method) & (df_accuracy["k"] == j)
            ]["accuracy"]
            mean_accuracy = df_subset.mean()
            ci_accuracy = 1.96 * df_subset.std() / np.sqrt(len(df_subset))

            # For TikZ plot
            print(f"({j},{mean_accuracy}) +=(0,{ci_accuracy}) -=(0,{ci_accuracy})")


def print_proportion_stats(df_proportions):
    """Print proportion statistics."""
    print("\nProportions")

    methods = df_proportions["method"].unique()
    groups = df_proportions["f"].unique()

    for method in methods:
        for group in groups:
            df_subset = df_proportions[
                (df_proportions["method"] == method) & (df_proportions["f"] == group)
            ]
            mean_prop = df_subset["proportion"].mean()
            ci_prop = 1.96 * df_subset["proportion"].std() / np.sqrt(len(df_subset))

            print(
                f"Method: {method}, f: {group}, mean proportion = {mean_prop}, 1.96*SE = {ci_prop}"
            )


def main():
    # Define constants
    SEEDS = list(range(5))
    ANALYSIS_K = 5
    n_women = 1000 * ANALYSIS_K
    D = AdultIncome(test_size=0.5, seed=0)
    X_test = D.get_test(n_women=n_women, return_labels=False)
    n_test = len(X_test)

    # Collect data
    df_accuracy = collect_accuracy_data(seeds=SEEDS)
    df_proportions = collect_proportion_data(seeds=SEEDS, k=ANALYSIS_K, n_test=n_test)

    # Print statistics
    print_accuracy_stats(df_accuracy)
    print_proportion_stats(df_proportions)


if __name__ == "__main__":
    main()
