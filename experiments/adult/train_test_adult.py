# stdlib
import logging
import warnings
from collections import defaultdict

# third party
import dill
import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# dagnosis absolute
from dagnosis.utils.conformal import analyse_conformal_dict
from dagnosis.utils.seed import set_random_seeds

warnings.filterwarnings("ignore")


def complement(a1, a2):
    return list(set(a2) - set(a1))


def get_consistent_indices_and_stats(df, conformal_dict, feature_index):
    n_samples = len(df)

    inconsistent_samples_indices = df.index[
        df["inconsistent"] == 1
    ].tolist()  # indices of the samples which are inconsistent

    inconsistent_specific_feature = np.intersect1d(
        df.index[df[feature_index] == 1].tolist(),
        conformal_dict[feature_index]
        .index[conformal_dict[feature_index]["true_val"] == 0]
        .tolist(),
    )
    inconsistent_any_feature = np.intersect1d(
        df.index[df["inconsistent"] == 1].tolist(),
        conformal_dict[feature_index]
        .index[conformal_dict[feature_index]["true_val"] == 0]
        .tolist(),
    )

    consistent_indices = complement(inconsistent_samples_indices, np.arange(n_samples))

    return (
        consistent_indices,
        inconsistent_specific_feature,
        inconsistent_any_feature,
        inconsistent_samples_indices,
    )


@hydra.main(
    version_base=None, config_path="../../conf", config_name="config_adult_income"
)
def main(cfg: DictConfig):
    for i in range(cfg.n_repeats):
        run_experiment(i, cfg)


def run_experiment(seed, cfg: DictConfig):

    set_random_seeds(seed)
    cfg.datamodule.seed = seed

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    sex_feature_index = cfg.sex_feature_index

    detection_on_sex = defaultdict(list)

    detection_all = defaultdict(list)
    detection = defaultdict(list)
    method_accuracy = defaultdict(list)
    D = instantiate(cfg.datamodule)

    X_train, y_train = D.get_train(return_labels=True)

    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train, y_train)

    for k in range(cfg.n_iterations):
        logger.info(f"Iteration: {k}")
        n_women_rich = 1000 * k
        X_test, y_test = D.get_test(n_women_rich, return_labels=True)

        acc_test_supervised = accuracy_score(y_test, clf.predict(X_test))
        method_accuracy["supervised"].append(acc_test_supervised)

    list_features = np.arange(D._get_dim())

    for method, pipeline_cfg in cfg.pipelines.items():
        logger.info(f"Training {method.upper()} method")
        pipeline = instantiate(pipeline_cfg)

        pipeline.conformal_evaluator = pipeline.conformal_evaluator(
            conf_predictor_cfg=cfg.conf_predictor_cfg, logger=logger
        )

        pipeline.train(D, list_features)

        for k in range(cfg.n_iterations):

            logger.info(f"Iteration: {k}")

            n_women_rich = 1000 * k
            X_test, y_test = D.get_test(n_women_rich, return_labels=True)

            logger.info(f"Number of test samples: {len(X_test)}")

            conformal_dict_corrupted = pipeline.predict(X_test, list_features)

            df = analyse_conformal_dict(
                conformal_dict_corrupted
            )  # This is a dataframe where each column denotes if the sample is corrupted or not on the i-th feature

            (
                consistent_indices,
                inconsistent_women,
                inconsistent_women_all,
                inconsistent_samples_indices,
            ) = get_consistent_indices_and_stats(
                df, conformal_dict_corrupted, sex_feature_index
            )

            y_pred = clf.predict(X_test[consistent_indices, :])
            accuracy = accuracy_score(y_test[consistent_indices], y_pred)
            method_accuracy[method].append(accuracy)

            detection_on_sex[method].append(len(inconsistent_women))
            detection_all[method].append(len(inconsistent_women_all))
            detection[method].append(len(inconsistent_samples_indices))

        artifacts = {}
        artifacts["accuracy"] = method_accuracy
        artifacts["detection_all"] = detection_all
        artifacts["detection_on_sex"] = detection_on_sex
        artifacts["detection"] = detection

        with open(f"artifacts_adult/artifacts_final_seed_{seed}", "wb") as f:
            dill.dump(artifacts, f)


if __name__ == "__main__":
    main()
