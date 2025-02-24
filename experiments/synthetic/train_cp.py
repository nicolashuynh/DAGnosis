# stdlib
import glob
import logging
from pathlib import Path
from typing import Any

# third party
import dill
import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

# dagnosis absolute
from dagnosis.utils.seed import set_random_seeds

logger = logging.getLogger(__name__)


def save_artifacts(
    artifacts: dict, filename: str, path: str, alpha: float, cal_ratio: float
):
    """Save artifacts to file."""
    id = filename.split("/")[-1] + f"_alpha_{alpha}_calratio_{cal_ratio}"
    with open(path + id, "wb") as f:
        dill.dump(artifacts, f)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> Any:

    PATH_LOAD = cfg.PATH_SAVE_DATA
    PATH_SAVE = cfg.PATH_SAVE_CP
    Path(PATH_SAVE).mkdir(parents=True, exist_ok=True)

    set_random_seeds(cfg.random_seed)

    # Configure parameters
    significance = cfg.significance
    cal_train_ratio = cfg.cal_train_ratio

    for filename in glob.glob(PATH_LOAD + "*"):
        logger.info(f"Processing {filename}")

        # Load data
        with open(filename, "rb") as f:
            data = dill.load(f)

        D = data["D"]
        dimension = D._get_dim()
        alpha = significance / dimension  # Bonferroni correction
        list_features = np.arange(dimension)

        # Initialize artifacts
        artifacts = {"evaluators": {}, "D": D}  # Store training data

        # Instantiate and train evaluators from config
        for method, pipeline_cfg in cfg.pipelines.items():
            logger.info(f"Training {method.upper()} method")
            pipeline = instantiate(pipeline_cfg)

            pipeline.conformal_evaluator = pipeline.conformal_evaluator(
                conf_predictor_cfg=cfg.conf_predictor_cfg, logger=logger
            )

            pipeline.train(D, list_features)

            artifacts["evaluators"][method] = pipeline.conformal_evaluator

        # Save results
        save_artifacts(artifacts, filename, PATH_SAVE, alpha, cal_train_ratio)
        logger.info(f"Saved results for {filename}")


if __name__ == "__main__":
    main()
