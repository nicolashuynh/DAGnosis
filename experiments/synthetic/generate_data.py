"""
Data generation script for synthetic experiments.

This module generates synthetic datasets using
various structural equation models (SEM) and graph types.
"""

# stdlib
import itertools
import logging

# third party
import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

# dagnosis absolute
from dagnosis.utils.seed import set_random_seeds

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Set the random seed
    random_seed = cfg.random_seed
    set_random_seeds(random_seed)

    # Initialize the data generator
    generator = instantiate(cfg.data_generator)

    sparsity_levels = cfg.sparsity_levels
    dimension = cfg.dimension
    n_repetitions = cfg.n_repetitions
    n_train = cfg.n_train
    n_test = cfg.n_test

    assert dimension > 0, "Dimension must be a positive integer"
    assert n_train > 0, "Number of training samples must be a positive integer"
    assert n_test > 0, "Number of test samples must be a positive integer"

    for sparsity, iteration in itertools.product(
        sparsity_levels, np.arange(n_repetitions)
    ):
        logger.info(f"Generating data for sem_type {cfg.sem_type}, s {sparsity}")

        generator.generate_and_save(
            sparsity=sparsity, dimension=dimension, n_train=n_train, n_test=n_test
        )


if __name__ == "__main__":
    main()
