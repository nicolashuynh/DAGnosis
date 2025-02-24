# stdlib
import random

# third party
import numpy as np
import torch


def set_random_seeds(random_seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
