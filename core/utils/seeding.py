"""RNG seeding."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python ``random``, NumPy, and PyTorch.

    Args:
        seed: Seed value applied to all three RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
