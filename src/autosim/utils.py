import random

import numpy as np
import torch


def set_random_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for Python, NumPy and PyTorch.

    Parameters
    ----------
    seed: int
        The random seed to use.
    deterministic: bool
        Use "deterministic" algorithms in PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
