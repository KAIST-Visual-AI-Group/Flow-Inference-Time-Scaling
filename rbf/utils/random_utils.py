"""
random_utils.py

Utility functions for controlling randomness.
"""

import os
import numpy as np
import torch
import random


def seed_everything(seed=0):
    """
    Seeds the random number generators of Python, Numpy and PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False