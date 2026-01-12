# utils_seed.py
import os
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If you ever move to GPU later
    torch.cuda.manual_seed_all(seed)

    # Make results more deterministic (can slightly reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Some libs read this for hashing randomness
    os.environ["PYTHONHASHSEED"] = str(seed)
