"""Central seeding so every stage of the project (data split, and later
training) uses one number, logged once, instead of scattered magic seeds."""
from __future__ import annotations

import random


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # Fine on machines that only run the data stage (no torch installed).
        pass
