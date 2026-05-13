"""Deterministik tohumlama yardımcıları."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Tekrar üretilebilirlik için Python, NumPy ve PyTorch'u (varsa) tohumlar."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(
            False
        )  # GAT scatter işlemleri deterministik değil
    except ImportError:
        pass
