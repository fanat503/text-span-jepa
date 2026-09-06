# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Reproducibility: deterministic seeding for all random sources
# From PyTorch reproducibility docs + I-JEPA + NextLat best practices

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """Seed all random sources for full reproducibility.

    Args:
        seed: master seed
        deterministic: if True, use torch.backends.cudnn.deterministic = True
                       (slower but fully reproducible)

    Returns:
        The seed that was set (for logging)

    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # I-JEPA pattern: benchmark=True for speed, accept minor non-determinism
        torch.backends.cudnn.benchmark = True

    return seed


def worker_init_fn(worker_id: int, base_seed: int = 42):
    """Init function for DataLoader workers to ensure reproducibility.

    Usage:
        DataLoader(..., worker_init_fn=lambda wid: worker_init_fn(wid, seed))

    Each worker gets a different seed derived from base_seed + worker_id.
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
