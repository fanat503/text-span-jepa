# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Safe torch.load: prefer weights_only=True, fall back for legacy pickles.

Checkpoints written by this repo contain only tensors, primitives and
(dict/list) containers, so weights_only=True is the right default — it
neutralizes arbitrary-pickle execution from untrusted checkpoint files.
Legacy checkpoints (or third-party files) that embed non-allowlisted
objects trigger a warning and a single weights_only=False retry, keeping
old workflows alive without silently weakening every load.
"""

import warnings

import torch


def safe_torch_load(path, map_location=None):
    """torch.load preferring weights_only=True; one audited fallback."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as first_err:
        warnings.warn(
            f"safe_torch_load: weights_only=True failed for {path} "
            f"({type(first_err).__name__}); retrying with weights_only=False. "
            "Only do this for checkpoints you trust."
        )
        return torch.load(path, map_location=map_location, weights_only=False)
