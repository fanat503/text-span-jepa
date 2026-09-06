# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Safe torch.load: prefer weights_only=True, fall back for legacy pickles.

Checkpoints written by this repo contain only tensors, primitives and
(dict/list) containers, so weights_only=True is the right default — it
neutralizes arbitrary-pickle execution from untrusted checkpoint files.
Legacy checkpoints (or third-party files) that embed non-allowlisted
objects trigger a warning and a single weights_only=False retry, keeping
old workflows alive without silently weakening every load.

Only ``pickle.UnpicklingError`` (= the file is a pickle whose objects are
not in the weights_only allowlist — the legacy-checkpoint case) triggers
the retry. All other errors (FileNotFoundError, truncated files,
non-pickle garbage) propagate untouched: an attacker can no longer craft
an exception that flips the strict unpickler off.
"""

import pickle
import warnings

import torch


def safe_torch_load(path, map_location=None):
    """torch.load preferring weights_only=True; fallback only for legacy pickles."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        warnings.warn(
            f"safe_torch_load: weights_only=True failed for {path} "
            "(UnpicklingError); retrying with weights_only=False. "
            "Only do this for checkpoints you trust."
        )
        return torch.load(path, map_location=map_location, weights_only=False)
