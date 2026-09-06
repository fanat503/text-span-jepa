# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""safe_torch_load hardening gates (security audit H1).

Contract:
- clean (weights_only-allowlisted) checkpoints load silently;
- legacy pickles (non-allowlisted objects) fall back to weights_only=False
  with an explicit warning;
- every OTHER error propagates untouched — no second unpickling attempt,
  no warning (an attacker cannot flip the strict unpickler off by raising
  an arbitrary exception from allowlisted primitives).
"""

import pytest
import torch

from src.utils.torchio import safe_torch_load


class _Legacy:
    """Object whose class is NOT in the weights_only allowlist."""

    def __init__(self):
        self.x = 1


def test_weights_only_clean_load_emits_no_warning(tmp_path, recwarn):
    path = tmp_path / "clean.pt"
    torch.save({"w": torch.ones(2)}, path)
    assert safe_torch_load(path)["w"].sum().item() == 2.0
    assert len(recwarn) == 0


def test_legacy_pickle_falls_back_with_warning(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"cfg": _Legacy()}, path)  # custom class -> weights_only=True raises
    with pytest.warns(UserWarning, match="weights_only=False"):
        ckpt = safe_torch_load(path)
    assert ckpt["cfg"].x == 1


def test_missing_file_raises_without_fallback(tmp_path, recwarn):
    with pytest.raises(FileNotFoundError):
        safe_torch_load(tmp_path / "nope.pt")
    assert len(recwarn) == 0
