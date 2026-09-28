# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Safe torch.load: strict ``weights_only=True``, no implicit downgrade.

Checkpoints written by this repo contain only tensors, primitives and
(dict/list/tuple) containers, so ``weights_only=True`` is correct and it
neutralises arbitrary-pickle execution from untrusted checkpoint files.

There is **no** automatic retry with ``weights_only=False``.

An earlier version of this module caught ``pickle.UnpicklingError`` and retried
with ``weights_only=False``, on the stated grounds that "only a legacy pickle
raises it, and an attacker cannot craft that". That reasoning was false, and
measurably so:

* a plain text file raises ``UnpicklingError`` -- it is not a zip container, so
  ``torch.load`` falls back to the legacy ``pickle.load`` path and the first
  opcode is garbage;
* a **truncated raw pickle** raises ``UnpicklingError`` too;
* most importantly, an attacker's pickle whose only GLOBAL is a non-allowlisted
  callable raises ``UnpicklingError`` *by design* -- that is precisely what the
  allowlist is for.

So the retry trigger was "the file is not safe to load strictly", and the
response was "load it unsafely". Verified before the fix: writing a payload
whose ``__reduce__`` calls a sentinel function left the sentinel file on disk.

Decision: a file that cannot be loaded strictly is reported, never downgraded.
If a caller genuinely owns a legacy checkpoint, it must say so by passing
``allow_unsafe_fallback=True`` -- an explicit, greppable, per-call-site decision
rather than an automatic behaviour triggered by attacker-controlled bytes.
"""

import warnings

import torch

__all__ = ["UnsafeCheckpointError", "safe_torch_load"]

_INSTALL_HINT = (
    "Re-save the checkpoint with torch.save() from a trusted process, or pass "
    "allow_unsafe_fallback=True if you have verified that this exact file is "
    "yours (that path executes arbitrary code from the file)."
)

# Filesystem-level failures say nothing about whether the *contents* are
# trustworthy. Wrapping them in UnsafeCheckpointError would both break
# `except FileNotFoundError` at the call site and tell the user their checkpoint
# is unsafe when it is simply absent.
_FS_ERRORS = (FileNotFoundError, IsADirectoryError, PermissionError)


class UnsafeCheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be loaded under ``weights_only=True``.

    Subclasses ``RuntimeError`` so existing ``except RuntimeError`` handlers in
    torch's own load path keep working, while callers can distinguish "this file
    is not safely loadable" from "this file is a different kind of problem".
    """


def _is_zip_container(path):
    """True when `path` is a torch zip-serialised checkpoint."""
    try:
        return torch.serialization._is_zipfile(path)
    except _FS_ERRORS:
        return False
    except Exception:  # pragma: no cover - defensive: unreadable/odd path
        return False


def safe_torch_load(path, map_location=None, allow_unsafe_fallback=False):
    """Load a checkpoint with ``weights_only=True``.

    Args:
        path: checkpoint file path.
        map_location: forwarded to ``torch.load``.
        allow_unsafe_fallback: opt in to a single ``weights_only=False``
            retry. Off by default. Only set this for a file you produced
            yourself and whose provenance you control.

    Raises:
        UnsafeCheckpointError: the file is readable but is not strictly
            loadable and the caller did not opt in to the unsafe path.
        FileNotFoundError, IsADirectoryError, PermissionError: propagated
            untouched, because they are filesystem problems rather than
            verdicts about the file's trustworthiness.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except _FS_ERRORS:
        # Not a statement about the contents. Let the caller handle it.
        raise
    except Exception as strict_error:
        if not allow_unsafe_fallback:
            kind = "zip container" if _is_zip_container(path) else "raw pickle stream"
            raise UnsafeCheckpointError(
                f"safe_torch_load: refusing to load {path} as a checkpoint. It is a "
                f"{kind} and failed strict (weights_only=True) loading with "
                f"{type(strict_error).__name__}. Either it is corrupt/truncated, it "
                f"was written by a torch version this build cannot read, or it "
                f"contains non-allowlisted objects. {_INSTALL_HINT}"
            ) from strict_error
        warnings.warn(
            f"safe_torch_load: weights_only=True failed for {path} "
            f"({type(strict_error).__name__}); retrying with weights_only=False "
            f"because allow_unsafe_fallback=True was passed. This executes "
            f"arbitrary code from the file. Only do this for checkpoints you trust.",
        )
        return torch.load(path, map_location=map_location, weights_only=False)
