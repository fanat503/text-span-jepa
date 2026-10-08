# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Security gates for ``safe_torch_load`` (audit H1, re-derived from measurement).

The previous version of this file asserted the OPPOSITE contract: that a
non-allowlisted pickle "falls back with a warning". That behaviour was the
vulnerability, and it was measurably exploitable -- a truncated file and a plain
text file both raise ``UnpicklingError``, which was the exact retry trigger, so
"strict load failed" was answered with "load it unsafely". A hostile payload
whose ``__reduce__`` calls a sentinel function ran that function.

The contract pinned here:

- clean (weights_only-allowlisted) checkpoints load silently;
- a file that fails strict loading RAISES -- it is never silently downgraded;
- the unsafe path exists only behind an explicit ``allow_unsafe_fallback=True``
  at the call site, and warns loudly when used;
- filesystem errors propagate untouched, because "absent" is not "untrusted";
- a hostile checkpoint must not execute its payload, ever, by any of the routes
  an attacker can reach.
"""

import pathlib

import pytest
import torch

from src.utils.torchio import UnsafeCheckpointError, safe_torch_load


class _Legacy:
    """Object whose class is NOT in the weights_only allowlist."""

    def __init__(self):
        self.x = 1


_SENTINEL = pathlib.Path(__file__).with_name("_torchio_pwned.txt")


def _write_reducer_payload(path):
    """A pickle whose __reduce__ calls a function that touches the filesystem.

    This is the benign stand-in for arbitrary code execution: if any code path
    unpickles this with weights_only=False, the sentinel file appears.
    """

    class _Exploit:
        def __reduce__(self):
            return (pathlib.Path.touch, (str(_SENTINEL),))

    torch.save({"cfg": _Exploit()}, path)


# --------------------------------------------------------------------------
# clean loads
# --------------------------------------------------------------------------


def test_weights_only_clean_load_emits_no_warning(tmp_path, recwarn):
    path = tmp_path / "clean.pt"
    torch.save({"w": torch.ones(2)}, path)
    assert safe_torch_load(path)["w"].sum().item() == 2.0
    assert len(recwarn) == 0


# --------------------------------------------------------------------------
# the actual fix: strict failure RAISES instead of downgrading
# --------------------------------------------------------------------------


def test_legacy_pickle_raises_instead_of_downgrading(tmp_path, recwarn):
    """The regression this file exists for.

    A non-allowlisted object used to trigger an automatic weights_only=False
    retry. It must now raise, and must not warn, because no retry happened.
    """
    path = tmp_path / "legacy.pt"
    torch.save({"cfg": _Legacy()}, path)

    with pytest.raises(UnsafeCheckpointError):
        safe_torch_load(path)

    # No warning is correct here: nothing was downgraded, so there is nothing
    # to warn about. A warning here would mean the fallback still ran.
    assert not [w for w in recwarn if "weights_only=False" in str(w.message)]


def test_truncated_file_raises_instead_of_downgrading(tmp_path, recwarn):
    """Truncation raised UnpicklingError -- the old retry trigger."""
    path = tmp_path / "truncated.pt"
    torch.save({"w": torch.ones(64)}, path)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])

    with pytest.raises(UnsafeCheckpointError):
        safe_torch_load(path)
    assert not [w for w in recwarn if "weights_only=False" in str(w.message)]


def test_plain_text_file_raises_instead_of_downgrading(tmp_path, recwarn):
    """A non-pickle file also raised UnpicklingError -- the old retry trigger."""
    path = tmp_path / "notes.txt"
    path.write_text("this was never a checkpoint, it is a README fragment\n")

    with pytest.raises(UnsafeCheckpointError):
        safe_torch_load(path)
    assert not [w for w in recwarn if "weights_only=False" in str(w.message)]


# --------------------------------------------------------------------------
# the security property itself
# --------------------------------------------------------------------------


def test_hostile_checkpoint_does_not_execute_its_payload(tmp_path):
    """No reachable route may run code from a hostile checkpoint.

    Covers the default call and the truncated variant an attacker would ship,
    because they raised different exception types originally and therefore hit
    different branches.
    """
    _SENTINEL.unlink(missing_ok=True)

    good = tmp_path / "hostile.pt"
    _write_reducer_payload(good)
    with pytest.raises(UnsafeCheckpointError):
        safe_torch_load(good)
    assert not _SENTINEL.exists(), "payload executed during strict load"

    data = good.read_bytes()
    truncated = tmp_path / "hostile_truncated.pt"
    truncated.write_bytes(data[: len(data) // 2])
    with pytest.raises(UnsafeCheckpointError):
        safe_torch_load(truncated)
    assert not _SENTINEL.exists(), "payload executed during truncated load"

    _SENTINEL.unlink(missing_ok=True)


def test_unsafe_fallback_is_opt_in_and_warns(tmp_path):
    """The escape hatch still works, but only when asked for, and it says so."""
    path = tmp_path / "legacy.pt"
    torch.save({"cfg": _Legacy()}, path)

    with pytest.warns(UserWarning, match="weights_only=False"):
        ckpt = safe_torch_load(path, allow_unsafe_fallback=True)
    assert ckpt["cfg"].x == 1


def test_fallback_is_off_by_default_for_every_hostile_shape(tmp_path):
    """No shape reaches the unsafe path without the explicit flag."""
    _SENTINEL.unlink(missing_ok=True)

    hostile = tmp_path / "h.pt"
    _write_reducer_payload(hostile)

    for candidate in (
        hostile,
        tmp_path / "text.txt",  # created below
        tmp_path / "truncated.pt",  # created below
    ):
        if candidate.name == "text.txt":
            candidate.write_text("not a checkpoint")
        elif candidate.name == "truncated.pt":
            raw = hostile.read_bytes()
            candidate.write_bytes(raw[: len(raw) // 2])
        with pytest.raises(UnsafeCheckpointError):
            safe_torch_load(candidate)

    assert not _SENTINEL.exists()
    _SENTINEL.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# filesystem errors are not verdicts about the file
# --------------------------------------------------------------------------


def test_missing_file_raises_filenotfound_not_unsafe(tmp_path):
    """ "Absent" must not be reported as "untrusted".

    Wrapping it would break ``except FileNotFoundError`` at the call site and
    send a user hunting for a corrupt checkpoint they never had.
    """
    with pytest.raises(FileNotFoundError):
        safe_torch_load(tmp_path / "nope.pt")


def test_directory_raises_a_filesystem_error_not_unsafe(tmp_path):
    """A directory is a filesystem problem, not an untrusted checkpoint.

    The concrete type is platform-dependent (POSIX raises IsADirectoryError,
    Windows raises PermissionError), so assert the property that matters: it is
    a filesystem error and it is NOT an UnsafeCheckpointError.
    """
    with pytest.raises((IsADirectoryError, PermissionError, OSError)) as excinfo:
        safe_torch_load(tmp_path)
    assert not isinstance(excinfo.value, UnsafeCheckpointError)


def test_unsafe_error_is_distinguishable_from_missing_file(tmp_path):
    """The two failure modes must be catchable separately."""
    bad = tmp_path / "bad.pt"
    torch.save({"cfg": _Legacy()}, bad)

    with pytest.raises(UnsafeCheckpointError):
        safe_torch_load(bad)
    # and NOT a FileNotFoundError, so callers can branch on it
    assert not issubclass(UnsafeCheckpointError, FileNotFoundError)
