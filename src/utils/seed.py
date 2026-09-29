# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Reproducibility: deterministic seeding for all random sources
# From PyTorch reproducibility docs + I-JEPA + NextLat best practices
"""Seeding, thread pinning and the determinism switches.

What a "reproducible" run needs, and which of those this module provides
----------------------------------------------------------------------
A run is only bitwise reproducible if ALL of the following are pinned. The
first was pinned here already; the other three were not, which is why the
audit measured two runs on one box agreeing bitwise and two runs on a
different core count disagreeing from optimizer step 3 onward:

1.  **The RNG streams** -- `random`, `numpy`, `torch` (CPU and CUDA).
    Pinned by :func:`seed_everything`.
2.  **The intra-op thread count** -- float addition is not associative, so a
    parallel reduction over ``N`` elements sums in a different order at
    ``N=1`` than at ``N=8``. This is not noise that averages out over a run;
    it perturbs the weights, and a training loop feeds every perturbation
    forward. Measured here (see ``.agent-notes/task-24.md``): 1 vs 2 threads,
    same seed, same data, 10 steps -- steps 0-2 bitwise equal, step 3 onwards
    differing by 1.19e-07 to 2.38e-07, final ``qkv`` weight not bitwise equal.
3.  **Deterministic algorithm selection** -- ``cudnn.benchmark`` picks the
    fastest convolution algorithm per shape, which is a machine-dependent
    choice, and a handful of ATen ops (``index_put_`` with ``accumulate``,
    ``scatter_add_``) are order-dependent by default.
4.  **The cuBLAS workspace** -- cuBLAS splits some GEMMs across threads, so
    its reduction order depends on the thread count. It is configured by
    ``CUBLAS_WORKSPACE_CONFIG``, which cuBLAS reads at handle creation, i.e.
    before the process is warm.

``deterministic=True`` used to set only ``cudnn.deterministic`` /
``cudnn.benchmark``, which are **no-ops on a CPU-only host** and are not even
read by cuBLAS. So the flag that named full reproducibility delivered none of
it here. It now covers all four axes, and :func:`reproducibility_state` reports
what was actually applied so a run can log it rather than assert it.

Reachability
------------
``deterministic`` and ``num_threads`` are reachable three ways:

* the API, :func:`seed_everything`;
* the environment, :data:`DETERMINISTIC_ENV` / :data:`NUM_THREADS_ENV` -- these
  work **today, with no call-site change**, which matters because the trainer
  (``src/train.py``) calls ``seed_everything(seed)`` with no way to pass
  anything else;
* a config key -- **not reachable yet**, and deliberately not faked. See
  ``defaults.yaml`` under ``meta.seed`` for the exact one-line change that
  would close it, and ``.agent-notes/task-24.md`` for the report. Declaring an
  unread key in ``defaults.yaml`` would silence the trainer's own typo
  warning and turn a loud "this did nothing" into a silent no-op, which is the
  defect class this repo treats as worse than a missing key.

What ``deterministic=True`` does and does not buy
-------------------------------------------------
On CPU, the thread count is the axis that moves the numbers, so
``deterministic=True`` pins the intra-op pool to 1 thread unless
``num_threads`` says otherwise. Pinning to ``N > 1`` is still useful -- it
makes a run repeatable *at that N* -- but it is not portable: two machines
with 4 threads can still differ, because the BLAS library behind ATen may
choose a different kernel for a different CPU. The honest contract is
"reproducible on a machine and a thread count that are both fixed", and no
amount of switching in this file makes it stronger than that on CPU.

``warn_only=True`` on :func:`torch.use_deterministic_algorithms` is deliberate
and load-bearing. Without it, the strict mode raises ``RuntimeError`` for any
op without a deterministic kernel, and this repo is full of them
(``torch.linalg.svd``, ``index_put_``, ``scatter``, the custom Grassmann
workspace ops). Strict mode would turn a reproducibility request into a crash
on the first batch, so determinism would never be requested and never be
tested. ``warn_only`` keeps the flag on and reports the gaps instead.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

#: Set to a truthy value to make :func:`seed_everything` behave as if
#: ``deterministic=True``. Accepted spellings are ``1/0``, ``true/false``,
#: ``yes/no``, ``on/off``, case-insensitive.
DETERMINISTIC_ENV = "TEXT_SPAN_JEPA_DETERMINISTIC"

#: Integer. Pins ``torch.set_num_threads`` to that many intra-op threads.
#: Unset or empty leaves torch's own default alone.
NUM_THREADS_ENV = "TEXT_SPAN_JEPA_NUM_THREADS"

#: cuBLAS reads this when it creates its handle, so it is only useful if it is
#: in the environment before the first CUDA matmul. It is set here anyway
#: because a CPU-only host also goes through this code path, and a host that
#: sets it earlier (a launcher, ``torchrun``) is unaffected -- we never
#: overwrite a value that is already present.
CUBLAS_WORKSPACE_CONFIG_ENV = "CUBLAS_WORKSPACE_CONFIG"
#: The value recommended by the PyTorch reproducibility documentation.
CUBLAS_WORKSPACE_CONFIG_VALUE = ":4096:8"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _env_flag(name: str) -> bool | None:
    """Read a boolean environment variable, or None when unset/unreadable.

    An unreadable value is an error, not a default: a typo in
    ``TEXT_SPAN_JEPA_DETERMINISTIC=treu`` must not silently mean "off" on the
    run that was supposed to be the reproducible one.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a boolean. Use one of "
        f"{sorted(_TRUE | _FALSE)} (case-insensitive), or unset it."
    )


def _env_int(name: str) -> int | None:
    """Read a positive-integer environment variable, or None when unset."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(
            f"{name}={raw!r} is not an integer. Unset it to leave torch's "
            f"thread count alone, or set it to 1 for a bitwise-reproducible run."
        ) from None
    if value < 1:
        raise ValueError(f"{name}={raw!r} must be >= 1; torch has no thread pools of size 0.")
    return value


def resolve_deterministic(deterministic: bool | None = None) -> bool:
    """Explicit argument wins, then the environment, then ``False``.

    The default is deliberately ``False`` and not ``True``. Threading the
    deterministic path through by default would slow every run on the box by
    the number of cores, and a reproducibility feature nobody pays for is a
    feature nobody turns on.
    """
    if deterministic is not None:
        return bool(deterministic)
    env = _env_flag(DETERMINISTIC_ENV)
    return False if env is None else env


def resolve_num_threads(num_threads: int | None = None) -> int | None:
    """Explicit argument wins, then the environment, then ``None`` (= don't touch).

    ``None`` means "leave torch alone", not "use 1": a run that never asked for
    determinism should keep the threads it has.
    """
    if num_threads is not None:
        value = int(num_threads)
        if value < 1:
            raise ValueError(f"num_threads={num_threads!r} must be >= 1.")
        return value
    return _env_int(NUM_THREADS_ENV)


def configure_threads(num_threads: int | None) -> dict[str, Any]:
    """Pin the intra-op (and, where possible, inter-op) thread pools.

    Returns what was actually applied, not what was asked for, because the
    inter-op pool can only be configured once per process and a second call
    raises. The trainer should log this dict rather than claim a thread count
    it never verified.
    """
    applied: dict[str, Any] = {
        "num_threads": None,
        "intraop_before": torch.get_num_threads(),
        "interop_before": torch.get_num_interop_threads(),
        "interop_pinned": False,
    }
    if num_threads is None:
        return applied

    applied["num_threads"] = int(num_threads)
    # This is the control that the 1-vs-2-thread measurement in the module
    # docstring exercised. OMP_NUM_THREADS/MKL_NUM_THREADS are deliberately
    # NOT set here: they are read when the process starts, so writing them
    # after torch has imported would advertise a setting that is not in
    # effect. A launcher that needs the environment itself sets it there.
    torch.set_num_threads(int(num_threads))
    try:
        torch.set_num_interop_threads(1)
        applied["interop_pinned"] = True
    except RuntimeError:
        # The pool already exists (anything ran in parallel earlier in this
        # process). Not fatal: on the sequential training path the intra-op
        # count is the axis that changes the numbers.
        pass
    applied["interop_after"] = torch.get_num_interop_threads()
    return applied


def reproducibility_state() -> dict[str, Any]:
    """Report the determinism settings currently in force in this process.

    The trainer has no way to know what a run was pinned to after the fact --
    a checkpoint carries weights, not a thread count -- so this is the only
    place the answer can come from. Logging it is the difference between "this
    run is reproducible" and "this run is reproducible and here is what it
    pinned".
    """
    return {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only_algorithms": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get(CUBLAS_WORKSPACE_CONFIG_ENV),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_env": os.environ.get(DETERMINISTIC_ENV),
        "num_threads_env": os.environ.get(NUM_THREADS_ENV),
    }


def seed_everything(
    seed: int,
    deterministic: bool | None = None,
    num_threads: int | None = None,
) -> int:
    """Seed all random sources, and optionally pin the process deterministic.

    Args:
        seed: master seed.
        deterministic: ``True`` pins the intra-op thread pool to one thread
            (or to ``num_threads``), turns on
            ``torch.use_deterministic_algorithms(warn_only=True)``, sets
            ``cudnn.deterministic``/``cudnn.benchmark``, and sets
            ``CUBLAS_WORKSPACE_CONFIG`` if it is not already set. ``False``
            turns the deterministic-algorithm switch back **off** and
            restores ``cudnn.benchmark``, so the function is symmetric: a
            caller that asks for determinism and then does not get it back to
            the fast path without this. ``None`` (the default) resolves from
            :data:`DETERMINISTIC_ENV`, then falls back to ``False``.
        num_threads: intra-op threads to pin to. ``None`` resolves from
            :data:`NUM_THREADS_ENV`; an unresolved ``None`` leaves torch's own
            thread count alone.

    Returns:
        The seed that was set (for logging).

    Note:
        ``deterministic=True`` is honoured by every caller, including the
        trainer's existing ``seed_everything(seed)``, which now picks it up
        from the environment. That is deliberate: it is the only way to reach
        the flag without editing ``src/train.py``.
    """
    want_deterministic = resolve_deterministic(deterministic)
    threads = resolve_num_threads(num_threads)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Always assigned, in both directions, so the flag cannot be left half
    # applied by a caller that switches modes mid-process.
    torch.use_deterministic_algorithms(want_deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = want_deterministic
    torch.backends.cudnn.benchmark = not want_deterministic

    if want_deterministic:
        # cuBLAS reads this once, at handle creation. Setting it here covers
        # the case where this call happens first; it is a no-op if a launcher
        # got there earlier, and we never clobber an existing choice.
        os.environ.setdefault(CUBLAS_WORKSPACE_CONFIG_ENV, CUBLAS_WORKSPACE_CONFIG_VALUE)
        # A deterministic run that is still multi-threaded is not
        # deterministic, so the thread count is not a separate, optional
        # extra here: it is the axis that moves the numbers on CPU.
        if threads is None:
            threads = 1

    configure_threads(threads)
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
