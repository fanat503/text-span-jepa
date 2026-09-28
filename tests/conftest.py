# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
"""Session-wide determinism enforcement for the test suite.

Why this file exists
--------------------
`AGENTS.md` says "Tests must be deterministic", and 12 of the 21 test files
contained no seeding call of any kind before this file existed. `src/interp/`
draws from the global RNG at 28 sites and never imports `src/utils/seed.py`.
Nothing enforced the rule; a suite like that is one test away from becoming
order-dependent, and the usual symptom is a failure that only reproduces on a
co-worker's machine.

A single constant seed is not the fix
------------------------------------
Seeding every test with the same constant makes the suite *reproducible* but
not *order-independent*: every test still starts from the same stream, so
test A's leftovers are indistinguishable from test B's, and reordering tests
silently re-labels the data. The stream is a function of the test, not a
shared resource. So the seed here is derived from the pytest node id, mixed
with a run-level base seed.

The node-id hash uses `hashlib.blake2b`, deliberately NOT the builtin
`hash()`: string hashing is randomised per interpreter process, so a builtin
`hash(nodeid)` would give a *different* seed on every run and destroy exactly
the property this file is here to create.

What is NOT here, on purpose
----------------------------
`torch.use_deterministic_algorithms(True)` is NOT called. It raises
`RuntimeError` for any op without a deterministic kernel, and this repo is
full of `torch.linalg.svd`, `index_put`/`scatter` and custom Grassmann
workspace ops. Turning it on would break the suite rather than harden it.
`src/utils/seed.seed_everything` is also bypassed on purpose: its
`deterministic=True` path sets `cudnn.deterministic`/`benchmark` flags that
mean nothing on this CPU-only box, and its `deterministic=False` default sets
`cudnn.benchmark = True`, which is an explicit *opt-out* of deterministic
cudnn algorithm selection on a CUDA host. Seeding the three primitives
directly says what it means.

Note on scope
-------------
The per-test seed does not cover module-scoped fixtures, which are built
before any function-scoped fixture runs. The suite currently has exactly one
(`future_clean` in `tests/test_probes_split.py`); it passes an explicit `seed`
and `tests/test_probes_split.py` asserts that `evaluate()` leaves the caller's
global RNG untouched, so it is not a live order-dependence vector today. A
new module-scoped fixture that draws from the global RNG would need the same
treatment, and that is a follow-up rather than speculative code here.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import pytest
import torch

#: Environment variable overriding the run-level base seed. Useful for
#: bisecting a stochastic flake: re-run with a different base and the failure
#: either persists (structural) or moves (data-dependent).
BASE_SEED_ENV = "PYTEST_BASE_SEED"

#: Used when BASE_SEED_ENV is unset.
DEFAULT_BASE_SEED = 0

#: Seeds are reduced modulo this. `np.random.seed` rejects anything >= 2**32,
#: so the modulus is the binding constraint; `random.seed` and
#: `torch.manual_seed` both accept the resulting range.
_SEED_MODULUS = 2**32


def _read_base_seed() -> int:
    """Read the run-level base seed, failing loudly on a typo."""
    raw = os.environ.get(BASE_SEED_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_BASE_SEED % _SEED_MODULUS
    try:
        base = int(raw.strip())
    except ValueError:
        # conftest is imported during collection, so this aborts the run with
        # a usage error instead of silently testing one seed.
        raise pytest.UsageError(
            f"{BASE_SEED_ENV}={raw!r} is not an integer. Unset it for the "
            f"default base seed, or set it to an integer."
        ) from None
    return base % _SEED_MODULUS


BASE_SEED = _read_base_seed()


def seed_for_node(node_id: str, base: int = BASE_SEED) -> int:
    """Return the per-test seed for a pytest node id.

    Deterministic across processes, machines and runs, and distinct for every
    distinct node id (including parametrised ids).
    """
    digest = hashlib.blake2b(node_id.encode("utf-8"), digest_size=8).digest()
    return (base + int.from_bytes(digest, "big")) % _SEED_MODULUS


@pytest.fixture(autouse=True)
def deterministic_rng(request):
    """Seed every RNG from the node id and pin torch to a single thread.

    Autouse, so it applies to tests that never asked for determinism -- which
    is the entire point, since the unseeded tests are the ones that break.
    """
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    seed = seed_for_node(request.node.nodeid)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Never let a teardown-relevant state escape: put the thread count back
        # before propagating.
        torch.set_num_threads(prev_threads)
        raise

    yield seed

    torch.set_num_threads(prev_threads)
