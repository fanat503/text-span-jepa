# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Detectors for `tests/conftest.py`, the suite's only determinism enforcement.

`tests/conftest.py` installs one autouse fixture that seeds `random`, `numpy` and
`torch` from a per-node-id digest and pins torch to a single thread. Without it,
12 of the 21 test files contain no seeding call at all and `src/interp/` draws
from the process-global RNG at 28 sites without ever importing
`src/utils/seed.py`.

Why this file had to exist
--------------------------
On its own the fixture was **unfalsifiable decoration**. `tests/conftest.py` can be
deleted outright and the suite stays green: measured 168 passed with the file
renamed away, across `test_cmc`, `test_cgn`, `test_probes_split`,
`test_index_and_cka`, `test_feature_composition` and `test_ablation_module`. Nothing
in the suite depended on it.

That is not a reason to delete the fixture -- a guard with no test today is still a
guard tomorrow, and the tests that would have caught the order-dependence it exists
to prevent have not been written yet. But it *is* a reason to make the guard
load-bearing, which is what this file is. The honest framing, and it cuts both ways:

  * The suite is **robustly non-flaky**. No test in it has ever failed because of
    unseeded randomness, which is a real and worth-keeping property.
  * The suite is also **blind to its own determinism**. Every assertion in it is a
    property-style bound -- loss non-negative, shape correct, top-1 in [0, 1],
    Gram matrix close -- and those are insensitive to *which* random draw they saw.
    So the insensitivity that makes the suite reliable is the same insensitivity
    that would let the seeding be deleted, or weakened to a constant, or switched
    to the builtin `hash()`, with a green run every time.

This file is what closes that gap. Every test below fails if `tests/conftest.py` is
deleted, if the fixture stops being autouse, if the per-test streams collapse onto
one shared stream, or if the seed stops being stable across processes.

The property that matters most, and the one that is easiest to lose
-------------------------------------------------------------------
The seed is derived from the pytest **node id** with `hashlib.blake2b`, *not* with
the builtin `hash()`. CPython randomises `str.__hash__` per interpreter process
(`PYTHONHASHSEED`), so a `hash(nodeid)`-based fixture would hand every test a
different seed on every run. Nothing would fail: the suite would be green, the
runs would be irreproducible, and a failure on a co-worker's machine would have no
explanation. `test_the_seed_does_not_depend_on_python_hash_randomisation` is the
detector for exactly that, and it is the one test here that could not be written
without spawning a real second interpreter.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import conftest as determinism

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

#: A node id that no run will ever collect. `seed_for_node` is a pure function of
#: the string, so a synthetic id is exactly as informative as a real one -- and it
#: means these tests never need re-baselining when this file is renamed.
PROBE_NODE_ID = "tests/test_determinism.py::test_probe_node"

#: Fingerprints of the three global streams, collected by a *separate* interpreter.
#: Deliberately not shared with the parent: each one imports `conftest`, computes a
#: seed for `PROBE_NODE_ID`, installs it exactly as the fixture does, and reports
#: what the process-global RNGs then look like. `PYTHONHASHSEED` differs between the
#: two calls, which is the whole point.
_CHILD_PROBE = """
import hashlib, json, random, sys
sys.path.insert(0, {tests_dir!r})
import numpy as np
import torch
import conftest


def fp(payload):
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


nodeid = {nodeid!r}
seed = conftest.seed_for_node(nodeid)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
print(json.dumps({{
    "seed": seed,
    "builtin_hash": hash(nodeid) & 0xFFFFFFFFFFFFFFFF,
    "live": [
        fp(torch.get_rng_state().numpy().tobytes()),
        fp(repr(random.getstate()).encode()),
        fp(np.random.get_state()[1].tobytes()),
    ],
    "draws": [random.random(), float(np.random.rand()), float(torch.rand(1)[0])],
}}))
"""


def _seed_in_a_fresh_process(nodeid: str, hash_seed: int) -> dict:
    """Install `seed_for_node(nodeid)` in a second interpreter and report the result.

    The child is given this process's own `BASE_SEED`, so the comparison against
    the in-process value is well defined even when the campaign is bisecting a
    flake with `PYTEST_BASE_SEED` set.
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hash_seed)
    env[determinism.BASE_SEED_ENV] = str(determinism.BASE_SEED)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_PROBE.format(tests_dir=str(TESTS_DIR), nodeid=nodeid)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        # Not check=True: the assert below reports the child's stderr verbatim,
        # which is the only way to diagnose a probe that stops importing.
        check=False,
    )
    assert proc.returncode == 0, (
        f"the determinism probe process failed (returncode={proc.returncode}).\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def two_processes():
    """Two runs in two interpreters, under deliberately different hash seeds.

    Module scope so the two ~4 s interpreter starts are paid for once and every
    cross-process property can be asserted against the same pair.
    """
    return (
        _seed_in_a_fresh_process(PROBE_NODE_ID, hash_seed=1),
        _seed_in_a_fresh_process(PROBE_NODE_ID, hash_seed=987654),
    )


def _live_fingerprint() -> tuple:
    """Fingerprint the three process-global RNGs, without consuming from them."""
    return (
        hashlib.blake2b(torch.get_rng_state().numpy().tobytes(), digest_size=8).hexdigest(),
        hashlib.blake2b(repr(random.getstate()).encode(), digest_size=8).hexdigest(),
        hashlib.blake2b(np.random.get_state()[1].tobytes(), digest_size=8).hexdigest(),
    )


def _reference_fingerprint(seed: int) -> tuple:
    """What the three global streams look like immediately after `seed`."""
    torch_gen = torch.Generator()
    torch_gen.manual_seed(seed)
    numpy_rng = np.random.RandomState(seed)
    return (
        hashlib.blake2b(torch_gen.get_state().numpy().tobytes(), digest_size=8).hexdigest(),
        hashlib.blake2b(repr(random.Random(seed).getstate()).encode(), digest_size=8).hexdigest(),
        hashlib.blake2b(numpy_rng.get_state()[1].tobytes(), digest_size=8).hexdigest(),
    )


#: `(nodeid, live fingerprint, seed)` recorded by the two observer tests below.
#: Module-level because a fixture would have to be function-scoped to be covered by
#: the autouse fixture being tested, and a function-scoped fixture cannot carry
#: state from one test to the next.
_OBSERVED: dict = {}


class TestFixtureIsLive:
    """The fixture ran for a test that never asked for it."""

    def test_the_determinism_fixture_is_autouse(self, request):
        """`autouse=True`, proven the only way that counts: by not asking.

        This test's signature contains no `deterministic_rng` parameter -- the
        fixture is invisible to it -- and yet `request.fixturenames` lists it,
        because pytest reports the fixtures it is *going to* set up, autouse ones
        included. If someone drops `autouse=True` this is the first test to go red,
        and nothing else in the suite would notice.

        Asserted on the public `request` API rather than on the fixture object's
        internals: `pytest.fixture` has wrapped its return value differently in
        every major version (`_pytestfixturefunction` in pytest 7, a
        `FixtureFunctionDefinition` in pytest 9), and a test that reads those
        attributes breaks on upgrade.
        """
        assert "deterministic_rng" in request.fixturenames, (
            "the conftest determinism fixture is not active for this test. It is "
            "declared autouse, so it must appear in request.fixturenames even "
            "though nothing requested it."
        )

    def test_every_global_rng_sits_exactly_at_the_seed_for_this_node_id(self, request):
        """The state at the start of a test is a *function of its node id*.

        Not merely "some seed was installed": the live state must equal a freshly
        built reference for `seed_for_node(request.node.nodeid)`, on all three
        primitives. That single assertion is simultaneously the autouse proof, the
        per-node-id proof, and the proof that nothing advanced the streams between
        the fixture and the test body.

        It goes red if the fixture is deleted, if it stops being autouse, if the
        seed becomes a constant, or if the derivation changes -- i.e. on every
        mutation that matters and none that does not.
        """
        seed = determinism.seed_for_node(request.node.nodeid)
        assert _live_fingerprint() == _reference_fingerprint(seed), (
            "the global RNG state at the start of this test is not the state "
            f"seed_for_node({request.node.nodeid!r}) installs. Either the "
            "autouse fixture did not run, or the seed is not derived from the "
            "node id."
        )

    def test_torch_is_pinned_to_a_single_thread(self):
        """The fixture's second job, and the reason it exists as a fixture at all.

        `tools/rt.py` also sets `OMP_NUM_THREADS=1`, so under the campaign runner
        this assertion is not sufficient on its own; under a bare `pytest` -- which
        is what GitHub Actions runs -- the thread count comes from this fixture
        alone, and dropping `torch.set_num_threads(1)` would silently re-enable
        every core on the box.
        """
        assert torch.get_num_threads() == 1, (
            "torch is running on "
            f"{torch.get_num_threads()} threads inside a test. The conftest "
            "fixture pins it to 1 and restores it on teardown."
        )


class TestPerTestStream:
    """Two different tests must observe two different streams."""

    def test_observer_alpha_sees_the_stream_of_its_own_node_id(self, request):
        """Records, then checks. The pair is meaningless without the comparison."""
        seed = determinism.seed_for_node(request.node.nodeid)
        live = _live_fingerprint()
        _OBSERVED["alpha"] = (request.node.nodeid, seed, live)
        assert live == _reference_fingerprint(seed)

    def test_observer_beta_sees_a_different_stream_from_alpha(self, request):
        """The heart of the card: per-test streams, not one shared stream.

        A single constant seed would make the suite *reproducible* but not
        *order-independent* -- every test would start from the same position, test
        A's leftovers would be indistinguishable from test B's, and reordering
        tests would silently re-label the data. This is the assertion that
        distinguishes the two, and it is the one a constant-seed mutation kills.

        Pytest runs a module's tests in definition order, so alpha has already run.
        Selecting this test on its own with `-k` will fail the `len` assert with a
        message that says so; that is deliberate, because a silent no-op
        comparison would be exactly the decoration this file exists to remove.
        """
        seed = determinism.seed_for_node(request.node.nodeid)
        live = _live_fingerprint()
        _OBSERVED["beta"] = (request.node.nodeid, seed, live)
        assert live == _reference_fingerprint(seed)

        assert len(_OBSERVED) == 2, (
            "test_observer_alpha must run for this comparison. Select the file, "
            "not a single test: pytest runs a module's tests in definition order."
        )
        alpha_node, alpha_seed, alpha_live = _OBSERVED["alpha"]
        assert alpha_node != request.node.nodeid, "the two observers share a node id"
        assert alpha_seed != seed, (
            "two different node ids collapsed onto one seed; every test in the "
            "suite now shares a single stream and reordering them re-labels data"
        )
        assert alpha_live != live, (
            "the two tests entered the test body with byte-identical global RNG "
            "state. The per-test stream is not per-test."
        )

    def test_the_observed_streams_are_those_two_node_ids_predict(self):
        """Closes the loop: the streams that were observed are the predicted ones."""
        assert set(_OBSERVED) == {"alpha", "beta"}
        for key, (nodeid, seed, live) in _OBSERVED.items():
            assert live == _reference_fingerprint(seed), f"{key} drifted mid-file"
            assert determinism.seed_for_node(nodeid) == seed

    def test_distinct_node_ids_get_distinct_seeds(self):
        """Distinctness must hold for parametrised ids, not just sibling tests.

        The node id of a parametrised case is `...::test_x[k1]`, so two cases of
        one parametrised test differ only inside the bracket. They must still get
        different streams, or adding a parametrisation silently makes every case
        share one.
        """
        ids = [
            "tests/test_a.py::test_one",
            "tests/test_a.py::test_two",
            "tests/test_b.py::test_one",
            "tests/test_b.py::test_one[k1]",
            "tests/test_b.py::test_one[k2]",
            "tests/test_b.py::TestClass::test_one",
        ]
        seeds = [determinism.seed_for_node(i) for i in ids]
        assert len(set(seeds)) == len(ids), "two distinct node ids collided on one seed"

    def test_distinctness_survives_a_realistic_number_of_node_ids(self):
        """The suite has ~1500 tests; 64-bit digests make a collision implausible.

        Not a proof of injectivity -- nothing can, over an unbounded id space --
        but a 1500-way birthday check on the actual scale of this suite is cheap
        and turns "implausible" into "not observed at this scale".
        """
        seeds = {
            determinism.seed_for_node(f"tests/test_determinism.py::test_generated[{i}]")
            for i in range(1500)
        }
        assert len(seeds) == 1500, "a node-id digest collision appeared in 1500 ids"


class TestSeedDerivation:
    """The seed is a pure, stable, in-range function of the node id."""

    def test_the_seed_is_recomputable_from_the_node_id_alone(self):
        """No hidden interpreter state leaks into the derivation.

        A pure function of the string is what makes the seed reproducible at all;
        anything derived from `id()`, a counter, a previously collected test, or
        the builtin `hash()` would make the same node id mean different things in
        different runs.
        """
        nodeid = PROBE_NODE_ID
        first = determinism.seed_for_node(nodeid)
        assert determinism.seed_for_node(nodeid) == first
        assert determinism.seed_for_node(nodeid) != determinism.seed_for_node(nodeid + "x")
        digest = int.from_bytes(
            hashlib.blake2b(nodeid.encode("utf-8"), digest_size=8).digest(), "big"
        )
        expected = (determinism.BASE_SEED + digest) % determinism._SEED_MODULUS
        assert first == expected, (
            "seed_for_node is no longer 'base + blake2b(node_id) mod 2**32'. A "
            "stable cryptographic digest is what makes the seed reproducible "
            "across processes. Nothing in the suite would notice a change here -- "
            "every test would silently get a new stream -- so if the derivation "
            "was changed on purpose, change the docstring that says 'blake2b' in "
            "the same commit."
        )

    def test_two_runs_in_separate_processes_agree(self, two_processes):
        """Seed and full live state must be byte-identical across interpreters.

        Compares the two children's seeds *and* their fingerprints of the three
        global streams *and* their first draws, and pins all of them to the
        in-process value. Agreeing on the seed but not on the state would mean the
        fixture's installation step is the non-reproducible part.
        """
        first, second = two_processes
        local = determinism.seed_for_node(PROBE_NODE_ID)
        assert first["seed"] == second["seed"] == local
        # json round-trips a tuple as a list, so compare list-to-list.
        assert first["live"] == second["live"] == list(_reference_fingerprint(local))
        assert first["draws"] == second["draws"], (
            "two processes with the same seed produced different draws; the "
            "streams agree at the seed but not downstream of it"
        )

    def test_the_seed_does_not_depend_on_python_hash_randomisation(self, two_processes):
        """The detector for the `hash()` regression, stated as a positive.

        CPython randomises `str.__hash__` per process via `PYTHONHASHSEED`, so a
        `hash(nodeid)`-based fixture hands every test a fresh seed on every run:
        the suite stays green and simply stops being reproducible, which is the
        exact failure this file exists to prevent and the one a green suite cannot
        report on its own.

        The two children ran under `PYTHONHASHSEED=1` and `=987654`. Their builtin
        hashes differ -- that is the premise, and it is asserted first so that a
        run where the env var failed to take effect fails here rather than
        silently proving nothing. Their seeds must nevertheless agree. Together
        the two asserts establish that the derivation cannot be the builtin hash.
        """
        first, second = two_processes
        assert first["builtin_hash"] != second["builtin_hash"], (
            "the two probe interpreters agreed on hash(nodeid), so PYTHONHASHSEED "
            "was not randomised and this test proves nothing. Fix the probe."
        )
        assert first["seed"] == second["seed"], (
            "the seed changed with PYTHONHASHSEED. seed_for_node is using the "
            "builtin hash() of the node id, which CPython randomises per process; "
            "every test now gets a new stream on every run and no test would fail."
        )

    def test_the_run_level_base_seed_shifts_every_node_uniformly(self):
        """`PYTEST_BASE_SEED` is the documented way to bisect a stochastic flake.

        Re-running with a different base either keeps the failure (structural) or
        moves it (data-dependent). It only works if every node's seed moves.
        """
        nodeid = PROBE_NODE_ID
        base_zero = determinism.seed_for_node(nodeid, base=0)
        assert determinism.seed_for_node(nodeid, base=1) != base_zero
        assert determinism.seed_for_node(nodeid, base=2) != determinism.seed_for_node(
            nodeid, base=1
        )
        # The modulus is the binding constraint, so it must wrap rather than raise.
        assert determinism.seed_for_node(nodeid, base=determinism._SEED_MODULUS) == base_zero

    def test_a_non_integer_base_seed_is_a_usage_error(self, monkeypatch):
        """A typo in the env var must abort, not silently test one seed.

        `conftest` is imported during collection, so raising here aborts the whole
        run with a usage error instead of quietly proceeding.
        """
        monkeypatch.setenv(determinism.BASE_SEED_ENV, "not-a-number")
        with pytest.raises(pytest.UsageError):
            determinism._read_base_seed()

    def test_every_seed_is_in_range_and_accepted_by_numpy(self):
        """`np.random.seed` rejects >= 2**32, so the modulus is load-bearing.

        The fixture seeds numpy first, so one out-of-range digest would turn into a
        hard error in *every* test rather than in a test about the fixture.
        """
        numpy_state = np.random.get_state()
        try:
            for i in range(500):
                seed = determinism.seed_for_node(f"tests/test_determinism.py::test_seed[{i}]")
                assert 0 <= seed < determinism._SEED_MODULUS
                np.random.seed(seed)
        finally:
            np.random.set_state(numpy_state)


class TestDeletionIsDetected:
    """The property that made this file necessary, asserted as a fact.

    Deleting `tests/conftest.py` fails the suite now, at import time, because this
    module imports it. That import *is* the deletion detector -- there is no
    gentler way to assert the absence of a file -- so these tests exist to say what
    the import is for and to check the imported surface is still the contract.
    """

    def test_the_conftest_module_still_exports_its_documented_surface(self):
        assert callable(determinism.seed_for_node)
        assert determinism.BASE_SEED_ENV == "PYTEST_BASE_SEED"
        assert determinism.DEFAULT_BASE_SEED == 0
        assert determinism._SEED_MODULUS == 2**32
        # The fixture itself, by name: the autouse assert in TestFixtureIsLive
        # looks this exact name up in request.fixturenames, so the two must agree.
        assert determinism.deterministic_rng is not None
        assert "deterministic_rng" in vars(determinism)
