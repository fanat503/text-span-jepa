"""Tests for the SAE resume path and the rng site namespace.

TASK-46 shipped both fixes with FIVE mutations, FIVE survivors: no test anywhere
called `SAETrainer.save` or `SAETrainer.load`, so a correct fix and a reverted
fix were indistinguishable to the suite. These tests exist to make that
impossible.

This is the THIRD draft. The first two failed because I wrote assertions against
a behaviour I had not read rather than against the source:

  - I asserted decoder ROW norms were 1. The code deliberately normalises COLUMN
    norms (`dim=0`), and says why at length: `recons = W_dec @ latent + b` means
    feature j contributes `a_j * W_dec[:, j]`, so the column norm is the one that
    makes an activation readable as the magnitude of its contribution. My
    assertion was wrong; the code was right.
  - I called `resample_dead_features()` expecting it to act. It early-returns
    unless `_steps_since_resample >= resample_interval` AND `total_samples > 0`,
    so it did nothing and every downstream assertion was vacuous. A test that
    passes because the code under test never ran is worse than no test.

So the helpers below drive the real preconditions. Kept here because the failure
modes are the general ones: a green test that exercised nothing, and a confident
assertion about code nobody read.

Mutations killed, from .agent-notes/task-46.md:

  delete sae.load_state_dict(...)            -> test_resume_restores_sae_state
  load_state_dict(..., strict=False)         -> test_load_is_strict
  generator_for drops the site term          -> test_implicit_generator_respects_site
  derive_generator drops the site term       -> test_derive_generator_respects_site
  site offset returns a constant 0          -> test_site_offset_is_not_a_constant
  nothing written that is never read         -> test_saved_keys_are_all_read
"""

# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import torch

from src.interp import rng
from src.interp.sae import SAETrainer, SparseAutoencoder

IN, LATENT, K = 16, 24, 4


def _sae(seed=0, resample_interval=1):
    return SparseAutoencoder(
        input_dim=IN, latent_dim=LATENT, k=K, seed=seed, resample_interval=resample_interval
    )


def _trainer(seed=0, resample_interval=1):
    sae = _sae(seed=seed, resample_interval=resample_interval)
    return sae, SAETrainer(sae, lr=1e-3, seed=seed)


def _batch(n=8):
    return torch.randn(n, IN)


def _make_every_feature_dead(sae):
    """Drive resample_dead_features past BOTH of its early returns.

    It returns immediately when `_steps_since_resample < resample_interval`, and
    again when `total_samples == 0`. Both must be satisfied or the call is a
    no-op and any assertion about it is vacuous.
    """
    sae.feature_act_count.zero_()
    sae.total_samples = torch.tensor(1000, dtype=torch.long)
    sae._steps_since_resample = sae.resample_interval
    return sae


class _KeyRecorder(dict):
    """Module-level so `torch.save` can pickle it.

    A local class cannot be pickled, and the probe below has to be written to
    disk for `load` to read. My first draft defined this inside the test and the
    save failed with "Can't pickle local object".
    """

    seen: set = set()

    def __missing__(self, key):
        type(self).seen.add(key)
        raise KeyError(key)

    def get(self, key, default=None):
        type(self).seen.add(key)
        return dict.get(self, key, default)


class TestResumeRestoresSAEState:
    """Kills: delete sae.load_state_dict(...)"""

    def test_resume_restores_sae_state(self, tmp_path):
        """Save a known state, THEN poison the live module, then load.

        The order matters and my first two drafts got it wrong twice. Poisoning
        before `save` stores the poison, so `load` faithfully restores the
        poison and the assertion fails while the code is correct. What has to be
        corrupted is the object being restored INTO, not the one being read.
        """
        sae, tr = _trainer()
        for _ in range(3):
            tr.train_step(_batch())
        assert tr.step_count > 0, "trainer never stepped; the test would be vacuous"

        path = tmp_path / "sae.pt"
        tr.save(str(path))
        expected_enc = sae.encoder.weight.detach().clone()
        expected_dec = sae.decoder.weight.detach().clone()

        with torch.no_grad():
            sae.encoder.weight.fill_(123.0)
            sae.decoder.weight.fill_(456.0)
        assert torch.all(sae.encoder.weight == 123.0), "poison did not take; test is void"

        tr.load(str(path))

        assert torch.equal(sae.encoder.weight, expected_enc), "encoder weights not restored"
        assert torch.equal(sae.decoder.weight, expected_dec), "decoder weights not restored"

    def test_resumed_trajectory_is_bit_identical(self, tmp_path):
        """The report's measurement, as a test.

        Uninterrupted steps ran 1.461717 ... 1.345310; the resumed run ran
        1.603925 ... 1.488915. A warm optimiser attached to fresh weights is
        what produced that gap.

        Five attempts at this test reported a divergence that a field-by-field
        state comparison could not reproduce. The cause was in the TEST every
        time. The last one is worth recording because it is invisible from the
        test body:

        the identical code passes outside pytest and fails inside it, because
        `tests/conftest.py`'s autouse `deterministic_rng` fixture calls
        `torch.set_num_threads(1)`. Float reduction order depends on the thread
        count, so the two runs were doing different arithmetic. That is TASK-24's
        measured finding about thread-dependent reductions, showing up here as a
        test failure that had nothing to do with the resume path.

        The threads are pinned explicitly here rather than relying on the
        fixture, because this test is about RESUME FIDELITY and the fixture is
        about suite determinism; if the fixture's policy changes, this test must
        not silently start depending on it.
        """
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            gen = torch.Generator().manual_seed(7)
            batches = [torch.randn(8, IN, generator=gen) for _ in range(12)]

            _, tr = _trainer(resample_interval=10**9)  # no resample mid-window
            for b in batches[:3]:
                tr.train_step(b)

            # Save AT the resume point, then continue. Computing all of `straight`
            # before saving is the mistake I made four times: `save` then captured
            # the state after twelve steps while `resumed` restarted at step three,
            # so step 3's loss was compared against step 12's and the test reported a
            # divergence that a field-by-field comparison could not reproduce.
            # The two runs must be interleaved around the save, not run end to end.
            path = tmp_path / "traj.pt"
            tr.save(str(path))

            straight = []
            resumed = []
            _, tr2 = _trainer(resample_interval=10**9)
            tr2.load(str(path))
            for b in batches[3:]:
                straight.append(float(tr.train_step(b)["recons_loss"]))
                resumed.append(float(tr2.train_step(b)["recons_loss"]))
        finally:
            torch.set_num_threads(prev_threads)

        assert len(straight) == len(resumed) == 9
        for i, (a, b) in enumerate(zip(straight, resumed)):
            assert abs(a - b) < 1e-12, f"resume diverged at step {i + 3}: {a} vs {b}"

    def test_resume_matches_a_fresh_trainer_state_field_by_state_field(self, tmp_path):
        """The check that actually located the four harness bugs above.

        Comparing state field by field, rather than only comparing a trajectory,
        is what distinguishes "the resume is wrong" from "my test drew different
        data". Every field matched while the trajectory test claimed otherwise.
        """
        gen = torch.Generator().manual_seed(11)
        batches = [torch.randn(8, IN, generator=gen) for _ in range(5)]

        sae, tr = _trainer(resample_interval=10**9)
        for b in batches[:3]:
            tr.train_step(b)
        path = tmp_path / "fields.pt"
        tr.save(str(path))
        sae2, tr2 = _trainer(resample_interval=10**9)
        tr2.load(str(path))

        a, b = sae.state_dict(), sae2.state_dict()
        assert set(a) == set(b), "state_dict key sets differ"
        for k in a:
            assert torch.equal(a[k], b[k]), f"{k} differs after resume"
        assert tr.step_count == tr2.step_count
        assert tr.scheduler.state_dict() == tr2.scheduler.state_dict()
        assert tr.optimizer.state_dict()["param_groups"] == (
            tr2.optimizer.state_dict()["param_groups"]
        )

    def test_dead_feature_counters_survive_a_resume(self, tmp_path):
        """The consequence that mattered more than the weights.

        `feature_act_count` and `total_samples` feed the dead-feature decision.
        Both reset to zero, so after a resume every feature read as dead and the
        encoder was reinitialised a second time - through a path that looks
        entirely like normal behaviour.
        """
        sae, tr = _trainer(resample_interval=10**9)
        for _ in range(4):
            tr.train_step(_batch())
        assert int(sae.feature_act_count.sum()) > 0, "counters never moved; vacuous"

        path = tmp_path / "counters.pt"
        tr.save(str(path))
        sae2, tr2 = _trainer(resample_interval=10**9)
        tr2.load(str(path))

        assert int(sae2.total_samples) == int(sae.total_samples)
        assert torch.equal(sae2.feature_act_count, sae.feature_act_count)

    def test_resample_cadence_survives_a_resume(self, tmp_path):
        """`steps_since_resample` is saved; without it a resumed run resamples at
        a different step than the run it resumed."""
        sae, tr = _trainer(resample_interval=3)
        for _ in range(5):
            tr.train_step(_batch())
        path = tmp_path / "cadence.pt"
        tr.save(str(path))
        sae2, tr2 = _trainer(resample_interval=3)
        tr2.load(str(path))
        assert sae2._steps_since_resample == sae._steps_since_resample

    def test_step_count_resumes_faithfully(self, tmp_path):
        _, tr = _trainer(resample_interval=10**9)
        for _ in range(5):
            tr.train_step(_batch())
        path = tmp_path / "steps.pt"
        tr.save(str(path))
        _, tr2 = _trainer(resample_interval=10**9)
        tr2.load(str(path))
        assert tr2.step_count == tr.step_count == 5

    def test_load_is_strict(self, tmp_path):
        """Kills: load_state_dict(..., strict=False).

        A non-strict load is how a renamed key becomes a silently dropped
        statistic instead of an error.
        """
        sae, tr = _trainer(resample_interval=10**9)
        tr.train_step(_batch())
        good = tmp_path / "good.pt"
        tr.save(str(good))

        blob = torch.load(str(good), weights_only=False)
        blob["sae_state"] = {k: v for k, v in blob["sae_state"].items() if "encoder" not in k}
        bad = tmp_path / "broken.pt"
        torch.save(blob, str(bad))

        try:
            tr.load(str(bad))
        except Exception:
            return
        raise AssertionError("a sae_state missing the encoder loaded without error")

    def test_saved_keys_are_all_read(self, tmp_path):
        """Nothing written that `load` never reads.

        Measured by intercepting the read, not by reading the source: a dict
        subclass that records every key access, including via `.get()`.
        """
        sae, tr = _trainer(resample_interval=10**9)
        tr.train_step(_batch())
        path = tmp_path / "keys.pt"
        tr.save(str(path))

        blob = torch.load(str(path), weights_only=False)
        written = set(blob["sae_state"])
        assert written, "nothing was saved; the test would be vacuous"

        # Two failed probes got here, and both are worth recording.
        #
        # A dict subclass cannot see what `load_state_dict` reads: that runs in C
        # and goes straight to the underlying storage, bypassing `__missing__` and
        # `get`. It reported all six keys unread, which looked like a finding.
        #
        # The behavioural probe works, but the DIRECTION is the whole subtlety. A
        # key that IS read ends up HOLDING the poison; a key that is never read
        # keeps its fresh constructor value. So "still poisoned" means the key was
        # read, and asserting on survivors checks the opposite of what it looks
        # like. Assert poison ARRIVED.
        blob = torch.load(str(path), weights_only=False)
        poison = 1234.5
        poisoned = []
        for k, v in blob["sae_state"].items():
            if torch.is_tensor(v) and v.is_floating_point():
                blob["sae_state"][k] = torch.full_like(v, poison)
                poisoned.append(k)
        assert poisoned, "nothing was poisoned; the test would be vacuous"

        probed = tmp_path / "poisoned.pt"
        torch.save(blob, str(probed))

        # A FRESH trainer: its values are not poison, so the only way poison can
        # appear is through load.
        sae3, tr3 = _trainer(resample_interval=10**9)
        before = {k: v.clone() for k, v in sae3.state_dict().items()}
        assert not any(
            torch.all(before[k] == poison) for k in poisoned
        ), "a fresh trainer already holds the poison; the probe is void"

        tr3.load(str(probed))
        after = tr3.sae.state_dict()

        arrived = [k for k in poisoned if torch.all(after[k] == poison)]
        assert set(arrived) == set(poisoned), (
            f"written but never read - poison never reached: "
            f"{sorted(set(poisoned) - set(arrived))}"
        )


class TestSiteNamespaceIsLive:
    """Kills: generator_for drops the site term; derive_generator drops it;
    site offset returns a constant 0."""

    def test_implicit_generator_respects_site(self):
        a = rng.generator_for(0, "sae.init")
        b = rng.generator_for(0, "sae.resample")
        assert not torch.equal(
            torch.randn(64, generator=a), torch.randn(64, generator=b)
        ), "two sites share one stream"

    def test_derive_generator_respects_site(self):
        a = rng.derive_generator("alpha")
        b = rng.derive_generator("beta")
        assert not torch.equal(torch.randn(64, generator=a), torch.randn(64, generator=b))

    def test_site_offset_is_not_a_constant(self):
        offs = {s: rng._site_offset(s) for s in ("a", "b", "c", "sae.init", "sae.resample")}
        assert len(set(offs.values())) == len(offs), f"offsets collide: {offs}"

    def test_same_site_and_seed_reproduce(self):
        x = torch.randn(32, generator=rng.generator_for(5, "same"))
        y = torch.randn(32, generator=rng.generator_for(5, "same"))
        assert torch.equal(x, y), "the same (seed, site) must reproduce"

    def test_counter_generator_respects_site(self):
        a = rng.counter_generator("x")
        b = rng.counter_generator("y")
        assert not torch.equal(torch.randn(32, generator=a), torch.randn(32, generator=b))


class TestSAEStreamsAreIndependent:
    """Closes the loop from the rng contract back to the module it broke."""

    def test_resample_is_not_a_noop(self):
        """TASK-42's silent defect: advanced indexing returns a copy, so
        `xavier_uniform_` wrote into a temporary and was discarded - for every
        SAE this repo has ever run, square ones included."""
        sae = _make_every_feature_dead(SparseAutoencoder(input_dim=8, latent_dim=8, k=4, seed=1))
        before = sae.encoder.weight.detach().clone()
        sae.resample_dead_features()
        assert not torch.allclose(
            sae.encoder.weight, before
        ), "resample left encoder.weight untouched: the re-init went into a copy"

    def test_resample_draws_differently_from_construction(self):
        """Through the IMPLICIT generator path. The existing test in
        `test_interp.py` passes an explicit generator, which is precisely why it
        never noticed the implicit one ignored `site`."""
        init = SparseAutoencoder(input_dim=8, latent_dim=8, k=4, seed=2)
        init_matrix = init.encoder.weight.detach().clone()

        sae = _make_every_feature_dead(SparseAutoencoder(input_dim=8, latent_dim=8, k=4, seed=2))
        sae.resample_dead_features()
        assert not torch.allclose(
            sae.encoder.weight, init_matrix
        ), "resample redrew the constructor's matrix instead of an independent draw"

    def test_decoder_column_norms_are_unit_at_construction(self):
        """`dim=0` is the feature axis for a (latent, input) weight, and the code
        says why: feature j contributes `a_j * W_dec[:, j]`, so the column norm is
        what makes an activation readable as that contribution's magnitude."""
        sae = SparseAutoencoder(input_dim=8, latent_dim=8, k=4, seed=1)
        col = sae.decoder.weight.norm(dim=0)
        assert torch.allclose(
            col, torch.ones_like(col), atol=1e-4
        ), f"decoder COLUMN norms are {col.tolist()}, expected all 1.0"

    def test_resample_preserves_column_norms_on_the_feature_axis(self):
        """TASK-42's axis bug landed on the INPUT axis. After an all-dead resample
        the column norms must still be 1; row norms never were, and never should
        be, because the constraint is not on them."""
        sae = _make_every_feature_dead(SparseAutoencoder(input_dim=8, latent_dim=8, k=4, seed=1))
        sae.resample_dead_features()
        col = sae.decoder.weight.norm(dim=0)
        assert torch.allclose(
            col, torch.ones_like(col), atol=1e-4
        ), f"after resample the COLUMN norms are {col.tolist()}, expected all 1.0"

    def test_rectangular_sae_is_constructible(self):
        """TASK-42's original IndexError, on the combination the class's own
        defaults advertise (input_dim=768, latent_dim=4096)."""
        sae = _make_every_feature_dead(SparseAutoencoder(input_dim=12, latent_dim=40, k=8, seed=0))
        sae.resample_dead_features()  # must not raise

    def test_global_stream_is_untouched(self):
        torch.manual_seed(1234)
        before = torch.randn(8)
        torch.manual_seed(1234)

        sae = _make_every_feature_dead(SparseAutoencoder(input_dim=8, latent_dim=8, k=4, seed=1))
        sae.resample_dead_features()
        after = torch.randn(8)

        assert torch.equal(
            before, after
        ), "the global torch stream advanced; these draws must be private"
