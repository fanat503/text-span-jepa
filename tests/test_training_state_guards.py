# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Training-state guards: eval() must not mutate mechanism state.

`src/train.py::_validate` runs every mechanism's loss under
``model.eval()`` + ``torch.no_grad()``.  Despite that, five mechanism
modules (wsd, sta, rdc, puc, gac) still wrote their EMA / step buffers on
that path, so the *trained weights* depended on whether a validation
split was loaded: two machines with the same seed produced different
models.

The contract enforced here:

  * eval() + no_grad() changes NOTHING (the bug this file exists for)
  * train() still changes everything (guards must not over-correct and
    freeze training)
  * a validation call does not perturb a training trajectory
  * the rule is reachable by grepping ONE name (`_mutate_state`)
  * SPC / CGN keep the guards they already had (pinned, so nobody
    regresses them)

Also covered: `wsr_mode: sam` must not silently substitute the
orthonormality proxy for the documented SAM quantity, and PUC's
100-step re-orthogonalization cadence is pinned.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from unittest import mock

import pytest
import torch

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _snapshot(module):
    """Clone every registered buffer of ``module``."""
    return {name: buf.detach().clone() for name, buf in module.named_buffers()}


def _deltas(before, after):
    """Max per-element absolute change per buffer (exact for ints/bools)."""
    out = {}
    for name, ref in before.items():
        cur = after[name]
        if cur.dtype == torch.bool or not cur.dtype.is_floating_point:
            out[name] = float((cur.long() - ref.long()).abs().max().item())
        else:
            out[name] = (cur - ref).abs().max().item()
    return out


def _changed(deltas):
    return sorted(n for n, d in deltas.items() if d != 0.0)


def _format(deltas):
    return "\n".join(f"    {n:<32} Δ={d:.6g}" for n, d in sorted(deltas.items()))


def _assert_frozen(module, run, label):
    """Run ``run()`` under eval()+no_grad() and demand zero buffer change."""
    module.eval()
    before = _snapshot(module)
    with torch.no_grad():
        run()
    after = _snapshot(module)
    deltas = _deltas(before, after)
    changed = _changed(deltas)
    assert not changed, (
        f"{label}: eval()+no_grad() mutated {len(changed)}/{len(deltas)} buffers "
        f"({', '.join(changed)}).\n{_format({k: v for k, v in deltas.items() if v != 0.0})}"
    )


def _make(name):
    """Build (module, run_callable) for one mechanism, at a tiny size."""
    D = 32
    if name == "wsd":
        from src.models.wsd import WorkspaceSyncDrift

        m = WorkspaceSyncDrift(embed_dim=D, k=4, sync_interval=5)
        Q, _ = torch.linalg.qr(torch.randn(D, 4))
        h = torch.randn(8, D)
        # step=5 hits the periodic-resync branch (5 % sync_interval == 0).
        return m, (lambda: m.compute_drift(Q, h_target=h, step=5))

    if name == "sta":
        from src.models.sta import SpectralTransportAlignment

        m = SpectralTransportAlignment(embed_dim=D, warmup_steps=0, update_interval=2)
        z = torch.randn(8, D)
        return m, (lambda: m(z, step=1))

    if name == "rdc":
        from src.models.rdc import RepresentationDriftCompensation

        m = RepresentationDriftCompensation(embed_dim=D, warmup_steps=0, k_workspace=8)
        z = torch.randn(4, 8, D)
        Q, _ = torch.linalg.qr(torch.randn(D, 8))
        return m, (lambda: m(z, workspace_Q=Q, step=1))

    if name == "puc":
        from src.models.puc import PredictionUncertaintyCalibration

        m = PredictionUncertaintyCalibration(embed_dim=D, n_components=4, warmup_steps=0)
        z = torch.randn(4, 8, D)
        return m, (lambda: m(z, step=1))

    if name == "gac":
        from src.models.gac import GradientAllocatedCapacity

        m = GradientAllocatedCapacity(embed_dim=D, warmup_steps=0)
        z = torch.randn(8, D)
        gn = torch.full((D,), 1e-6)  # every dimension starved -> non-zero EMA
        return m, (lambda: m(z, gn, step=1))

    if name == "spc":
        from src.models.spc import SpectralPredictiveCoding

        m = SpectralPredictiveCoding(embed_dim=D, n_bands=4)
        z_pred = torch.randn(4, D)
        z_target = torch.randn(4, D)
        return m, (lambda: m(z_pred, z_target))

    if name == "cgn":
        from src.models.cgn import ContextualGatingNetwork

        m = ContextualGatingNetwork(embed_dim=D, n_groups=2, anneal_steps=100)
        z = torch.randn(4, 8, D)
        mask = torch.zeros(4, 8)
        mask[:, :4] = 1.0
        return m, (lambda: m(z, mask, step=1))

    raise AssertionError(f"unknown mechanism {name!r}")


#: Mechanisms that were unguarded when this file was written.
BUGGY = ("wsd", "sta", "rdc", "puc", "gac")
#: Mechanisms that were already guarded; pinned so nobody regresses them.
ALREADY_GUARDED = ("spc", "cgn")
ALL_MECHANISMS = BUGGY + ALREADY_GUARDED

#: Buffers each mechanism is *supposed* to move while training.
#: NOTE: `sta.running_w1` is deliberately absent — on the very first call STA
#: initialises BOTH the reference and the current spectrum from the same batch
#: (audit R11), so W1 is exactly 0 and the EMA stays at 0.0.
TRAIN_MUTATES = {
    "wsd": {"target_cov", "target_Q", "is_initialized", "step_count", "running_drift"},
    "sta": {
        "ref_cov",
        "ref_eigenvalues",
        "current_eigenvalues",
        "running_spectral_gap",
        "is_initialized",
        "step_count",
    },
    "rdc": {
        "running_drift_norm",
        "running_ortho_drift_norm",
        "running_workspace_drift_norm",
        "running_drift_ratio",
        "total_steps",
        "z_previous",
    },
    "puc": {
        "running_mean",
        "running_eigenvalues",
        "proj_vectors",
        "running_entropy",
        "running_overconfidence",
        "total_steps",
    },
    "gac": {"running_grad_norms", "running_starved_fraction", "total_gac_steps"},
    "spc": {"adapt_step", "running_residual_vars"},
    "cgn": {"total_steps"},
}


# ═══════════════════════════════════════════════════════════════════════════
#  DEFECT 1 — eval() must be side-effect free
# ═══════════════════════════════════════════════════════════════════════════


class TestEvalDoesNotMutateState:
    @pytest.mark.parametrize("name", ALL_MECHANISMS)
    def test_no_buffer_changes_under_eval(self, name):
        module, run = _make(name)
        _assert_frozen(module, run, name)

    @pytest.mark.parametrize("name", ALL_MECHANISMS)
    def test_repeated_eval_is_a_pure_function(self, name):
        """Same inputs -> bit-identical loss, and still no state change."""
        module, run = _make(name)
        module.eval()
        with torch.no_grad():
            first, second = run()[0], run()[0]
        assert torch.is_tensor(first), f"{name} did not return a tensor loss"
        assert torch.equal(first, second), f"{name}: two identical eval() calls disagreed"

    @pytest.mark.parametrize("name", ALL_MECHANISMS)
    def test_eval_call_does_not_perturb_training(self, name):
        """The headline consequence: a validation pass must not change
        what training subsequently does.  Two runs from the same seed --
        one of them interrupted by a validation call -- must end in the
        same state."""
        torch.manual_seed(20260928)
        clean, clean_run = _make(name)
        torch.manual_seed(20260928)
        dirty, dirty_run = _make(name)
        clean.train()
        dirty.train()

        with torch.no_grad():
            clean_run()
            clean_run()
            # interleaved: train -> validate -> train
            dirty_run()
            dirty.eval()
            dirty_run()
            dirty.train()
            dirty_run()

        deltas = _deltas(_snapshot(clean), _snapshot(dirty))
        changed = _changed(deltas)
        assert not changed, (
            f"{name}: a validation call changed the training trajectory in "
            f"{len(changed)} buffers ({', '.join(changed)}).\n{_format(deltas)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  The guards must not over-correct: training still moves state
# ═══════════════════════════════════════════════════════════════════════════


class TestTrainingModeStillMutates:
    @pytest.mark.parametrize("name", ALL_MECHANISMS)
    def test_training_updates_its_buffers(self, name):
        module, run = _make(name)
        module.train()
        before = _snapshot(module)
        with torch.no_grad():
            run()
        changed = set(_changed(_deltas(before, _snapshot(module))))
        missing = TRAIN_MUTATES[name] - changed
        assert not missing, (
            f"{name}: guard over-corrected — training no longer updates " f"{sorted(missing)}"
        )

    @pytest.mark.parametrize("name", ALL_MECHANISMS)
    def test_train_and_eval_buffers_both_pinned(self, name):
        """Guard is a switch, not a deletion: the same buffers move in
        train mode and stay put in eval mode."""
        module, run = _make(name)
        module.train()
        with torch.no_grad():
            run()
        after_train = _snapshot(module)
        assert set(_changed(_deltas(_snapshot(module), after_train))) == set()

        module.eval()
        with torch.no_grad():
            run()
        assert not _changed(_deltas(after_train, _snapshot(module)))


# ═══════════════════════════════════════════════════════════════════════════
#  The rule must be greppable from one name
# ═══════════════════════════════════════════════════════════════════════════


class TestGuardIsGreppable:
    @pytest.mark.parametrize("name", ALL_MECHANISMS)
    def test_module_uses_the_shared_guard(self, name):
        src = (Path(__file__).resolve().parent.parent / "src" / "models" / f"{name}.py").read_text()
        assert (
            "_mutate_state" in src
        ), f"src/models/{name}.py does not route state writes through _mutate_state"

    def test_guard_is_defined_once(self):
        from src.models.wsd import TrainingStateGuard

        assert hasattr(TrainingStateGuard, "_mutate_state")

    def test_guard_is_a_noop_in_eval(self):
        from src.models.wsd import TrainingStateGuard

        class Dummy(TrainingStateGuard):
            def __init__(self):
                super().__init__()
                self.hits = 0

            def _work(self):
                self.hits += 1
                return "done"

        d = Dummy()
        d.train()
        assert d._mutate_state(d._work) == "done"
        assert d.hits == 1
        d.eval()
        assert d._mutate_state(d._work) is None
        assert d.hits == 1, "_mutate_state ran the callable in eval mode"


# ═══════════════════════════════════════════════════════════════════════════
#  DEFECT 2/3 — WSR must not silently substitute a different loss
# ═══════════════════════════════════════════════════════════════════════════


def _sam_fixture(D=32, k=4, seed=3):
    """A loss whose Q-argument really matters, plus its exact dL/dQ.

    The gradient is computed by autograd from the SAME closure that WSR
    calls, so the SAM perturbation is a genuine ascent direction and
    L(Q+Δ) - L(Q) > 0 must hold.  With the orthonormality proxy instead,
    Δ = 0 and the sharpness is exactly 0.

    Returns:
        (loss_fn, Q, z_pred, z_target, grad)
    """
    gen = torch.Generator().manual_seed(seed)
    z_pred = torch.randn(4, D, generator=gen)
    z_target = torch.randn(4, D, generator=gen)
    target = torch.randn(4, k, generator=gen)

    def loss_fn(Q, _zp, _zt):
        return ((z_pred @ Q) - target).pow(2).mean()

    Q, _ = torch.linalg.qr(torch.randn(D, k, generator=gen))
    Qv = Q.clone().requires_grad_(True)
    loss_fn(Qv, z_pred, z_target).backward()
    return loss_fn, Q, z_pred, z_target, Qv.grad.detach().clone()


class TestWSRSamNoSilentSubstitute:
    def test_orthonormality_proxy_is_degenerate_for_orthonormal_Q(self):
        """Documents the bug.  Q is exactly orthonormal, so the
        orthonormality 'gradient' is exactly 0, the perturbation is 0, and
        L_sam is identically 0 — no regularization at all, under the
        name of the documented SAM loss."""
        from src.models.wsr import WorkspaceSharpnessRegularization

        loss_fn, Q, z_pred, z_target, _grad = _sam_fixture()
        wsr = WorkspaceSharpnessRegularization(embed_dim=32, mode="sam")
        assert Q.grad is None and getattr(wsr, "_lagged_gradient", None) is None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loss, info = wsr(Q, loss_fn, z_pred, z_target, step=1000)
        assert info["wsr_sharpness"] == 0.0
        assert loss.item() == 0.0
        assert loss_fn(Q, z_pred, z_target).item() > 0.0, "the test loss must be non-trivial"

    def test_sam_reports_missing_gradient(self):
        """No gradient anywhere -> loud warning + explicit info flag."""
        from src.models.wsr import WorkspaceSharpnessRegularization

        loss_fn, Q, z_pred, z_target, _grad = _sam_fixture()
        wsr = WorkspaceSharpnessRegularization(embed_dim=32, mode="sam")

        with pytest.warns(UserWarning, match="gradient"):
            _loss, info = wsr(Q, loss_fn, z_pred, z_target, step=1000)
        assert info["wsr_grad_source"] == "unavailable"
        assert info["wsr_gradient_substituted"] is True

    def test_sam_uses_lagged_gradient_when_available(self):
        """The one-step-lagged snapshot is a real ∇_Q L, so sam must
        use it instead of the degenerate proxy: with the proxy the
        perturbation is exactly 0 and the perturbed loss equals the
        current one; with a real gradient it does not."""
        from src.models.wsr import WorkspaceSharpnessRegularization

        loss_fn, Q, z_pred, z_target, grad = _sam_fixture()
        wsr = WorkspaceSharpnessRegularization(embed_dim=32, mode="sam")
        wsr.set_lagged_gradient(grad)

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # must not warn: gradient IS available
            _loss, info = wsr(Q, loss_fn, z_pred, z_target, step=1000)
        assert info["wsr_grad_source"] == "lagged"
        assert info["wsr_gradient_substituted"] is False
        assert info["wsr_grad_norm"] > 0.0
        assert info["wsr_loss_perturbed"] != info["wsr_loss_current"], (
            "SAM perturbation was not applied — the documented L(Q+Δ)-L(Q) " "was never evaluated"
        )

    def test_sam_proxy_evaluates_no_perturbation(self):
        """The orthonormality proxy leaves Q untouched, so L_perturbed ==
        L_current and the documented quantity is identically 0."""
        from src.models.wsr import WorkspaceSharpnessRegularization

        loss_fn, Q, z_pred, z_target, _grad = _sam_fixture()
        wsr = WorkspaceSharpnessRegularization(embed_dim=32, mode="sam")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _loss, info = wsr(Q, loss_fn, z_pred, z_target, step=1000)
        assert info["wsr_grad_norm"] < 1e-9
        assert info["wsr_loss_perturbed"] == info["wsr_loss_current"]
        assert info["wsr_sharpness"] == 0.0

    def test_sam_lagged_perturbs_where_proxy_does_not(self):
        """Property: a real gradient direction actually moves Q, so the
        documented L(Q+Δ)-L(Q) is evaluated; the orthonormality proxy
        never does."""
        from src.models.wsr import WorkspaceSharpnessRegularization

        loss_fn, Q, z_pred, z_target, grad = _sam_fixture()

        def build(lagged):
            wsr = WorkspaceSharpnessRegularization(embed_dim=32, mode="sam")
            if lagged is not None:
                wsr.set_lagged_gradient(lagged)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _l, info = wsr(Q, loss_fn, z_pred, z_target, step=1000)
            return info

        proxy = build(None)
        real = build(grad)
        assert proxy["wsr_loss_perturbed"] == proxy["wsr_loss_current"]
        assert real["wsr_loss_perturbed"] != real["wsr_loss_current"]
        assert not math.isclose(
            real["wsr_loss_perturbed"],
            proxy["wsr_loss_perturbed"],
            rel_tol=1e-6,
            abs_tol=1e-9,
        )

    def test_gradient_mode_reports_missing_gradient(self):
        """DEFECT 3: the MechanismBundle path feeds WSR a non-leaf,
        grad-less Q slice, so the proxy is taken there too."""
        from src.models.wsr import WorkspaceSharpnessRegularization

        wsr = WorkspaceSharpnessRegularization(embed_dim=32, mode="gradient")
        Q = torch.randn(32, 4, requires_grad=True)  # non-leaf slice, as in mechanisms.py
        Q2 = Q[:, :4] * 1.0
        Q2.retain_grad()
        with pytest.warns(UserWarning, match="set_lagged_gradient"):
            _loss, info = wsr(Q2, step=1000)
        assert info["wsr_grad_source"] == "unavailable"
        assert info["wsr_gradient_substituted"] is True

    def test_gradient_mode_quiet_when_gradient_available(self):
        from src.models.wsr import WorkspaceSharpnessRegularization

        _fn, Q, _zp, _zt, grad = _sam_fixture()
        wsr = WorkspaceSharpnessRegularization(embed_dim=32, mode="gradient")
        wsr.set_lagged_gradient(grad)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _loss, info = wsr(Q, step=1000)
        assert info["wsr_grad_source"] == "lagged"
        assert info["wsr_gradient_substituted"] is False

    def test_live_leaf_grad_is_used(self):
        """A leaf Q that still holds a grad (same-step forward) is a
        legitimate source and must not warn."""
        from src.models.wsr import WorkspaceSharpnessRegularization

        _fn, Q, _zp, _zt, grad = _sam_fixture()
        wsr = WorkspaceSharpnessRegularization(embed_dim=32, mode="gradient")
        Q.grad = grad.clone()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _loss, info = wsr(Q, step=1000)
        assert info["wsr_grad_source"] == "live"

    def test_eval_does_not_move_wsr_statistics(self):
        from src.models.wsr import WorkspaceSharpnessRegularization

        _fn, Q, _zp, _zt, grad = _sam_fixture()
        wsr = WorkspaceSharpnessRegularization(embed_dim=32, mode="gradient")
        wsr.set_lagged_gradient(grad)
        wsr.eval()
        before = {n: b.clone() for n, b in wsr.named_buffers()}
        with torch.no_grad():
            wsr(Q, step=1000)
        for name, buf in wsr.named_buffers():
            assert torch.equal(buf, before[name]), f"wsr eval mutated {name}"

    @pytest.mark.xfail(
        reason=(
            "OPEN DEFECT (not mine to decide): _stiefel_retract picks its "
            "column signs from diag((Q R)[:k, :]) instead of diag(R). For a "
            "generic orthonormal Q that block is not triangular, so the "
            "signs are arbitrary and the retraction permutes/flips columns "
            "rather than nudging the subspace by rho. mode='sam' therefore "
            "still perturbs in a near-arbitrary direction even now that the "
            "real dL/dQ is available. Fixing it changes what "
            "wsr_mode='sam' computes, so it needs a human decision."
        ),
        strict=False,
    )
    def test_retraction_preserves_column_orientation(self):
        """Expected property: retract(Q + ε·tangent) stays within ε of Q."""
        from src.models.wsr import WorkspaceSharpnessRegularization

        wsr = WorkspaceSharpnessRegularization(embed_dim=32, rho=0.05)
        Q, _ = torch.linalg.qr(torch.randn(32, 4))
        t = torch.randn(32, 4)
        t = t - Q @ (Q.T @ t)
        t = t / t.norm()
        Q_ret = wsr._stiefel_retract(Q + 0.05 * t)
        # Sign/rotation-invariant distance: same subspace means 0.
        assert (Q @ Q.T - Q_ret @ Q_ret.T).norm().item() < 1e-3
        assert (Q - Q_ret).abs().max().item() < 0.2


# ═══════════════════════════════════════════════════════════════════════════
#  DEFECT 4 — RNG / cadence
# ═══════════════════════════════════════════════════════════════════════════


class TestCGNGateIsDeterministicInEval:
    def test_gumbel_noise_is_training_only(self):
        """cgn's Gumbel-Softmax draw must be training-gated: under eval
        the gate must not depend on the global RNG stream."""
        from src.models.cgn import ContextualGatingNetwork

        cgn = ContextualGatingNetwork(embed_dim=32, n_groups=2, anneal_steps=100)
        z = torch.randn(4, 8, 32)
        mask = torch.zeros(4, 8)
        mask[:, :4] = 1.0

        cgn.eval()
        torch.manual_seed(0)
        a, _ = cgn(z, mask, step=1)
        torch.manual_seed(999_999)
        b, _ = cgn(z, mask, step=1)
        assert torch.equal(a, b), "eval-mode gate consumed RNG"

        cgn.train()
        torch.manual_seed(0)
        c, _ = cgn(z, mask, step=1)
        torch.manual_seed(999_999)
        d, _ = cgn(z, mask, step=1)
        assert not torch.equal(c, d), "training-mode gate is not sampling Gumbel noise"

    def test_eval_does_not_advance_anneal_counter(self):
        from src.models.cgn import ContextualGatingNetwork

        cgn = ContextualGatingNetwork(embed_dim=32, n_groups=2, anneal_steps=1000)
        z = torch.randn(4, 8, 32)
        mask = torch.zeros(4, 8)
        cgn.eval()
        cgn(z, mask, step=500)
        cgn(z, mask, step=900)
        assert cgn.total_steps.item() == 0
        cgn.train()
        cgn(z, mask, step=500)
        assert cgn.total_steps.item() == 500


class TestPUCReorthogonalizationCadence:
    def test_runs_every_100_steps(self):
        """Pins the re-orthogonalization cadence.  It is keyed off
        `step`, so it is coupled to the resume point — a checkpoint
        restored at step 137 re-orthogonalizes on the next 200-boundary,
        not 137 steps later."""
        from src.models.puc import PredictionUncertaintyCalibration

        puc = PredictionUncertaintyCalibration(embed_dim=16, n_components=4, warmup_steps=0)
        calls = []
        original = PredictionUncertaintyCalibration._orthogonalize_projections

        def counting(self):
            calls.append(self.total_steps.item())
            return original(self)

        z = torch.randn(2, 4, 16)
        with mock.patch.object(
            PredictionUncertaintyCalibration, "_orthogonalize_projections", counting
        ):
            for step in range(1, 251):
                puc(z, step=step)

        assert len(calls) == 2, f"expected 2 re-orthogonalizations in steps 1..250, got {calls}"

    def test_projections_start_orthonormal(self):
        from src.models.puc import PredictionUncertaintyCalibration

        puc = PredictionUncertaintyCalibration(embed_dim=16, n_components=4)
        gram = puc.proj_vectors @ puc.proj_vectors.T
        assert torch.allclose(gram, torch.eye(4), atol=1e-5)

    def test_no_projection_vector_reseeding_in_eval(self):
        """The randn_like fallback in _orthogonalize_projections is a
        rare degenerate branch, but it must not become reachable from a
        validation pass."""
        from src.models.puc import PredictionUncertaintyCalibration

        puc = PredictionUncertaintyCalibration(embed_dim=16, n_components=4, warmup_steps=0)
        z = torch.randn(2, 4, 16)
        puc.train()
        puc(z, step=100)  # triggers the 100-step re-orthogonalization
        after_train = puc.proj_vectors.clone()

        puc.eval()
        with torch.no_grad():
            for step in (100, 150, 200, 250):
                puc(z, step=step)
        assert torch.equal(puc.proj_vectors, after_train)
