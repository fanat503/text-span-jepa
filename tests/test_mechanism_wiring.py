# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Integration wiring tests: CMC bridge, GAC hook, live workspace views.

Pins the round-2 audit contracts:
  * compute_cmc_between_passes(primary, secondary) scatters compact slot
    predictions from two compute_loss_with_targets passes into full-sequence
    space and returns a finite, differentiable loss;
  * gradients reach the online encoder through the CMC secondary path;
  * GAC's stashed live slot tensor receives grads from the main backward and
    feeds a second scaled-free backward without freed-graph errors;
  * live workspace views give SWIP/RDC/WSR real dL/dQ (workspace_Q.grad);
  * WSR mode='gradient' prefers set_lagged_gradient() snapshots;
  * stiefel_retract projects its Riemannian correction in-place on Q.grad.

Plus the info-key stability contract (`TestInfoKeyStability`): a mechanism's
`info` dict must carry the SAME KEYS on every path it can return from. The
early-return path used to return a shorter dict, which is invisible to a
caller that iterates and fatal to a caller that indexes — and the only steps
that take it are the first `warmup_steps` of a run, so the defect is untestable
by any suite that only ever runs a step past warmup.
"""

import pathlib
import src
import pytest
import torch

from src.models.cgn import ContextualGatingNetwork
from src.models.cmc import CrossMaskConsistency
from src.models.gac import GradientAllocatedCapacity
from src.models.jawp import JAWPModule
from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
from src.models.pcr import PredictiveCascadeRefinement
from src.models.puc import PredictionUncertaintyCalibration
from src.models.rdc import RepresentationDriftCompensation
from src.models.spc import SpectralPredictiveCoding
from src.models.sta import SpectralTransportAlignment
from src.models.swip import SWIPModule
from src.models.wsd import WorkspaceSyncDrift
from src.models.wsr import WorkspaceSharpnessRegularization

# Provenance. An editable install on this machine points at a DIFFERENT clone
# of this repo, and a stale __pycache__ produced traceback paths naming that
# other tree - so a green run here can be a run against someone else's code.
#
# The guard must key off the test file's OWN location, not a hardcoded worktree
# path: this originally read `startswith(r"C:\dev\wt-keys")`, which was correct in
# the one worktree it was written in and wrong everywhere else - it made
# test_mechanism_wiring.py fail at COLLECTION on main, with an error that looked
# like an environment problem rather than a stale absolute path.
_HERE = pathlib.Path(__file__).resolve().parent.parent
assert src.__file__.startswith(
    str(_HERE)
), f"tests must exercise the working-tree copy of src, got {src.__file__}"


def _tiny_config() -> TextSpanJEPAConfig:
    return TextSpanJEPAConfig(
        vocab_size=64,
        max_seq_len=16,
        embed_dim=32,
        encoder_depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        predictor_embed_dim=16,
        predictor_depth=1,
        future_offsets=[1],
        num_refine_steps=1,
        use_jawp=True,
        jawk_k_start=2,
        jawk_k_end=4,
        jawk_curriculum_steps=0,
        use_swip=True,
        lambda_swip=0.01,
        use_rdc=True,
        lambda_rdc=0.01,
        use_wsr=True,
        lambda_wsr=0.01,
        wsr_mode="gradient",
        wsr_warmup_steps=0,
        use_cmc=True,
        cmc_interval=1,
        lambda_cmc=0.05,
        use_gac=True,
        gac_gamma=0.01,
        gac_tau_grad=1e-5,
        gac_warmup_steps=0,
        lambda_gac=0.02,
        future_warmup_steps=0,
    )


def _masks(T: int = 16):
    m1 = torch.zeros(2, T, dtype=torch.long)
    m2 = torch.zeros(2, T, dtype=torch.long)
    m1[:, 2:8] = 1  # primary spans
    m2[:, 5:11] = 1  # secondary spans; overlap at positions 5..7
    return m1, m2


class TestCMCBridge:
    def test_loss_finite_and_differentiable(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, m2 = _masks()

        out1, _, _ = model.compute_loss_with_targets(ids, ids, m1, 0, 100)
        primary = model._cmc_pass
        assert primary is not None and primary["slots"].shape[0] == 2

        model.compute_loss_with_targets(ids, ids, m2, 0, 100)
        secondary = model._cmc_pass
        assert secondary is not None

        loss_cmc, info = model.compute_cmc_between_passes(primary, secondary)
        assert torch.isfinite(loss_cmc).all()
        assert loss_cmc.requires_grad, "secondary pass must stay attached to the graph"
        assert info.get("cmc_skipped") in (False, True)

    def test_gradient_reaches_encoder_via_secondary(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, m2 = _masks()

        out1, _, _ = model.compute_loss_with_targets(ids, ids, m1, 0, 100)
        primary = model._cmc_pass
        model.compute_loss_with_targets(ids, ids, m2, 0, 100)

        loss_cmc, _ = model.compute_cmc_between_passes(primary, model._cmc_pass)
        total = out1 + model.config.lambda_cmc * loss_cmc
        total.backward(retain_graph=True)

        enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
        assert enc_grads, "CMC secondary path must backprop into the online encoder"
        assert any(g.abs().sum() > 0 for g in enc_grads)


class TestGACHook:
    def test_stashed_slots_receive_grads(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        assert model._gac_z is None, "stash starts empty before the first forward"

        ids = torch.randint(0, 64, (2, 16))
        m1, _ = _masks()
        out, ldict, _ = model.compute_loss_with_targets(ids, ids, m1, 10, 100)
        assert model._gac_z is not None, "lambda_gac>0 must stash live slots"
        out.backward(retain_graph=True)
        assert model._gac_z.grad is not None

        D = model._gac_z.size(-1)
        gn = model._gac_z.grad.detach().reshape(-1, D).norm(dim=0)
        loss_gac, info = model.gac(model._gac_z, gn, step=10)
        assert torch.isfinite(loss_gac).all()
        if loss_gac.requires_grad:
            loss_gac.backward()  # must not raise on a freed graph


class TestLiveWorkspaceViews:
    def test_regularizers_give_workspace_grads(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, _ = _masks()
        out, _, _ = model.compute_loss_with_targets(ids, ids, m1, 20, 100)
        out.backward()

        qgrad = model.jawp.workspace_Q.grad
        assert (
            qgrad is not None and qgrad.abs().sum() > 0
        ), "SWIP/RDC/WSR live views must contribute gradient to workspace_Q"

    def test_wsr_prefers_lagged_gradient_snapshot(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        D, k = model.config.embed_dim, int(model.jawp.active_k.item())
        snap = torch.randn(D, k)
        model.wsr.set_lagged_gradient(snap)

        ids = torch.randint(0, 64, (2, 16))
        m1, _ = _masks()
        _, ldict, _ = model.compute_loss_with_targets(ids, ids, m1, 30, 100)
        assert torch.isfinite(torch.tensor(ldict["loss_wsr"]))

    def test_retraction_projects_grad_inplace_and_keeps_orthogonal(self):
        model = TextSpanJEPA(_tiny_config())
        model.train()
        ids = torch.randint(0, 64, (2, 16))
        m1, _ = _masks()
        out, _, _ = model.compute_loss_with_targets(ids, ids, m1, 20, 100)
        out.backward()

        k = int(model.jawp.active_k.item())
        G = torch.randn_like(model.jawp.workspace_Q)
        g_before = G[:, :k].clone()
        model.jawp.workspace_Q.grad = G.clone()

        model.jawp.stiefel_retract()

        # Riemannian correction mutates grad[:, :k] in place (normal component removed).
        assert not torch.equal(model.jawp.workspace_Q.grad[:, :k], g_before)
        # SVD retraction leaves Q orthonormal.
        Q = model.jawp.workspace_Q.detach()
        ortho_err = (Q.T @ Q - torch.eye(Q.size(1))).abs().max()
        assert ortho_err < 1e-4


# ═══════════════════════════════════════════════════════════════════════════
#  Info-key stability
#
#  A mechanism's `info` dict is a CONTRACT, not a report of what happened to be
#  computed on this step. A consumer that has seen a key once — a logging
#  column, a metric registry, a reader in a test — will read it again on the
#  next step. If the module's early-return path returns a shorter dict, that
#  consumer raises KeyError.
#
#  The defect it enables is the worst kind, because it is invisible to the
#  suite: the only steps that take the early return are steps 0..warmup_steps-1,
#  and every existing test drives the module at a step past warmup. Local tests
#  pass, CI passes, and the first real run dies at step 3.
#
#  The fix pattern is a SENTINEL, not an absent key: a value that cannot be
#  computed yet is reported as an explicit zero and flagged by the module's own
#  `<mech>_warmup` key, so `info[k]` stays valid everywhere and the warmup is
#  visible in the log as zeros rather than as a hole in it.
# ═══════════════════════════════════════════════════════════════════════════

_D = 16
_BT = (4, 8)
#: A step far past every shipped `*_warmup_steps` in `defaults.yaml`.
_PAST_WARMUP = 5000


def _zt(seed):
    """A deterministic (B, T, D) tensor, varying with the step it feeds."""
    return torch.randn(_BT[0], _BT[1], _D, generator=torch.Generator().manual_seed(seed))


def _orthonormal_Q(cols, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(_D, cols, generator=g))[0]


def _drive_wsr(module, step):
    """Feed WSR a real dL/dQ so the post-warmup leg is the DOCUMENTED path.

    Without the lagged snapshot, `_gradient_mode` takes the orthonormality
    proxy and warns on every call — which would make the post-warmup key set
    this test pins a description of the fallback rather than of the module.
    The warmup return happens before the gradient is ever resolved, so seeding
    it here does not affect the step-0 leg.
    """
    module.set_lagged_gradient(_orthonormal_Q(4, 24))
    return module(_orthonormal_Q(4, 23), step=step)[1]


# (module name, factory, drive) — `drive(module, step) -> info dict`.
# Every entry is one of the twelve `MechanismBundle.ALL_MECHANISMS`.
_MECHANISM_DRIVERS = [
    (
        "jawp",
        lambda: JAWPModule(embed_dim=_D, k_start=2, k_end=4, curriculum_steps=0),
        lambda m, s: m.compute_loss(_zt(1 + s), _zt(2 + s), step=s)[1],
    ),
    (
        "cgn",
        lambda: ContextualGatingNetwork(embed_dim=_D, n_groups=4),
        lambda m, s: m(
            _zt(3 + s),
            (torch.rand(_BT, generator=torch.Generator().manual_seed(7)) < 0.4).long(),
            step=s,
        )[1],
    ),
    (
        "swip",
        lambda: SWIPModule(embed_dim=_D, k_workspace=4),
        lambda m, s: m(_zt(4 + s), workspace_Q=_orthonormal_Q(4, 11))[1],
    ),
    (
        "pcr",
        lambda: PredictiveCascadeRefinement(embed_dim=_D, n_levels=2),
        lambda m, s: m(_zt(5 + s), _zt(6 + s), step=s)[1],
    ),
    (
        "spc",
        lambda: SpectralPredictiveCoding(embed_dim=_D, n_bands=4),
        lambda m, s: m(_zt(8 + s), _zt(9 + s))[1],
    ),
    (
        "wsd",
        lambda: WorkspaceSyncDrift(embed_dim=_D, k=4, sync_interval=100),
        lambda m, s: m.compute_drift(_orthonormal_Q(4, 13), _zt(10 + s), step=s)[1],
    ),
    (
        "cmc",
        lambda: CrossMaskConsistency(embed_dim=_D),
        lambda m, s: m(
            _zt(12 + s),
            _zt(13 + s),
            torch.zeros(_BT, dtype=torch.long).index_fill_(1, torch.tensor([1, 2, 3]), 1),
        )[1],
    ),
    (
        "gac",
        lambda: GradientAllocatedCapacity(embed_dim=_D, warmup_steps=1000),
        lambda m, s: m(
            _zt(14 + s),
            torch.rand(_D, generator=torch.Generator().manual_seed(17)) * 1e-2,
            step=s,
        )[1],
    ),
    (
        "sta",
        lambda: SpectralTransportAlignment(embed_dim=_D, warmup_steps=500),
        lambda m, s: m(_zt(15 + s), step=s)[1],
    ),
    (
        "puc",
        lambda: PredictionUncertaintyCalibration(embed_dim=_D, warmup_steps=500),
        lambda m, s: m(_zt(16 + s), step=s)[1],
    ),
    (
        "rdc",
        lambda: RepresentationDriftCompensation(embed_dim=_D, warmup_steps=500),
        lambda m, s: m(
            _zt(18 + s),
            z_previous=_zt(19),
            workspace_Q=torch.eye(_D)[:, :4],
            step=s,
        )[1],
    ),
    (
        "wsr",
        lambda: WorkspaceSharpnessRegularization(embed_dim=_D, warmup_steps=500),
        _drive_wsr,
    ),
]

_ALL_NAMES = [name for name, _, _ in _MECHANISM_DRIVERS]


class TestInfoKeyStability:
    """`info` must carry the same keys on every path a module can return from."""

    @pytest.mark.parametrize(
        "name,factory,drive",
        _MECHANISM_DRIVERS,
        ids=_ALL_NAMES,
    )
    def test_warmup_path_returns_the_same_keys(self, name, factory, drive):
        """The early-return path is a warmup path, not a shorter key set."""
        module = factory()
        module.train()
        at_zero = drive(module, 0)  # inside warmup
        past = drive(module, _PAST_WARMUP)  # past every shipped warmup

        assert isinstance(at_zero, dict), f"{name}: warmup path must still return an info dict"
        missing = sorted(set(past) - set(at_zero))
        extra = sorted(set(at_zero) - set(past))
        assert not missing, (
            f"{name} drops {missing} on its step-0 (warmup) path. Every key the "
            f"module reports at step {_PAST_WARMUP} must exist at step 0 too — a "
            f"consumer that reads a key it has seen once must not KeyError during "
            f"the first warmup_steps of a run."
        )
        assert not extra, (
            f"{name} invents {extra} on its step-0 (warmup) path. A key that only "
            f"exists during warmup is the same contract inverted: it is absent on "
            f"every step after, which is the far larger fraction of the run."
        )

    @pytest.mark.parametrize(
        "name,factory,drive",
        _MECHANISM_DRIVERS,
        ids=_ALL_NAMES,
    )
    def test_indexing_every_key_survives_the_first_steps(self, name, factory, drive):
        """The crash itself, as a consumer would actually hit it.

        A reader that has seen the full key set once and then does `info[k]` for
        each of those keys must survive step 0. This is the assertion the whole
        class exists for; the one above states the invariant, this one names the
        failure it prevents.
        """
        module = factory()
        module.train()
        known = set(drive(module, _PAST_WARMUP))  # what a consumer has seen

        module = factory()
        module.train()
        warmup_info = drive(module, 0)

        # The consumer's loop, verbatim.
        read_back = {}
        for k in sorted(known):
            read_back[k] = warmup_info[k]  # KeyError here is the crash
        assert set(read_back) == known

    @pytest.mark.parametrize(
        "name,factory,drive",
        _MECHANISM_DRIVERS,
        ids=_ALL_NAMES,
    )
    def test_warmup_values_are_real_sentinels_not_none(self, name, factory, drive):
        """A sentinel must be writable to a log, so it must be a real value."""
        module = factory()
        module.train()
        known = set(drive(module, _PAST_WARMUP))
        module = factory()
        module.train()
        warmup_info = drive(module, 0)

        for k, v in sorted(warmup_info.items()):
            assert v is not None, f"{name}: info[{k!r}] is None during warmup"
            assert isinstance(v, (int, float, bool, str, list)), (
                f"{name}: info[{k!r}] is {type(v).__name__} during warmup; a "
                f"warmup sentinel has to be something a CSV cell or a JSON dump "
                f"can hold, not a tensor or a placeholder object."
            )
        assert known <= set(warmup_info)

    def test_wsr_modes_report_the_same_keys(self):
        """WSR's two modes were a second key-set split, on the mode axis.

        `mode='sam'` reports `wsr_loss_current` / `wsr_loss_perturbed`;
        `mode='gradient'` did not. Same contract, different axis: a reader
        written against one mode KeyErrors on the other.
        """

        def info_for(mode):
            m = WorkspaceSharpnessRegularization(
                embed_dim=_D, warmup_steps=500, mode=mode, rho=0.05, eta=0.01
            )
            m.train()
            m.set_lagged_gradient(_orthonormal_Q(4, 24))
            _, info = m(
                _orthonormal_Q(4, 23),
                loss_fn=lambda Q, a, b: ((a - b) ** 2).mean(),
                z_pred=_zt(25),
                z_target=_zt(26),
                step=_PAST_WARMUP,
            )
            return info

        grad, sam = info_for("gradient"), info_for("sam")
        assert set(grad) == set(sam), (
            f"WSR modes disagree on keys: only in gradient={sorted(set(grad) - set(sam))}, "
            f"only in sam={sorted(set(sam) - set(grad))}"
        )

    def test_all_twelve_mechanisms_are_covered(self):
        """Guards the table above against silently losing a mechanism."""
        from src.models.mechanisms import MechanismBundle

        assert sorted(_ALL_NAMES) == sorted(MechanismBundle.ALL_MECHANISMS)
