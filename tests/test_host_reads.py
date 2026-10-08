# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""The per-step path must not read scalars back to the host one at a time.

`.item()`, `.tolist()` and `bool(tensor)` on a CUDA tensor are hard device
synchronisations. In a per-step path they serialise the step against the
device, and this repo's training loop calls these three mechanisms on every
step. The values themselves were kept bit-identical while the reads were
batched; these tests pin BOTH halves of that claim, because either alone is
worthless:

  1. no per-value host read in the step path (a count, not a wall clock), and
  2. the batched read returns exactly the float the old `.item()` returned
     (a reference implementation of the old code, compared for equality).

`torch.cuda.synchronize()` cannot be used: the suite is CPU-only, so the
count is taken by patching `torch.Tensor.item` / `.tolist` / `.__bool__` and
recording the call sites. That counts the operations which WOULD synchronise
on CUDA. It is exact for that purpose and needs no GPU.

What this file does NOT establish: any wall-clock speedup. There is no CUDA
on this host, so "faster" is not measurable here — see .agent-notes/task-21.md.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src.models.jawp import JAWPModule
from src.models.pcr import PredictiveCascadeRefinement
from src.models.spc import SpectralPredictiveCoding

D = 64


# ═══════════════════════════════════════════════════════════════════════
#  The counter
# ═══════════════════════════════════════════════════════════════════════


class ReadCounter:
    """Count host reads, tagged with the file that made them."""

    def __init__(self, files):
        self.files = files
        self.sites = []
        self.active = False

    def _note(self, kind):
        if not self.active:
            return
        # Walk out to the nearest frame belonging to a watched file. The depth
        # differs per entry point (`item` is reached through one extra Python
        # frame for a list comprehension, `.tolist` not at all), so search
        # rather than index blindly. Frames above the patched method are
        # skipped so a read from torch internals is never attributed to a
        # mechanism file.
        f = sys._getframe(1)
        while f is not None:
            name = os.path.basename(f.f_code.co_filename)
            if name in self.files:
                # The kind is part of the tag: `bool` is the one that must never
                # appear in a step path, and a bare "jawp.py:416" would hide it.
                self.sites.append(f"{name}:{f.f_lineno} [{kind}]")
                return
            f = f.f_back
        self.sites.append(f"<outside watched files> [{kind}]")

    def __enter__(self):
        self.sites = []
        self.active = True
        ctr = self

        self._orig = {
            "item": torch.Tensor.item,
            "tolist": torch.Tensor.tolist,
            "bool": torch.Tensor.__bool__,
        }

        def item(t):
            ctr._note("item")
            return ctr._orig["item"](t)

        def tolist(t):
            ctr._note("tolist")
            return ctr._orig["tolist"](t)

        def bool_(t):
            ctr._note("bool")
            return ctr._orig["bool"](t)

        torch.Tensor.item = item
        torch.Tensor.tolist = tolist
        torch.Tensor.__bool__ = bool_
        return self

    def __exit__(self, *exc):
        torch.Tensor.item = self._orig["item"]
        torch.Tensor.tolist = self._orig["tolist"]
        torch.Tensor.__bool__ = self._orig["bool"]
        self.active = False
        return False

    @property
    def count(self):
        return len(self.sites)


# ═══════════════════════════════════════════════════════════════════════
#  1. The step path batches its reads
# ═══════════════════════════════════════════════════════════════════════


def _spc_step(spc, z_pred, z_target, step=5):
    spc(z_pred, z_target)
    loss, _ = spc(z_pred, z_target)
    spc.stiefel_retract()
    return loss


class TestSPCReadBatching:
    """SPC reported 24 host reads per step; n_bands of them were per-band."""

    def test_read_count_is_independent_of_n_bands(self):
        """The point of batching: doubling the bands must not add reads.

        This is the load-bearing property. The old code did one `.item()` per
        band for `band_residuals`, one for `band_losses` and one inside
        `_update_running_statistics`, so the count grew linearly with
        `n_bands`. A batched read does not.
        """
        counts = {}
        for n_bands in (4, 8, 16):
            torch.manual_seed(0)
            spc = SpectralPredictiveCoding(embed_dim=D, n_bands=n_bands, init="dct")
            spc.train()
            z_pred = torch.randn(8, D)
            z_target = torch.randn(8, D)
            spc(z_pred, z_target)  # warm any lazy init
            with ReadCounter({"spc.py"}) as ctr:
                spc(z_pred, z_target)
            counts[n_bands] = ctr.count
        assert (
            counts[4] == counts[8] == counts[16]
        ), f"host reads scale with n_bands, so the reads were not batched: {counts}"

    def test_no_per_value_item_in_the_step_path(self):
        """Every remaining read must be a batched one, not a per-value `.item()`.

        A `.tolist()` reads a whole vector in one transfer; a bare `.item()`
        reads one scalar and blocks. A per-value `.item()` is what this card
        removed, so its return is a failure even though the total is small.
        """
        torch.manual_seed(0)
        spc = SpectralPredictiveCoding(embed_dim=D, n_bands=8, init="dct")
        spc.train()
        z_pred = torch.randn(8, D)
        z_target = torch.randn(8, D)
        spc(z_pred, z_target)
        with ReadCounter({"spc.py"}) as ctr:
            spc(z_pred, z_target)
        assert ctr.sites, "counter saw nothing at all; the patch is not working"
        # No site is called more than twice: the read is per-vector, not per-band.
        from collections import Counter

        per_site = Counter(ctr.sites)
        worst = max(per_site.values())
        assert worst <= 2, f"a host-read site runs {worst}x per step: {per_site.most_common(3)}"


class TestPCRReadBatching:
    """PCR reported 21 host reads per step: 4 per level plus 3 overall."""

    def test_read_count_is_independent_of_n_levels(self):
        counts = {}
        for n_levels, dims in ((1, [16]), (2, [16, 8]), (3, [16, 8, 4])):
            torch.manual_seed(0)
            pcr = PredictiveCascadeRefinement(embed_dim=D, n_levels=n_levels, level_dims=dims)
            with torch.no_grad():
                for g in pcr.level_gates:
                    g.copy_(torch.tensor(0.5))
            z_pred = torch.randn(8, D)
            z_target = torch.randn(8, D)
            pcr(z_pred, z_target, step=0)  # warm
            with ReadCounter({"pcr.py"}) as ctr:
                pcr(z_pred, z_target, step=0)
            counts[n_levels] = ctr.count
        assert (
            counts[1] == counts[2] == counts[3]
        ), f"host reads scale with n_levels, so the per-level reads were not batched: {counts}"

    def test_level_offsets_mirror_matches_the_buffer(self):
        """The host offset mirror must equal the `level_offsets` buffer.

        `_get_subspace_proj` slices with the mirror instead of reading the
        buffer, so if the two ever disagreed the cascade would silently refine
        in the wrong subspaces. The buffer stays in the state_dict, so this is
        the assertion that keeps the mirror honest.
        """
        for n_levels, dims in (
            (1, [16]),
            (2, [32, 16]),
            (3, None),
            (4, [8, 8, 4, 2]),
            (3, [20, 10, 5]),
        ):
            pcr = PredictiveCascadeRefinement(embed_dim=D, n_levels=n_levels, level_dims=dims)
            assert pcr._level_offsets_py == pcr.level_offsets.tolist()
            for l in range(pcr.n_levels):
                off = pcr.level_offsets[l].item()
                dim = pcr.level_dims[l]
                assert torch.equal(
                    pcr._get_subspace_proj(l), pcr.workspace_Q[:, off : off + dim]
                ), f"level {l} projection differs from the buffer-derived slice"


class TestJAWPReadBatching:
    """JAWP reported 13 host reads per step, 2 of them `bool(tensor)`."""

    def test_no_bool_of_a_tensor_in_the_step_path(self):
        """`if some_tensor > x:` is a sync. The cosine guard must be a where().

        This is the one that is easy to reintroduce by accident: the guard
        reads naturally as a Python `if` over two norms.
        """
        torch.manual_seed(0)
        jawp = JAWPModule(embed_dim=D, k_start=1, k_end=8, curriculum_steps=10)
        z_pred = torch.randn(8, D)
        z_target = torch.randn(8, D)
        jawp.compute_loss(z_pred, z_target, step=5)  # warm
        with ReadCounter({"jawp.py"}) as ctr:
            jawp.compute_loss(z_pred, z_target, step=5)
        assert ctr.sites, "counter saw nothing at all; the patch is not working"
        bools = [s for s in ctr.sites if "[bool]" in s]
        assert not bools, f"a tensor was converted to bool in the step path: {bools}"

    def test_active_k_mirror_tracks_the_buffer(self):
        """`active_k` is a state_dict buffer; the host mirror must match it.

        Every slice in the mechanism takes its width from the mirror, so a
        desync would mean refining/retracting over the wrong number of columns.
        """
        torch.manual_seed(0)
        jawp = JAWPModule(embed_dim=D, k_start=1, k_end=8, curriculum_steps=10)
        for step in (0, 3, 9, 10, 500):
            jawp.compute_loss(torch.randn(4, D), torch.randn(4, D), step=step)
            assert jawp.active_k_value() == int(jawp.active_k.item()), (
                f"mirror {jawp.active_k_value()} != buffer {int(jawp.active_k.item())} "
                f"at step {step}"
            )

    def test_active_k_mirror_resyncs_on_checkpoint_load(self):
        """A resumed run must not keep reporting the constructor's k_start.

        `active_k` travels in the state_dict, so a checkpoint written deep into
        the curriculum restores a width the fresh module never saw. Without the
        re-read in `_load_from_state_dict` the mirror would keep slicing at
        `k_start` for the rest of the run.
        """
        torch.manual_seed(0)
        deep = JAWPModule(embed_dim=D, k_start=1, k_end=8, curriculum_steps=10)
        deep.compute_loss(torch.randn(4, D), torch.randn(4, D), step=500)
        assert deep.active_k_value() == 8

        fresh = JAWPModule(embed_dim=D, k_start=1, k_end=8, curriculum_steps=10)
        assert fresh.active_k_value() == 1, "precondition: a fresh module starts at k_start"
        fresh.load_state_dict(deep.state_dict())
        assert fresh.active_k_value() == int(fresh.active_k.item())
        assert fresh.active_k_value() == 8, (
            f"after load the mirror says {fresh.active_k_value()}, "
            f"but the buffer says {int(fresh.active_k.item())}"
        )

    def test_active_k_width_is_not_read_back_from_the_buffer(self):
        """Taking the width from the mirror must not cost a device sync.

        Value-agreement tests cannot catch a regression here: an implementation
        that returns the RIGHT number by reading `active_k.item()` still passes
        them, while reintroducing a sync on every retraction and every tangent
        projection. Only the read count separates the two. (Found by mutating
        `active_k_value` to `int(self.active_k.item())` — correct, green, and
        still a sync.)
        """
        torch.manual_seed(0)
        jawp = JAWPModule(embed_dim=D, k_start=1, k_end=8, curriculum_steps=10)
        z_pred = torch.randn(8, D)
        z_target = torch.randn(8, D)
        loss, _ = jawp.compute_loss(z_pred, z_target, step=5)
        loss.backward()  # so project_tangent_gradient has a grad to work on
        jawp.project_tangent_gradient()
        jawp.stiefel_retract()  # warm
        with ReadCounter({"jawp.py"}) as ctr:
            jawp.project_tangent_gradient()
            jawp.stiefel_retract()
        reads = [s for s in ctr.sites if "item" in s]
        assert not reads, (
            f"the active width is being read back from the buffer: {reads}. "
            f"Use active_k_value(), which serves it from the host mirror."
        )


# ═══════════════════════════════════════════════════════════════════════
#  2. The batched read returns the same float
# ═══════════════════════════════════════════════════════════════════════


def _ref_spc_info(spc, z_pred, z_target):
    """The pre-change SPC forward, transcribed: one `.item()` per value."""
    Dd = z_pred.size(-1)
    pred_bands = spc._decompose_bands(z_pred)
    target_bands = spc._decompose_bands(z_target.detach())
    weights = spc.get_band_weights()
    band_residuals, band_losses = [], []
    total = torch.tensor(0.0, device=z_pred.device)
    for b in range(spc.n_bands):
        rv = (pred_bands[b] - target_bands[b]).pow(2).mean()
        band_residuals.append(rv.item())
        bl = weights[b] * rv
        band_losses.append(bl.item())
        total = total + bl
    with torch.no_grad():
        new_vars = torch.tensor(band_residuals, device=z_pred.device)
        tvars = [max(tb.pow(2).mean().item(), spc.eps) for tb in target_bands]
        target_var_t = torch.tensor(tvars, device=z_pred.device)
        predictability = (1.0 - new_vars / (target_var_t + spc.eps)).clamp(0, 1)
        spc.running_residual_vars.mul_(spc.adapt_momentum).add_((1 - spc.adapt_momentum) * new_vars)
        spc.running_predictability.mul_(spc.adapt_momentum).add_(
            (1 - spc.adapt_momentum) * predictability
        )
        w_norm = weights / weights.sum()
        entropy = -(w_norm * (w_norm + spc.eps).log()).sum()
        lo, hi = weights[: spc.n_bands // 2].mean(), weights[spc.n_bands // 2 :].mean()
        return {
            "spc_total_loss": total.item(),
            "spc_uniform_loss": sum(band_residuals) / spc.n_bands,
            "spc_weight_entropy": entropy.item(),
            "spc_n_significant_bands": (weights > 0.5).sum().item(),
            "spc_spectral_tilt": (hi / (lo + spc.eps)).log().item(),
            "spc_ortho_error": (
                (spc.freq_basis.T @ spc.freq_basis - torch.eye(Dd, device=spc.freq_basis.device))
                .abs()
                .max()
                .item()
            ),
            "spc_band_weights": weights.tolist(),
            "spc_band_residuals": band_residuals,
            "spc_band_losses": band_losses,
            "spc_band_predictability": spc.running_predictability.tolist(),
        }


def _ref_jawp_info(jawp, z_pred, z_target, step):
    """The pre-change JAWP diagnostics, transcribed."""
    import torch.nn.functional as F

    k = jawp.current_k(step)
    Dd = z_pred.size(-1)
    Q = jawp.workspace_Q[:, :k]
    zpf, ztf = z_pred.reshape(-1, Dd), z_target.reshape(-1, Dd).detach()
    pred_ws, target_ws = zpf @ Q, ztf @ Q
    with torch.no_grad():
        p, t = pred_ws.detach(), target_ws.detach()
        pred_ws_recon = p @ Q.detach().T
        loss_focus = ((zpf - pred_ws_recon) ** 2).mean()
        ws_util = ((p**2).sum() / ((zpf**2).sum() + 1e-10)).clamp(0, 1).item()
        t_ws_frac = ((t**2).sum() / ((ztf**2).sum() + 1e-10)).clamp(0, 1).item()
        if p.norm() > 1e-10 and t.norm() > 1e-10:
            cos = (
                F.cosine_similarity(p.flatten().unsqueeze(0), t.flatten().unsqueeze(0))
                .clamp(-1, 1)
                .item()
            )
        else:
            cos = 0.0
        gram = Q.T @ Q
        od = gram.clone()
        od.fill_diagonal_(0)
        ortho = 1.0 - od.abs().mean().clamp(0, 1).item()
        bg = ((zpf - ztf) ** 2).mean()
        ws = ((p - t) ** 2).mean()
        rel = max(0.0, 1.0 - (ws / bg).item()) if bg.item() > 1e-10 else 1.0
        # pca alignment, the old host-side policy
        try:
            N, Dm = ztf.shape
            if N <= 1 or k > Dm or k < 1:
                pca = 0.0
            else:
                cen = ztf - ztf.mean(dim=0)
                cov = (cen.T @ cen) / max(N - 1, 1)
                V = torch.linalg.eigh(cov)[1].flip(1)[:, :k]
                val = (((Q.T @ V) ** 2).sum() / k).item()
                pca = 0.0 if not math.isfinite(val) else max(0.0, min(1.0, val))
        except Exception:
            pca = 0.0
    return {
        "loss_workspace": F.mse_loss(pred_ws, target_ws).item(),
        "loss_predictor_focus": loss_focus.item(),
        "k": k,
        "workspace_utilization": ws_util,
        "target_ws_fraction": t_ws_frac,
        "workspace_cosine": cos,
        "ortho_score": ortho,
        "predictive_relevance": rel,
        "pca_alignment": pca,
    }


def _ref_pcr_info(pcr, z_pred, z_target, step):
    """The pre-change PCR forward, transcribed: four `.item()` per level."""
    Dd = z_pred.size(-1)
    shape = z_pred.shape
    zpf, ztf = z_pred.reshape(-1, Dd), z_target.reshape(-1, Dd).detach()
    z_cur = zpf.clone()
    residual = ztf - z_cur
    warmup = (
        0.0
        if step < pcr.warmup_steps
        else min((step - pcr.warmup_steps) / max(pcr.warmup_steps, 1), 1.0)
    )
    total_norm = 0.0
    info = {}
    for l in range(pcr.n_levels):
        P_l = pcr._get_subspace_proj(l)
        r_proj = residual @ P_l
        corr = pcr.refine_blocks[l](r_proj) @ P_l.T
        gate = torch.sigmoid(pcr.level_gates[l]) * warmup
        z_cur = z_cur + gate * corr
        residual = ztf - z_cur
        with torch.no_grad():
            cn = (gate * corr).norm().item()
            total_norm += cn
            r_e = (r_proj**2).sum().item()
            t_r = (residual**2).sum().item() + 1e-10
            info[f"pcr_level_{l}_correction_norm"] = cn
            info[f"pcr_level_{l}_gate"] = gate.item() if isinstance(gate, torch.Tensor) else gate
            info[f"pcr_level_{l}_subspace_fraction"] = r_e / (t_r + r_e)
            info[f"pcr_level_{l}_dim"] = pcr.level_dims[l]
    with torch.no_grad():
        ir = (ztf - zpf).norm().item()
        fr = (ztf - z_cur).norm().item()
        imp = 1.0 - fr / ir if ir > 1e-10 else 0.0
        gram = pcr.workspace_Q.T @ pcr.workspace_Q
        od = gram.clone()
        od.fill_diagonal_(0)
        info["pcr_ortho_score"] = 1.0 - od.abs().mean().clamp(0, 1).item()
    info.update(
        {
            "pcr_improvement": max(imp, 0.0),
            "pcr_total_refinement_norm": total_norm,
            "pcr_initial_residual": ir,
            "pcr_final_residual": fr,
            "pcr_n_levels": pcr.n_levels,
            "pcr_warmup_factor": warmup,
        }
    )
    return info, z_cur.reshape(shape)


class TestBatchedReadsReturnTheSameFloat:
    """Batching is only allowed if the reported numbers do not move."""

    @pytest.mark.parametrize("n_bands,scale", [(8, 1.0), (8, 1e-6), (4, 10.0), (2, 1.0)])
    def test_spc_info_matches_the_original(self, n_bands, scale):
        torch.manual_seed(11)
        spc = SpectralPredictiveCoding(embed_dim=D, n_bands=n_bands, init="dct")
        spc.train()
        spc.running_residual_vars.copy_(torch.ones(n_bands))
        spc.running_predictability.copy_(torch.zeros(n_bands))
        z_pred = torch.randn(8, D) * scale
        z_target = torch.randn(8, D) * scale

        _, got = spc(z_pred, z_target)
        new_vars = spc.running_residual_vars.clone()
        new_pred = spc.running_predictability.clone()

        spc2 = SpectralPredictiveCoding(embed_dim=D, n_bands=n_bands, init="dct")
        spc2.train()
        spc2.load_state_dict(spc.state_dict())
        spc2.running_residual_vars.copy_(torch.ones(n_bands))
        spc2.running_predictability.copy_(torch.zeros(n_bands))
        want = _ref_spc_info(spc2, z_pred, z_target)

        assert got == want, (
            f"info dict moved: "
            f"{ {k: (got[k], want[k]) for k in want if got.get(k) != want[k]} }"
        )
        assert torch.equal(new_vars, spc2.running_residual_vars), "running_residual_vars moved"
        assert torch.equal(new_pred, spc2.running_predictability), "running_predictability moved"

    @pytest.mark.parametrize("scale,shape", [(1.0, (8, D)), (1e-7, (2, 3, D)), (10.0, (5, D))])
    def test_jawp_info_matches_the_original(self, scale, shape):
        torch.manual_seed(12)
        jawp = JAWPModule(embed_dim=D, k_start=1, k_end=8, curriculum_steps=10)
        z_pred = torch.randn(*shape) * scale
        z_target = torch.randn(*shape) * scale
        for step in (0, 5, 50):
            _, got = jawp.compute_loss(z_pred, z_target, step=step)
            want = _ref_jawp_info(jawp, z_pred, z_target, step)
            assert got == want, (
                f"info dict moved at step {step}: "
                f"{ {k: (got[k], want[k]) for k in want if got.get(k) != want[k]} }"
            )

    def test_jawp_guarded_branches_still_fire(self):
        """The device-side `where()` must reproduce the old `if` exactly.

        Zero inputs are the case the old guard existed for: both norms are 0,
        so the old code returned a hard 0.0 for the cosine. If the guard were
        dropped rather than moved to the device, the cosine would come back as
        the cosine of two zero vectors instead.
        """
        torch.manual_seed(13)
        jawp = JAWPModule(embed_dim=D, k_start=1, k_end=8, curriculum_steps=10)
        z = torch.zeros(4, D)
        _, info = jawp.compute_loss(z, z, step=5)
        assert info["workspace_cosine"] == 0.0
        assert info["predictive_relevance"] == 1.0, "zero residual takes the else branch"
        assert info["pca_alignment"] == 0.0

    @pytest.mark.parametrize(
        "n_levels,dims,shape", [(1, [16], (8, D)), (2, [32, 16], (5, D)), (3, None, (4, D))]
    )
    def test_pcr_info_matches_the_original(self, n_levels, dims, shape):
        torch.manual_seed(14)
        pcr = PredictiveCascadeRefinement(embed_dim=D, n_levels=n_levels, level_dims=dims)
        with torch.no_grad():
            for g in pcr.level_gates:
                g.copy_(torch.randn(()))
        z_pred = torch.randn(*shape)
        z_target = torch.randn(*shape)
        for step in (0, 999, 2500):
            z_refined, got = pcr(z_pred, z_target, step=step)
            want, want_refined = _ref_pcr_info(pcr, z_pred, z_target, step)
            assert got == want, (
                f"info dict moved at step {step}: "
                f"{ {k: (got[k], want[k]) for k in want if got.get(k) != want[k]} }"
            )
            assert torch.equal(z_refined, want_refined), "the refined tensor moved"
