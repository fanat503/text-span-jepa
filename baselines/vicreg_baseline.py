# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""VICReg baseline (Bardes et al., ICLR 2022) on the same encoder class as JEPA.

WHY THIS FILE EXISTS
--------------------
VICReg was cited in prose across `src/` and implemented nowhere: the ablation grid
was 38 internal-mechanism arms and 0 external baselines. This is one of the four
missing arms; the others are `byol_baseline.py`, `barlow_baseline.py` and
`simsiam_baseline.py`.

THE METHOD, AND THE PART A NAIVE PORT DROPS
-------------------------------------------
VICReg's loss is three terms over the projector output ``z``, for a batch of B
samples in D dimensions:

    invariance    mean_i ||z_a[i] - z_b[i]||^2      -- the two views agree
    variance      mean_i max(0, gamma - std(z[:, i]))^2   -- no dimension dies
    covariance    sum_{i != j} C_ij^2                -- dimensions decorrelate

THE VARIANCE TERM IS NOT OPTIONAL. It is the difference between a method that
prevents collapse and one that does not, and dropping it is the single most
common VICReg port bug -- it produces a baseline whose loss falls smoothly while
its representation goes constant, which is indistinguishable from success on
every loss curve a reviewer looks at. The term is implemented here, is exposed as
a constructor argument, and `tests/test_ssl_baselines.py` asserts that setting it
to zero collapses the representation.

A subtlety worth naming: the covariance term alone does NOT prevent collapse. It
penalises *correlation between* dimensions but is completely indifferent to the
overall scale of each dimension, so a representation in which every dimension is
a copy of one signal has covariance zero and is perfectly happy. Only the
variance hinge, which asks each dimension's standard deviation to stay near
`gamma`, forces information through. That is why the negative control in the
test file zeroes the variance term and leaves covariance at full strength: it
isolates the term that actually does the work, instead of ablating two at once
and proving less.

PARAMETER MATCHING
------------------
`MLMBaseline` carries 1.281x `TextSpanJEPA`'s trainable parameters because it
holds an untied ``(embed_dim, vocab_size)`` output matrix. VICReg's trainable
parameters beyond the encoder are a projector MLP and a predictor MLP -- the
same pair of heads BYOL has, and the same budget.

    D=640 depth=10   TextSpanJEPA trainable   88,947,841
                     VICRegBaseline           88,322,560   0.993x
    D=768 depth=12   TextSpanJEPA trainable  134,390,017
                     VICRegBaseline          133,532,160   0.994x

Arithmetic: both arms share the identical encoder, so the whole trainable gap is
head-vs-head against JEPA's 7,189,121-parameter budget at D=640. Two MLPs of
width ``4 * embed_dim`` cost ``4 * D * 4D = 16 D^2``, plus ``4 * hidden`` for the
two *affine* BatchNorms -- which is why this arm's count is 10,240 above the
other three's at D=640 rather than equal to it. The ratio is asserted into a
band, not to equality, because JEPA's head budget includes a fixed-count
mechanism term that does not scale with width.

TRAINER CONTRACT
----------------
`src/train.py::create_model` builds every arm from ``(vocab_size, max_seq_len,
embed_dim, depth, num_heads, mlp_ratio, drop_rate)`` plus method kwargs, and
`src/train.py::compute_loss` dispatches on duck-typing, in this order:
``compute_loss_with_targets`` (JEPA), then ``forward`` **and** ``regression_head``
(data2vec), then ``compute_loss`` (MLM). This module must therefore not define
``compute_loss_with_targets`` and must not own an attribute named
``regression_head``; either one silently routes this arm to another arm's loss.
`tests/test_ssl_baselines.py::TestTrainerContract` pins both absences.

`compute_loss(view_a, view_b, mask_positions=None)` matches the trainer's
three-argument call site. `mask_positions` is accepted and ignored: this is a
non-instance-discrimination method with no per-position target set. The two views
are the trainer's masked and unmasked inputs.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.encoder import TextSpanJEPAEncoder

#: Projector and predictor width as a multiple of ``embed_dim``. Same budget as
#: BYOL's, because both arms spend it on the same two MLPs. See the module header.
HIDDEN_MULTIPLE = 4.0

#: Bardes et al. weights: 25 / 25 / 1 on invariance / variance / covariance.
DEFAULT_SIM_WEIGHT = 25.0
DEFAULT_VAR_WEIGHT = 25.0
DEFAULT_COV_WEIGHT = 1.0

#: The variance hinge target. Bardes et al. use 1.0; this is exposed because the
#: scale of a projected representation is a free choice, so a reviewer comparing
#: arms needs to know which value the arm was given rather than assume it.
DEFAULT_GAMMA = 1.0

#: Guards the covariance sum against 1/sqrt(B) -> 0 on a degenerate batch. The
#: paper computes the covariance over the centred batch; with B == 1 the centred
#: batch is exactly zero, which is finite but carries no information.
_EPS = 1e-4


def round_hidden(hidden: int, num_heads: int) -> int:
    """Round a head width down to a positive multiple of ``num_heads``.

    The heads are plain MLPs, so this is not a shape requirement -- it keeps the
    parameter arithmetic in the header exact at every shape. Floors at one head.
    """
    num_heads = max(int(num_heads), 1)
    return max(int(hidden) // num_heads, 1) * num_heads


class VICRegBaseline(nn.Module):
    """VICReg baseline using the same encoder architecture as `TextSpanJEPA`.

    Architecture::

        encoder -> projector -> predictor   (both views, all trainable)

    Unlike BYOL there is no EMA target and no stop-gradient: VICReg does not need
    them, because the variance term prevents collapse directly.

    Loss: ``sim_w * invariance + var_w * variance_hinge + cov_w * covariance``
    summed over both views and averaged, following Bardes et al.

    Verified edge cases:
        - the variance hinge is one-sided: a dimension whose std already exceeds
          `gamma` contributes exactly zero and cannot be pushed down
        - the covariance term excludes the diagonal by construction, so a
          dimension's own variance is not double-counted into it
        - B == 1 does not raise
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        max_seq_len: int = 512,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        projector_hidden: int | None = None,
        predictor_hidden: int | None = None,
        sim_weight: float = DEFAULT_SIM_WEIGHT,
        var_weight: float = DEFAULT_VAR_WEIGHT,
        cov_weight: float = DEFAULT_COV_WEIGHT,
        gamma: float = DEFAULT_GAMMA,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sim_weight = float(sim_weight)
        self.var_weight = float(var_weight)
        self.cov_weight = float(cov_weight)
        self.gamma = float(gamma)

        if projector_hidden is None:
            projector_hidden = round_hidden(int(HIDDEN_MULTIPLE * embed_dim), num_heads)
        if predictor_hidden is None:
            predictor_hidden = round_hidden(int(HIDDEN_MULTIPLE * embed_dim), num_heads)

        self.encoder = TextSpanJEPAEncoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        def _head(width: int) -> nn.Sequential:
            # BatchNorm is AFFINE here, so it owns 2*width parameters per head --
            # the reason this arm's count is not identical to BYOL's and
            # SimSiam's, which use `affine=False`. It is counted in the header's
            # arithmetic rather than assumed away.
            return nn.Sequential(
                nn.Linear(embed_dim, width, bias=False),
                nn.BatchNorm1d(width),
                nn.GELU(),
                nn.Linear(width, embed_dim, bias=False),
            )

        self.projector = _head(projector_hidden)
        self.predictor = _head(predictor_hidden)

    @staticmethod
    def pool(h: torch.Tensor) -> torch.Tensor:
        """Mean-pool a ``(B, T, D)`` encoder output to ``(B, D)``.

        Identical to the pooling in the other three SSL baselines, so their
        collapse statistics are measured on the same kind of object.
        """
        return h.mean(dim=1)

    def forward(
        self, view_a: torch.Tensor, view_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode, project and predict both views.

        Returns:
            ``(z_a, z_b)``, each ``(B, embed_dim)`` -- the representation the
            three loss terms are computed on.
        """
        h_a, _ = self.encoder(view_a)
        h_b, _ = self.encoder(view_b)
        z_a = self.predictor(self.projector(self.pool(h_a)))
        z_b = self.predictor(self.projector(self.pool(h_b)))
        return z_a, z_b

    def invariance_loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """``mean_i ||z_a[i] - z_b[i]||^2`` -- pull the two views together.

        Bardes et al. sum over the embedding dimension and average over the
        batch, so a wider embedding is not silently down-weighted.
        """
        return (z_a - z_b).pow(2).sum(dim=1).mean()

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """One-sided hinge asking each dimension's std to stay near `gamma`.

        THIS IS THE TERM THAT MAKES VICReg WORK. It is computed with an unbiased
        estimator (Bardes et al.), which is the ``B - 1`` denominator, and the
        hinge is ``relu(gamma - std)``: a dimension already at or above the
        target contributes exactly zero and is never penalised for being large.

        Two numerical details, both load-bearing, and both found by a test rather
        than by reading the paper:

        - ``_EPS`` sits INSIDE the square root, so at exactly zero variance the
          hinge saturates at ``gamma - sqrt(_EPS)`` -- 0.99 at the default gamma
          -- rather than at ``gamma``. That is deliberate: without it,
          ``d/dz sqrt(var)`` is infinite at ``var = 0``, and the exact state
          this term exists to correct would produce a NaN gradient instead of a
          restoring force.
        - The unbiased estimator is UNDEFINED for ``B == 1`` and returns NaN,
          which would make this arm emit a NaN loss on a one-sample batch. It is
          therefore selected only when ``B > 1``, falling back to the biased
          estimator otherwise -- which is 0 for a single sample, making the hinge
          return its maximum, which is the correct response to a batch that
          cannot be spread.

        Args:
            z: ``(B, D)``.

        Returns:
            Scalar, zero for a perfectly healthy representation.
        """
        unbiased = z.shape[0] > 1
        std = torch.sqrt(z.var(dim=0, unbiased=unbiased) + _EPS)
        return torch.relu(self.gamma - std).mean()

    def covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """``sum_{i != j} C_ij^2`` over the centred batch -- decorrelate dimensions.

        Computed by centring, taking the D x D Gram matrix, and subtracting the
        squared diagonal. Subtracting the diagonal after the fact is exact in
        float32 for the sizes involved and avoids allocating a D x D mask that
        would hold the same information.

        Note this term is scale-blind: a representation whose every dimension is
        a copy of one signal has covariance zero. It cannot substitute for the
        variance hinge, which is why the ablation in the test file zeroes
        `var_weight` and leaves this one at full strength.
        """
        B, D = z.shape
        z_c = z - z.mean(dim=0)
        cov = (z_c.transpose(0, 1) @ z_c) / max(B - 1, 1)
        off_diag = cov.pow(2).sum() - torch.diagonal(cov).pow(2).sum()
        return off_diag / D

    def compute_loss(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        mask_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """VICReg loss over both views.

        ``sim_w * invariance + var_w * (var_a + var_b)/2 + cov_w * (cov_a + cov_b)/2``.

        The variance and covariance terms are averaged over the two views, as in
        Bardes et al., so the arm is symmetric in its two inputs.

        Args:
            view_a: ``(B, T)`` token indices -- the trainer's SPAN-MASKED input,
                at the `model.mask_ratio_start/end` curriculum.
            view_b: ``(B, T)`` token indices -- the trainer's CLEAN input.
            mask_positions: accepted for the trainer's call signature and
                ignored. No per-position target set exists in this method.

        Returns:
            ``(loss, info)`` with `loss_vicreg` and all three terms separately.
            The terms are reported individually because their relative size is
            the diagnostic: an arm whose `var_vicreg` is pinned at 0 is not
            learning a representation, whatever `loss_vicreg` says.

        NOT THE PUBLISHED VIEW CONSTRUCTION
        ------------------------------------
        Bardes et al. define VICReg over two AUGMENTED views of the same input.
        `src.train.compute_loss` hands this arm `(masked, clean)`: one view is
        span-masked and the other is untouched, so the pair is nested rather
        than two independent augmentation draws. The invariance/variance/
        covariance objective and its published weights are the published ones;
        the view construction is this repo's, and a run of this arm is therefore
        NOT a reproduction of VICReg as published. Wiring a real
        two-augmentation pipeline is a science change that re-measures this
        repository's collapse thresholds -- see `.agent-notes/fairness.md`.
        Stated here, at the signature, so an arm that is not the published
        method cannot read as one.
        """
        z_a, z_b = self.forward(view_a, view_b)

        sim = self.invariance_loss(z_a, z_b)
        var = 0.5 * (self.variance_loss(z_a) + self.variance_loss(z_b))
        cov = 0.5 * (self.covariance_loss(z_a) + self.covariance_loss(z_b))

        loss = self.sim_weight * sim + self.var_weight * var + self.cov_weight * cov

        return loss, {
            "loss_vicreg": float(loss.item()),
            "invariance": float(sim.item()),
            "var_vicreg": float(var.item()),
            "cov_vicreg": float(cov.item()),
        }

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"projector_hidden={self.projector[0].out_features}, "
            f"predictor_hidden={self.predictor[0].out_features}, "
            f"weights=({self.sim_weight}, {self.var_weight}, {self.cov_weight}), "
            f"gamma={self.gamma}"
        )

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Total parameter count: encoder plus projector and predictor.

        The default is `False` so a bare call returns the whole model, matching
        the other three arms. This arm holds no frozen copy, so total and
        trainable coincide.

        Args:
            non_embedding: subtract the token and position embedding tables. Kept
                because "non-embedding" is a standard published convention, but
                it is not the model's size and must not be quoted as one.
        """
        total = sum(p.numel() for p in self.parameters())
        if not non_embedding:
            return total
        return total - (
            self.encoder.token_embedding.weight.numel() + self.encoder.pos_embedding.numel()
        )

    def get_num_params_trainable(self) -> int:
        """Parameters that receive a gradient.

        Equal to `get_num_params()` here, since there is no EMA teacher to
        freeze. Compare against `TextSpanJEPA.get_num_params_trainable()`, not
        against its total -- the number this returns is what the header's ratio
        is computed from.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
