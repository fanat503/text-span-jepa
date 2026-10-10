# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""Barlow Twins baseline (Zbontar et al., NeurIPS 2021) on the JEPA encoder class.

WHY THIS FILE EXISTS
--------------------
Barlow Twins was cited in prose across `src/` and implemented nowhere: the
ablation grid was 38 internal-mechanism arms and 0 external baselines. This is
one of the four missing arms; the others are `byol_baseline.py`,
`vicreg_baseline.py` and `simsiam_baseline.py`.

THE METHOD, AND THE PART A NAIVE PORT DROPS
-------------------------------------------
Barlow Twins is the simplest of the four to state and the easiest to break. Its
loss has two halves:

    invariance     sum_i  (C_ii - 1)^2      -- views agree dimension by dimension
    decorrelation  lambda * sum_{i != j} C_ij^2  -- dimensions stop duplicating

where ``C`` is the cross-correlation matrix of the two views' *batch-normalised*
projector outputs. The second half is the ENTIRE anti-collapse mechanism. With
``lambda = 0`` the objective is minimised by a rank-1 representation -- every
dimension a copy of one direction -- which still reports a falling loss. That
is not a subtle degradation, it is a baseline that has learned nothing.

There is no EMA target and no predictor here, by design: those belong to BYOL
and SimSiam, and importing them into this arm would be a different method. The
only anti-collapse structure Barlow Twins has is the off-diagonal penalty, so
the collapse test in `tests/test_ssl_baselines.py` exercises exactly that term.

PARAMETER MATCHING
------------------
`MLMBaseline` carries 1.281x `TextSpanJEPA`'s trainable parameters because it
holds an untied ``(embed_dim, vocab_size)`` matrix. Barlow Twins has no such
head: its only trainable parameters beyond the encoder are one projector MLP.
That single MLP is therefore sized to absorb JEPA's whole non-encoder trainable
budget on its own.

Measured at the two production rungs (`tests/test_ssl_baselines.py` pins these):

    D=640 depth=10   TextSpanJEPA trainable   88,947,841
                     BarlowTwinsBaseline      88,312,320   0.993x
    D=768 depth=12   TextSpanJEPA trainable  134,390,017
                     BarlowTwinsBaseline     133,519,872   0.994x

The mechanism is arithmetic. Both arms share the identical encoder, so the whole
trainable gap is head-vs-head, and JEPA's head budget at D=640 is 7,189,121
parameters. One MLP of width ``8 * embed_dim`` costs ``2 * D * 8D = 16 D^2`` =
6,553,600 at D=640. A two-head method (BYOL, SimSiam) spends the same budget as
two MLPs of width ``4 * D``, which is why the two constants differ by exactly 2.

The ratio is not asserted to be exactly 1.0 and cannot be: JEPA's head budget
includes a fixed-count mechanism term that does not scale with width, so the
exact ratio legitimately moves with shape. The test asserts a band.

A GENUINE CONSTRAINT THIS METHOD HAS
------------------------------------
Barlow Twins needs the batch to be at least as large as the projection width,
otherwise the cross-correlation matrix is rank-deficient and full decorrelation
is impossible no matter what ``lambda`` is. This arm projects to ``embed_dim``
to stay parameter-matched, so it wants ``batch_size >= embed_dim`` -- at D=640
that is a batch of 640, which is affordable only under gradient accumulation.
This is a property of the method, not of this port, and it is stated here
because it is the first thing a reviewer will run into when configuring the arm.

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
are the trainer's masked and unmasked inputs -- the two distinct views of the
same sequences this repo's data pipeline produces.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.encoder import TextSpanJEPAEncoder

#: Projector width as a multiple of ``embed_dim``.
#:
#: Twice BYOL's ``4.0`` because Barlow Twins has one head where BYOL has two.
#: Both spend the same ``16 * D^2`` parameters on heads, which is what makes them
#: parameter-comparable with each other AND with JEPA. See the module header for
#: the measured ratios; `tests/test_ssl_baselines.py::TestParameterMatching` pins
#: them.
HIDDEN_MULTIPLE = 8.0

#: Zbontar et al. use lambda = 5e-3 (ResNet) / 1e-3 (ViT). This is the paper's
#: own value and it is also this arm's only anti-collapse term, so it is exposed
#: as a constructor argument and its ablation is a first-class test rather than
#: a line edit.
DEFAULT_LAMBDA = 5e-3


def round_hidden(hidden: int, num_heads: int) -> int:
    """Round a head width down to a positive multiple of ``num_heads``.

    The projector is a plain MLP, so this is not a shape requirement -- it keeps
    the parameter arithmetic in the header exact at every shape rather than
    approximate. Floors at one head: a zero-width MLP is not a smaller baseline,
    it is a broken one.
    """
    num_heads = max(int(num_heads), 1)
    return max(int(hidden) // num_heads, 1) * num_heads


class BarlowTwinsBaseline(nn.Module):
    """Barlow Twins baseline using the same encoder architecture as JEPA.

    Architecture::

        encoder -> projector (one MLP, non-affine BatchNorm at the output)

    Loss: cross-correlation of the two views' batch-normalised embeddings pushed
    toward the identity matrix -- diagonal to 1, off-diagonal weighted by
    ``lambda_offdiag`` toward 0.

    Verified edge cases:
        - the batch-normalisation is non-affine, so it owns no parameters and the
          head's parameter count is exactly ``2 * embed_dim * hidden``
        - a single-sample batch does not raise (BatchNorm in eval mode)
        - gradients reach the encoder and the projector, and nothing else
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
        hidden: int | None = None,
        lambda_offdiag: float = DEFAULT_LAMBDA,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.lambda_offdiag = float(lambda_offdiag)

        if hidden is None:
            hidden = round_hidden(int(HIDDEN_MULTIPLE * embed_dim), num_heads)

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

        # Non-affine BatchNorm: zero parameters, so the head's parameter count is
        # exactly 2 * embed_dim * hidden and the header's arithmetic is exact.
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden, affine=False),
            nn.GELU(),
            nn.Linear(hidden, embed_dim, bias=False),
        )

    @staticmethod
    def pool(h: torch.Tensor) -> torch.Tensor:
        """Mean-pool a ``(B, T, D)`` encoder output to ``(B, D)``.

        Identical to the pooling in the other three SSL baselines, so their
        collapse statistics are measured on the same kind of object and are
        comparable.
        """
        return h.mean(dim=1)

    def forward(
        self, view_a: torch.Tensor, view_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project both views.

        Args:
            view_a, view_b: ``(B, T)`` token indices, two views of the same
                sequences.

        Returns:
            ``(z_a, z_b)``, each ``(B, embed_dim)``. These are *unnormalised*
            projector outputs -- the batch normalisation that Barlow Twins applies
            before the cross-correlation is part of the loss, not of the
            representation, and folding it in here would hide it.
        """
        h_a, _ = self.encoder(view_a)
        h_b, _ = self.encoder(view_b)
        return self.projector(self.pool(h_a)), self.projector(self.pool(h_b))

    @staticmethod
    def cross_correlation(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Cross-correlation matrix of two batch-normalised embeddings.

        Args:
            z_a, z_b: ``(B, D)``.

        Returns:
            ``(D, D)`` matrix ``C`` with ``C[i, j] = <z_a[:, i], z_b[:, j]> / B``
            computed on per-dimension batch-normalised inputs.

        Note:
            BatchNorm is applied here in ``train()`` mode rather than carried by a
            module, so a caller inspecting `forward`'s output sees the raw
            projection. A B == 1 batch would make the per-dimension standard
            deviation zero; that is handled by the module's own eps, not here.
        """
        B = z_a.shape[0]
        z_a_bn = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0, unbiased=False) + 1e-6)
        z_b_bn = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0, unbiased=False) + 1e-6)
        return (z_a_bn.transpose(0, 1) @ z_b_bn) / B

    def compute_loss(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        mask_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Barlow Twins loss.

        ``sum_i (C_ii - 1)^2 + lambda * sum_{i != j} C_ij^2``.

        The off-diagonal term is the whole anti-collapse mechanism and it is
        written as its own named loss so that a reader (and a mutation) can see
        exactly which term does what. `lambda_offdiag=0` is therefore a supported
        configuration -- it is the paper's ablation -- and
        `tests/test_ssl_baselines.py` asserts that it collapses.

        Args:
            view_a: ``(B, T)`` token indices -- the trainer's SPAN-MASKED
                input, at the `model.mask_ratio_start/end` curriculum.
            view_b: ``(B, T)`` token indices -- the trainer's CLEAN input.
            mask_positions: accepted for the trainer's call signature and
                ignored. No per-position target set exists in this method.

        NOT THE PUBLISHED VIEW CONSTRUCTION
        ------------------------------------
        Zbontar et al. define Barlow Twins over two AUGMENTED views of the same
        input. `src.train.compute_loss` hands this arm `(masked, clean)`: one
        view is span-masked and the other is untouched, so the pair is nested
        rather than two independent augmentation draws. The cross-correlation
        objective and `lambda_offdiag` are the published ones; the view
        construction is this repo's, and a run of this arm is therefore NOT a
        reproduction of Barlow Twins as published. Wiring a real two-augmentation
        pipeline is a science change that re-measures this repository's collapse
        thresholds -- see `.agent-notes/fairness.md`. Stated here, at the
        signature, so an arm that is not the published method cannot read as one.

        Returns:
            ``(loss, info)`` with `loss_barlow`, the two halves separately, and
            `lambda_offdiag`.
        """
        z_a, z_b = self.forward(view_a, view_b)
        c = self.cross_correlation(z_a, z_b)

        on_diag = torch.diagonal(c)
        loss_invariance = ((on_diag - 1.0) ** 2).sum()

        # Off-diagonal without materialising a D x D mask: the squared sum of
        # every entry minus the squared sum of the diagonal.
        loss_corr = (c**2).sum() - (on_diag**2).sum()
        loss = loss_invariance + self.lambda_offdiag * loss_corr

        return loss, {
            "loss_barlow": float(loss.item()),
            "loss_invariance": float(loss_invariance.item()),
            "loss_correlation": float(loss_corr.item()),
            "lambda_offdiag": self.lambda_offdiag,
        }

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, hidden={self.projector[0].out_features}, "
            f"lambda_offdiag={self.lambda_offdiag}"
        )

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Total parameter count: encoder plus projector, the model's size.

        The default is `False` so a bare call returns the whole model, matching
        `TextSpanJEPA`, `MLMBaseline` and `Data2VecTextBaseline`. This arm holds
        no frozen copy, so total and trainable coincide here -- see
        `get_num_params_trainable`.

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

        Equal to `get_num_params()` for this arm -- there is no EMA teacher to
        freeze -- which is exactly why it must not be compared against JEPA's
        *total*. `get_num_params_trainable` on both arms is the like-for-like
        comparison, and it is the number the header's ratio uses.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
