# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""SimSiam baseline (Chen & He, ICML 2021) on the same encoder class as JEPA.

WHY THIS FILE EXISTS
--------------------
SimSiam was cited in prose across `src/` and implemented nowhere: the ablation
grid was 38 internal-mechanism arms and 0 external baselines. This is one of the
four missing arms; the others are `byol_baseline.py`, `barlow_baseline.py` and
`vicreg_baseline.py`.

THE METHOD, AND THE PART A NAIVE PORT DROPS
-------------------------------------------
SimSiam's entire loss is:

    -0.5 * ( cos(p_a, z_b) + cos(p_b, z_a) )

where ``z`` is the projector output under a STOP-GRADIENT and ``p`` is the
predictor output without one. There is no EMA target, no negatives, no variance
term, no decorrelation term. One structure does all the work, and it is the
thing a refactor deletes because it looks like a no-op:

    THE STOP-GRADIENT. Without it the loss is minimised by making the projector
    constant. Chen & He show this collapsing; a port that writes
    ``z_b.detach()`` in one place and not the other, or that computes both
    branches in a single `torch.cat` under one `no_grad`, has produced exactly
    that silently.

This module makes the stop-gradient STRUCTURAL rather than a `.detach()` that a
later refactor can drop: `target_branch` returns a tensor that has no graph at
all, computed inside `torch.no_grad()`. There is nothing for a refactor to
forget -- if a future change routes the target through the graph, it has to
construct the gradient path deliberately, which is a visible act rather than an
omission.

Note the asymmetry with BYOL, which is easy to get backwards: BYOL's predictor
guards against an EMA target that is moving; SimSiam's predictor guards against a
target that is *the same weights*. Both need the predictor, neither needs the
other method's machinery, and importing an EMA teacher here would make this a
different method.

PARAMETER MATCHING
------------------
`MLMBaseline` carries 1.281x `TextSpanJEPA`'s trainable parameters because it
holds an untied ``(embed_dim, vocab_size)`` matrix. SimSiam's heads are a
projector MLP and a predictor MLP -- the same pair BYOL and VICReg have, and the
same budget.

    D=640 depth=10   TextSpanJEPA trainable   88,947,841
                     SimSiamBaseline          88,312,320   0.993x
    D=768 depth=12   TextSpanJEPA trainable  134,390,017
                     SimSiamBaseline         133,519,872   0.994x

Arithmetic: both arms share the identical encoder, so the whole trainable gap is
head-vs-head against JEPA's 7,189,121-parameter budget at D=640. Two MLPs of
width ``4 * embed_dim`` cost ``4 * D * 4D = 16 D^2``. The ratio is asserted into
a band, not to equality, because JEPA's head budget includes a fixed-count
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
import torch.nn.functional as F
from torch import nn

from src.models.encoder import TextSpanJEPAEncoder

#: Projector and predictor width as a multiple of ``embed_dim``. Same budget as
#: BYOL and VICReg -- all three arms spend it on the same two MLPs.
HIDDEN_MULTIPLE = 4.0


def round_hidden(hidden: int, num_heads: int) -> int:
    """Round a head width down to a positive multiple of ``num_heads``.

    The heads are plain MLPs, so this is not a shape requirement -- it keeps the
    parameter arithmetic in the header exact at every shape. Floors at one head.
    """
    num_heads = max(int(num_heads), 1)
    return max(int(hidden) // num_heads, 1) * num_heads


class SimSiamBaseline(nn.Module):
    """SimSiam baseline using the same encoder architecture as `TextSpanJEPA`.

    Architecture::

        encoder -> projector -> predictor   (both views, all trainable)

    There is deliberately NO target encoder and NO EMA update. The stop-gradient
    is the collapse mechanism, and `target_branch` makes it structural.

    Loss: symmetric negative cosine similarity between predictor output and the
    stop-gradient projector output.

    Verified edge cases:
        - `target_branch` output has `requires_grad == False` and `grad_fn is
          None`, for any input
        - B == 1 does not raise
        - gradient reaches the encoder, the projector and the predictor
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
        projector_hidden: int | None = None,
        predictor_hidden: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim

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
        )

        def _head(width: int) -> nn.Sequential:
            # BatchNorm1d(affine=False): zero parameters, matching Chen & He's
            # projector and keeping the header's arithmetic exact.
            return nn.Sequential(
                nn.Linear(embed_dim, width, bias=False),
                nn.BatchNorm1d(width, affine=False),
                nn.ReLU(),
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

    def online_branch(self, view: torch.Tensor) -> torch.Tensor:
        """Online path: encoder -> projector -> predictor. Trainable end to end."""
        h, _ = self.encoder(view)
        return self.predictor(self.projector(self.pool(h)))

    def target_branch(self, view: torch.Tensor) -> torch.Tensor:
        """Target path: encoder -> projector, computed under `torch.no_grad`.

        This is the stop-gradient, and it is structural. The returned tensor has
        no `grad_fn` and `requires_grad == False` because it was never built in
        a tracking context -- not because a `.detach()` was applied to a graph
        that had already been constructed. There is no line here that a refactor
        can delete to remove the stop-gradient without also having to write a new
        one.
        """
        with torch.no_grad():
            h, _ = self.encoder(view)
            return self.projector(self.pool(h))

    def forward(
        self, view_a: torch.Tensor, view_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """All four tensors the loss needs, in one pass pair.

        Returns:
            ``(p_a, p_b, z_a, z_b)`` -- the two online predictions and the two
            stop-gradient targets.
        """
        return (
            self.online_branch(view_a),
            self.online_branch(view_b),
            self.target_branch(view_a),
            self.target_branch(view_b),
        )

    def compute_loss(
        self,
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        mask_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Symmetric SimSiam loss.

        ``-0.5 * (mean cos(p_a, z_b) + mean cos(p_b, z_a))``.

        No `.detach()` is written in this method on purpose. The targets already
        carry no gradient (`target_branch` computed them under `no_grad`), so a
        detach here would be a second, redundant, deletable copy of the same
        guarantee. The test file asserts the no-grad property at the source.

        Args:
            view_a, view_b: ``(B, T)`` token indices, two views.
            mask_positions: accepted for the trainer's call signature and
                ignored. No per-position target set exists in this method.

        Returns:
            ``(loss, info)`` with `loss_simsiam` and the two cosine terms.
        """
        p_a, p_b, z_a, z_b = self.forward(view_a, view_b)

        cos_ab = F.cosine_similarity(p_a, z_b, dim=-1).mean()
        cos_ba = F.cosine_similarity(p_b, z_a, dim=-1).mean()
        loss = -0.5 * (cos_ab + cos_ba)

        return loss, {
            "loss_simsiam": float(loss.item()),
            "cos_ab": float(cos_ab.item()),
            "cos_ba": float(cos_ba.item()),
        }

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"projector_hidden={self.projector[0].out_features}, "
            f"predictor_hidden={self.predictor[0].out_features}"
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
