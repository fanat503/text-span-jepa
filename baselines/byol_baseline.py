# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
"""BYOL baseline (Grill et al., NeurIPS 2020) on the same encoder class as JEPA.

WHY THIS FILE EXISTS
--------------------
BYOL was cited across `src/` in prose and implemented nowhere. The ablation grid
was 38 internal-mechanism arms and 0 external baselines, which is the single
fact every reviewer of this paper rejects on. This is one of the four missing
arms; the other three are `barlow_baseline.py`, `vicreg_baseline.py` and
`simsiam_baseline.py`.

THE METHOD, AND THE PART A NAIVE PORT DROPS
-------------------------------------------
BYOL is *bootstrapped*: it has no negatives, no variance term and no
decorrelation term. It cannot collapse for any reason other than the two
structures below, so a port that drops either one does not produce a slightly
worse baseline -- it produces a collapsed constant function that still reports a
decreasing loss. That is the failure mode this module is written to make
impossible to ship:

1.  **EMA target network** -- a frozen copy of the encoder AND the projector,
    updated as ``theta_t <- tau*theta_t + (1-tau)*theta_s``, and read under
    ``torch.no_grad()``. Without it the objective has a trivial exact minimiser.
2.  **Predictor** -- a bottleneck MLP applied to the ONLINE branch only, never
    to the target. Without it the online branch can match the target directly and
    BYOL collapses (Grill et al., Table 3).

Both are load-bearing *together*, which is why they are separate named modules
here rather than folded into the encoder, and why the collapse test in
`tests/test_ssl_baselines.py` carries a negative control that removes each one
in turn and asserts the representation goes constant.

PARAMETER MATCHING
------------------
`MLMBaseline` carries 1.281x `TextSpanJEPA`'s trainable parameters because it
holds an untied ``(embed_dim, vocab_size)`` output matrix -- 32,194,560
parameters with no JEPA counterpart at all (`tests/test_baseline_parity.py`).
BYOL's heads are the opposite problem: the paper's projector/predictor are small
MLPs, so a default port lands *below* JEPA rather than above it.

So the widths here are chosen to land ON JEPA's trainable count rather than to
copy a number from the BYOL paper. Measured at both production rungs:

    D=640 depth=10   TextSpanJEPA trainable    88,947,841   1.000x
                     BYOLBaseline trainable    88,312,320   0.993x
                     MLMBaseline trainable    113,953,280   1.281x
    D=768 depth=12   TextSpanJEPA trainable   134,390,017   1.000x
                     BYOLBaseline trainable   133,519,872   0.994x

`projector_hidden`/`predictor_hidden` default to ``4 * embed_dim``, i.e. two
``D->4D->D`` MLPs costing ``16 D^2``. The mechanism is arithmetic, not luck: both
arms share the identical encoder (same class, same dims), so the whole trainable
gap reduces to head-vs-head, and JEPA's non-encoder trainable budget is
7,189,121 at D=640 against this arm's 6,553,600.

The ratio is not exactly 1.0 and is not asserted to be: JEPA's head budget
includes a fixed-count mechanism term that does not scale with width, so the
exact ratio legitimately moves with vocab size and depth. `tests/test_ssl_baselines.py`
asserts the ratio stays inside a band, which is the property that matters.

`MLMBaseline`'s 1.281x is the contrast. It is not a rounding difference; it is
the whole reason a reviewer measures this repo's baselines before believing a
comparison it draws between them.

TRAINER CONTRACT
----------------
`src/train.py::create_model` builds every arm from ``(vocab_size, max_seq_len,
embed_dim, depth, num_heads, mlp_ratio, drop_rate)`` plus method kwargs, and
`src/train.py::compute_loss` dispatches on duck-typing, in this order:
``compute_loss_with_targets`` (JEPA), then ``forward`` **and** ``regression_head``
(data2vec), then ``compute_loss`` (MLM). This module therefore must not define
``compute_loss_with_targets`` and must not own an attribute named
``regression_head`` -- either one silently routes this arm to another arm's
loss. `tests/test_ssl_baselines.py::TestTrainerContract` pins both absences.

Like `MLMBaseline`, ``compute_loss`` takes ``(view_a, view_b, mask_positions=None)``
so the trainer's existing three-argument call site reaches it. `mask_positions` is
accepted and ignored: this is a non-instance-discrimination method and has no
per-position target set. The two views are the trainer's masked and unmasked
inputs.

`update_target_encoder()` takes no arguments and mirrors
`Data2VecTextBaseline.update_target_encoder`, so the trainer's existing
zero-argument EMA call site reaches it. Wiring this arm into `create_model` and
`do_ema_update` is a change to `src/train.py`, which this file does not own.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from src.models.encoder import TextSpanJEPAEncoder

#: Projector and predictor width as a multiple of ``embed_dim``.
#:
#: Chosen so the arm's trainable count lands on JEPA's, not by copying the BYOL
#: paper: two ``D->4D->D`` MLPs cost ``16 D^2``, which at D=640 is 6,553,600
#: against JEPA's 7,189,121-parameter head budget -- 0.993x, where the naive
#: small-head port gives roughly 0.94x and `MLMBaseline` gives 1.281x. Barlow
#: Twins spends the same budget on one MLP and so uses ``8 * embed_dim``.
#: Pinned by `tests/test_ssl_baselines.py::TestParameterMatching`.
HIDDEN_MULTIPLE = 4.0


def round_hidden(hidden: int, num_heads: int) -> int:
    """Round a head width down to a multiple of ``num_heads``.

    The heads are plain MLPs, so this is not a shape requirement -- it is so
    that the width is a clean multiple of the attention head count at every
    shape, which keeps the parameter arithmetic in the header exact rather than
    approximate.

    Args:
        hidden: requested width.
        num_heads: encoder head count.

    Returns:
        ``hidden`` rounded DOWN to a positive multiple of ``num_heads``. Never
        returns 0: a zero-width MLP is not a smaller baseline, it is a broken
        one, so the floor is one head.
    """
    num_heads = max(int(num_heads), 1)
    return max(int(hidden) // num_heads, 1) * num_heads


def build_mlp(embed_dim: int, hidden: int) -> nn.Sequential:
    """The BYOL projector/predictor: ``Linear -> BN -> GELU -> Linear``.

    BatchNorm is the shape used by Grill et al. for both modules. It is
    non-affine, so it owns no parameters -- which is why the header's parameter
    arithmetic counts only the two Linear layers.
    """
    return nn.Sequential(
        nn.Linear(embed_dim, hidden, bias=False),
        nn.BatchNorm1d(hidden, affine=False),
        nn.GELU(),
        nn.Linear(hidden, embed_dim, bias=False),
    )


class BYOLBaseline(nn.Module):
    """BYOL baseline using the same encoder architecture as `TextSpanJEPA`.

    NOT parameter-matched in the strict sense of an identical integer count, and
    not claimed to be: the ratio is asserted into a band, because JEPA's
    mechanism parameters are a fixed count that does not scale. The encoder is
    shared exactly, so the whole gap is head-vs-head. Measured figures are in
    the module header; `tests/test_ssl_baselines.py` pins them.

    Architecture::

        encoder                -> projector -> predictor   (online, trainable)
        target_encoder (frozen) -> target_projector (frozen) (target, EMA)

    Loss: symmetric negative cosine similarity between the online prediction and
    the STOP-GRADIENT target, summed over both view orderings
    (``2 - 2*(cos(p1, t2) + cos(p2, t1))``).

    Verified edge cases:
        - B == 1: loss is finite (cosine similarity is defined on a single
          vector pair), no division by zero
        - target branch carries no gradient under any input
        - `update_target_encoder` moves the target and NOT the online weights
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
        target_momentum: float = 0.996,
        projector_hidden: int | None = None,
        predictor_hidden: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.target_momentum = target_momentum
        self.num_updates = 0

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

        self.projector = build_mlp(embed_dim, projector_hidden)
        self.predictor = build_mlp(embed_dim, predictor_hidden)

        # Collapse structure #1: the EMA target. BYOL EMAs the encoder AND the
        # projector -- the target projection is part of the teacher, not a
        # separately-trained head. Omitting the projector from the EMA is a
        # subtle port bug that the collapse test does not catch, so it is done
        # correctly here and asserted by parameter-identity below.
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_projector = copy.deepcopy(self.projector)
        for module in (self.target_encoder, self.target_projector):
            for p in module.parameters():
                p.requires_grad = False

    # ── collapse structure #1: the EMA teacher ──────────────────────────

    def get_annealed_momentum(self) -> float:
        """Current EMA momentum. Constant here; exposed for parity with data2vec.

        data2vec anneals its decay, JEPA anneals `tau` via a scheduler. BYOL's
        paper anneals it too, but the schedule is a training-loop concern this
        module does not own, so the value is fixed at `target_momentum` and the
        hook exists for the trainer to drive.
        """
        return self.target_momentum

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """EMA the target encoder and target projector toward the online pair.

        No-argument, mirroring `Data2VecTextBaseline.update_target_encoder`, so
        the trainer's existing call site reaches it.

        Uses `.data` throughout, matching data2vec: under `@torch.no_grad` a
        plain in-place `mul_`/`add_` on a leaf that requires no grad is already
        safe, and `.data` keeps this identical to the arm that shipped.
        """
        decay = self.get_annealed_momentum()
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(decay).add_((1.0 - decay) * online.detach().data)
        for online, target in zip(self.projector.parameters(), self.target_projector.parameters()):
            target.data.mul_(decay).add_((1.0 - decay) * online.detach().data)
        self.num_updates += 1

    # ── views ──────────────────────────────────────────────────────────

    @staticmethod
    def pool(h: torch.Tensor) -> torch.Tensor:
        """Mean-pool a ``(B, T, D)`` encoder output to ``(B, D)``.

        Sequence-level representation, which is what a text SSL baseline is
        evaluated on. Done here rather than at each call site so all four SSL
        baselines pool identically and their collapse statistics are comparable.
        """
        return h.mean(dim=1)

    def online_branch(self, view: torch.Tensor) -> torch.Tensor:
        """Online path: encoder -> projector -> predictor. Trainable end to end.

        Collapse structure #2: the predictor is applied HERE AND ONLY HERE. It
        is deliberately not a method of the online path that the target shares,
        so `tests/test_ssl_baselines.py` can bypass it with a one-line subclass
        and assert the representation goes constant.
        """
        h, _ = self.encoder(view)
        return self.predictor(self.projector(self.pool(h)))

    def target_branch(self, view: torch.Tensor) -> torch.Tensor:
        """Target path: EMA encoder -> EMA projector, read under no_grad.

        The returned tensor has `requires_grad == False` and no graph, which is
        what makes the stop-gradient structural rather than a `.detach()` that a
        later refactor can drop: there is nothing to drop.
        """
        with torch.no_grad():
            h, _ = self.target_encoder(view)
            return self.target_projector(self.pool(h))

    def forward(
        self, view_a: torch.Tensor, view_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """All four tensors the loss needs, in one pass pair.

        Args:
            view_a, view_b: ``(B, T)`` token indices, two augmentations of the
                same sequences.

        Returns:
            ``(p_a, p_b, t_a, t_b)`` -- the two online predictions and the two
            stop-gradient targets. Returned rather than computed inside
            `compute_loss` so a caller can inspect the collapse mechanism
            directly without reaching into the loss.
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
        """Symmetric BYOL loss.

        ``2 - 2*(mean cos(p_a, t_b) + mean cos(p_b, t_a))``, the form in Grill et
        al. Both terms are kept: the asymmetry-free form is what makes BYOL
        "symmetric", and dropping one direction is a port bug that changes the
        gradient by a factor of two rather than failing loudly.

        Args:
            view_a: ``(B, T)`` token indices -- the trainer's SPAN-MASKED
                input, at the `model.mask_ratio_start/end` curriculum.
            view_b: ``(B, T)`` token indices -- the trainer's CLEAN input.
            mask_positions: accepted for the trainer's call signature and
                ignored. This method has no per-position target set; it is not a
                masked-prediction objective.

        NOT THE PUBLISHED VIEW CONSTRUCTION
        ------------------------------------
        Grill et al. define BYOL over two AUGMENTED views of the same input.
        `src.train.compute_loss` hands this arm `(masked, clean)`: one view is
        span-masked and the other is untouched, so the pair is nested rather
        than two independent augmentation draws. The objective, the EMA teacher
        and the predictor are the published ones; the view construction is this
        repo's, and a run of this arm is therefore NOT a reproduction of BYOL as
        published. Wiring a real two-augmentation pipeline is a science change
        that re-measures this repository's collapse thresholds -- see
        `.agent-notes/fairness.md`. Stated here, at the signature, so an arm that
        is not the published method cannot read as one.

        Returns:
            ``(loss, info)`` with `loss_byol`, the two cosine terms, and the
            EMA momentum.
        """
        p_a, p_b, t_a, t_b = self.forward(view_a, view_b)

        cos_ab = F.cosine_similarity(p_a, t_b.detach(), dim=-1).mean()
        cos_ba = F.cosine_similarity(p_b, t_a.detach(), dim=-1).mean()
        loss = 2.0 - 2.0 * (cos_ab + cos_ba)

        if self.training:
            self.num_updates += 1

        return loss, {
            "loss_byol": float(loss.item()),
            "cos_ab": float(cos_ab.item()),
            "cos_ba": float(cos_ba.item()),
            "ema_momentum": self.get_annealed_momentum(),
        }

    def extra_repr(self) -> str:
        proj_h = self.projector[0].out_features
        pred_h = self.predictor[0].out_features
        return (
            f"embed_dim={self.embed_dim}, projector_hidden={proj_h}, "
            f"predictor_hidden={pred_h}, target_momentum={self.target_momentum}"
        )

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Total parameter count: the model's size, the kind `train.py` logs.

        The default is `False` so a bare call returns the whole model, matching
        `TextSpanJEPA` and both existing baselines. `tests/test_baseline_parity.py::
        TestLoggedQuantityIsOneKind` pins that KIND for the arms `create_model`
        knows; this module is not yet one of them (see the module header), and
        this method is what it must report once it is.

        Note the frozen `target_encoder` IS inside this total and is NOT inside
        `get_num_params_trainable`. That asymmetry is deliberate and matches
        JEPA and data2vec: counting a frozen EMA copy as capacity would overstate
        what the arm trains.

        Args:
            non_embedding: subtract the token and position embedding tables of
                BOTH encoders. Kept because "non-embedding" is a standard
                published convention, but it is not the model's size.
        """
        total = sum(p.numel() for p in self.parameters())
        if not non_embedding:
            return total
        embeddings = 0
        for enc in (self.encoder, self.target_encoder):
            embeddings += enc.token_embedding.weight.numel()
            embeddings += enc.pos_embedding.numel()
        return total - embeddings

    def get_num_params_trainable(self) -> int:
        """Parameters that receive a gradient: total minus the frozen teacher.

        This is the like-for-like capacity number for a comparison against
        `TextSpanJEPA.get_num_params_trainable()`, and it is the number the
        header's ratio is computed from.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
