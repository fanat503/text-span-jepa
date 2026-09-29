# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# MLM baseline: BERT-style masked language model on the same encoder class.
#
# WHAT IS MATCHED, AND WHAT IS NOT
# --------------------------------
# This file used to state a fair-comparison guarantee -- equal model capacity on
# both arms, and equal FLOPs per forward pass, with only the training objective
# differing. (The exact wording is quoted in the module docstring, inside the
# "Historical claim" markers, so the test that forbids a LIVE guarantee can tell
# a quotation from an assertion.) Both quantitative halves were false, and the
# class docstring repeated the guarantee. The claim is removed rather than
# softened, because a reviewer reads it and then measures. The measured facts, at
# the shipped 640/10 rung (V=50304, max_seq_len 512), are these:
#
#   one encoder (either arm)                 81,758,720   <- MATCHED, exactly
#   this arm: mlm_head, untied                32,194,560
#   this arm: total == trainable            113,953,280
#   TextSpanJEPA: total                     170,706,561   (1.498x this arm)
#   TextSpanJEPA: trainable                  88,947,841
#
# So the encoder is identical, and the arms are NOT capacity-matched: this arm
# carries 1.281x JEPA's trainable parameters, 25,005,439 more. The whole excess
# is the output head. This arm holds a separate, untied (embed_dim, vocab_size)
# matrix; JEPA's `TiedTokenDecoder` is 1,639,680 parameters and gets its vocab
# projection for free by reusing `encoder.token_embedding.weight`
# (`src/models/decoder.py`), so this head has no JEPA counterpart at all. The
# asymmetry favours the control, which is the direction that makes it worth
# stating.
#
# JEPA's higher *total* is the opposite story and is not a capacity advantage:
# 81,758,720 of its 170,706,561 is a `target_encoder` deepcopy with
# requires_grad=False, so it carries no gradient path.
#
# Compute is not matched either, and cannot be. Beyond the head, JEPA runs a
# second encoder forward plus a predictor where this arm runs one encoder. And
# `compute_loss` below projects all B*T positions before gathering, so at the
# 64x512 micro-batch this arm materialises a dense (B, T, 50304) fp32 logit
# tensor -- 6.14 GiB -- against JEPA's masked-row-only decode at 2.15 GiB for
# its 0.35 mask budget. Moving this arm's gather above the head would remove that
# 2.86x activation penalty; it would not make the FLOP counts equal. That is a
# behaviour change to a training path and is deliberately not bundled here.
#
# What genuine parity would cost, named exactly, because it is an experiment
# decision and not a code fix: tie `mlm_head` to the token embedding (drops this
# arm to 81,758,720 trainable, i.e. 0.919x JEPA, so the sign of the asymmetry
# flips rather than vanishing, and it changes what the control is), or widen
# JEPA's predictor by ~25M parameters to absorb the head, or train both arms
# under a published parameter budget and report the budget. Any of those changes
# a trained result and needs human sign-off.
#
# Guard: tests/test_baseline_parity.py pins every number above, so this comment
# cannot drift away from the code without a test going red.
"""BERT-style MLM baseline on the same encoder class as `TextSpanJEPA`.

NOT capacity-matched and NOT compute-matched to the JEPA arm. Only the encoder is
shared. The measured asymmetry, the mechanism behind it, and what genuine parity
would cost are in the module header; `tests/test_baseline_parity.py` pins every
number in both places.

--- BEGIN HISTORICAL CLAIM (quotation, not a live guarantee) ---
This module previously asserted: "Fair comparison guarantee: identical model
capacity (encoder params match exactly), identical compute (same FLOPs per
forward pass), only the training objective differs."
--- END HISTORICAL CLAIM ---

The parenthetical was true; the headline it qualified was not, and the two
together are a contradiction. Do not restore the wording above as an assertion.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.encoder import TextSpanJEPAEncoder


class MLMBaseline(nn.Module):
    """BERT-style MLM baseline using the same encoder architecture.

    NOT capacity-matched and NOT compute-matched to `TextSpanJEPA`. Only the
    encoder is shared. The measured asymmetry, and what parity would cost, are
    in the module header; `tests/test_baseline_parity.py` pins the numbers.

    Architecture:
        encoder → mlm_head (Linear: embed_dim → vocab_size, untied)

    Loss: cross-entropy on masked positions only.

    Verified edge cases:
        - Empty mask (num_masked=0): returns 0.0 loss in computation graph
        - Eval mode: F.cross_entropy exactly (diff < 1e-5)
        - Gradient flows to all trainable params
        - decoder is the same object as mlm_head (an alias, not a weight tie:
          `mlm_head` is a separate matrix from `encoder.token_embedding`, which
          is precisely what JEPA's `TiedTokenDecoder` does not have)
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        max_seq_len: int = 512,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.encoder = TextSpanJEPAEncoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
        )
        self.mlm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Decoder attribute for train.py compatibility (same object, weight-tied)
        self.decoder = self.mlm_head

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, embed_dim={self.embed_dim}"

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode and project to vocabulary logits.

        Args:
            input_ids: (B, T) token indices

        Returns:
            logits: (B, T, vocab_size)

        """
        h, _ = self.encoder(input_ids)
        logits = self.mlm_head(h)
        return logits

    def compute_loss(
        self,
        masked_input_ids: torch.Tensor,
        original_input_ids: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute MLM cross-entropy loss on masked positions.

        Args:
            masked_input_ids: (B, T) token indices with mask tokens
            original_input_ids: (B, T) original token indices (targets)
            mask_positions: (B, T) binary mask, 1=masked

        Returns:
            loss: scalar tensor (differentiable)
            info: dict with loss_mlm and mlm_accuracy

        """
        logits = self.forward(masked_input_ids)
        # Boolean indexing for masked positions — vectorized, no loop
        masked_logits = logits[mask_positions.bool()]
        masked_targets = original_input_ids[mask_positions.bool()]

        # Guard against empty mask (no masked positions).
        # cross_entropy on empty tensors produces NaN — return zero loss
        # that participates in the computation graph (for gradient accumulation).
        if masked_logits.size(0) == 0:
            zero = logits.sum() * 0.0
            return zero, {"loss_mlm": 0.0, "mlm_accuracy": 0.0}

        loss = F.cross_entropy(masked_logits, masked_targets)
        # Micro-opt: compute accuracy under no_grad to avoid storing graph.
        # Use argmax on dim=-1 for (N, V) logits → (N,) predictions.
        with torch.no_grad():
            accuracy = (masked_logits.argmax(dim=-1) == masked_targets).float().mean()
        # Type-safe: ensure info dict values are plain Python floats (not torch scalars)
        return loss, {"loss_mlm": float(loss.item()), "mlm_accuracy": float(accuracy.item())}

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Count model parameters."""
        enc = self.encoder.get_num_params(non_embedding)
        head = sum(p.numel() for p in self.mlm_head.parameters())
        return enc + head

    def get_num_params_trainable(self) -> int:
        """Count parameters that receive a gradient.

        Every parameter of this arm is trainable, so this equals the total. It
        exists to mirror `TextSpanJEPA.get_num_params_trainable`, where the two
        differ by the frozen `target_encoder` deepcopy. Comparing the two arms
        on this number -- not on the total -- is the only like-for-like capacity
        comparison, and it is the one the module header reports: 113,953,280
        against 88,947,841, i.e. 1.281x in this arm's favour.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
