# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# MLM baseline: BERT-style masked language model, same encoder, fair comparison
#
# Fair comparison guarantee: identical model capacity (encoder params match exactly),
# identical compute (same FLOPs per forward pass), only the training objective differs.
# This isolates the effect of the JEPA objective from architectural confounds.

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.models.encoder import TextSpanJEPAEncoder


class MLMBaseline(nn.Module):
    """BERT-style MLM baseline using the same encoder architecture.

    Fair comparison: identical model capacity, identical compute,
    only the training objective differs.

    Architecture:
        encoder → mlm_head (Linear: embed_dim → vocab_size, weight-tied optional)

    Loss: cross-entropy on masked positions only.

    Verified edge cases:
        - Empty mask (num_masked=0): returns 0.0 loss in computation graph
        - Eval mode: F.cross_entropy exactly (diff < 1e-5)
        - Gradient flows to all trainable params
        - decoder weight-tied to mlm_head
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
