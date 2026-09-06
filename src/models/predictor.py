# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Predictor: span + future prediction with iterative refinement
# Uses torch.gather for efficient masked-position extraction (Fix #3)
# Predictor architecture from I-JEPA, adapted for text with query embeddings

import math

import torch
import torch.nn.functional as F
from torch import nn


class PredictorBlock(nn.Module):
    """Lightweight transformer block for the predictor — from I-JEPA."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class TextSpanJEPApredictor(nn.Module):
    """Predictor for Text-Span JEPA.

    Architecture follows I-JEPA predictor with text-specific additions:
    - Project from encoder dim to predictor dim
    - Learned query embeddings per task type (span_query, future_queries)
    - Insert mask tokens at target positions
    - Iterative refinement: multiple cheap predictor passes
    - Project back to encoder dim

    Fix #3: torch.gather for masked positions (no boolean indexing).
    """

    def __init__(
        self,
        embed_dim=768,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        max_seq_len=512,
        future_offsets=(1, 4, 16),
        num_refine_steps=3,
        refine_step_size=0.1,
        init_std=0.02,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.predictor_embed_dim = predictor_embed_dim
        self.future_offsets = future_offsets
        self.num_refine_steps = num_refine_steps
        self.refine_step_size = refine_step_size
        self.init_std = init_std

        # Project from encoder dim to predictor dim (I-JEPA pattern)
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Learned query embeddings for different prediction tasks
        self.span_query = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        self.future_queries = nn.ParameterDict(
            {
                f"offset_{d}": nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
                for d in future_offsets
            },
        )

        # Learned positional embedding for predictor (I-JEPA: learned, not frozen)
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, predictor_embed_dim),
            requires_grad=True,
        )

        # Mask token at positions to be predicted (I-JEPA pattern)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # Refinement gate: learned residual weight
        self.refine_gate = nn.Parameter(torch.tensor(0.1))

        # Predictor transformer blocks (I-JEPA pattern)
        self.predictor_blocks = nn.ModuleList(
            [
                PredictorBlock(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ],
        )
        self.predictor_norm = nn.LayerNorm(predictor_embed_dim)

        # Project back to encoder dim
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # I-JEPA init: trunc_normal_ + depth-wise rescaling
        nn.init.trunc_normal_(self.span_query, std=init_std)
        nn.init.trunc_normal_(self.predictor_pos_embed, std=init_std)
        for v in self.future_queries.values():
            nn.init.trunc_normal_(v, std=init_std)
        nn.init.trunc_normal_(self.mask_token, std=init_std)
        self.apply(self._init_weights)
        self._fix_init_weight()

    def _fix_init_weight(self):
        """Depth-wise rescaling from I-JEPA / CaiT."""

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.out_proj.weight.data, layer_id + 1)
            rescale(layer.mlp[-1].weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _forward_predictor(self, x):
        """Single forward pass through predictor blocks."""
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)
        return x

    def _iterative_refine(self, x):
        """Iterative refinement: multiple cheap steps in latent space.

        The encoder is NOT re-run. Only the predictor does extra passes.
        z_0 = predictor(h)
        z_{k+1} = z_k + α * sigmoid(gate) * (predictor(z_k) - z_k)
        """
        for _ in range(self.num_refine_steps):
            x_refined = self._forward_predictor(x)
            gate = torch.sigmoid(self.refine_gate)
            x = x + gate * self.refine_step_size * (x_refined - x)
        return x

    @staticmethod
    def _gather_masked(h, mask_positions):
        """Efficiently extract hidden states at masked positions using torch.gather.

        Fix #3: Instead of boolean indexing (irregular shapes),
        we use gather with precomputed index tensors + valid_mask.

        Args:
            h: (B, T, D) hidden states
            mask_positions: (B, T) binary mask, 1=masked
        Returns:
            gathered: (B, max_num_masked, D) padded with zeros
            num_masked_per_sample: (B,)
            valid_mask: (B, max_num_masked) bool, True for real masked positions

        """
        B, _T, D = h.shape
        num_masked_per_sample = mask_positions.sum(dim=1)
        max_num_masked = num_masked_per_sample.max().item()

        if max_num_masked == 0:
            return (
                torch.zeros(B, 1, D, device=h.device),
                num_masked_per_sample,
                torch.zeros(B, 1, dtype=torch.bool, device=h.device),
            )

        indices = torch.zeros(B, max_num_masked, dtype=torch.long, device=h.device)
        valid_mask = torch.zeros(B, max_num_masked, dtype=torch.bool, device=h.device)
        for b in range(B):
            masked_idx = mask_positions[b].nonzero(as_tuple=True)[0]
            n = min(len(masked_idx), max_num_masked)
            indices[b, :n] = masked_idx[:n]
            valid_mask[b, :n] = True

        indices_expanded = indices.unsqueeze(-1).expand(B, max_num_masked, D)
        gathered = torch.gather(h, dim=1, index=indices_expanded)
        gathered[~valid_mask.unsqueeze(-1).expand_as(gathered)] = 0.0

        return gathered, num_masked_per_sample, valid_mask

    def forward_span_prediction(self, h_online, mask_positions):
        """Predict target latent states at masked span positions."""
        B, T, _D = h_online.shape

        x = self.predictor_embed(h_online)
        pos_emb = self.predictor_pos_embed[:, :T, :]
        x = x + pos_emb

        span_q = self.span_query.expand(B, 1, self.predictor_embed_dim)
        mask_expanded = mask_positions.unsqueeze(-1).float()
        mask_tokens = self.mask_token.expand(B, T, -1) + span_q + pos_emb
        x = mask_expanded * mask_tokens + (1 - mask_expanded) * x

        x_out = self._iterative_refine(x)
        predictions = self.predictor_proj(x_out)

        gathered, num_masked, valid_mask = self._gather_masked(predictions, mask_positions)
        return gathered, num_masked, valid_mask

    def forward_future_prediction(self, h_online, token_embeds, target_h):
        """Predict future target latent states — lightweight (no iterative refinement)."""
        B, T, _D = h_online.shape
        future_losses = {}
        future_predictions = {}

        for d in self.future_offsets:
            if d >= T:
                continue
            h_curr = h_online[:, : T - d, :]
            x = self.predictor_embed(h_curr)
            future_q = self.future_queries[f"offset_{d}"].expand(B, T - d, -1)
            pos_emb = self.predictor_pos_embed[:, : T - d, :]
            x = x + future_q + pos_emb

            x_out = self._forward_predictor(x)  # no refinement for future
            predictions = self.predictor_proj(x_out)
            h_target_future = target_h[:, d:, :]

            future_predictions[d] = predictions
            future_losses[d] = F.smooth_l1_loss(predictions, h_target_future)

        return future_losses, future_predictions

    def forward(self, h_online, mask_positions, token_embeds, target_h):
        """Combined forward: span + future prediction."""
        span_preds, num_masked, valid_mask = self.forward_span_prediction(h_online, mask_positions)
        future_losses, future_preds = self.forward_future_prediction(
            h_online,
            token_embeds,
            target_h,
        )
        return span_preds, num_masked, valid_mask, future_losses, future_preds

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# Backward-compatible alias
TextSpanJPAPredictor = TextSpanJEPApredictor
