# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# data2vec baseline: EMA teacher + regression head on masked token representations
# Directly adapted from fairseq/examples/data2vec/models/data2vec_text.py
# (Baevski et al., ICML 2022)

from __future__ import annotations

import copy
import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from src.models.encoder import TextSpanJEPAEncoder


def get_annealed_rate(start: float, end: float, curr_step: int, total_steps: int) -> float:
    """EMA decay annealing from data2vec official."""
    r = end - start
    pct_remaining = 1.0 - curr_step / max(total_steps, 1)
    return end - r * pct_remaining


class Data2VecTextBaseline(nn.Module):
    """data2vec-style baseline using the same encoder architecture.

    From the official fairseq implementation (data2vec_text.py):
    - Online encoder processes masked input
    - EMA teacher (target encoder) processes original input
    - Regression head: Linear → GELU → Linear (head_layers=2)
    - Loss: smooth_l1_loss or mse_loss on masked positions
    - Target: layer_norm of top-K hidden layers from teacher
    - EMA tau: annealed from ema_decay to ema_end_decay
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        max_seq_len: int = 512,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        average_top_k_layers: int = 8,
        loss_beta: float = 0.0,
        loss_scale: float | None = None,
        ema_decay: float = 0.999,
        ema_end_decay: float = 0.9999,
        ema_anneal_end_step: int = 100000,
        head_layers: int = 1,
        mask_token_id: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.average_top_k_layers = average_top_k_layers
        self.loss_beta = loss_beta
        if loss_scale is None:
            self.loss_scale = -1
        else:
            self.loss_scale = loss_scale
        self.ema_decay = ema_decay
        self.ema_end_decay = ema_end_decay
        self.ema_anneal_end_step = max(ema_anneal_end_step, 1)
        self.mask_token_id = mask_token_id
        self.num_updates = 0

        self.encoder = TextSpanJEPAEncoder(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        curr_dim = embed_dim
        projs: list[nn.Module] = []
        for i in range(head_layers - 1):
            next_dim = embed_dim * 2 if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim
        projs.append(nn.Linear(curr_dim, embed_dim))
        self.regression_head = nn.Sequential(*projs)

    def get_annealed_decay(self) -> float:
        if self.num_updates >= self.ema_anneal_end_step:
            return self.ema_end_decay
        return get_annealed_rate(
            self.ema_decay, self.ema_end_decay, self.num_updates, self.ema_anneal_end_step
        )

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        decay = self.get_annealed_decay()
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(decay).add_((1.0 - decay) * param_q.detach().data)

    def forward(
        self,
        masked_input_ids: torch.Tensor,
        original_input_ids: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        h_online, _ = self.encoder(masked_input_ids)

        with torch.no_grad():
            intermediates = self.target_encoder.get_intermediate_layers(original_input_ids)
            k = min(self.average_top_k_layers, len(intermediates))
            if len(intermediates) < self.average_top_k_layers:
                warnings.warn(
                    f"data2vec target depth truncated: encoder exposes "
                    f"{len(intermediates)} layers, average_top_k_layers="
                    f"{self.average_top_k_layers}"
                )
            if k == 0:
                h_target, _ = self.target_encoder(original_input_ids)
            else:
                target_layers = intermediates[-k:]
                h_target = torch.stack(target_layers, dim=0).mean(dim=0)
            h_target = F.layer_norm(h_target.float(), h_target.shape[-1:])

        masked_indices = mask_positions.bool()
        x = h_online[masked_indices]
        y = h_target[masked_indices]

        if x.size(0) == 0:
            zero = h_online.sum() * 0.0
            return zero, {
                "loss_data2vec": 0.0,
                "ema_decay": self.get_annealed_decay(),
                "num_masked": 0,
            }

        x = self.regression_head(x)

        sz = x.size(-1)
        if self.loss_beta == 0:
            loss_per_token = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss_per_token = F.smooth_l1_loss(
                x.float(), y.float(), reduction="none", beta=self.loss_beta
            ).sum(dim=-1)

        loss_total = loss_per_token.sum()
        if self.loss_scale <= 0:
            loss_total = loss_total / math.sqrt(sz)
        else:
            loss_total = loss_total * self.loss_scale

        sample_size = mask_positions.sum().item()
        loss = loss_total / max(sample_size, 1)

        if self.training:
            self.num_updates += 1

        return loss, {
            "loss_data2vec": float(loss.item()),
            "ema_decay": self.get_annealed_decay(),
            "num_masked": int(sample_size),
        }

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.encoder.token_embedding.num_embeddings}, "
            f"embed_dim={self.encoder.embed_dim}, "
            f"head_layers={len([m for m in self.regression_head if isinstance(m, nn.Linear)])}, "
            f"average_top_k_layers={self.average_top_k_layers}, "
            f"loss_beta={self.loss_beta}"
        )

    def get_num_params(self, non_embedding: bool = True) -> int:
        enc = self.encoder.get_num_params(non_embedding)
        reg = sum(p.numel() for p in self.regression_head.parameters())
        return enc + reg
