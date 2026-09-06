# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Bidirectional Transformer encoder (online + target shared architecture)
# Architecture adapted from I-JEPA (Assran et al., CVPR 2023) for 1D text
# Init patterns: trunc_normal_ + depth-wise rescaling from I-JEPA/CaiT

import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn


class DropPath(nn.Module):
    """Stochastic depth — from I-JEPA / timm."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Attention(nn.Module):
    """Multi-head self-attention for bidirectional encoder."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # Micro-opt: fused reshape+permute avoids extra copy
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # micro-opt: unbind instead of indexing (avoids contiguity issues)
        # Micro-opt: use scaled_dot_product_attention when available (PyTorch 2.0+)
        if hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TextSpanJEPLEncoder(nn.Module):
    """Bidirectional Transformer encoder for Text-Span JEPA.

    Shared architecture between online encoder and target encoder (EMA copy).
    Init: trunc_normal_ + depth-wise rescaling from I-JEPA/CaiT.
    """

    def __init__(
        self,
        vocab_size=50304,
        max_seq_len=512,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        init_std=0.02,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.init_std = init_std

        self.gradient_checkpointing = gradient_checkpointing
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Token + position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim),
            requires_grad=True,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ],
        )
        self.norm = norm_layer(embed_dim)

        # I-JEPA init: trunc_normal_ + depth-wise rescaling
        nn.init.trunc_normal_(self.pos_embedding, std=init_std)
        self.apply(self._init_weights)
        self._fix_init_weight()

    def _fix_init_weight(self):
        """Depth-wise rescaling from I-JEPA / CaiT."""

        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=self.init_std)

    def forward(self, input_ids, return_intermediates=False):
        """Encode token sequence.

        Args:
            input_ids: (B, T) token indices
            return_intermediates: if True, return per-layer hidden states

        Returns:
            hidden_states: (B, T, embed_dim)
            token_embeds: (B, T, embed_dim) raw token embeddings
            intermediates: list of (B, T, embed_dim) per layer (only if return_intermediates=True)

        """
        _B, T = input_ids.shape
        token_embeds = self.token_embedding(input_ids)
        x = token_embeds + self.pos_embedding[:, :T, :]

        intermediates = []
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if return_intermediates:
                intermediates.append(x)

        x = self.norm(x)

        if return_intermediates:
            return x, token_embeds, intermediates
        return x, token_embeds

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.pos_embedding.numel()
        return n_params

    def get_intermediate_layers(self, input_ids):
        """Get hidden states from each transformer block.

        Used by interpretability analysis (layer-wise CKA, probing, etc.).

        Args:
            input_ids: (B, T) token indices

        Returns:
            list of (B, T, embed_dim) tensors, one per block

        """
        _B, T = input_ids.shape
        token_embeds = self.token_embedding(input_ids)
        x = token_embeds + self.pos_embedding[:, :T, :]

        intermediates = []
        for blk in self.blocks:
            x = blk(x)
            intermediates.append(x.clone())
        return intermediates


# Backward-compatible alias
TextSpanJEPAEncoder = TextSpanJEPLEncoder
