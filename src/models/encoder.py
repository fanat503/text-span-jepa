# Copyright 2026 Slyatski Ilya
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


class QKNorm(nn.Module):
    """QK-normalisation: RMS-normalise ``q`` and ``k`` per head, before the dot product.

    Added from the RoboJEPA literature scan (docs/RESEARCH_BACKLOG.md item 1,
    arXiv:2610.10515: "We found QK-normalization to be of particular importance
    for training stability when scaling the dataset size and the model size.").

    WHY THIS IS NOT A DUPLICATE OF `norm_layer`
    -------------------------------------------
    The blocks here are PRE-norm: `Block.forward` applies `self.norm1` to the
    residual stream *before* `self.attn` projects it into q/k/v. That LayerNorm
    bounds the block INPUT. It does not bound the attention LOGITS, which grow
    with `||W_q x|| * ||W_k x||` and therefore with the projection weights --
    the thing that actually grows during a long run and produces the saturated
    softmax (ViT-22B / Gemma-2 "attention logit growth"). Normalising q and k
    after projection removes that dependence entirely: the logit is a pure
    cosine, so its magnitude is fixed by `head_dim` alone.

    WHY RMS AND NOT LAYERNORM, AND WHY NO LEARNABLE GAIN
    ---------------------------------------------------
    1. Order. QK-norm is defined on the post-projection q/k and must be applied
       BEFORE any rotation is added to them. This module has no positional
       rotation to order against -- see the note on RoPE below.
    2. RMS (not LayerNorm) because RMS is the cheaper of the two and, with no
       mean-subtraction, preserves the sign of every component of q and k,
       which a cosine on a zero-mean vector does not.
    3. NO learnable gain, deliberately. A per-channel gain `g` would preserve
       scale invariance (g does not depend on the token), but it would add
       2 * head_dim parameters per head-set, and this repo compares arms that
       are parameter-matched by construction. As written, `use_qk_norm=True` and
       `use_qk_norm=False` have IDENTICAL parameter counts and IDENTICAL
       state_dicts, so the ablation isolates the normalisation and nothing else.
       `tests/test_qk_norm.py` pins that property.

    THE RoPE ORDERING, STATED EXPLICITLY
    -----------------------------------
    The brief for this change requires q/k to be RMS-normalised BEFORE RoPE is
    added, because normalising after a rotation only partially undoes it: the
    rotation mixes coordinates of different magnitude into the RMS, so the
    result is still scale-dependent, and the error is invisible to any shape
    check. This codebase HAS NO RoPE -- `TextSpanJEPLEncoder` uses a learned
    absolute `pos_embedding`, and `grep -rn "rotary|RoPE|rotate_half" src/`
    returns nothing. Positional information is therefore added to the token
    embedding, i.e. before the qkv projection, so q and k are normalised here
    with no rotation anywhere in the path. The ordering requirement is satisfied
    vacuously. Adding RoPE is a separate backlog item and is on the
    "does not transfer" list for this modality (no views, sequence positions
    only), so it is NOT introduced here -- doing so would be scope creep into
    another worker's file.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """Normalise the head dimension (last axis) of q and k.

        Args:
            q, k: (B, num_heads, N, head_dim)

        Returns:
            (q_hat, k_hat), same shapes, each with RMS exactly 1 per head-vector.

        """
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + self.eps)
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + self.eps)
        return q, k


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
        use_qk_norm=False,
        qk_norm_eps=1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        head_dim = self.head_dim
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # `None` (not an Identity) when off, so the disabled arm's state_dict is
        # byte-identical to the pre-QK-norm model and old checkpoints load.
        self.qk_norm = QKNorm(eps=qk_norm_eps) if use_qk_norm else None

    def _attend(self, q, k, v):
        """Attention over pre-split (B, num_heads, N, head_dim) tensors.

        Split out from `forward` so a test can drive the logits with q/k of a
        chosen magnitude. Scale invariance of the logits is a property of this
        function's inputs, and it cannot be observed through `forward` -- the
        pre-norm LayerNorm already hides the input's scale, so an
        end-to-end scaling test passes with QK-norm off and proves nothing.
        """
        B, _H, N, _D = q.shape
        C = self.num_heads * self.head_dim
        # Micro-opt: use scaled_dot_product_attention when available (PyTorch 2.0+)
        if hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
            return x.transpose(1, 2).reshape(B, N, C)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        return (attn @ v).transpose(1, 2).reshape(B, N, C)

    def forward(self, x):
        B, N, C = x.shape
        # Micro-opt: fused reshape+permute avoids extra copy
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # micro-opt: unbind instead of indexing (avoids contiguity issues)
        if self.qk_norm is not None:
            # Per head, before the dot product. See QKNorm for the RoPE ordering.
            q, k = self.qk_norm(q, k)
        x = self._attend(q, k, v)
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
        use_qk_norm=False,
        qk_norm_eps=1e-6,
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
            use_qk_norm=use_qk_norm,
            qk_norm_eps=qk_norm_eps,
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
        use_qk_norm=False,
        qk_norm_eps=1e-6,
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
                    use_qk_norm=use_qk_norm,
                    qk_norm_eps=qk_norm_eps,
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
