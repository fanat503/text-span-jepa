# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
# Tied token decoder: auxiliary head grounding latent representations in token space
# Regression head architecture from data2vec (Baevski et al., ICML 2022):
#   Linear → GELU → Linear (same pattern as data2vec_text.py head_layers=2)

import torch.nn.functional as F
from torch import nn


class TiedTokenDecoder(nn.Module):
    """Lightweight decoder that projects predicted latent states to token logits.

    Uses weight-tying with the encoder's token embedding:
        logits = normalize(project(predicted_latent)) @ W_embed^T

    If latents collapse to a uniform vector, the decoder cannot predict
    different tokens → acts as anti-collapse grounding.

    data2vec regression head: Linear → GELU → Linear (head_layers=2 pattern)
    """

    def __init__(self, embed_dim=768, vocab_size=50304, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        # data2vec regression head pattern: head_layers=2 →
        #   first layer expands 2x with GELU, second projects back
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2, bias=bias),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim, bias=bias),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, predicted_latents, token_embedding_weight):
        """Project predicted latents to token logits via weight-tying.

        Args:
            predicted_latents: (..., D) predicted latent representations
            token_embedding_weight: (V, D) encoder's token_embedding weight

        Returns:
            logits: (..., V) token logits

        """
        x = self.proj(predicted_latents)
        x = self.norm(x)
        logits = F.linear(x, token_embedding_weight)
        return logits
