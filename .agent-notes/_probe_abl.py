"""Ablate to find the residual: leave in only encoder + target_encoder + predictor."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

import src
from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__

B, D, NE, NP, DP, V, HEADS, R = 4, 768, 12, 6, 384, 4096, 12, 4.0
OFFSETS, REFINE, MASK = (1, 4, 16), 3, 0.35


def make(T):
    cfg = TextSpanJEPAConfig(
        vocab_size=V, max_seq_len=512, embed_dim=D, encoder_depth=NE, num_heads=HEADS,
        mlp_ratio=R, predictor_embed_dim=DP, predictor_depth=NP,
        future_offsets=OFFSETS, num_refine_steps=REFINE,
    )
    torch.manual_seed(0)
    m = TextSpanJEPA(cfg)
    m.train()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, V, (B, T), generator=g)
    n = max(1, int(T * MASK))
    mask = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        s = (b * 3) % max(1, T - n)
        mask[b, s : s + n] = 1
    masked = ids.clone()
    masked[mask.bool()] = 0
    return m, masked, ids, mask


class CheapDecoder(nn.Module):
    """Same output shape, negligible flops -> isolates the decoder's cost."""

    def __init__(self, dim, vocab):
        super().__init__()
        self.vocab = vocab
        self.dim = dim

    def forward(self, h, w):
        return h.new_zeros(h.shape[0], self.vocab)


class ZeroCov(nn.Module):
    def forward(self, h):
        return h.sum() * 0.0


def run(T, cheap_decoder=False, zero_cov=False):
    m, masked, ids, mask = make(T)
    if cheap_decoder:
        m.decoder = CheapDecoder(D, V)
    if zero_cov:
        m.covariance_reg = ZeroCov()
    with FlopCounterMode(display=False) as fc:
        loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
        loss.backward()
    return fc.get_total_flops()


for T in (256,):
    offs = [d for d in OFFSETS if d < T]
    enc = 24 * B * T * D * D * NE
    S = REFINE * T + sum(T - d for d in offs)
    L = T + sum(T - d for d in offs)
    pa = 24 * B * S * DP * DP * NP + 4 * B * D * DP * L
    core = 3 * (enc + pa) + enc
    nm = B * int(T * MASK)
    dec = 3 * 2 * nm * (4 * D * D + D * V)
    cov = 3 * 2 * B * T * D * D

    full = run(T)
    nodec = run(T, cheap_decoder=True)
    nocov = run(T, zero_cov=True)
    neither = run(T, True, True)
    print("T={}".format(T))
    print("  full            = {:.6e}".format(full))
    print("  cheap decoder   = {:.6e}   (saves {:.6e}; model says dec={:.6e})".format(
        nodec, full - nodec, dec))
    print("  zero covariance = {:.6e}   (saves {:.6e}; model says cov={:.6e})".format(
        nocov, full - nocov, cov))
    print("  neither         = {:.6e}   (saves {:.6e}; model says dec+cov={:.6e})".format(
        neither, full - neither, dec + cov))
    print("  core (enc+tenc+pred) model = {:.6e}  residual = {:.6e} ({:.2%})".format(
        core, neither - core, (neither - core) / neither))
