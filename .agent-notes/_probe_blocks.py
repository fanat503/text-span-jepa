"""PredictorBlock accounting in train() mode (the mode the real step runs in)."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils.flop_counter import FlopCounterMode

import src
from src.models.predictor import PredictorBlock

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__


def cnt(fn, train, grad=False):
    pb = PredictorBlock(dim=Cp, num_heads=H, mlp_ratio=r)
    pb.train(train)
    x = torch.randn(B, L, Cp, requires_grad=grad)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with FlopCounterMode(display=False) as fc:
        with ctx:
            fn(pb, x)
    return fc.get_total_flops()


B, L, Cp, C, r, H = 4, 512, 384, 768, 4.0, 12
run = lambda pb, x: pb(x)

for train, grad in ((True, False), (True, True), (False, False)):
    got = cnt(run, train, grad)
    lin = 24 * B * L * Cp * Cp
    att = 4 * B * L * L * Cp
    print(
        "train={} grad={}  got={:.6e}  linear={:.6e} attn={:.6e} "
        "got/lin={:.4f} lin+attn={:.6e} ratio={:.4f}".format(
            train, grad, got, lin, att, got / lin, lin + att, got / (lin + att)
        )
    )

print()
# sanity: does the real predictor's measured total match the train-mode model?
from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

cfg = TextSpanJEPAConfig(
    vocab_size=4096, max_seq_len=512, embed_dim=768, encoder_depth=12,
    num_heads=12, mlp_ratio=4.0, predictor_embed_dim=384, predictor_depth=6,
    future_offsets=(1, 4, 16), num_refine_steps=3,
)
m = TextSpanJEPA(cfg)
m.train()
for T in (128, 256, 512):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 4096, (B, T), generator=g)
    n = max(1, int(T * 0.35))
    mask = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        s = (b * 3) % max(1, T - n)
        mask[b, s : s + n] = 1
    with torch.no_grad():
        ho, te = m.encoder(ids)
        ht, _ = m.target_encoder(ids)
        with FlopCounterMode(display=False) as fc:
            m.predictor(ho, mask, te, ht)
    got = fc.get_total_flops()
    # span: refine passes over L=T ; future: one pass over L=T-d per offset
    spanL = m.predictor.num_refine_steps * T
    futL = sum(T - d for d in m.predictor.future_offsets if d < T)
    Lsum = spanL + futL
    blocks = 24 * B * Lsum * Cp * Cp
    attn = 4 * B * cfg.predictor_depth * (m.predictor.num_refine_steps * T ** 2
                                          + sum((T - d) ** 2 for d in m.predictor.future_offsets
                                                if d < T)) * Cp
    emb = 2 * B * T * C * Cp
    proj = 2 * B * C * Cp * (T + sum(T - d for d in m.predictor.future_offsets if d < T))
    print("T={:<4} got={:.6e}  blocks={:.6e} attn={:.6e} emb+proj={:.6e} sum={:.6e} "
          "ratio={:.4f}".format(T, got, blocks, attn, emb + proj,
                                blocks + attn + emb + proj,
                                got / (blocks + attn + emb + proj)))
