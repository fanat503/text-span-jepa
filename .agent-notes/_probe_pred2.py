"""Definitive: exact predictor analytic vs measured, and the residual."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
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


def pred_analytic(T):
    offs = [d for d in OFFSETS if d < T]
    S = REFINE * T + sum(T - d for d in offs)          # total predictor token-passes
    L = T + sum(T - d for d in offs)                  # embed + proj token count
    blocks = 24 * B * S * DP * DP * NP
    proj = 4 * B * D * DP * L                         # predictor_embed + predictor_proj
    return blocks + proj, S, L


for T in (128, 256, 512):
    m, masked, ids, mask = make(T)
    with torch.no_grad():
        ho, te = m.encoder(masked)
        ht, _ = m.target_encoder(ids)
    ho, te, ht = ho.detach(), te.detach(), ht.detach()

    def build():
        sp, _n, _v, fl, _fp = m.predictor(ho, mask, te, ht)
        tot = sp.sum()
        for v in fl.values():
            tot = tot + v.sum()
        return tot

    with FlopCounterMode(display=False) as fc:
        build().backward()
    fb = fc.get_total_flops()
    an, S, L = pred_analytic(T)
    print("T={:<4} predictor fwd+bwd={:.6e}  analytic_fwd={:.6e}  ratio={:.5f}  "
          "(S={}, L={})".format(T, fb, an, fb / (3 * an), S, L))

print()
for T in (128, 256, 512):
    m, masked, ids, mask = make(T)
    with FlopCounterMode(display=False) as fc:
        loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
        loss.backward()
    meas = fc.get_total_flops()

    offs = [d for d in OFFSETS if d < T]
    enc = 24 * B * T * D * D * NE
    tenc = enc
    pa, S, L = pred_analytic(T)
    nm = B * int(T * MASK)
    dec = 2 * nm * (4 * D * D + D * V)
    cov = 2 * B * T * D * D
    model = 3 * (enc + pa + dec + cov) + tenc
    print("T={:<4} meas={:.6e} model={:.6e} ratio={:.5f} resid={:.4e} ({:.2%})".format(
        T, meas, model, model / meas, meas - model, (meas - model) / meas))
    # residual shape hints
    print("      resid / (B*T*D*D) = {:.4f}   resid/(B*T) = {:.4e}   resid/(B*T*T) = {:.4e}".format(
        (meas - model) / (B * T * D * D), (meas - model) / (B * T), (meas - model) / (B * T * T)))
