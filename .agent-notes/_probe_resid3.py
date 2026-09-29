"""Isolate the residual: zero every auxiliary loss lambda and re-measure."""

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

LAMBDAS = [
    "lambda_span", "lambda_decoder", "lambda_variance", "lambda_covariance",
    "lambda_sigreg", "lambda_predictive_rank", "lambda_future",
]


def make(T, zero=False):
    cfg = TextSpanJEPAConfig(
        vocab_size=V, max_seq_len=512, embed_dim=D, encoder_depth=NE, num_heads=HEADS,
        mlp_ratio=R, predictor_embed_dim=DP, predictor_depth=NP,
        future_offsets=OFFSETS, num_refine_steps=REFINE,
    )
    if zero:
        for k in LAMBDAS:
            if hasattr(cfg, k):
                setattr(cfg, k, 0.0)
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


def run(T, zero):
    m, masked, ids, mask = make(T, zero)
    with FlopCounterMode(display=False) as fc:
        loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
        loss.backward()
    return fc.get_total_flops()


for T in (128, 256, 512):
    a = run(T, False)
    b = run(T, True)
    enc = 24 * B * T * D * D * NE
    offs = [d for d in OFFSETS if d < T]
    S = REFINE * T + sum(T - d for d in offs)
    pred = (24 * B * S * DP * DP * NP
            + 2 * 2 * B * D * DP * (T + sum(T - d for d in offs)))
    dec = 2 * B * int(T * MASK) * (4 * D * D + D * V)
    cov = 2 * B * T * D * D
    model = 3 * (enc + pred + dec + cov) + enc
    print("T={:<4} default={:.6e}  zeroed={:.6e}  delta={:.4e}  model={:.6e} "
          "resid={:.4e} ({:.2%})".format(
              T, a, b, a - b, model, a - model, (a - model) / a))
