"""What does the unmodelled residual scale with? Sweep D, T, B independently."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils.flop_counter import FlopCounterMode

import src

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig  # noqa: E402
from src.utils.flops import estimate_jepa_step_flops  # noqa: E402


def one(D=32, NE=2, DP=16, NP=2, H=4, R=2.0, off=(1, 4), ref=1, T=16, B=2, V=97, mr=0.2):
    cfg = TextSpanJEPAConfig(
        vocab_size=V, max_seq_len=T, embed_dim=D, encoder_depth=NE, num_heads=H,
        mlp_ratio=R, predictor_embed_dim=DP, predictor_depth=NP,
        future_offsets=off, num_refine_steps=ref,
    )
    torch.manual_seed(0)
    m = TextSpanJEPA(cfg)
    m.train()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, V, (B, T), generator=g)
    mask = torch.zeros(B, T, dtype=torch.long)
    n = int(round(T * mr))
    for b in range(B):
        mask[b, 1 : 1 + n] = 1
    masked = ids.clone()
    masked[mask.bool()] = 0
    with FlopCounterMode(display=False) as fc:
        loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
        loss.backward()
    meas = fc.get_total_flops()
    est = estimate_jepa_step_flops(
        embed_dim=D, encoder_depth=NE, predictor_embed_dim=DP, predictor_depth=NP,
        seq_len=T, batch_size=B, vocab_size=V, mlp_ratio=R, predictor_mlp_ratio=R,
        num_refine_steps=ref, future_offsets=off, mask_ratio=mr,
    )
    return meas, est["total_flops"]


base = dict()
m0, e0 = one(**base)
print("baseline: meas={:.6e} est={:.6e} resid={:.6e}".format(m0, e0, m0 - e0))
print()
for name, kw in (("D=64", {"D": 64}), ("D=16", {"D": 16}),
                 ("T=32", {"T": 32}), ("T=8", {"T": 8, "off": (1, 4)}),
                 ("B=4", {"B": 4}), ("B=1", {"B": 1}),
                 ("V=194", {"V": 194}), ("mr=0.4", {"mr": 0.4}), ("mr=0.0", {"mr": 0.0})):
    try:
        mm, ee = one(**{**base, **kw})
        print("{:>7}: meas={:.6e} resid={:.6e}  ratio={:.5f}".format(
            name, mm, mm - ee, ee / mm))
    except Exception as exc:
        print("{:>7}: FAILED {}".format(name, type(exc).__name__))
