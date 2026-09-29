"""Does the structural estimator hold at SMALL dims (cheap enough for a test)?"""

import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils.flop_counter import FlopCounterMode

import src

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig  # noqa: E402
from src.utils.flops import estimate_jepa_step_flops  # noqa: E402

CASES = [
    # D,  NE, DP, NP, H, R, offsets, refine, T, B, V, mask
    (32, 2, 16, 2, 4, 2.0, (1, 4), 1, 16, 2, 97, 3 / 16),
    (32, 2, 16, 2, 4, 2.0, (1, 4), 1, 32, 2, 97, 0.25),
    (64, 3, 32, 2, 4, 4.0, (1, 4), 2, 32, 4, 256, 0.25),
    (32, 2, 16, 2, 4, 2.0, (1, 4), 1, 16, 2, 97, 0.0),
    (48, 2, 24, 3, 4, 2.0, (2, 8), 2, 24, 3, 128, 0.2),
]

print("{:>34} {:>15} {:>15} {:>9} {:>7}".format(
    "case", "measured", "estimate", "est/act", "secs"))
for (D, NE, DP, NP, H, R, off, ref, T, B, V, mr) in CASES:
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

    t0 = time.monotonic()
    with FlopCounterMode(display=False) as fc:
        loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
        loss.backward()
    meas = fc.get_total_flops()
    dt = time.monotonic() - t0

    est = estimate_jepa_step_flops(
        embed_dim=D, encoder_depth=NE, predictor_embed_dim=DP, predictor_depth=NP,
        seq_len=T, batch_size=B, vocab_size=V, mlp_ratio=R, predictor_mlp_ratio=R,
        num_refine_steps=ref, future_offsets=off, mask_ratio=mr,
    )
    e = est["total_flops"]
    print("D{:<3} NE{:<2} T{:<3} B{:<2} mr{:<5} {:>15.6e} {:>15.6e} {:>9.5f} {:>7.2f}".format(
        D, NE, T, B, round(mr, 3), meas, e, e / meas, dt))
