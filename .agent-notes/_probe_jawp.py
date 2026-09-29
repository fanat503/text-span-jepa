"""Is the residual the JAWP SVD path? Toggle use_jawp and re-measure."""

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


def one(D, NE, DP, NP, H, R, off, ref, T, B, V, mr, use_jawp=True, lam_rank=None):
    cfg = TextSpanJEPAConfig(
        vocab_size=V, max_seq_len=T, embed_dim=D, encoder_depth=NE, num_heads=H,
        mlp_ratio=R, predictor_embed_dim=DP, predictor_depth=NP,
        future_offsets=off, num_refine_steps=ref,
        use_jawp=use_jawp,
        **({} if lam_rank is None else {"lambda_predictive_rank": lam_rank}),
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
    est = estimate_jepa_step_flops(
        embed_dim=D, encoder_depth=NE, predictor_embed_dim=DP, predictor_depth=NP,
        seq_len=T, batch_size=B, vocab_size=V, mlp_ratio=R, predictor_mlp_ratio=R,
        num_refine_steps=ref, future_offsets=off, mask_ratio=mr,
    )
    return fc.get_total_flops(), est["total_flops"], cfg


base = dict(D=32, NE=2, DP=16, NP=2, H=4, R=2.0, off=(1, 4), ref=1,
            T=16, B=2, V=97, mr=0.2)
for label, kw in (("default", {}),
                  ("use_jawp=False", {"use_jawp": False}),
                  ("lam_rank=0", {"lam_rank": 0.0})):
    meas, est, cfg = one(**{**base, **kw})
    print("{:>16}: meas={:.6e} est={:.6e} resid={:.6e} ratio={:.5f}  "
          "(use_jawp={} lam_rank={})".format(
              label, meas, est, meas - est, est / meas,
              cfg.use_jawp, cfg.lambda_predictive_rank))

print()
b2 = dict(base, D=64, T=32, B=4, V=256, mr=0.25, R=4.0, H=4, NE=3, DP=32, NP=2, ref=2)
for label, kw in (("default", {}), ("use_jawp=False", {"use_jawp": False})):
    meas, est, cfg = one(**{**b2, **kw})
    print("{:>16}: meas={:.6e} est={:.6e} resid={:.6e} ratio={:.5f}".format(
        label, meas, est, meas - est, est / meas))
