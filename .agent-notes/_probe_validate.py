"""Validate the rewritten src/utils/flops.py against FlopCounterMode.

Stale-clone guard: `pip install -e .` maps `src` to C:/Users/<u>/tmp/clone-check-1.
sys.path[0] (this script's dir) wins because the editable finder is APPENDED
to sys.meta_path, after PathFinder. Assert it rather than trust it.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils.flop_counter import FlopCounterMode

import src

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), (
    "STALE-CLONE HAZARD: src -> %s" % src.__file__
)
print("src guard OK ->", src.__file__)

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig  # noqa: E402
from src.utils.flops import (  # noqa: E402
    estimate_jepa_step_flops,
    estimate_training_flops,
    estimate_transformer_flops,
)

D, NE, DP, NP, HEADS, R = 768, 12, 384, 6, 12, 4.0
OFFSETS, REFINE, MASK, V = (1, 4, 16), 3, 0.35, 4096
B = 4

print("\n{:>4} {:>15} {:>15} {:>8} {:>15} {:>8} {:>15}".format(
    "T", "measured", "estimate", "est/act", "attention", "measured+a", "est/(m+a)"))
for T in (128, 256, 512):
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

    with FlopCounterMode(display=False) as fc:
        loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
        loss.backward()
    meas = fc.get_total_flops()

    est = estimate_jepa_step_flops(
        embed_dim=D, encoder_depth=NE, predictor_embed_dim=DP, predictor_depth=NP,
        seq_len=T, batch_size=B, vocab_size=V, mlp_ratio=R, predictor_mlp_ratio=R,
        num_refine_steps=REFINE, future_offsets=OFFSETS, mask_ratio=MASK,
    )
    e = est["total_flops"]
    a = est["attention_flops"]
    print("{:>4} {:>15.6e} {:>15.6e} {:>8.5f} {:>15.6e} {:>8.5f} {:>15.6e}".format(
        T, meas, e, e / meas, a, (e + a) / meas, a / meas))

    if T == 512:
        print("\n   breakdown @T=512:")
        for k, v in sorted(est.items(), key=lambda kv: -kv[1] if isinstance(kv[1], float) else 0):
            print("     {:>22} {:.6e}".format(k, v))
        kaplan = estimate_transformer_flops(
            sum(p.numel() for p in m.parameters()), T, batch_size=B)
        print("\n   kaplan_6nd on real params: {:.6e}  ratio {:.5f}  method={}".format(
            kaplan["total_flops"], kaplan["total_flops"] / meas, kaplan["method"]))
        struct = estimate_transformer_flops(
            sum(p.numel() for p in m.parameters()), T, batch_size=B,
            embed_dim=D, num_layers=NE, mlp_ratio=R)
        print("   structural encoder-only:   {:.6e}  ratio {:.5f}  method={}".format(
            struct["total_flops"], struct["total_flops"] / meas, struct["method"]))
        tr = estimate_training_flops(e, 100000)
        print("   training 100k steps: {:.6e} ({:.2f} PFLOPs)".format(
            tr["total_flops"], tr["pflops"]))
