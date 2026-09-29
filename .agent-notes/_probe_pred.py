"""Instrument the predictor: how many block passes, over how many tokens."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils.flop_counter import FlopCounterMode

import src
from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
from src.models.predictor import PredictorBlock

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__

cfg = TextSpanJEPAConfig(
    vocab_size=4096, max_seq_len=512, embed_dim=768, encoder_depth=12,
    num_heads=12, mlp_ratio=4.0, predictor_embed_dim=384, predictor_depth=6,
    future_offsets=(1, 4, 16), num_refine_steps=3,
)
m = TextSpanJEPA(cfg)
p = m.predictor
print("predictor depth      =", len(p.predictor_blocks))
print("predictor mlp_ratio  =", cfg.predictor_mlp_ratio if hasattr(cfg, "predictor_mlp_ratio")
      else getattr(cfg, "mlp_ratio", "?"))
print("refine steps         =", p.num_refine_steps)
print("future offsets       =", p.future_offsets)
print("predictor block mlp  =", p.predictor_blocks[0].mlp)

B, T = 4, 512
g = torch.Generator().manual_seed(0)
ids = torch.randint(0, 4096, (B, T), generator=g)
n_mask = int(T * 0.35)
mask = torch.zeros(B, T, dtype=torch.long)
for b in range(B):
    start = (b * 3) % (T - n_mask)
    mask[b, start : start + n_mask] = 1

counts = {"passes": 0, "tokens": 0}
with torch.no_grad():
    ho, te = m.encoder(ids)
    ht, _ = m.target_encoder(ids)
    hooks = []
    for blk in p.predictor_blocks:
        def hook(mod, args, output, _c=counts):
            _c["passes"] += 1
            _c["tokens"] += args[0].shape[0] * args[0].shape[1]
        hooks.append(blk.register_forward_hook(hook))
    with FlopCounterMode(display=False) as fc:
        p(ho, mask, te, ht)
    for h in hooks:
        h.remove()
print("predictor fwd flops  =", fc.get_total_flops())
print("block passes         =", counts["passes"], " (expect depth*(refine+len(offsets)) =",
      len(p.predictor_blocks), "*", p.num_refine_steps + len(p.future_offsets), "=",
      len(p.predictor_blocks) * (p.num_refine_steps + len(p.future_offsets)), ")")
print("block tokens         =", counts["tokens"])
C, Cp, r = 768, 384, 4.0
print("analytic blocks      =", (8 + 4 * r) * B * counts["tokens"] * Cp * Cp)
resid = fc.get_total_flops() - (8 + 4 * r) * B * counts["tokens"] * Cp * Cp
print("residual             =", resid)
print("  embed proj         =", 2 * B * T * C * Cp, "+proj", 2 * B * T * Cp * C)
for d in p.future_offsets:
    print("  future d=%-3d proj =", 2 * B * (T - d) * Cp * C)
