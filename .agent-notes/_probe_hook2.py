"""Attribute each recorded matmul to a module via forward hooks on Linear/MHA."""

import os
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

import src

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig  # noqa: E402

D, NE, DP, NP, H, R = 32, 2, 16, 2, 4, 2.0
OFF, REF, T, B, V, MR = (1, 4), 1, 16, 2, 97, 0.2
n_mask = int(round(T * MR))

cfg = TextSpanJEPAConfig(
    vocab_size=V, max_seq_len=T, embed_dim=D, encoder_depth=NE, num_heads=H,
    mlp_ratio=R, predictor_embed_dim=DP, predictor_depth=NP,
    future_offsets=OFF, num_refine_steps=REF,
)
torch.manual_seed(0)
m = TextSpanJEPA(cfg)
m.train()
g = torch.Generator().manual_seed(0)
ids = torch.randint(0, V, (B, T), generator=g)
mask = torch.zeros(B, T, dtype=torch.long)
for b in range(B):
    mask[b, 1 : 1 + n_mask] = 1
masked = ids.clone()
masked[mask.bool()] = 0

tally = defaultdict(float)
handles = []
for name, mod in m.named_modules():
    if isinstance(mod, (nn.Linear, nn.MultiheadAttention)):
        # Derive the cost from the module's own weight shapes instead of
        # re-running it: a pre-hook that calls mod() re-enters the hook.
        def post(mod, inp, out, _n=name):
            is_mha = isinstance(mod, nn.MultiheadAttention)
            tok = inp[0].shape[0] * inp[0].shape[1] if inp[0].dim() == 3 else inp[0].shape[0]
            if is_mha:
                e = mod.embed_dim
                f = 2 * tok * (3 * e * e + e * e)
            else:
                f = 2 * tok * mod.in_features * mod.out_features
            tally[_n] += f
        handles.append(mod.register_forward_hook(post))

with FlopCounterMode(display=False) as fc:
    with torch.no_grad():
        m.compute_loss_with_targets(masked, ids, mask)
total = fc.get_total_flops()
for h in handles:
    h.remove()

print("forward total = {}".format(total))
agg = defaultdict(float)
for k, v in tally.items():
    parts = k.split(".")
    if k.startswith("target_encoder"):
        agg["target_encoder"] += v
    elif "predictor_blocks" in k:
        agg["predictor.blocks"] += v
    elif k.startswith("encoder.blocks"):
        agg["encoder.blocks"] += v
    elif k.startswith("predictor."):
        agg["predictor.proj"] += v
    elif k.startswith("decoder"):
        agg["decoder"] += v
    else:
        agg[k] += v
for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
    print("  {:>22} {:>12.0f}".format(k, v))
print("  {:>22} {:>12.0f}".format("SUM", sum(agg.values())))
print("  {:>22} {:>12.0f}".format("UNATTRIBUTED", total - sum(agg.values())))
