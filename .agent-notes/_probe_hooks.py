"""Per-nn.Linear FLOP attribution over the whole step (forward), via hooks."""

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


T = 256
m, masked, ids, mask = make(T)
groups = {}
handles = []
for name, mod in m.named_modules():
    if isinstance(mod, (nn.Linear, nn.MultiheadAttention, nn.Embedding)):
        key = name.rsplit(".", 1)[0] if name.count(".") >= 2 else name
        if name.count(".") >= 2 and not name.startswith("target_encoder"):
            key = name.rsplit(".", 2)[0] if "predictor_blocks" in name else "encoder"
        elif name.startswith("target_encoder"):
            key = "target_encoder"
        elif "predictor_blocks" in name:
            key = "predictor.blocks"
        elif "decoder" in name:
            key = "decoder"
        elif name.startswith("predictor."):
            key = "predictor.other"

        def hook(mod, inp, _k=key):
            with FlopCounterMode(display=False) as fc:
                mod(inp[0])
            groups[_k] = groups.get(_k, 0) + fc.get_total_flops()

        handles.append(mod.register_forward_hook(hook))

with FlopCounterMode(display=False) as fc:
    loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
    loss.backward()
total = fc.get_total_flops()
for h in handles:
    h.remove()

print("T={} step fwd+bwd total={:.6e}".format(T, total))
for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
    print("   {:>20} fwd={:.6e}".format(k, v))
print("   sum(hooked)={:.6e}".format(sum(groups.values())))
