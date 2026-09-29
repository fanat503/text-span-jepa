"""Why was encoder fwd/bwd 2.0 in the big-model probe but 3.0 in the small one?"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils.flop_counter import FlopCounterMode

import src
from src.models.encoder import TextSpanJEPAEncoder
from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__
print("src ->", src.__file__)


def run(enc, ids, tag):
    enc.train()
    with FlopCounterMode(display=False) as f:
        h, _ = enc(ids)
    fwd = f.get_total_flops()
    with FlopCounterMode(display=False) as f2:
        h.sum().backward()
    fb = f2.get_total_flops()
    enc.zero_grad(set_to_none=True)
    print(
        "{:<34} gc={} fwd={:.6e} fwd+bwd={:.6e} ratio={:.4f}".format(
            tag, getattr(enc, "gradient_checkpointing", "?"), fwd, fb, fb / fwd
        )
    )
    return fwd, fb


ids = torch.randint(0, 256, (4, 32))

enc = TextSpanJEPAEncoder(vocab_size=256, embed_dim=64, depth=2, num_heads=8, max_seq_len=64)
run(enc, ids, "small standalone")

cfg = TextSpanJEPAConfig(
    vocab_size=4096, max_seq_len=512, embed_dim=768, encoder_depth=12,
    num_heads=12, mlp_ratio=4.0, predictor_embed_dim=384, predictor_depth=6,
)
print("cfg.gradient_checkpointing =", cfg.gradient_checkpointing)
m = TextSpanJEPA(cfg)
print("model.encoder.gradient_checkpointing =", m.encoder.gradient_checkpointing)
run(m.encoder, ids, "full model encoder (gc off)")

m.encoder.gradient_checkpointing = False
run(m.encoder, ids, "full model encoder gc=False")

cfg2 = TextSpanJEPAConfig(
    vocab_size=4096, max_seq_len=512, embed_dim=768, encoder_depth=12,
    num_heads=12, mlp_ratio=4.0, predictor_embed_dim=384, predictor_depth=6,
    gradient_checkpointing=False,
)
m2 = TextSpanJEPA(cfg2)
run(m2.encoder, ids, "fresh model gc=False")
