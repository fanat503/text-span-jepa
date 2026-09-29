"""Which parameter count produced the card's 3.147e11 / 6.295e11 / 1.259e12?"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch

import src

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig  # noqa: E402

cfg = TextSpanJEPAConfig(
    vocab_size=4096, max_seq_len=512, embed_dim=768, encoder_depth=12, num_heads=12,
    mlp_ratio=4.0, predictor_embed_dim=384, predictor_depth=6,
    future_offsets=(1, 4, 16), num_refine_steps=3,
)
torch.manual_seed(0)
m = TextSpanJEPA(cfg)

total = sum(p.numel() for p in m.parameters())
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
enc_only = sum(p.numel() for p in m.encoder.parameters())
nonemb = m.get_num_params()
nonemb_all = m.encoder.get_num_params()

print("total params          = {:,}".format(total))
print("trainable             = {:,}".format(trainable))
print("encoder only          = {:,}".format(enc_only))
print("get_num_params()      = {:,}   <- what train.py logs".format(nonemb))
print("encoder non-embedding = {:,}".format(nonemb_all))

print("\ncard's estimate at T=512 is 1.259e12;  6*N*512*4 -> N = {:.4e}".format(
    1.259e12 / (6 * 512 * 4)))
for name, n in (("total", total), ("trainable", trainable), ("encoder", enc_only),
                ("get_num_params", nonemb), ("enc non-emb", nonemb_all)):
    print("  6*N*512*4 with {:>16} = {:.6e}".format(name, 6 * n * 512 * 4))
