"""Residual is inside enc+tenc+pred. Bisect: real encoder vs a stub encoder."""

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


class CheapDecoder(nn.Module):
    def __init__(self, dim, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, h, w):
        return h.new_zeros(h.shape[0], self.vocab)


def make(T, cheap_decoder=True):
    cfg = TextSpanJEPAConfig(
        vocab_size=V, max_seq_len=512, embed_dim=D, encoder_depth=NE, num_heads=HEADS,
        mlp_ratio=R, predictor_embed_dim=DP, predictor_depth=NP,
        future_offsets=OFFSETS, num_refine_steps=REFINE,
    )
    torch.manual_seed(0)
    m = TextSpanJEPA(cfg)
    m.train()
    if cheap_decoder:
        m.decoder = CheapDecoder(D, V)
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
offs = [d for d in OFFSETS if d < T]
enc = 24 * B * T * D * D * NE
S = REFINE * T + sum(T - d for d in offs)
L = T + sum(T - d for d in offs)
pa = 24 * B * S * DP * DP * NP + 4 * B * D * DP * L

m, masked, ids, mask = make(T)

# 1) full (cheap decoder)
with FlopCounterMode(display=False) as fc:
    loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
    loss.backward()
full = fc.get_total_flops()

# 2) with the predictor's future branch disabled
m2, mk2, i2, ms2 = make(T)
m2.predictor.future_offsets = ()
with FlopCounterMode(display=False) as fc:
    loss2, _i, _d = m2.compute_loss_with_targets(mk2, i2, ms2)
    loss2.backward()
nofut = fc.get_total_flops()
fut_model = 3 * (24 * B * sum(T - d for d in offs) * DP * DP * NP
                 + 4 * B * D * DP * sum(T - d for d in offs))
print("full        = {:.6e}".format(full))
print("no-future   = {:.6e}  saves {:.6e}   model future={:.6e}  delta={:.3e}".format(
    nofut, full - nofut, fut_model, (full - nofut) - fut_model))

# 3) with num_refine_steps = 0
m3, mk3, i3, ms3 = make(T)
m3.predictor.num_refine_steps = 0
with FlopCounterMode(display=False) as fc:
    loss3, _i, _d = m3.compute_loss_with_targets(mk3, i3, ms3)
    loss3.backward()
norefine = fc.get_total_flops()
ref_model = 3 * (24 * B * REFINE * T * DP * DP * NP + 4 * B * D * DP * 0)
print("no-refine   = {:.6e}  saves {:.6e}   model refine-blocks={:.6e}  delta={:.3e}".format(
    norefine, full - norefine, ref_model, (full - norefine) - ref_model))

# 4) with a stubbed target encoder (identity-ish)
class StubTarget(nn.Module):
    def __init__(self, inner, dim):
        super().__init__()
        self.inner = inner
        self.dim = dim

    def forward(self, ids):
        return self.inner(ids)

m4, mk4, i4, ms4 = make(T)
with FlopCounterMode(display=False) as fc:
    loss4, _i, _d = m4.compute_loss_with_targets(mk4, i4, ms4)
    loss4.backward()
print()
print("target-encoder no_grad contribution should be {:.6e}".format(enc))
print("core model (3*(enc+pred) + enc) = {:.6e}".format(3 * (enc + pa) + enc))
print("full                                          = {:.6e}".format(full))
print("residual = {:.6e} ({:.2%})".format(full - (3 * (enc + pa) + enc),
                                          (full - (3 * (enc + pa) + enc)) / full))
