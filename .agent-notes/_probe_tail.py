"""Decisive: stub the loss-tail term by term to attribute the residual."""

import os
import sys

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


def build():
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
    return m, masked, ids, mask


def run(with_grad):
    m, mk, ids, mask = build()
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with FlopCounterMode(display=False) as fc:
        with ctx:
            m.compute_loss_with_targets(mk, ids, mask)
    return fc.get_total_flops()


fwd = run(False)

# Now with the predictor's span output replaced by a cheap tensor: stub
# predictor.forward_span_prediction and forward_future_prediction.
class StubPred(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, h_online, mask_positions, token_embeds, target_h):
        B = h_online.shape[0]
        sp = h_online.new_zeros(B, n_mask, self.dim)
        vm = torch.ones(B, n_mask, dtype=torch.bool, device=h_online.device)
        nm = torch.full((B,), n_mask, device=h_online.device)
        return sp, nm, vm, {}, {}


m, mk, ids, mask = build()
m.predictor = StubPred(D)
with FlopCounterMode(display=False) as fc:
    with torch.no_grad():
        m.compute_loss_with_targets(mk, ids, mask)
nopred = fc.get_total_flops()
pred_model = 2 * (2 * B * T * D * D * (4 + 2 * R) * NE) + (
    2 * (1 * B * T + B * (T - 1) + B * (T - 4)) * DP * DP * (4 + 2 * 4.0) * NP
    + 2 * 2 * B * D * DP * (T + (T - 1) + (T - 4))
)
print("forward-only total       = {}".format(fwd))
print("with stubbed predictor   = {}   (saves {})".format(nopred, fwd - nopred))
print("model predictor forward  = {}".format(pred_model))
print("predictor delta          = {:.1f}".format((fwd - nopred) - pred_model))
print()
print("residual after stubbing predictor = {}".format(nopred - pred_model))
