"""Wrap each loss call site on the live model and attribute its FLOPs exactly."""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils.flop_counter import FlopCounterMode

import src
from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), src.__file__

B, D, NE, NP, DP, V, HEADS, R = 4, 768, 12, 6, 384, 4096, 12, 4.0
OFFSETS, REFINE, MASK = (1, 4, 16), 3, 0.35

NAMES = ["_pred_rank_loss", "_cgn_ortho_loss", "_sigreg_loss", "_wsd_loss", "_sta_loss",
         "_rdc_loss", "_wsr_loss", "_puc_loss", "_swip_loss", "_spc_loss"]


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


def instrument(m, tally):
    for name in NAMES:
        orig = getattr(m, name)

        def wrap(*a, _o=orig, _n=name, **k):
            with FlopCounterMode(display=False) as fc:
                out = _o(*a, **k)
            tally[_n] = tally.get(_n, 0) + fc.get_total_flops()
            return out

        setattr(m, name, wrap)
    for name in ("variance_reg", "covariance_reg"):
        m._modules[name] = _Counter(m._modules[name], tally, name)


class _Counter:
    """Delegating proxy -- avoids nn.Module.__setattr__'s type check."""

    def __init__(self, inner, tally, name):
        self.inner = inner
        self.tally = tally
        self.name = name

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def __call__(self, *a, **k):
        with FlopCounterMode(display=False) as fc:
            out = self.inner(*a, **k)
        self.tally[self.name] = self.tally.get(self.name, 0) + fc.get_total_flops()
        return out

    def forward(self, *a, **k):
        with FlopCounterMode(display=False) as fc:
            out = self.inner(*a, **k)
        self.tally[self.name] = self.tally.get(self.name, 0) + fc.get_total_flops()
        return out


for T in (256,):
    m, masked, ids, mask = make(T)
    tally = {}
    instrument(m, tally)
    # Instance __dict__ shadows nn.Module.__getattr__, so __setattr__'s
    # Module-type check does not apply and m.encoder.token_embedding still resolves.
    for nm in ("encoder", "target_encoder", "predictor", "decoder"):
        sub = getattr(m, nm)
        m.__dict__[nm] = _Counter(sub, tally, nm)
        object.__setattr__(m, nm, m.__dict__[nm])
    with FlopCounterMode(display=False) as fc:
        loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
        loss.backward()
    full = fc.get_total_flops()
    print("T={} full={:.6e}".format(T, full))
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        if v:
            print("   {:>18} fwd={:.4e}  ({:.3%} of step)".format(k, v, v / full))
    s = sum(tally.values())
    print("   subtotal instrumented = {:.4e}  unaccounted = {:.4e}".format(s, full - s))
