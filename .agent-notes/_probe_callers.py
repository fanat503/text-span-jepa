"""Print the Python caller frame of every unmodelled matmul during the forward."""

import os
import sys
import traceback
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils._python_dispatch import TorchDispatchMode

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

INTERESTING = (("mm", (32, 32), (32, 32)), ("mm", (32, 6), (6, 32)),
               ("mm", (1, 32), (32, 31)))


SKIP = ("__probe_callers", "eval_frame.py", "flop_counter.py", "<torch", "functorch",
        "module.py", "_functions.py", "module_tracker.py")


def caller():
    """Nearest frame in src/ (i.e. the actual model code), not torch plumbing."""
    for fr in reversed(traceback.extract_stack()[:-2]):
        f = fr.filename.replace("\\", "/")
        if any(s in f for s in SKIP):
            continue
        if "wt-22" not in f:
            continue
        return "{}:{}  in {}()  |  {}".format(
            os.path.basename(fr.filename), fr.lineno, fr.name, fr.line)
    return "?"


class Rec(TorchDispatchMode):
    def __init__(self):
        self.hits = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        op = str(func).split(".")[1]
        ts = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]
        key = None
        if op in ("mm", "bmm") and len(ts) >= 2:
            key = (op, ts[0], ts[1])
        elif op in ("addmm", "baddbmm") and len(ts) >= 3:
            key = (op, ts[1], ts[2])
        if key in INTERESTING:
            self.hits[(key, caller())] += 1
        return func(*args, **kwargs)


rec = Rec()
with rec:
    with torch.no_grad():
        m.compute_loss_with_targets(masked, ids, mask)

for (key, where), c in sorted(rec.hits.items(), key=lambda kv: -kv[1]):
    fl = 2 * key[1][0] * key[1][1] * key[2][1] * c
    print("x{}  {:>10.0f}  {}  <-  {}".format(c, fl, key, where))
