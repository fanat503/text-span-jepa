"""Shared harness for the metrics inventory.

Every driver imports `guard()` first. The editable install on this box maps
`src` to a DIFFERENT clone, so an unasserted `import src` can silently measure
another tree. `guard()` refuses to continue if that happens.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = r"C:\dev\wt-metrics"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

import src  # noqa: E402


def guard():
    got = os.path.abspath(src.__file__)
    want = os.path.abspath(ROOT)
    assert got.startswith(want), f"WRONG TREE: src.__file__ = {got}, expected under {want}"
    print(f"[provenance] src.__file__ = {got}")
    return got


RESULTS = {}


def _num(v):
    """Coerce a 0-dim/1-element tensor to float; leave anything else alone."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    n = getattr(v, "numel", None)
    if callable(n) and n() == 1:
        return float(v)
    return v


def record(mech, metric, values, note=""):
    """Record a measured sweep: the list of values a metric took under inputs
    designed to vary the quantity its name refers to."""
    values = [_num(v) for v in values]
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    rec = {"values": values, "note": note}
    if nums:
        lo, hi = min(nums), max(nums)
        rec["min"] = lo
        rec["max"] = hi
        rec["range"] = hi - lo
        rec["n_numeric"] = len(nums)
        rec["constant"] = (hi - lo) == 0.0
    else:
        rec["constant"] = len({repr(v) for v in values}) == 1
    RESULTS.setdefault(mech, {})[metric] = rec
    flag = "CONST " if rec.get("constant") else "moves "
    print("  " + f"{mech:5s} {metric:38s} {flag} " + repr(values))
    return rec


def save(path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, indent=1, default=str)
    print(f"\n[saved] {path}  ({sum(len(v) for v in RESULTS.values())} metric rows)")


def ortho_Q(D, k, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(D, D, generator=g, dtype=dtype)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q[:, :k].contiguous()