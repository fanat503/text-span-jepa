# Verification for docs/THEORY_AND_PROOFS.md - WSD findings.
# Run: & $PY .agent-notes/verify_wsd.py   (from C:\dev\wt-th)
# Read-only: imports src, builds nothing persistent, no training.

import glob
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.abspath("."))
import src  # noqa: E402

assert src.__file__.startswith(os.path.abspath(".")), f"wrong src: {src.__file__}"
print("PROVENANCE src =", src.__file__)

from src.models.wsd import WorkspaceSyncDrift  # noqa: E402

print("=== 1. Is wsd_steady_state_error an algebraic restatement of wsd_running_drift? ===")
torch.manual_seed(0)
w = WorkspaceSyncDrift(embed_dim=16, k=3, sync_interval=7, ema_beta=0.5)
info = None
for step in range(12):
    h = torch.randn(8, 16)
    Q = torch.linalg.qr(torch.randn(16, 3))[0]
    loss, info = w.compute_drift(Q, h_target=h, step=step)
    sse = info["wsd_steady_state_error"]
    rd = info["wsd_running_drift"]
    nu = info["wsd_nu_max_estimate"]
    lam = info["wsd_lambda_estimate"]
    print(
        f"step={step:2d} lam={lam:.6f} nu={nu:.6f} "
        f"sse={sse:.10f} running_drift={rd:.10f} sse-rd={sse - rd:+.3e}"
    )
print("lambda_estimate == 1/sync_interval ?", abs(info["wsd_lambda_estimate"] - 1.0 / 7) < 1e-12)
print(
    "nu_max == running_drift/sync_interval ?",
    abs(info["wsd_nu_max_estimate"] - info["wsd_running_drift"] / 7) < 1e-12,
)
print(
    "=> sse = nu/lam = (rd/T)/(1/T) = rd. Tautology:",
    abs(info["wsd_steady_state_error"] - info["wsd_running_drift"]) < 1e-12,
)

print()
print("=== 2. Does the shipped default lambda_wsd equal the code's lambda_estimate? ===")
print(
    "defaults.yaml lambda_wsd = 0.01 ; wsd_lambda_estimate = 1/wsd_sync_interval = 1/100 =",
    1.0 / 100,
)
print("=> the number the code REPORTS as lambda is 1/sync_interval, not config.lambda_wsd")

print()
print("=== 3. Non-orthonormal Q: does the clamp mask it as drift=0? ===")
w2 = WorkspaceSyncDrift(embed_dim=8, k=2, sync_interval=1, ema_beta=0.5)
Q_scaled = torch.linalg.qr(torch.randn(8, 2))[0] * 3.0
h = torch.randn(10, 8)
_, i_scaled = w2.compute_drift(Q_scaled, h_target=h, step=0)
_, i_orth = w2.compute_drift(Q_scaled / 3.0, h_target=h, step=1)
print(f"scaled ||Q||: drift_sq={i_scaled['wsd_drift_sq']:.6f}")
print(f"orthonormal  : drift_sq={i_orth['wsd_drift_sq']:.6f}")
print(f"the 3x-scaled Q reads drift_sq = 0 (clamped) vs {i_orth['wsd_drift_sq']:.6f} orthonormal:")
print(
    f"  -> clamp turns an OUT-OF-RANGE chordal value into 'perfect alignment': "
    f"{i_scaled['wsd_drift_sq'] < i_orth['wsd_drift_sq'] / 2}"
)

print()
print("=== 4. Reachable configs with use_wsd true (deep-merge over defaults.yaml) ===")


def deep_merge(base, over):
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


with open("defaults.yaml") as f:
    defaults = yaml.safe_load(f)


# find the subtree that actually carries use_wsd
def find_subtree(node, key):
    if isinstance(node, dict):
        if key in node:
            return node
        for v in node.values():
            r = find_subtree(v, key)
            if r is not None:
                return r
    return None


root = find_subtree(defaults, "use_wsd")
print(
    "defaults.yaml subtree carrying use_wsd: use_wsd =",
    root.get("use_wsd"),
    "lambda_wsd =",
    root.get("lambda_wsd"),
)

files = sorted(glob.glob("config/**/*.yaml", recursive=True))
on_both = []
for path in files:
    with open(path) as f:
        over = yaml.safe_load(f) or {}
    merged = deep_merge(defaults, over)
    sub = find_subtree(merged, "use_wsd")
    if sub.get("use_wsd") and sub.get("use_jawp"):
        on_both.append(path)
print(f"configs scanned: {len(files)}")
print(f"resolve use_wsd=True AND use_jawp=True: {len(on_both)}")
for p in on_both:
    print("   ", p)
