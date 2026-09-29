"""FINAL VERIFY for TASK-22 -- pasted into the report verbatim.

Stale-clone guard: `pip install -e .` installs an __editable__ finder mapping
`src` -> C:/Users/<u>/tmp/clone-check-1/src. It is APPENDED to sys.meta_path,
i.e. after PathFinder, so it only wins when the running script's own directory
does not contain `src`. Running from inside the worktree is necessary but not
sufficient -- we assert.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch
from torch.utils.flop_counter import FlopCounterMode

import src

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), (
    "STALE-CLONE HAZARD: src -> %s (not under %s)" % (src.__file__, REPO)
)
print("src guard OK ->", src.__file__)
print("torch", torch.__version__, "| threads", torch.get_num_threads())
print()

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig  # noqa: E402
from src.utils.flops import estimate_jepa_step_flops  # noqa: E402

D, NE, DP, NP, HEADS, R = 768, 12, 384, 6, 12, 4.0
OFFSETS, REFINE, B, V, MASK_RATIO = (1, 4, 16), 3, 4, 4096, 0.35


class Off(torch.nn.Module):
    def compute(self, *a, **k):
        return {}


print("=== OLD estimator (Kaplan 6*N*L*B) vs NEW vs FlopCounterMode ===")
print("    base_140m dims (768/12, predictor 384/6), vocab 4096, B=4, mask 0.35")
print()
print("{:>5} {:>15} {:>15} {:>9} | {:>15} {:>9} {:>8}".format(
    "T", "real fwd+bwd", "OLD 6ND", "OLD/real", "NEW est", "NEW/real", "OLD/NEW"))
for T in (128, 256, 512):
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
    n = max(1, int(T * MASK_RATIO))
    mask = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        s = (b * 3) % max(1, T - n)
        mask[b, s : s + n] = 1
    masked = ids.clone()
    masked[mask.bool()] = 0

    with FlopCounterMode(display=False) as fc:
        loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
        loss.backward()
    real = fc.get_total_flops()

    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    old = 6 * n_params * T * B

    new = estimate_jepa_step_flops(
        embed_dim=D, encoder_depth=NE, predictor_embed_dim=DP, predictor_depth=NP,
        seq_len=T, batch_size=B, vocab_size=V, mlp_ratio=R, predictor_mlp_ratio=R,
        num_refine_steps=REFINE, future_offsets=OFFSETS, mask_ratio=MASK_RATIO,
    )
    e = new["total_flops"]
    print("{:>5} {:>15.6e} {:>15.6e} {:>9.5f} | {:>15.6e} {:>9.5f} {:>8.4f}".format(
        T, real, old, old / real, e, e / real, old / e))
print()
print("(OLD/real degrades 0.58 -> 0.52 with T; NEW/real is flat. The NEW number")
print(" still counts the per-step diagnostics, which the OLD one ignores too.)")
print()

print("=== NEW vs FlopCounterMode with the per-step diagnostics disabled ===")
print("    (CollapseDiagnostics.compute + JSpaceMetrics.compute, called")
print("     unconditionally by compute_loss_with_targets)")
print()
print("{:>5} {:>15} {:>15} {:>9} {:>15}".format(
    "T", "real (no diag)", "NEW est", "est/real", "diag cost"))
for T in (128, 256, 512):
    measured = {}
    for label, disable in (("off", True), ("on", False)):
        cfg = TextSpanJEPAConfig(
            vocab_size=V, max_seq_len=512, embed_dim=D, encoder_depth=NE, num_heads=HEADS,
            mlp_ratio=R, predictor_embed_dim=DP, predictor_depth=NP,
            future_offsets=OFFSETS, num_refine_steps=REFINE,
        )
        torch.manual_seed(0)
        m = TextSpanJEPA(cfg)
        m.train()
        if disable:
            m.diagnostics = Off()
            m.jspace_metrics = Off()
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(0, V, (B, T), generator=g)
        n = max(1, int(T * MASK_RATIO))
        mask = torch.zeros(B, T, dtype=torch.long)
        for b in range(B):
            s = (b * 3) % max(1, T - n)
            mask[b, s : s + n] = 1
        masked = ids.clone()
        masked[mask.bool()] = 0
        with FlopCounterMode(display=False) as fc:
            loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
            loss.backward()
        measured[label] = fc.get_total_flops()
    real = measured["off"]
    est = estimate_jepa_step_flops(
        embed_dim=D, encoder_depth=NE, predictor_embed_dim=DP, predictor_depth=NP,
        seq_len=T, batch_size=B, vocab_size=V, mlp_ratio=R, predictor_mlp_ratio=R,
        num_refine_steps=REFINE, future_offsets=OFFSETS, mask_ratio=MASK_RATIO,
    )
    print("{:>5} {:>15.6e} {:>15.6e} {:>9.5f} {:>15.6e}".format(
        T, real, est["total_flops"], est["total_flops"] / real,
        measured["on"] - real))
print()
print("=== component breakdown @ T=512 ===")
est = estimate_jepa_step_flops(
    embed_dim=D, encoder_depth=NE, predictor_embed_dim=DP, predictor_depth=NP,
    seq_len=512, batch_size=B, vocab_size=V, mlp_ratio=R, predictor_mlp_ratio=R,
    num_refine_steps=REFINE, future_offsets=OFFSETS, mask_ratio=MASK_RATIO,
)
for k in ("total_flops", "encoder_flops", "target_encoder_flops", "predictor_flops",
          "decoder_flops", "covariance_flops", "attention_flops"):
    print("  {:>22}  {:.6e}".format(k, est[k]))
print()
print("=== FlopCounterMode counts NO attention on this build ===")
import torch.nn.functional as F  # noqa: E402

q = torch.randn(B, HEADS, 64, D // HEADS)
with FlopCounterMode(display=False) as fc:
    F.scaled_dot_product_attention(q, q, q)
print("  F.scaled_dot_product_attention -> {:.1f} FLOPs".format(fc.get_total_flops()))
print("  => attention_flops is reported separately, never folded into total_flops.")
