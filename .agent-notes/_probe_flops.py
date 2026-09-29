"""Ad-hoc FlopCounterMode probe for TASK-22. NOT part of the test suite.

Guards against the known hazard: `pip install -e .` installs an
__editable__ finder that maps `src` to a STALE clone
(C:/Users/<user>/tmp/clone-check-1/src). That finder is APPENDED to
sys.meta_path, i.e. AFTER PathFinder, so it only wins when the running
script's own directory does not contain `src`. Running this file from
inside the worktree is therefore necessary but not sufficient -- we assert.
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

import src  # noqa: E402

assert os.path.realpath(src.__file__).startswith(os.path.realpath(REPO)), (
    "STALE-CLONE HAZARD: src resolved to %s, not under %s" % (src.__file__, REPO)
)
print("src guard OK ->", src.__file__)
print("torch", torch.__version__, "threads", torch.get_num_threads())

from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig  # noqa: E402

VOCAB = 4096
B = 4
DEPTH = 12
DIM = 768
HEADS = 12
MLP = 4.0
P_DIM = 384
P_DEPTH = 6
OFFSETS = (1, 4, 16)
REFINE = 3
MASK_RATIO = 0.35


def build():
    cfg = TextSpanJEPAConfig(
        vocab_size=VOCAB,
        max_seq_len=512,
        embed_dim=DIM,
        encoder_depth=DEPTH,
        num_heads=HEADS,
        mlp_ratio=MLP,
        predictor_embed_dim=P_DIM,
        predictor_depth=P_DEPTH,
        future_offsets=OFFSETS,
        num_refine_steps=REFINE,
    )
    return TextSpanJEPA(cfg)


def make_inputs(T):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, VOCAB, (B, T), generator=g)
    masked = ids.clone()
    n_mask = max(1, int(T * MASK_RATIO))
    mask = torch.zeros(B, T, dtype=torch.long)
    for b in range(B):
        start = (b * 3) % max(1, T - n_mask)
        masked[b, start : start + n_mask] = 0
        mask[b, start : start + n_mask] = 1
    return masked, ids, mask


def measure(T):
    torch.manual_seed(0)
    model = build()
    model.train()
    masked, ids, mask = make_inputs(T)

    with FlopCounterMode(display=False) as fc:
        loss, _info, _d = model.compute_loss_with_targets(masked, ids, mask)
        loss.backward()
    total = fc.get_total_flops()

    parts = {}

    # forward-only, no_grad  (target-encoder path)
    with torch.no_grad():
        with FlopCounterMode(display=False) as f:
            model.encoder(ids)
        parts["enc_fwd_nograd"] = f.get_total_flops()
        with FlopCounterMode(display=False) as f:
            model.target_encoder(ids)
        parts["tenc_fwd_nograd"] = f.get_total_flops()

    # forward with graph
    with FlopCounterMode(display=False) as f:
        ho, te = model.encoder(masked)
    parts["enc_fwd"] = f.get_total_flops()
    with FlopCounterMode(display=False) as f:
        ho.sum().backward()
    model.zero_grad(set_to_none=True)
    parts["enc_fwdbwd"] = f.get_total_flops()

    with torch.no_grad():
        ht, _ = model.target_encoder(ids)
    with FlopCounterMode(display=False) as f:
        model.predictor(ho.detach(), mask, te.detach(), ht.detach())
    parts["pred_fwd_nograd"] = f.get_total_flops()

    with FlopCounterMode(display=False) as f:
        model.decoder(ho.detach()[mask.bool()], model.encoder.token_embedding.weight)
    parts["dec_fwd"] = f.get_total_flops()

    return total, parts


if __name__ == "__main__":
    print("\n{:>5} {:>14} {:>14} {:>8}".format("T", "fwd+bwd", "encoder_fwd", "ratio"))
    rows = []
    for T in (128, 256, 512):
        total, p = measure(T)
        rows.append((T, total, p))
        print(
            "{:>5} {:>14.6e} {:>14.6e} {:>8.4f}".format(
                T, total, p["enc_fwd"], total / p["enc_fwd"]
            )
        )

    print("\n--- component breakdown (forward-only unless noted) ---")
    hdr = ["enc_fwd_nograd", "tenc_fwd_nograd", "enc_fwd", "enc_fwdbwd", "pred_fwd_nograd", "dec_fwd"]
    print("{:>5}".format("T") + "".join("{:>18}".format(h) for h in hdr))
    for T, total, p in rows:
        print("{:>5}".format(T) + "".join("{:>18.6e}".format(p[h]) for h in hdr))
    T, total, p = rows[-1]
    print("\nT=512 shares of total fwd+bwd:")
    for h in hdr:
        print("  {:>18} {:>8.2f}%".format(h, 100 * p[h] / total))
    print("  {:>18} {:>8.4f}".format("enc_fwd/enc_fwdbwd", p["enc_fwdbwd"] / p["enc_fwd"]))
