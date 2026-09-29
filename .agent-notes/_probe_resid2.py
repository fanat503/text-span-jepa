"""Clean per-component fwd and bwd, with detached inputs so gradients do not
leak into the encoder. Residual hunt."""

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


def fwd_only(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()


def fwd_bwd(build):
    with FlopCounterMode(display=False) as fc:
        out = build()
        torch.autograd.backward(out, torch.ones_like(out) if out.dim() else out)
    return fc.get_total_flops()


for T in (256,):
    m, masked, ids, mask = make(T)
    print("T={}".format(T))

    # ---- encoder: detached input, grad only reaches encoder params ----
    x = masked.clone()
    with FlopCounterMode(display=False) as fc:
        h, _ = m.encoder(x)
        h.sum().backward()
    enc_fb = fc.get_total_flops()
    e_fwd = 24 * B * T * D * D * NE
    print("  encoder   fwd+bwd={:.6e}  fwd_analytic={:.6e}  ratio={:.5f}".format(
        enc_fb, e_fwd, enc_fb / e_fwd))

    # ---- target encoder (no_grad) ----
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m.target_encoder(ids)
    print("  target_encoder no_grad fwd={:.6e}  analytic={:.6e}".format(
        fc.get_total_flops(), e_fwd))

    # ---- predictor: detach inputs, grad only reaches predictor params ----
    with torch.no_grad():
        ho, te = m.encoder(masked)
        ht, _ = m.target_encoder(ids)
    ho, te, ht = ho.detach(), te.detach(), ht.detach()

    def build_pred():
        sp, _nm, _vm, fl, _fp = m.predictor(ho, mask, te, ht)
        tot = sp.sum()
        for v in fl.values():
            tot = tot + v.sum()
        return tot

    with FlopCounterMode(display=False) as fc:
        build_pred().backward()
    pred_fb = fc.get_total_flops()
    offs = [d for d in OFFSETS if d < T]
    S = REFINE * T + sum(T - d for d in offs)
    e_pred = (24 * B * S * DP * DP * NP
              + 2 * B * D * DP * (T + sum(T - d for d in offs)))
    print("  predictor fwd+bwd={:.6e}  fwd_analytic={:.6e}  ratio={:.5f}".format(
        pred_fb, e_pred, pred_fb / e_pred))

    # ---- decoder ----
    with torch.no_grad():
        hdm = ho[mask.bool()].detach().requires_grad_(True)
    with FlopCounterMode(display=False) as fc:
        lg = m.decoder(hdm, m.encoder.token_embedding.weight)
        lg.sum().backward()
    dec_fb = fc.get_total_flops()
    nm = int(mask.sum())
    e_dec = 2 * nm * (4 * D * D + D * V)
    print("  decoder   fwd+bwd={:.6e}  fwd_analytic={:.6e}  ratio={:.5f}  n_masked={}".format(
        dec_fb, e_dec, dec_fb / e_dec, nm))

    # ---- covariance reg ----
    hd = ho.detach().requires_grad_(True)
    with FlopCounterMode(display=False) as fc:
        m.covariance_reg(hd).backward()
    print("  covariance_reg fwd+bwd={:.6e}  3*2*B*T*D^2={:.6e}".format(
        fc.get_total_flops(), 3 * 2 * B * T * D * D))

    total_model = 3 * (e_fwd + e_pred + e_dec + 2 * B * T * D * D) + e_fwd
    with FlopCounterMode(display=False) as fc:
        loss, _i, _d = m.compute_loss_with_targets(masked, ids, mask)
        loss.backward()
    print("\n  measured step = {:.6e}".format(fc.get_total_flops()))
    print("  assembled     = {:.6e}".format(total_model))
