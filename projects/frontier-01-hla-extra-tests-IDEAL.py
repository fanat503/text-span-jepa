# OLD_MODEL-only toy dead end remains abandoned; gate always lemma + multi-mechanism required
#!/usr/bin/env python3
"""
frontier-01-hla-extra-tests-IDEAL.py
IDEAL implementation for OLD_MODEL extra tests - дотестить OLD_MODEL до конца.

This file demonstrates IDEAL (maximal sterility, reproducibility, variance) for OLD_MODEL testing:
- SVD всех механизмов
- Второй сид
- Длинные seq_len loss в конце
- Зануление laplace механизмов

Usage:
  python frontier-01-hla-extra-tests-IDEAL.py --mode svd
  python frontier-01-hla-extra-tests-IDEAL.py --mode long_context
  python frontier-01-hla-extra-tests-IDEAL.py --mode zero_laplace
  python frontier-01-hla-extra-tests-IDEAL.py --mode all

All modes work on synthetic demo without real checkpoints, and on real checkpoints if provided.
"""

import argparse, json, os, sys, math, torch
import torch.nn.functional as F

# Add OLD_MODEL repo to path
OLD_MODEL_ROOT = "./reviews/laplace-attention"
if OLD_MODEL_ROOT not in sys.path:
    sys.path.insert(0, OLD_MODEL_ROOT)

try:
    from src.model import GPT, GPTConfig
    from src.data import FixedDataset
    OLD_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"OLD_MODEL repo not available: {e}, using mock")
    OLD_MODEL_AVAILABLE = False
    GPT = None
    GPTConfig = None

def make_tiny_hla_model(device, trained=False, seed=42):
    """Tiny OLD_MODEL model for IDEAL demo (avoid OOM)"""
    torch.manual_seed(seed)
    cfg = {
        "block_size": 512, "vocab_size": 50257, "n_layer": 2, "n_head": 4, "n_embd": 256,
        "dropout":0.0,"bias":False,"gradient_checkpointing":False,"fused_swiglu":True,
        "use_rope":True,"phase_mult":0.15,"use_laplace":True,"laplace_alpha":1.0,
        "laplace_range_k":0.8,"laplace_range_v":0.5,"beta_k":0.7,"beta_v":0.5,
        "k_log_clip":2.0,"v_log_clip":1.5,"layer_dependent_gate":False,
        "use_distance_laplace":True,"distance_laplace_alpha":0.5,"distance_laplace_range":1.0,"distance_laplace_clip":1.0,
        "use_salience_bias":True,"salience_alpha":1.0,"salience_range":2.0,"salience_clip":3.0,
        "use_wpe":False
    }
    if not OLD_MODEL_AVAILABLE:
        return None, cfg
    model = GPT(GPTConfig(**cfg))
    if trained:
        for blk in model.transformer.h:
            blk.attn.W_phase_q.data = torch.randn_like(blk.attn.W_phase_q) * 0.03
            blk.attn.W_phase_k.data = torch.randn_like(blk.attn.W_phase_k) * 0.03
            blk.attn.W_gate_k.weight.data = torch.randn_like(blk.attn.W_gate_k.weight) * 0.08
            blk.attn.W_gate_v.weight.data = torch.randn_like(blk.attn.W_gate_v.weight) * 0.08
            blk.attn.W_gate_sal.weight.data = torch.randn_like(blk.attn.W_gate_sal.weight) * 0.06
            blk.attn.W_gate_d.weight.data = torch.randn_like(blk.attn.W_gate_d.weight) * 0.06
    model.to(device).eval()
    return model, cfg

def svals(mat):
    return torch.linalg.svdvals(mat.detach().float().cpu())
def erank(s, eps=1e-12):
    total = float(s.sum())
    if total <= 0:
        return 0.0
    p = (s/total).clamp_min(eps)
    return float(torch.exp(-(p*p.log()).sum()))
def stable_rank(s):
    if s.numel()==0 or float(s[0])==0:
        return 0.0
    return float((s*s).sum()/(s[0]*s[0]))

def mode_svd(device):
    print("\n" + "="*60)
    print("MODE: SVD - до конца посмотреть все наши SVD и ТД")
    print("="*60)

    print("\n--- At Init (Identity, Theorem 1) ---")
    model_init, _ = make_tiny_hla_model(device, trained=False, seed=42)
    if model_init:
        for li, blk in enumerate(model_init.transformer.h):
            attn = blk.attn
            gk_s = svals(attn.W_gate_k.weight)
            print(f"L{li}: gate_k norm {float(attn.W_gate_k.weight.norm()):.4f} erank {erank(gk_s):.2f} (expected 0 at init)")
            print(f"     phase_q norm {float(attn.W_phase_q.norm()):.4f} (expected 0 at init)")

    print("\n--- After Training (Simulated) ---")
    model_tr, _ = make_tiny_hla_model(device, trained=True, seed=42)
    if model_tr:
        for li, blk in enumerate(model_tr.transformer.h):
            attn = blk.attn
            gk_s = svals(attn.W_gate_k.weight)
            gv_s = svals(attn.W_gate_v.weight)
            gsal_s = svals(attn.W_gate_sal.weight)
            gd_s = svals(attn.W_gate_d.weight)
            # Q/K/V stable rank
            W = attn.c_attn.weight
            C = attn.n_embd
            s_q = svals(W[:C,:])
            s_k = svals(W[C:2*C,:])
            s_v = svals(W[2*C:,:])
            # gate correlations
            gk = attn.W_gate_k.weight.detach().float().flatten()
            gv = attn.W_gate_v.weight.detach().float().flatten()
            def corr(a,b):
                a = a - a.mean()
                b = b - b.mean()
                return float((a*b).sum() / (a.norm()*b.norm()+1e-12))
            print(f"L{li}: gate_k erank {erank(gk_s):.2f} norm {float(attn.W_gate_k.weight.norm()):.3f} | gate_v erank {erank(gv_s):.2f} | sal {erank(gsal_s):.2f} d {erank(gd_s):.2f} | qk_stable {0.5*(stable_rank(s_q)+stable_rank(s_k)):.1f} v_stable {stable_rank(s_v):.1f} | k-v corr {corr(gk,gv):.3f} (threshold |corr|>0.9 => merge)")

    print("\nExpected for real 200M (12L 1024d 16H):")
    print("  phase_q_erank 4-8 (not rank-1 collapse, not full hd/2=32)")
    print("  gate_k_erank 3-8 (specialized, not 1 and not H=16)")
    print("  gate redundancy |corr|<0.9 => keep separate, >0.9 merge in v6 (R-B)")
    print("  qk_interference DROP vs base, ov_interference preserved, separation INCREASE (H2)")
    print("  qk_stable DIVERGE from base (retrieval specializes), v_stable comparable")

def mode_long_context(device):
    print("\n" + "="*60)
    print("MODE: Long Context Loss At End - замерить именно в конце насколько loss отличается")
    print("="*60)

    print("\nSimulating base vs OLD_MODEL loss at different seq_lens (real 200M would use real val data):")
    seq_lens = [256,512,1024,2048]
    for T in seq_lens:
        # Simulated expected behavior: base degrades more at end, OLD_MODEL less
        base_first = 3.2
        base_last = 3.2 + 0.15 * math.log(T/128+1)
        hla_first = 3.18
        hla_last = 3.18 + 0.05 * math.log(T/128+1)
        delta_last = base_last - hla_last
        print(f"T={T}: base last10% {base_last:.4f} OLD_MODEL last10% {hla_last:.4f} delta {delta_last:+.4f} (should GROW with T)")

    print("\nFor REAL measurement:")
    print("  python scripts/analyze_checkpoint.py --checkpoint <ckpt> --config <config> --out out.json --seq-len 2048 --batches 32")
    print("Check:")
    print("  per_position_loss_curve: posloss_bin_00 vs posloss_bin_07, early_late_ratio, mid_bump")
    print("  positional_recall_curve: litm_middle_drop (OLD_MODEL smaller), litm_worst_frac (OLD_MODEL closer to 1)")
    print("  knockout_by_context_length: delta GROWS with T = causal evidence for long-range")

    # Also run tiny synthetic eval
    model_base, _ = make_tiny_hla_model(device, trained=False, seed=42)
    model_hla, _ = make_tiny_hla_model(device, trained=True, seed=43)
    if model_base and model_hla:
        # zero base mechanisms
        for blk in model_base.transformer.h:
            blk.attn.phase_mult = 0.0
            blk.attn.laplace_alpha = 0.0
            blk.attn.distance_laplace_alpha = 0.0
            blk.attn.salience_alpha = 0.0
        # quick per-pos loss on synthetic tokens - need 513 to cover T=512
        g = torch.Generator(device="cpu")
        g.manual_seed(42)
        toks = torch.randint(0, 50257, (4, 513), generator=g)
        def per_pos_loss(model, toks, T):
            per_sum = torch.zeros(T)
            for i in range(toks.shape[0]):
                x = toks[i:i+1, :T].to(device)
                y = toks[i:i+1, 1:T+1].to(device)
                logits,_ = model(x)
                from src.eval import _mask_padded_logits
                logits = _mask_padded_logits(model, logits.float())
                # logits [1,T,V], y [1,T] -> per-pos loss [T]
                loss = F.cross_entropy(logits[0], y[0], reduction='none')
                per_sum += loss.cpu()
                del logits
            return per_sum / toks.shape[0]
        for T in [128,256,512]:
            per_base = per_pos_loss(model_base, toks, T)
            per_hla = per_pos_loss(model_hla, toks, T)
            last10_base = float(per_base[-T//10:].mean())
            last10_hla = float(per_hla[-T//10:].mean())
            print(f"Synthetic T={T}: base last10% {last10_base:.4f} OLD_MODEL {last10_hla:.4f} delta {last10_base-last10_hla:+.4f}")

def mode_zero_laplace(device):
    print("\n" + "="*60)
    print("MODE: Zero Laplace Mechanisms - сравнить что будет если занулить")
    print("="*60)

    model, _ = make_tiny_hla_model(device, trained=True, seed=42)
    if not model:
        print("OLD_MODEL not available")
        return

    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    x = torch.randint(0, 50257, (2, 256), generator=g).to(device)
    y = torch.randint(0, 50257, (2, 256), generator=g).to(device)

    from src.eval import mechanism_knockout
    _, base_loss = model(x,y)
    base_loss = float(base_loss)
    print(f"Full OLD_MODEL loss: {base_loss:.4f} (at init, would be same as base)")

    ko = mechanism_knockout(model, x, y, mechanisms=["phase","gates","salience","distance","forget","qtemp"])
    for k,v in ko.items():
        print(f"  {k}: {v}")

    print("\nAt init (identity): all deltas 0 (Theorem 1)")
    print("After real training expected:")
    print("  phase +0.02-0.05 (content-conditioned Q/K rotation)")
    print("  gates +0.03-0.08 (K/V Laplace suppression)")
    print("  salience +0.01-0.03 (importance bias)")
    print("  distance +0.02-0.04 (content-conditioned distance, flattens LITM)")
    print("  forget/qtemp OFF in primary recipe (delta 0)")

    # Family zeroing
    saved = {}
    for blk in model.transformer.h:
        saved[id(blk)] = {
            "phase_mult": blk.attn.phase_mult,
            "laplace_alpha": blk.attn.laplace_alpha,
            "distance_alpha": blk.attn.distance_laplace_alpha,
            "salience_alpha": blk.attn.salience_alpha,
        }
    def eval_zeroed(zero_list):
        for blk in model.transformer.h:
            if "phase" in zero_list:
                blk.attn.phase_mult = 0.0
            if "gates" in zero_list:
                blk.attn.laplace_alpha = 0.0
            if "distance" in zero_list:
                blk.attn.distance_laplace_alpha = 0.0
            if "salience" in zero_list:
                blk.attn.salience_alpha = 0.0
        _, loss = model(x,y)
        for blk in model.transformer.h:
            s = saved[id(blk)]
            blk.attn.phase_mult = s["phase_mult"]
            blk.attn.laplace_alpha = s["laplace_alpha"]
            blk.attn.distance_laplace_alpha = s["distance_alpha"]
            blk.attn.salience_alpha = s["salience_alpha"]
        return float(loss)

    for zl, desc in [
        (["phase"], "Zero ONLY phase"),
        (["gates"], "Zero ONLY K/V gates"),
        (["salience"], "Zero ONLY salience"),
        (["distance"], "Zero ONLY distance"),
        (["gates","salience","distance"], "Zero ALL laplace family"),
        (["phase","gates","salience","distance"], "Zero ALL OLD_MODEL"),
    ]:
        loss = eval_zeroed(zl)
        print(f"{desc}: loss {loss:.4f} delta {loss-base_loss:+.4f}")

def mode_seed43():
    print("\n" + "="*60)
    print("MODE: Second Seed 43 - обучить еще на одном сиде")
    print("="*60)
    print("\nConfigs already ready:")
    print("  configs/kaggle_200m_base_9h_s42.json - base seed 42")
    print("  configs/kaggle_200m_hla_9h_s42.json - OLD_MODEL seed 42")
    print("  configs/kaggle_200m_base_9h_s43.json - base seed 43")
    print("  configs/kaggle_200m_hla_9h_s43.json - OLD_MODEL seed 43")
    print("\nDifference only seed and init_ckpt - sterile pair same init/data, differ only via mechanisms")
    print("\nTraining commands (Kaggle V5e-8, batch 1 accum 16 eff 128 seq 2048, 15000 steps 3.93B tokens):")
    print("  python src/train_xla.py --config configs/kaggle_200m_base_9h_s43.json")
    print("  python src/train_xla.py --config configs/kaggle_200m_hla_9h_s43.json")
    print("\nTPU month grant (28B dataset, 20000 steps):")
    print("  python src/train_xla.py --config configs/200m_base_s43.json")
    print("  python src/train_xla.py --config configs/200m_hla_s43.json")
    print("\nExpected:")
    print("  Val loss gap OLD_MODEL < base sign-stability across seeds")
    print("  Gap ~0.01-0.03 on 200M (from v3/v4 historical)")
    print("  If gap <5x seed std, add seeds 45,46 (pre-registered)")
    print("  Paired t-test across seeds (same seed = same init pair)")
    print("\nAnalysis after training:")
    print("  python scripts/analyze_checkpoint.py --checkpoint runs/kaggle_200m_hla_9h_s43/checkpoint.pt --config configs/kaggle_200m_hla_9h_s43.json --out runs/hla_s43_analysis.json --seq-len 2048 --batches 32")

def main():
    ap = argparse.ArgumentParser(description="OLD_MODEL Extra Tests IDEAL")
    ap.add_argument("--mode", default="all", choices=["svd","long_context","zero_laplace","seed43","all"], help="which test to run")
    ap.add_argument("--device", default="cpu", help="cpu|cuda")
    args = ap.parse_args()

    device = torch.device(args.device)

    if args.mode in ("svd","all"):
        mode_svd(device)
    if args.mode in ("long_context","all"):
        mode_long_context(device)
    if args.mode in ("zero_laplace","all"):
        mode_zero_laplace(device)
    if args.mode in ("seed43","all"):
        mode_seed43()

    print("\n" + "="*60)
    print("All extra tests done. See docs/EXTRA_TESTS.md for full documentation.")
    print("Scripts:")
    print("  scripts/svd_full_audit.py")
    print("  scripts/eval_long_context_end.py")
    print("  scripts/zero_laplace_ablation.py")
    print("  scripts/eval_hla_extra.py (all-in-one)")
    print("  scripts/eval_hla_trained_sim.py (simulated trained)")
    print("="*60)

if __name__ == "__main__":
    main()
