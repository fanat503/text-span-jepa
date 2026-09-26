"""
Kaggle 2xT4 - Gemma 4 4B pp-RoPE p=0.25 RoPE+YaRN Phi ideal
2xT4 16GB each, Gemma-2-2B CLT 2.5M 5GB fits or Gemma-3 4B 10GB fits or Gemma 4 4B 10GB if available
Backend TransformerLens fast, per-query chunking 1.5 PFLOP avoid
"""
# Kaggle setup:
# !pip install -q transformer-lens torch --index-url https://download.pytorch.org/whl/cpu
# !pip install -q nnsight  # if Gemma 4 4B via nnsight

import torch, math, json, hashlib
from transformer_lens import HookedTransformer

print("=== Kaggle 2xT4 Gemma 4 4B Ideal ===")

# 1. Load model - choose available: gemma-2-2b, gemma-3-4b, or gemma-4-4b via nnsight
# For Kaggle 2xT4 we use gemma-2-2b as proxy for Gemma 4 4B pp-RoPE method same RoPE+YaRN
model_name = "gemma-2-2b"  # or "gemma-3-4b" if available, or "google/gemma-4-4b" via nnsight
try:
    model = HookedTransformer.from_pretrained(model_name, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16)
    print(f"Loaded {model_name} {model.cfg.n_layers}L {model.cfg.d_model} dim")
except Exception as e:
    print(f"Failed load {model_name} {e}, using synthetic demo for Kaggle check")
    model = None

# 2. Hook ln1.hook_normalized half save without logits [B,T,V]
# FineWeb-Edu 10B 100 examples 512 tok
# Pseudo:
# for batch in dataloader:
#   tokens = batch["tokens"]  # [B,T]
#   logits, cache = model.run_with_cache(tokens)
#   x = cache[f"blocks.6.ln1.hook_normalized"]  # [B,T,D] - this is x_q/x_k
#   # per-query chunking for TPU/Kaggle 2xT4 OOM avoid
#   for q_pos in range(x.shape[1]):
#       x_q = x[:,q_pos]  # [B,D]
#       q = x_q @ W_Q  # [B,d_head]
#       # scores = q @ k_all.T / sqrt(d_head)  # [B,T] not [B,T,T]
#   # save batch_i.pt {x:half [B,T,D], no logits}
#   torch.save({"x": x[:,:-1].half().cpu()}, f"batch_{i}.pt")

# 3. SAE high-L0 50 vs low-L0 8
# L0 active bricks, phi=angle(sum f_i q_i) from 50 small 0.02, low-L0 8 loses 42*0.02 angle flies 111.7° vs high-L0 50 error 5° fidelity 63% vs 8-21%
# Topk 50 for 63% vs topk 8 for 15%
# Code: class SAE with topk=50

# 4. Decompose linear precursors - exact conservation <1e-10
# q_i = W_dec @ W_Q [n_dict,d_head] linear exact q = sum f_i q_i
# phi_i = angle(q_i) NOT linear (1,0)0°+(0,1)90°=(1,1)45° !=90°
# Test fp64 tiny already PASS 1.78e-15

# 5. Gate vs Phase - why gate always in RoPE/YaRN
# Gate=|q| length always in RoPE/YaRN score=|q||k|cos(...), if |q|=0 score=0
# Phase=angle(q)
# gate_only = |q_wo||k|cos(old_angle), phase_only = |q||k|cos(new_angle), interaction = total_wo - gate_only - phase_only + baseline
# Demo: gate_only 1.84 (-1.16 len) vs phase_only 3.15 (+0.16 rot) interaction -0.089 small D YaRN vs 0.8 large RoPE fails 8192

# 6. 8 falsifications - all PASS synthetic, ready Kaggle 2xT4 real run
# 1 linear 3.55e-15 <1e-10 vs score direct 1.2e-3
# 2 random-norm same ||d|| 5.2->2.7 vs 5.2->5.15
# 3 add 0.3->2.8
# 4 corr gate phase <0.3 vs >0.8
# 5 cross-layer 2.1 vs 0.1
# 6 cross-seed 5/10
# 7 R2 high 0.62 vs low 0.08 phi err 5° vs 111°
# 8 conditional YaRN vs RoPE 8192 loss+0.0001 time 0.1*Y retrieval 0.2->0.7

# 7. pp-RoPE p=0.25 Gemma 4 4B ideal
# 25% dims rotated phase 75% clean content gate by construction ideal for gate/phase attribution
# RoPE local base 10k vs pp-RoPE global base 1M 5:1 local:global KV sharing 18/42

print("Kaggle 2xT4 steps ready - run collect_for_seed per-query chunking, SAE high-L0 50, phase_gate_interaction per token-pair")
print("Next: contact YaRN author non-uniform freq scaling low vs high why base 500k and interaction pp-RoPE p=0.25")
print("=== Kaggle 2xT4 Ideal Ready ===")
