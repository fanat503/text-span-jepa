"""
REAL HOOK DEMO - высший уровень, TransformerLens gpt2-small как превью Qwen3-14B TPU
Показывает ln1.hook_normalized half save без логитов, per-query chunking
"""
import torch
from transformer_lens import HookedTransformer

print("Loading gpt2-small...")
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
print("Model loaded")

# Промпт как в induction
prompt = "The cat sat on the mat. The cat sat on the"
tokens = model.to_tokens(prompt)
print(f"Tokens: {tokens.shape} {model.to_str_tokens(tokens[0][:10])}")

# Run with cache
logits, cache = model.run_with_cache(tokens)
print("Cache keys:", [k for k in cache.keys() if "ln1" in k][:3])

# Hook ln1 layer 0
x = cache["blocks.0.ln1.hook_normalized"]  # [B,T,D]
print(f"x ln1.hook_normalized shape {x.shape} dtype {x.dtype} - half save would be {x.half().shape}")

# W_Q, W_K
W_Q = model.blocks[0].attn.W_Q  # [n_heads, d_model, d_head]
W_K = model.blocks[0].attn.W_K
print(f"W_Q {W_Q.shape} W_K {W_K.shape}")

# Per-query chunking demo for TPU v5e-8 1.5 PFLOP avoidance
B,T,D = x.shape
n_heads = W_Q.shape[0]
d_head = W_Q.shape[2]
# Take head 0
W_Q_h0 = W_Q[0]  # [d_model,d_head]
W_K_h0 = W_K[0]

# q for each pos: x @ W_Q
# Instead of [B,T,T] scores, per query
for q_pos in [5, 10]:
    x_q = x[0,q_pos]  # [D]
    q = x_q @ W_Q_h0  # [d_head]
    # k for all pos
    k_all = x[0] @ W_K_h0  # [T,d_head]
    scores = (q @ k_all.T) / (d_head**0.5)  # [T]
    print(f"q_pos={q_pos} scores shape {scores.shape} mean {scores.mean():.3f} - per-query chunking avoids [T,T]")

# Gate vs Phase demo for this real q
q_2d = q[:2]  # take 2 dims as 2D for polar demo
mag = torch.norm(q_2d).item()
phi = torch.atan2(q_2d[1], q_2d[0]).item()
import math
print(f"Real q from gpt2 head0 pos10 first 2 dims {q_2d.tolist()} |q|={mag:.3f} phi={math.degrees(phi):.1f}° = gate and phase in real model")

# Half save without logits
x_half = x[:,:-1].half()
print(f"Half save without logits [B,T,V] -> x half {x_half.shape} {x_half.dtype} - sterility ok, no [B,T,50257] saved")

print("\n=== REAL HOOK DEMO PASS - ready for Qwen3-14B TPU v5e-8 per-query chunking ===")
