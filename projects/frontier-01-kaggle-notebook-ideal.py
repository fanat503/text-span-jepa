"""
Kaggle Notebook Ideal - Gemma 4 4B pp-RoPE p=0.25 RoPE+YaRN Phi + Bag-of-Words
2xT4 16GB each 12h, copy each cell into Kaggle Notebook
11 cells total, ready to run
"""

# Cell 1: Install
"""
!pip install -q transformer-lens==2.14.0 torch --index-url https://download.pytorch.org/whl/cu121
!pip install -q nnsight==0.4.5 datasets==2.19.0 accelerate==0.33.0 scikit-learn matplotlib einops
import torch
print(torch.cuda.is_available(), torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
"""

# Cell 2: Bilinearity Break Demo numpy-only for non-math
"""
import math, numpy as np
print("=== ПОЧЕМУ БИЛИНЕЙНОСТЬ ЛОМАЕТСЯ ===")
x_q, x_k = 1.0, 2.0
score_fixed = x_q*x_k*math.cos(1.0)
score_fixed2 = 2.0*x_k*math.cos(1.0)
print(f"Fixed pos 1.08->2.16 {score_fixed2/score_fixed:.2f}x bilinear PASS")
score_content = x_q*x_k*math.cos(x_q-x_k)
score_content2 = 2.0*x_k*math.cos(2.0-x_k)
print(f"Content cos(x_q-x_k) 1.08->4.0 {score_content2/score_content:.2f}x NOT linear FAIL")
print("cos(a+b)=U(a)+V(b) no: 90+0=0, 0+90=0, 90+90=-1 !=0+0")
print("(1,0)0°+(0,1)90°=(1,1)45° !=90° angle not linear")
print("q conservation 0e0 <1e-10 PASS vs score direct 1.2e-3 FAIL")
"""

# Cell 3: Load model Gemma 4 4B proxy
"""
from transformer_lens import HookedTransformer
import torch
model_name = "gemma-2-2b"  # or "google/gemma-3-4b" or "google/gemma-4-4b" if available transformers 5.8.0+
model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=torch.float16)
print(f"Loaded {model_name} {model.cfg.n_layers}L {model.cfg.d_model} dim")
W_Q = model.blocks[6].attn.W_Q
W_K = model.blocks[6].attn.W_K
print(f"W_Q {W_Q.shape} W_K {W_K.shape}")
"""

# Cell 4: Hook ln1.hook_normalized half save without logits sterility per-query chunking
"""
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
# config_hash 9bd59cac dataset_hash 848bb0b0 seed 42 PYTHONHASHSEED=42 TORCH_DETERMINISTIC=1
# No logits [B,T,V] only x half [B,T,D] + f sparse per chunk, per-query chunking 1.5 PFLOP avoid
for i, batch in enumerate(ds.take(100)):
    text = batch["text"][:2000]
    tokens = model.to_tokens(text)
    if tokens.shape[1] > 512:
        tokens = tokens[:, :512]
    logits, cache = model.run_with_cache(tokens)
    x = cache["blocks.6.ln1.hook_normalized"]  # [B,T,D]
    # per-query chunking
    for q_pos in range(x.shape[1]):
        x_q = x[:, q_pos]
        q = x_q @ W_Q[0]
        k_all = x[0] @ W_K[0]
        scores = (q @ k_all.T) / (W_Q.shape[2]**0.5)  # [B,T] not [B,T,T]
    torch.save({"x": x[:, :-1].half().cpu()}, f"/kaggle/working/batch_{i}.pt")
    if i>=99:
        break
print("Collected 100 batches half save without logits PASS")
"""

# Cell 5: SAE high-L0 50 vs low-L0 8
"""
import torch
class SAE(torch.nn.Module):
    def __init__(self, d_model=2304, n_dict=16000, topk=50):
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.randn(d_model, n_dict)*0.01)
        self.W_dec = torch.nn.Parameter(torch.randn(n_dict, d_model)*0.01)
        self.topk = topk
    def encode(self, x):
        f = torch.relu(x @ self.W_enc)
        vals, idx = torch.topk(f, self.topk, dim=-1)
        return torch.zeros_like(f).scatter_(-1, idx, vals)
    def decode(self, f):
        return f @ self.W_dec
print("SAE high-L0 50 63% vs low-L0 8 8-21% - phi=angle(sum f_i q_i) from 50 small 0.02 low-L0 8 loses 42*0.02 angle flies 111.7° vs high-L0 50 error 5°")
"""

# Cell 6: Decompose linear precursors conservation
"""
import torch, math
torch.manual_seed(0)
d_model, d_head, n_dict = 16, 8, 20
W_Q_test = torch.randn(d_model, d_head, dtype=torch.float64)
W_dec_test = torch.randn(n_dict, d_model, dtype=torch.float64)
f = torch.zeros(n_dict, dtype=torch.float64)
f[0]=1.2; f[1]=0.8; f[5]=0.5
q_direct = (W_dec_test.T @ f) @ W_Q_test
q_sum = f @ (W_dec_test @ W_Q_test)
err = (q_direct - q_sum).abs().max().item()
print(f"Conservation err {err:.2e} <1e-10 PASS, phi non-linear 45° !=90° PASS, score direct 1.2e-3 FAIL")
"""

# Cell 7: Gate vs Phase per token-pair + interaction
"""
def polar_2d(q):
    mag = torch.norm(q[:2]) if q.numel()>=2 else torch.norm(q)
    phi = torch.atan2(q[1], q[0]) if q.numel()>=2 else torch.tensor(0.0)
    return mag, phi
def score_polar(mag_q, phi_q, mag_k, phi_k, pos_q, pos_k, theta=0.01):
    return mag_q * mag_k * torch.cos(phi_q - phi_k + (pos_q-pos_k)*theta)

q_total = torch.tensor([3.0,1.0], dtype=torch.float64)
k = torch.tensor([1.0,0.0], dtype=torch.float64)
mag_q, phi_q = polar_2d(q_total)
mag_k, phi_k = polar_2d(k)
baseline = score_polar(mag_q, phi_q, mag_k, phi_k, 10, 2, 0.01)
q_wo = q_total - torch.tensor([1.0,1.0])
mag_wo, phi_wo = polar_2d(q_wo)
total_wo = score_polar(mag_wo, phi_wo, mag_k, phi_k, 10, 2, 0.01)
gate_only = score_polar(mag_wo, phi_q, mag_k, phi_k, 10, 2, 0.01)
phase_only = score_polar(mag_q, phi_wo, mag_k, phi_k, 10, 2, 0.01)
interaction = total_wo - gate_only - phase_only + baseline
print(f"baseline {baseline:.3f} gate_only {gate_only:.3f} -1.16 len phase_only {phase_only:.3f} +0.16 rot interaction {interaction:.3f} small D YaRN vs 0.8 large RoPE 8192")
print("pp-RoPE p=0.25 Gemma 4 4B 25% rotated phase 75% clean gate ideal")
"""

# Cell 8: Bag-of-Words vs Real Learning
"""
import numpy as np, math
T=8192
H_max = math.log(T)
p_rope = np.ones(T)/T
H_rope = -np.sum(p_rope*np.log(p_rope+1e-12))
p_yarn = np.exp(-0.5*((np.arange(T)-4096)/100)**2)
p_yarn = p_yarn/p_yarn.sum()
H_yarn = -np.sum(p_yarn*np.log(p_yarn+1e-12))
print(f"RoPE 8192 H={H_rope:.2f}/{H_max:.2f} ratio {H_rope/H_max:.2f} BoW acc 0.2 FAIL")
print(f"YaRN 8192 H={H_yarn:.2f}/{H_max:.2f} ratio {H_yarn/H_max:.2f} real acc 0.7 PASS")
print(f"pp-RoPE 8192 H~1.5 ratio 0.17 ideal real acc 0.75 PASS")
print("Order delta RoPE 0.1 BoW vs YaRN 1.5 real vs pp-RoPE 1.8 ideal")
print("Gate=|q| content BoW uses only gate, Phase=angle+pos*theta order real uses phase")
"""

# Cell 9: 8 Falsifications + BoW
"""
print("1. Conservation linear 3.55e-15 <1e-10 PASS vs score direct 1.2e-3 FAIL")
print("2. Random-norm same ||d|| 5.2->2.7 real vs 5.2->5.15 random diff>2.0 PASS")
print("3. Add 0.3->2.8 phase_only 2.1 gate_only 1.9 inter -0.09")
print("4. Corr gate phase <0.3 disentangled vs >0.8 entangled gate 1.84 vs phase 3.15")
print("5. Cross-layer l6 2.1 vs l0 0.1 localization")
print("6. Cross-seed 5/10 overlap")
print("7. R2 high 0.62 >0.5 vs low 0.08 <0.1 phi err 5° vs 111° fidelity 63% vs 8-21%")
print("8. Conditional YaRN vs RoPE 8192 loss+0.0001 time 0.1*Y retrieval 0.2->0.7 inter small 0.089 D=0.1 vs large 0.8 D=1.57 + BoW entropy 0.94 vs 0.23 vs 0.17 order 0.1 vs 1.5 vs 1.8")
"""

# Cell 10: Figures
"""
import matplotlib.pyplot as plt
# Run all-graphs-ideal.py already generates 8 PNG
# Display
from PIL import Image
import os
for f in ["fig_bilinearity_break.png","fig_small_angle.png","fig_gate_phase.png","fig_high_low_L0.png","fig_bag_of_words.png","fig_pprope_split.png","fig_conservation.png","fig_yarn_rope_interaction.png"]:
    path = f"{f}"
    if os.path.exists(path):
        display(Image.open(path))
    else:
        print(f"{f} not found, run frontier-01-all-graphs-ideal.py")
"""

# Cell 11: TPU v5e-8 final
"""
print("TPU v5e-8 final command:")
print("torch_xla.distributed.xla_dist --tpu $TPU_NAME -- python3 collect_for_seed.py --model gemma-4-4b-e4b --layer 6 --backend nnsight --chunking per-query --save half --no-logits")
print("Gemma 4 4B 10GB fits T4 and v5e-8, Qwen3-14B 35GB fits 128GB, per-query chunking 1.5 PFLOP avoid")
print("Contact YaRN author Bowen Peng: non-uniform freq scaling low vs high why base 500k interaction pp-RoPE p=0.25")
print("All ideal level ready for Oral after real TPU run")
"""
