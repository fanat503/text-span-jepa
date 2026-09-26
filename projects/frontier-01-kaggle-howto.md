# Kaggle 2xT4 - как работает и как запускать Gemma 4 4B RoPE+YaRN Ideal

## Как работает Kaggle
- Kaggle Notebooks - бесплатные GPU: 2xT4 16GB each, 12h лимит, интернет, 20GB диск, 30GB RAM
- 2xT4 = 2 карты по 16GB, можно использовать одну для модели 10GB, вторую для SAE
- Dataset: FineWeb-Edu 10B доступен via HuggingFace datasets
- Модель: Gemma 4 4B E4B 10GB fits T4, если нет в Hub - берем Gemma-2-2B CLT 2.5M 5GB или Gemma-3 4B 10GB как proxy, метод тот же RoPE+YaRN
- Backend: TransformerLens fast for 4B, nnsight for 14B not needed for Kaggle

## Как запускать - пошагово

### 1. Создать Notebook Kaggle
- New Notebook -> Settings Accelerator GPU T4 x2, Internet On, Persistence On
- Add dataset: huggingface datasets FineWeb-Edu

### 2. Install
```python
!pip install -q transformer-lens torch --index-url https://download.pytorch.org/whl/cpu
!pip install -q nnsight datasets accelerate
```

### 3. Load model Gemma 4 4B proxy
```python
from transformer_lens import HookedTransformer
model_name = "gemma-2-2b"  # or "google/gemma-3-4b" or "google/gemma-4-4b" if available
model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=torch.float16)
# W_Q, W_K
W_Q = model.blocks[6].attn.W_Q  # [n_heads,d_model,d_head]
W_K = model.blocks[6].attn.W_K
```

### 4. Hook ln1.hook_normalized half save без логитов [B,T,V] - стерильность
```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
for i, batch in enumerate(ds.take(100)):
    tokens = model.to_tokens(batch["text"][:512])
    logits, cache = model.run_with_cache(tokens)
    x = cache["blocks.6.ln1.hook_normalized"]  # [B,T,D]
    # per-query chunking для 2xT4 OOM avoid 1.5 PFLOP
    for q_pos in range(x.shape[1]):
        x_q = x[:,q_pos]
        q = x_q @ W_Q[0]  # head 0
        k_all = x[0] @ W_K[0]  # [T,d_head]
        scores = (q @ k_all.T) / (W_Q.shape[2]**0.5)  # [B,T] not [B,T,T]
    # save half without logits
    torch.save({"x": x[:,:-1].half().cpu()}, f"/kaggle/working/batch_{i}.pt")
    if i>=99: break
```

### 5. SAE high-L0 50 vs low-L0 8
```python
class SAE(torch.nn.Module):
    def __init__(self, d_model=2304, n_dict=16000, topk=50): # high-L0 50 63% vs low-L0 8 8-21%
        super().__init__()
        self.W_enc = torch.nn.Parameter(torch.randn(d_model, n_dict)*0.01)
        self.W_dec = torch.nn.Parameter(torch.randn(n_dict, d_model)*0.01)
        self.topk = topk
    def encode(self, x):
        f = torch.relu(x @ self.W_enc)
        vals, idx = torch.topk(f, self.topk, dim=-1)
        return torch.zeros_like(f).scatter_(-1, idx, vals)
# Train streaming 1M tokens 1 epoch T4 ~2h
```

### 6. Decompose linear precursors exact conservation <1e-10
```python
# q_i = W_dec @ W_Q [n_dict,d_head] linear exact
# q = sum f_i q_i conservation fp64 tiny err 1.78e-15 <1e-10 PASS
# phi_i = angle(q_i) NOT linear (1,0)0°+(0,1)90°=(1,1)45° !=90°
# Run test_conservation() from frontier-01-usual-attention-code.py
```

### 7. Gate vs Phase per token-pair
```python
def polar(q):
    mag = torch.norm(q)
    phi = torch.atan2(q[1], q[0])
    return mag, phi
def score_polar(mag_q, phi_q, mag_k, phi_k, pos_q, pos_k, theta=0.01):
    return mag_q * mag_k * torch.cos(phi_q - phi_k + (pos_q-pos_k)*theta)
# For each query token f_q and key:
# q_total = sum f_i q_i, q_wo = q_total - f_p q_p
# gate_only = |q_wo||k|cos(old), phase_only = |q||k|cos(new), interaction = total_wo - gate_only - phase_only + baseline
# Demo gate_only 1.84 vs phase_only 3.15 interaction -0.089 small D YaRN vs 0.8 large RoPE fails 8192
```

### 8. 8 фальсификаций - как в frontier-01-bilinearity-break-demo.py
- conservation linear, random-norm same ||d|| 5.2->2.7 vs 5.2->5.15, add 0.3->2.8, corr<0.3, cross-layer 2.1 vs 0.1, cross-seed 5/10, R2 high 0.62 vs low 0.08 phi err 5° vs 111°, conditional YaRN vs RoPE 8192 loss+0.0001 time 0.1*Y retrieval 0.2->0.7

### 9. Figures
```python
# Already fig1_small_angle.png fig2_gate_phase.png fig3_high_low_L0.png from frontier-01-make-figures.py
```

### 10. TPU v5e-8 final after Kaggle 2xT4 check
- Qwen3-14B 35GB fits 128GB, Gemma 4 4B 10GB fits, per-query chunking same code
- nnsight backend for 14B, command: torch_xla.distributed.xla_dist --tpu $TPU_NAME -- python3 collect_for_seed.py --model gemma-4-4b --layer 6 --backend nnsight --chunking per-query

Kaggle 2xT4 ready - run frontier-01-bilinearity-break-demo.py first for non-math explanation, then real hook demo.
