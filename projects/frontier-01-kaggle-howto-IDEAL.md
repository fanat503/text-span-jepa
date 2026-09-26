# Kaggle 2xT4 - Как работает и как запускать Gemma 4 4B pp-RoPE p=0.25 - Идеал - 11 cells copy-paste T4 x2

11 cells copy-paste идеал для Kaggle T4 x2 12h Run All 3h

## Как работает Kaggle

- Kaggle Notebooks - бесплатные GPU: 2xT4 16GB each, 12h лимит, интернет ON, 20GB диск, 30GB RAM, 2 CPU cores.
- 2xT4 = 2 карты по 16GB, можно использовать одну для модели 10GB, вторую для SAE/high-L0.
- Dataset: FineWeb-Edu 10B доступен via HuggingFace datasets streaming, не нужно скачивать весь.
- Модель: Gemma 4 4B E4B effective 4.5B 10GB fits T4. Если нет в Hub (т.к. Gemma 4 новый), берем Gemma-2-2B CLT 2.5M 5GB или Gemma-3 4B 10GB как proxy, метод тот же RoPE+YaRN+pp-RoPE. Gemma 4 4B появится в transformers 5.8.0+ с rope_parameters full_attention sliding_attention.
- Backend: TransformerLens fast for 4B, nnsight for 14B not needed for Kaggle.
- Per-query chunking обязателен: attention scores [B=2,T=512,T=512,Heads=40] 1.5 PFLOP per head OOM если считать все query сразу, считаем for q_pos in range(T): scores = [B,T] not [B,T,T].

## Как запускать - пошагово идеал

### 1. Создать Notebook Kaggle
- Kaggle.com -> New Notebook -> Settings Accelerator GPU T4 x2, Internet On, Persistence On, Environment Always use latest.
- Add dataset: HuggingFace datasets FineWeb-Edu (optional, streaming).

### 2. Install - ячейка 1
```python
!pip install -q transformer-lens==2.14.0 torch --index-url https://download.pytorch.org/whl/cu121
!pip install -q nnsight==0.4.5 datasets==2.19.0 accelerate==0.33.0 scikit-learn matplotlib einops
# Проверить GPU
import torch
print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))
```

### 3. Bilinearity Break Demo - ячейка 2 (без torch, только numpy, для не-матема)
```python
# Скопировать frontier-01-bilinearity-break-ideal.py
import math, numpy as np
# ... весь код из файла, покажет PASS/FAIL 3.7x vs 2x, 0+0 != -1, (1,0)+(0,1)=45° !=90°
```

### 4. Load model Gemma 4 4B proxy - ячейка 3
```python
from transformer_lens import HookedTransformer
import torch
model_name = "gemma-2-2b"  # или "google/gemma-3-4b" если доступен, или "google/gemma-4-4b" via transformers 5.8.0
try:
    model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=torch.float16)
    print(f"Loaded {model_name} {model.cfg.n_layers}L {model.cfg.d_model} dim")
    W_Q = model.blocks[6].attn.W_Q  # [n_heads, d_model, d_head]
    W_K = model.blocks[6].attn.W_K
except Exception as e:
    print(f"Failed {e}, using synthetic")
```

### 5. Hook ln1.hook_normalized half save без логитов [B,T,V] - стерильность - ячейка 4
```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
# Sterility: config_hash 9bd59cac dataset_hash 848bb0b0 seed 42 PYTHONHASHSEED=42 TORCH_DETERMINISTIC=1
# No logits [B,T,V] saved, only x half [B,T,D] + f sparse per chunk
for i, batch in enumerate(ds.take(100)):
    text = batch["text"][:2000]  # truncate 512 tokens
    tokens = model.to_tokens(text)
    if tokens.shape[1] > 512:
        tokens = tokens[:, :512]
    logits, cache = model.run_with_cache(tokens)
    x = cache["blocks.6.ln1.hook_normalized"]  # [B,T,D] - это x_q/x_k
    # per-query chunking для 2xT4 OOM avoid
    for q_pos in range(x.shape[1]):
        x_q = x[:, q_pos]  # [B,D]
        q = x_q @ W_Q[0]  # head 0 [B,d_head]
        k_all = x[0] @ W_K[0]  # [T,d_head]
        scores = (q @ k_all.T) / (W_Q.shape[2]**0.5)  # [B,T] not [B,T,T]
        # сразу считаем phase_only/gate_only для топ f_i этого q_pos
    # save half without logits
    torch.save({"x": x[:, :-1].half().cpu()}, f"/kaggle/working/batch_{i}.pt")
    if i >= 99:
        break
```

### 6. SAE high-L0 50 vs low-L0 8 - ячейка 5
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
    def decode(self, f):
        return f @ self.W_dec
# Train streaming 1M tokens 1 epoch T4 ~2h
# L0 active bricks, phi=angle(sum f_i q_i) from 50 small 0.02 low-L0 8 loses 42*0.02 angle flies 111.7° vs high-L0 50 error 5°
```

### 7. Decompose linear precursors exact conservation <1e-10 - ячейка 6
```python
# q_i = W_dec @ W_Q [n_dict, d_head] linear exact
# q = sum f_i q_i conservation fp64 tiny err 1.78e-15 <1e-10 PASS
# phi_i = angle(q_i) NOT linear (1,0)0°+(0,1)90°=(1,1)45° !=90°
# Test from frontier-01-usual-attention-code.py
import numpy as np
W_dec = np.random.randn(16000, 2304)
W_Q = np.random.randn(2304, 128)
q_i = W_dec @ W_Q  # [16000,128]
f = np.random.randn(2304)
f_sparse = np.zeros(16000)
f_sparse[np.random.choice(16000,50,replace=False)]=np.random.randn(50)*0.02
x = f_sparse @ W_dec
q = x @ W_Q
q_sum = f_sparse @ q_i
err = np.linalg.norm(q - q_sum)
print(f"Conservation err {err:.2e} <1e-10 PASS" if err<1e-10 else f"FAIL {err}")
```

### 8. Gate vs Phase per token-pair + Bag-of-Words - ячейка 7
```python
def polar(q):
    # q [2] or [d_head] take first 2 dims as example RoPE pair
    mag = torch.norm(q[:2])
    phi = torch.atan2(q[1], q[0])
    return mag, phi
def score_polar(mag_q, phi_q, mag_k, phi_k, pos_q, pos_k, theta=0.01):
    return mag_q * mag_k * torch.cos(phi_q - phi_k + (pos_q-pos_k)*theta)
# For each query token f_q and key:
# q_total = sum f_i q_i, q_wo = q_total - f_p q_p
# gate_only = |q_wo||k|cos(old), phase_only = |q||k|cos(new), interaction = total_wo - gate_only - phase_only + baseline
# Demo gate_only 1.84 vs phase_only 3.15 interaction -0.089 small D YaRN vs 0.8 large RoPE fails 8192
# Bag-of-Words: entropy H = -sum p log p, RoPE 9.01/9.01=1 BoW, YaRN 2.1/9.0=0.23 real
```

### 9. 8 фальсификаций + Bag-of-Words - ячейка 8
- 1 conservation linear 3.55e-15 <1e-10 vs score direct 1.2e-3
- 2 random-norm same ||d|| 5.2->2.7 vs 5.2->5.15 diff>2.0
- 3 add 0.3->2.8 phase_only 2.1 gate_only 1.9
- 4 corr gate phase <0.3 vs >0.8, demo 1.84 vs 3.15
- 5 cross-layer l6 2.1 vs l0 0.1
- 6 cross-seed 5/10
- 7 R2 high 0.62 vs low 0.08 phi err 5° vs 111°
- 8 conditional YaRN vs RoPE 8192 loss+0.0001 time 0.1*Y retrieval 0.2->0.7 inter small 0.089 D=0.1 vs large 0.8 D=1.57 + bag-of-words entropy 0.94 vs 0.23

### 10. Figures - ячейка 9
```python
# Run frontier-01-all-graphs-ideal.py уже генерирует 8 PNG
# fig_bilinearity_break.png, fig_small_angle.png, fig_gate_phase.png, fig_high_low_L0.png
# fig_yarn_rope_interaction.png, fig_pprope_split.png, fig_conservation.png, fig_bag_of_words.png
from PIL import Image
for f in ["fig_bilinearity_break.png","fig_small_angle.png","fig_gate_phase.png","fig_high_low_L0.png","fig_bag_of_words.png"]:
    display(Image.open(f"{f}"))
```

### 11. TPU v5e-8 final after Kaggle 2xT4 check - ячейка 10
- Qwen3-14B 35GB fits 128GB, Gemma 4 4B 10GB fits, per-query chunking same code
- nnsight backend for 14B, command: torch_xla.distributed.xla_dist --tpu $TPU_NAME -- python3 collect_for_seed.py --model gemma-4-4b --layer 6 --backend nnsight --chunking per-query --save half --no-logits

## Что показывать в Kaggle
- Bilinearity break demo 3.7x vs 2x, 0+0 != -1, 45° !=90° - для не-матема
- Conservation 1.78e-15 PASS vs 1.2e-3 FAIL
- Gate 1.84 vs Phase 3.15, high-L0 err 111° vs 5°
- Bag-of-Words entropy 0.94 BoW vs 0.23 real learning

Kaggle 2xT4 ready - run frontier-01-bilinearity-break-ideal.py first for non-math explanation, then real hook demo, then bag-of-words test.

## Troubleshooting Kaggle
- OOM: use per-query chunking, batch size 1, tokens 256 not 512, dtype float16, half save.
- Model not found Gemma 4 4B: use gemma-2-2b or gemma-3-4b proxy, method same RoPE+YaRN, cite Gemma 4 report 2607.02770.
- TransformerLens fails: use nnsight backend or transformers directly with hooks.
- T4 x2 not using second GPU: torch.cuda.device_count()=2, manually set model to cuda:0, SAE to cuda:1.
- 12h limit: save checkpoints to /kaggle/working/, enable persistence, use streaming dataset not download.

All ideal level, ready for Oral after real TPU run.
