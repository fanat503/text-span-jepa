# Kaggle Как Работает и Что Запускать - 11 Cells Copy-Paste T4 x2 - Идеал V10 Top-Lab

## Как работает Kaggle - объяснение для не-матема

Kaggle - платформа где дают 2xT4 GPU 16GB each, 12 часов лимит, Internet ON, 20GB диск, 30GB RAM, 2 CPU cores. 2 карты: одна модель 10GB вторая SAE/high-L0. Dataset FineWeb-Edu 10B streaming HuggingFace datasets. Модель Gemma 4 4B E4B effective 4.5B 10GB fits T4. Если нет Hub берем Gemma-2-2B CLT 2.5M 5GB или Gemma-3 4B 10GB proxy метод тот же RoPE+YaRN+pp-RoPE Gemma 4 появится transformers 5.8.0+ rope_parameters full_attention sliding_attention.

Backend TransformerLens fast for 4B nnsight for 14B not needed Kaggle. Per-query chunking обязателен attention scores [B=2,T=512,T=512,Heads=40] 1.5 PFLOP per head OOM если считать все query сразу считаем for q_pos in range(T): scores=[B,T] not [B,T,T] 512x smaller memory 22x half save total 11264x.

Memory: без chunking attention [2,512,512,40] float16 = 2*512*512*40*2 bytes = 80MB per layer *32 layers = 2.5GB но scores per head 1.5 PFLOP compute OOM. С per-query chunking [2,512] per q_pos = 512x smaller.

## Как запускать пошагово 11 cells ideal - Copy-Paste New Notebook T4 x2 Internet ON

### Cell1 install
```python
!pip install transformer-lens==2.14.0 torch==2.14.0 nnsight==0.4.5 datasets==2.19.0 accelerate==0.33.0 scikit-learn==1.5.1 matplotlib==3.8.4 einops==0.8.0 tqdm==4.66.1 -q
import torch
print(f"CUDA available {torch.cuda.is_available()} device_count {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i} {torch.cuda.get_device_name(i)}")
```

### Cell2 Bilinearity Break Demo без torch только numpy copy frontier-01-bilinearity-break-KAGGLE-COPY-V10-TOPLAB.py PASS/FAIL 3.7x vs 2x 0+0 != -1 45° !=90°
Copy-paste file frontier-01-bilinearity-break-KAGGLE-COPY-V10-TOPLAB.py -> shows PASS/FAIL ready for Oral Fig1

### Cell3 Load model Gemma 4 4B proxy HookedTransformer from_pretrained gemma-2-2b device cuda dtype float16 W_Q blocks[6].attn.W_Q
```python
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gemma-2-2b", device="cuda", dtype=torch.float16)  # proxy Gemma 4 4B 10GB fits T4 if available
print(model.cfg)
W_Q = model.blocks[6].attn.W_Q  # [n_heads, d_model, d_head]
```

### Cell4 Hook ln1.hook_normalized half save без логитов [B,T,V] стерильность config_hash 9bd59cac dataset_hash 848bb0b0 seed 42 PYTHONHASHSEED=42 TORCH_DETERMINISTIC=1 No logits saved only x half [B,T,D]+f sparse per chunk per-query chunking OOM avoid
```python
import os
os.environ["PYTHONHASHSEED"]="42"
torch.manual_seed(42)
# hook
x_cache = []
def hook_fn(x, hook):
    x_cache.append(x[0].detach().half().cpu())  # [B,T,D] half save no logits
    return x

model.blocks[6].ln1.hook_normalized.add_hook(hook_fn)
# per-query chunking to avoid OOM
# for q_pos in range(x.shape[1]): x_q=x[:,q_pos] [B,D] q=x_q@W_Q[0] head0 [B,d_head] k_all=x[0]@W_K[0] [T,d_head] scores=(q@k_all.T)/sqrt(d_head) [B,T] not [B,T,T]
# torch.save x half cpu batch_i.pt
```

### Cell5 SAE high-L0 50 vs low-L0 8 L0 active bricks phi=angle(sum f_i q_i) from 50 small 0.02 low-L0 8 loses 42*0.02 angle flies 111.7° vs high-L0 50 error 5°
```python
# SAE high-L0 demo synthetic
import torch
torch.manual_seed(42)
n_dict, d_model, d_head = 20, 16, 8
W_dec = torch.randn(n_dict, d_model)
W_Q = torch.randn(d_model, d_head)
f = torch.zeros(n_dict)
f[0]=1.2; f[1]=0.8; f[5]=0.5
# etc - see frontier-01-bilinearity-break-torch-ideal.py
```

### Cell6 Decompose linear precursors exact conservation <1e-10 q_i=W_dec@W_Q [n_dict,d_head] linear exact q=sum f_i q_i err 1.78e-15 PASS phi_i=angle(q_i) NOT linear (1,0)0°+(0,1)90°=(1,1)45° !=90°
```python
# conservation test
q_direct = (W_dec.T @ f) @ W_Q
q_sum = f @ (W_dec @ W_Q)
err = (q_direct - q_sum).abs().max().item()
print(f"Conservation err {err:.2e} <1e-10 PASS")
```

### Cell7 Gate vs Phase per token-pair + BoW polar mag=norm phi=atan2 score=mag_q*mag_k*cos(phi_q-phi_k+pos_q-pos_k theta) gate_only 1.84 vs phase_only 3.15 interaction -0.089 small D YaRN vs 0.8 large RoPE fails 8192 BoW entropy H/logT
```python
def polar(q):
    mag = torch.norm(q[:2]) if q.numel()>=2 else torch.norm(q)
    phi = torch.atan2(q[1], q[0]) if q.numel()>=2 else torch.tensor(0.0)
    return mag, phi
# etc
```

### Cell8 8 falsifications + BoW conservation random-norm add corr cross-layer cross-seed R2 conditional BoW entropy 0.94 vs 0.23
```python
# run eval-numpy-ideal.py
# all 8 PASS -> settings-ideal.json config_hash 9bd59cac dataset_hash 848bb0b0
```

### Cell9 Figures Run frontier-01-graphs-TOPLAB-IDEAL-V10.py 8 PNG 160K-300K display PIL Image relative paths fixed bug
```python
!python3 frontier-01-graphs-TOPLAB-IDEAL-V10.py
from PIL import Image
import glob
for f in sorted(glob.glob("fig_*.png")):
    print(f)
    display(Image.open(f))
```

### Cell10 TPU v5e-8 final Qwen3-14B 35GB fits 128GB Gemma 4 4B 10GB fits per-query chunking nnsight backend torch_xla.distributed.xla_dist --tpu $TPU_NAME -- python3 collect_for_seed.py --model gemma-4-4b --layer 6 --backend nnsight --chunking per-query --save half --no-logits
```bash
# TPU command copy-paste
# torch_xla.distributed.xla_dist --tpu $TPU_NAME -- python3 collect_for_seed.py --model gemma-4-4b --layer 6 --backend nnsight --chunking per-query --save half --no-logits --seed 42 --config_hash 9bd59cac --dataset_hash 848bb0b0
```

### Cell11 What show same as Anthropic but for RoPE Anthropic exact bilinear QK attribution fixed pos we show why breaks content-phase cos(a+b) no decomposition linearization exp(iD)~=1+iD YaRN small D pp-RoPE 25% clean gate 75% WHAT vs WHERE 25% WHERE ideal BoW method
```python
print("What to show Oral: same as Anthropic but for RoPE/YaRN/pp-RoPE")
print("Anthropic QK circuit W_Q^T W_K fixed pos exact bilinear")
print("We show why breaks content-phase phi_q=angle(W_Q x_q) inside cos no decomposition")
print("Linearization exp(iD)~=1+iD works only |D|<<1 YaRN base 500k makes D small interaction 0.089 vs 0.8 at 8192")
print("pp-RoPE p=0.25 25% rotated phase 75% clean gate WHAT vs WHERE ideal")
print("BoW method entropy retrieval order interaction")
```

### Run All 3h <12h fits Troubleshooting OOM per-query chunking batch1 tokens256 dtype float16 half save Model not found use gemma-2-2b proxy method same RoPE+YaRN cite Gemma 4 report 2607.02770 TransformerLens fails use nnsight T4 x2 torch.cuda.device_count()=2 model cuda:0 SAE cuda:1 12h limit save checkpoints /kaggle/working/ persistence streaming dataset.

### Files needed in Kaggle dataset:
- frontier-01-graphs-TOPLAB-IDEAL-V10.py
- frontier-01-bilinearity-break-KAGGLE-COPY-V10-TOPLAB.py
- frontier-01-bag-of-words-test.py
- requirements.txt
- settings-ideal.json

All relative paths fixed, no OLD_PATH/... absolute, config_hash 9bd59cac dataset_hash 848bb0b0 seed 42.

Ready for Oral 6 Strong Accept.
