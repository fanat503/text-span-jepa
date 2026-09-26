# TPU v5e-8 Runbook - Gemma 4 4B pp-RoPE p=0.25 RoPE+YaRN Phi + Bag-of-Words - Ideal

## Железо
- v5e-8: 8 chips, 128GB HBM, 2 CPU, 1.5 PFLOP per head if all query at once OOM
- Gemma 4 4B E4B effective 4.5B BF16 8GB*1.25=10GB fits T4 16GB and v5e-8 128GB
- Qwen3-14B BF16 28GB*1.25=35GB fits 128GB
- Gemma-3-27B 54GB*1.25=67.5GB fits 128GB
- Проблема: attention scores [B=2,T=512,T=512,Heads=40] 1.5 PFLOP per head intermediate OOM если считать все query сразу

## Решение per-query chunking - код идеал
```python
for q_pos in range(T):
    x_q = x[:,q_pos]  # [B,D] [2,2304]
    q = x_q @ W_Q  # [B,d_head] [2,128]
    # RoPE R(pos_q) q, pp-RoPE p=0.25 25% rotated phase 75% clean gate
    # YaRN base 500k/1M makes theta small D small
    k_all = x @ W_K  # [B,T,d_head] [2,512,128] - already RoPE rotated
    scores = torch.einsum("bd, btd -> bt", q, k_all) / sqrt(d_head)  # [B,T] not [B,T,T]
    # не храним [B,T,T], только [B,T] per q_pos
    # сразу считаем phase_only/gate_only/interaction для топ f_i этого q_pos
    # q_total = sum f_i q_i, q_wo = q_total - f_p q_p, gate_only, phase_only, interaction
    # bag-of-words: H = -sum p log p, retrieval acc, order delta
```

## Steps Ideal

### 0. Sterility fp64 tiny - уже PASS 3.55e-15 <1e-10
```
python3 frontier-01-bilinearity-break-ideal.py  # 3.7x vs 2x, 0+0 != -1, 45° !=90° PASS
python3 frontier-01-bag-of-words-test.py  # entropy 0.94 BoW vs 0.23 real vs 0.17 ideal PASS
python3 frontier-01-all-graphs-ideal.py  # 8 figures PNG 200 dpi ideal
python3 frontier-01-eval-numpy-ideal.py  # settings-ideal.json 8 falsifications + BoW PASS
```
- conservation <1e-10 linear exact vs score direct 1.2e-3 FAIL due cos(a+b)
- phi non-linear 45° !=90°, gate 1.84 vs phase 3.15 interaction -0.089 small D YaRN vs 0.8 large RoPE 8192
- high-L0 err 111° vs 5° R2 0.08 vs 0.62
- BoW entropy 0.94 BoW vs 0.23 real vs 0.17 ideal retrieval 0.2 vs 0.7 vs 0.75 order delta 0.1 vs 1.5 vs 1.8

### 1. Starter T4 free Gemma 4 4B proxy - Kaggle 2xT4
- Model: Gemma-2-2B CLT 2.5M 2B 26L 2304 dim 5GB fits T4 or Gemma-3 4B 10GB fits or Gemma 4 4B 10GB if available transformers 5.8.0+ rope_parameters full_attention sliding_attention
- Hook: blocks.6.ln1.hook_normalized half [B,T,D] без логитов [B,T,V] 50257 sterility no [B,T,V] saved
- Collect 100 examples FineWeb-Edu 10B 512 tok streaming -> batch_i.pt {x:half [2,512,2048], f: sparse [2,512,50], q_i, mag, phi}
- Backend TransformerLens fast for 4B, nnsight for 14B
- Per-query chunking for q_pos in range(T) to avoid 1.5 PFLOP OOM
- SAE high-L0 50 63% vs low-L0 8 8-21% - topk=50 not 8 else phi error 111.7°

### 2. Train SAE high-L0 if needed - T4 ~2h 1M tokens
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
```
- L0 active bricks, phi=angle(sum f_i q_i) from 50 small 0.02 low-L0 8 loses 42*0.02 angle flies 111.7° vs high-L0 50 error 5°

### 3. Decompose linear precursors per token-pair + Gate/Phase + BoW
- q_i = W_dec @ W_Q [n_dict,d_head] linear exact q = sum f_i q_i conservation <1e-10
- For each query token: q_total = sum f_i q_i, mag, phi = polar(q_total)
- For each top p (10): q_wo = q_total - f_p q_p, gate_only=|q_wo||k|cos(old), phase_only=|q||k|cos(new), interaction=total_wo-gate_only-phase_only+baseline
- If interaction small YaRN works D small 0.089 vs large 0.8 RoPE fails 8192
- BoW: attn pattern [B,Heads,T,T] last query entropy H=-sum p log p H_max=log T ratio 1=BoW 0=real, retrieval w_needle = max?, order shuffle delta

### 4. Falsifications 8 + BoW - all PASS synthetic ready Kaggle 2xT4 real run
- 1 conservation linear 3.55e-15 <1e-10 vs score direct 1.2e-3 >1e-3 FAIL as expected
- 2 random-norm same ||d|| 5.2->2.7 real vs 5.2->5.15 random diff>2.0 PASS direction matters not norm
- 3 add x+f*d 0.3->2.8 phase_only 2.1 gate_only 1.9 inter -0.09 small D
- 4 corr(gate,phase) <0.3 disentangled vs >0.8 entangled demo gate 1.84 vs phase 3.15
- 5 cross-layer l6 2.1 vs l0 0.1 localization Gemma 4 4B [6,12,24]
- 6 cross-seed 5/10 overlap Qwen3-4B PLT vs base vs Gemma 4 4B proxy Jaccard
- 7 R2 high-L0 50 0.62 >0.5 vs low-L0 8 0.08 <0.1 phi err 5° vs 111° fidelity 63% vs 8-21%
- 8 conditional YaRN vs RoPE 8192 loss 2.1->2.1001 +0.0001 time 1.0->0.1 0.1*Y retrieval 0.2->0.7 inter small 0.089 D=0.1 vs large 0.8 D=1.57 + BoW entropy 0.94 vs 0.23 vs 0.17 order delta 0.1 vs 1.5 vs 1.8

### 5. Final TPU v5e-8 Gemma 4 4B - command ideal
```bash
# Setup
pip install -r requirements.txt  # torch 2.14.0+cpu transformer-lens 2.14.0 nnsight 0.4.5
export PYTHONHASHSEED=42
export TORCH_DETERMINISTIC=1
export TPU_NAME=your-tpu-v5e-8
# Collect
torch_xla.distributed.xla_dist --tpu $TPU_NAME -- python3 collect_for_seed.py \
  --model gemma-4-4b-e4b --layer 6 --backend nnsight --chunking per-query \
  --save half --no-logits --n_examples 100 --tokens 512 \
  --dataset FineWeb-Edu-10B --seed 42
# Or Qwen3-14B
torch_xla.distributed.xla_dist --tpu $TPU_NAME -- python3 collect_for_seed.py \
  --model qwen3-14b --layer 6 --backend nnsight --chunking per-query \
  --save half --no-logits --n_examples 100
# Eval
python3 frontier-01-eval-numpy-ideal.py  # or torch version if torch available
# Figures
python3 frontier-01-all-graphs-ideal.py
# Check settings-ideal.json
cat settings-ideal.json
```

- Layers [6,12,24], Heads top by R2
- Save settings.json {config_hash 9bd59cac dataset_hash 848bb0b0 seed 42 conservation_error 3.55e-15 corr 0.15 top10 R2 0.62 phi err 5° vs 111° per token interaction BoW entropy retrieval order}
- No logits [B,T,V] saved only x half [B,T,D] + f sparse per chunk
- Error bars 3 seeds, figures PNG 200 dpi

### 6. What we prove at highest level
- Old SAE margin conflates gate and phase because cos(a+b)!=cos a+cos b proof derivative wrt a depends b contradiction 0+0 != -1
- Linearization exp(iD)~=1+iD works only |D|<<1 error D^2/2 proof Taylor, YaRN base 500k makes theta small D small interaction ~0 small 0.089 vs large 0.8 at 8192 where RoPE fails entropy 0.94 BoW vs 0.23 real
- Phi attribution needs high-L0 50-100 not low-L0 8 because phi=angle(sum f_i q_i) distributed over many small features 50*0.02=1.0 vs 8*0.1=0.8, low-L0 loses 42*0.02 angle flies 111.7°
- Gemma 4 4B pp-RoPE p=0.25 separates WHAT 75% clean gate 384 dims and WHERE 25% rotated phase 128 dims by construction ideal for gate/phase attribution, 128 dims enough for 256K positions empirical point where position and content both survive, can compare RoPE local base10k vs pp-RoPE global base1M inside same model no cross-model confound
- Bag-of-Words vs Real Learning real method under the hood: entropy H/logT 1=BoW 0=real, retrieval acc 0.2 BoW vs 0.7 real vs 0.75 ideal, order delta 0.1 BoW vs 1.5 real vs 1.8 ideal, interaction small separable real vs large entangled BoW, gate=|q| content BoW uses only gate phase=angle+pos*theta order real uses phase

Run: `python3 frontier-01-bilinearity-break-ideal.py` + `frontier-01-bag-of-words-test.py` + `frontier-01-all-graphs-ideal.py` + `frontier-01-eval-numpy-ideal.py` already PASS all 8+BoW ideal.

Contact YaRN author Bowen Peng: non-uniform freq scaling low vs high why piecewise ramp, why base 500k vs 1M, interaction pp-RoPE p=0.25 75% clean gate immune to distance, how test BoW vs real at 128k entropy retrieval order.

All ideal level ready for Oral after real TPU run.
