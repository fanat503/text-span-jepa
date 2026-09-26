# TPU v5e-8 Runbook - Qwen3 Phi на высшем уровне

## Железо
- v5e-8: 8 chips, 128GB HBM
- Qwen3-14B BF16 14B*2B=28GB *1.25 overhead=35GB fits
- Gemma-3-27B 54GB*1.25=67.5GB fits
- Проблема: attention scores [B=2,T=512,T=512,Heads=40] 1.5 PFLOP per head intermediate OOM если считать все query сразу

## Решение per-query chunking
```python
for q_pos in range(T):
    x_q = x[:,q_pos]  # [B,D]
    q = x_q @ W_Q  # [B,d_head]
    # RoPE R(pos_q) q
    scores = torch.einsum("bd, btd -> bt", q, k_rope) / sqrt(d_head)  # [B,T]
    # не храним [B,T,T], только [B,T] per q_pos
    # сразу считаем phase_only/gate_only для топ f_i этого q_pos
```

## Steps

### 0. Sterility fp64 tiny (уже PASS 1.78e-15)
```
python3 frontier-01-usual-attention-code.py
python3 frontier-01-high-level-demo.py
```
- conservation <1e-10, phi non-linear 45° !=90°, gate 1.84 vs phase 3.15, interaction -0.089 при D=0.1, high-L0 err 111°

### 1. Starter T4 free Qwen3-4B PLT
- Model: Qwen/Qwen3-4B + mwhanna/qwen3-4b-transcoders L0_50 63%
- Hook: blocks.6.ln1.hook_normalized half, без логитов [B,T,V]
- Collect 100 examples FineWeb-Edu 512 tok -> batch_i.pt {x:half [2,512,2048], f: sparse [2,512,50]}
- Backend TransformerLens fast

### 2. Train SAE high-L0 if needed
- topk=50 not 8, fidelity 63% vs 8-21%, иначе phi ошибка 111° как в демо

### 3. Decompose linear precursors per token-pair
- q_i = W_dec @ W_Q [n_dict,d_head] - линейно точно
- Для каждого query token: q_total = sum f_i q_i, mag, phi = polar(q_total)
- Для каждого топ p (10): q_wo = q_total - f_p q_p, gate_only, phase_only, interaction

### 4. Falsifications 8 штук
- conservation linear <1e-10 vs score >1e-3
- random-norm same ||d|| 5.2->2.7 vs 5.2->5.15
- add x+f*d 0.3->2.8
- corr(gate,phase) <0.3 vs >0.8
- cross-layer 6 vs 0: 2.1 vs 0.1
- cross-seed 5/10 overlap Qwen3-4B vs base
- R2>0.5 vs <0.1, high-L0 63% vs low-L0
- conditional benefit 8192 retrieval 0.2->0.7 loss+0.0001 time 0.1*Y, interaction большой при D=1.57, маленький при D=0.1

### 5. Final TPU v5e-8 Qwen3-14B
- nnsight backend, same hooks but per-query chunking
- Layers [6,12,24], Heads top by R2
- Save settings.json {config_hash, dataset_hash, git_commit, conservation_error, corr, top10, R2, per_token interaction}
- No logits saved

### 6. What we prove at highest level
- Old SAE margin conflates gate and phase, because cos(a+b) != cos a + cos b, exp(a+b)=exp(a)exp(b) multiplicative
- Linearization exp(iD)~=1+iD works only |D|<<1, YaRN base 500k makes theta small, D small, interaction ~0, but at 8192 D~1.57 interaction big -> RoPE fails
- Phi attribution needs high-L0 50-100, not low-L0 8, because phi = angle(sum f_i q_i) distributed over many small features
- Qwen3 Phi is same problem as CARoPE s_i = W_s^T d_i, both content-dependent phase, PoPE fixes by removing phi_k-phi_q

Run: `python3 frontier-01-high-level-demo.py` already PASS all 6 checks.
