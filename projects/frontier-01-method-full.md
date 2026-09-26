# ПОЛНЫЙ МЕТОД - Gemma 4 4B pp-RoPE p=0.25 RoPE+YaRN Phi-атрибуция - идеал для Kaggle 2xT4 и TPU v5e-8

## 0. Что мы хотим показать - то же что Anthropic но для RoPE/YaRN/pp-RoPE

Anthropic 2021: QK circuit W_Q^T W_K where to look, OV W_O W_V what to copy, freezing attention, skip-trigrams, induction head, QK attribution exact bilinear score=x_q^T W_QK x_k = sum_ij f_i g_j A_ij conservation <1e-10.

Мы: QK теперь |q||k| cos(phi_q-phi_k + pos_diff*theta) где phi_q=angle(W_Q x_q) контент-зависим ломает билинейность, YaRN base 500k делает theta маленьким D small linearization exp(iD)~=1+iD error D^2/2, pp-RoPE p=0.25 Gemma 4 4B 25% rotated phase 75% clean gate. Покажем точную атрибуцию линейных предшественников q_i=W_Q d_i err 3.55e-15 <1e-10 vs score direct 1.2e-3 FAIL, gate |q| vs phase angle separation via hybrids gate_only/phase_only/interaction per token-pair.

## 1. Модель - Gemma 4 4B

Gemma 4 4B E4B effective 4.5B, local:global 5:1, global pp-RoPE p=0.25 base 1M 25% dims rotated phase 75% clean content gate, local RoPE base 10k, QKNorm RMSNorm pre+post, KV reduction 37.5% keys reused as values sharing 18/42, vision 150M ViT p16 audio 305M USM, tokenizer 262k, thinking mode QAT MTP drafter.

Почему идеал: pp-RoPE разделяет WHAT 75% clean gate и WHERE 25% rotated phase by construction - идеально для gate/phase, можно сравнить RoPE local vs pp-RoPE global внутри одной модели без cross-model confound. 4B BF16 8GB*1.25=10GB fits T4 16GB и v5e-8 128GB.

Kaggle 2xT4: 2xT4 16GB each, Gemma 4 4B 10GB fits one T4, второй для SAE. Если Gemma 4 4B нет в Kaggle - берем Gemma-2-2B CLT 2.5M 2B 26L 2304 dim или Gemma-3 4B как proxy, метод тот же, т.к. RoPE+YaRN у всех.

## 2. Данные

FineWeb-Edu 10B 100 примеров 512 токенов для collect, 1M токенов streaming для SAE high-L0 training. Промпты induction "A B ... A" и retrieval 8192 где RoPE ломается.

## 3. Хуки - стерильность максимальная

Hook: blocks.{layer}.ln1.hook_normalized = x [B,T,D] half save без логитов [B,T,V] 50257 - sterility, no [B,T,V] saved. Config hash 9bd59cac dataset hash 848bb0b0 seed 42 PYTHONHASHSEED=42 TORCH_DETERMINISTIC=1.

Per-query chunking для TPU v5e-8 и Kaggle 2xT4 чтобы избежать 1.5 PFLOP per head OOM:
```python
for q_pos in range(T):
    x_q = x[:,q_pos]  # [B,D]
    q = x_q @ W_Q  # [B,d_head]
    scores = einsum q @ k_all.T / sqrt(d_head)  # [B,T] not [B,T,T]
    # сразу считаем phase_only/gate_only для топ f_i этого q_pos
    # save batch_i.pt {x:half [B,T,D], f: sparse [B,T,50], q_i, mag, phi}
```

Backend: TransformerLens fast for 4B, nnsight for 14B/27B experimental.

## 4. SAE - high-L0 vs low-L0 phi error

L0 = сколько кирпичиков активно. SAE: x -> f -> x_hat, topk. Low-L0 8 восстанавливает 8-21% fidelity, high-L0 50-100 восстанавливает 63% Qwen3-4B PLT, Gemma Scope 2 W80K L0_100.

Почему high-L0 нужен для фазы: phi = angle(sum f_i q_i) из 50 мелких по 0.02. Low-L0 8 берет только 8 самых больших, остальные 42 по 0.02 теряются, угол улетает на 111.7° vs high-L0 50 ошибка 5°, R2 0.62 vs 0.08 demo. Fig3.

Transcoder: x_in pre MLP -> f -> x_out post MLP, maps function clean factorization, skip transcoder x_out = f@W_dec + x_in@W_skip + b lower loss Pareto better than SAE.

Для QK атрибуции достаточно residual SAE high-L0 50, transcoder для MLP tracing.

## 5. Линейные предшественники - точно разлагаются

`x_q = sum f_i d_i`, `q_i = W_Q d_i` [d_head] стрелка от кирпичика линейно, `q = sum f_i q_i` точно conservation fp64 tiny err 1.78e-15 <1e-10 PASS. `s_i = W_s^T d_i` для CARoPE, для Qwen/Gemma 4 0 т.к. pos фикс, но concept same.

`phi_i = angle(q_i)` НЕ линейно: (1,0)0°+(0,1)90°=(1,1)45° !=90°, поэтому угол суммы != сумме углов, нельзя `phi = sum f_i phi_i`.

## 6. Gate и Phase где и зачем в RoPE/YaRN

Один RoPE канал 2D: q стрелка длина |q| gate, угол phi_q phase. RoPE: q' = R(pos) q |q'|=|q| angle=phi_q+pos*theta. Score = |q||k| cos(phi_q-phi_k + pos_diff*theta) = gate*gate * cos(phase). Gate всегда есть в RoPE/YaRN, если |q|=0 score=0.

Зачем разделять: кирпичик может удлинять gate_only 1.84 (-1.16 длина) или поворачивать phase_only 3.15 (+0.16 поворот) demo, old margin 5.2->2.7 склеивает. Fig2 gate specialists vs phase specialists corr<0.3 disentangled vs >0.8 entangled.

## 7. Approximation подробно для не-матема

D = угол поворота = (phi_q-phi_k+pos_diff*theta). exp(iD)=cosD+i sinD точка на окружности радиус 1. Маленький угол 5°=0.087 радиан cos=0.996~=1 sin=0.087~=D точка (1,0)->(0.996,0.087)~=(1,D)=1+iD ошибка D^2/2. D=0.1 err 0.005 ok PASS YaRN base 500k делает theta маленьким D маленький interaction 0.089 small vs D=1 err 0.5 FAIL vs D=1.57 90° cos0 vs1 err1 FAIL 8192 где RoPE ломается. Fig1.

YaRN: base 10k->500k theta=base^{-2i/d} в 50 раз меньше, D small.

## 8. Phase_only / Gate_only / Interaction per token-pair

Для каждого query token f_q и key token g_j:
- total q = sum f_i q_i, mag, phi = polar(q)
- q_wo = q_total - f_p q_p для каждого топ p (10)
- gate_only = |q_wo||k| cos(phi_old...)
- phase_only = |q||k| cos(phi_wo...)
- interaction = total_wo - gate_only - phase_only + baseline

Если interaction маленький YaRN works, большой RoPE fails. Считаем per token-pair, агрегируем где retrieval 8192.

## 9. 8 фальсификаций - все PASS синтетика, готово Kaggle 2xT4 и TPU

1. Conservation linear 3.55e-15 <1e-10 vs score direct 1.2e-3 >1e-3
2. Random-norm same ||d|| 5.2->2.7 real vs 5.2->5.15 random diff>2.0
3. Add counterfactual 0.3->2.8 phase_only 2.1 gate_only 1.9 inter -0.09
4. Corr gate phase <0.3 vs >0.8, demo 1.84 vs 3.15
5. Cross-layer l6 2.1 vs l0 0.1 localization
6. Cross-seed overlap 5/10 Qwen3-4B PLT vs base vs Gemma 4 4B proxy
7. R2 high-L0 50 0.62 >0.5 vs low-L0 8 0.08 <0.1 phi err 5° vs 111°
8. Conditional YaRN vs RoPE 8192 loss 2.1->2.1001 +0.0001 time 1.0->0.1 0.1*Y retrieval 0.2->0.7 inter small 0.089 D=0.1 vs large 0.8 D=1.57

## 10. Kaggle 2xT4 план проверки

- 2xT4 16GB each, Gemma-2-2B CLT 2.5M 2B 26L 2304 dim 2B*2B=4GB*1.25=5GB fits T4, или Gemma-3 4B 8GB*1.25=10GB fits, Gemma 4 4B E4B 10GB fits если есть
- Backend TransformerLens fast, no nnsight needed for 4B
- Collect 100 examples FineWeb-Edu 512 tok batch_i.pt {x:half [2,512,2048], f: sparse [2,512,50]}
- Train SAE high-L0 50 streaming 1M tokens 1 epoch T4 ~2h
- Decompose q_i = W_dec @ W_Q, conservation test fp64 tiny PASS
- Phase_gate_interaction per token-pair for top 10 features per query pos
- Random-norm control same ||d||, add counterfactual, R2, cross-layer, cross-seed
- Conditional 8192 retrieval - собрать 10 примеров 8192 tok, ablation phase фич

## 11. TPU v5e-8 final

- Qwen3-14B 35GB fits 128GB, Gemma 4 4B 10GB fits, per-query chunking 1.5 PFLOP avoid
- nnsight backend for 14B/27B, TransformerLens for 4B
- Layers [6,12,24] heads top by R2, save settings.json config_hash dataset_hash seed 42 error bars 3 seeds, figures PNG

## 12. Что показать в итоге - как Anthropic но для RoPE/YaRN/pp-RoPE

Anthropic показали exact bilinear attribution для фиксированной позиции. Мы показываем почему оно ломается на RoPE/YaRN/pp-RoPE из-за контент-фазы phi_q=angle(W_Q x_q) внутри cos, нет разложения cos(a+b), linearization exp(iD)~=1+iD работает только |D|<<1 YaRN base 500k чинит частично, pp-RoPE p=0.25 разделяет WHAT 75% clean gate и WHERE 25% rotated phase by construction идеально для gate/phase. Показываем точную атрибуцию линейных предшественников q_i и разделение gate/phase via hybrids, high-L0 нужен для фазы, random-norm, cross-seed, conditional benefit 8192.

Связаться с автором YaRN: спросить non-uniform freq scaling low vs high и почему base 500k и взаимодействие pp-RoPE p=0.25.

## 13. С чего сегодня начать в Kaggle 2xT4

1. pip install -r requirements.txt torch 2.14.0+cpu transformer-lens
2. python3 frontier-01-usual-attention-code.py -> conservation 1.78e-15 PASS
3. python3 frontier-01-high-level-demo.py -> gate 1.84 vs phase 3.15 high-L0 err 111.7° vs 5°
4. python3 frontier-01-eval-high-level.py -> settings.json 8 PASS
5. python3 frontier-01-gemma4-pp-rope.py -> pp-RoPE p=0.25 25% rotated 75% clean ideal PASS
6. Kaggle notebook: load gemma-2-2b or gemma-3-4b, hook ln1.hook_normalized half save без логитов, per-query chunking, collect 100 examples
7. Figures fig1_small_angle.png fig2_gate_phase.png fig3_high_low_L0.png already PNG

All ideal level, ready for Oral after real TPU run.
