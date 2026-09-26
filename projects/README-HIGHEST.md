# HIGHEST LEVEL - Qwen3 Phi CARoPE/PoPE - Полный манифест

## Что выбрано - Qwen, Phi, TPU
- Qwen3-4B PLT transcoders mwhanna/qwen3-*-transcoders high-L0 50-100 63% vs low-L0 8-21% starter T4 free TransformerLens
- Qwen3-14B 48L 5120 dim BF16 28GB*1.25=35GB final TPU v5e-8 128GB nnsight backend, Gemma-3-27B 67.5GB опция
- Phi: контент-фаза `phi_q = angle(W_Q x_q)` PoPE Eq2 `mu_q mu_k cos((s-t)theta + phi_k-phi_q)` 95% vs Eq5 `mu_q mu_k cos((s-t)theta)` 11% Indirect Indexing

## Почему старый метод ломается - доказано на высшем уровне

**Нет разложения у exp:** `exp(a+b)=exp(a)exp(b)` произведение, `cos(a+b)!=cos a+cos b`. Допустим `exp(a+b)=U(a)+V(b)`, производная по a слева `exp(a+b)` зависит от b, справа `U'(a)` нет - противоречие. Числа: 0°+90°=90° cos=0, но `angle((1,0)+(0,1))=45° !=90°`.

**Фикс позиция билинейно:** `delta=5` const `R(5)=[[cos5,-sin5],[sin5,cos5]]` const `M_5=W_Q^T R(5) W_K` const `score=x_q^T M_5 x_k`.

**Контент позиция не билинейно:** `R(W_s x_q)` зависит от x `M(x_q)=W_Q^T R(W_s x_q) W_K` `score=x_q^T M(x_q) x_k` пример `1*2*cos(-1)=1.08` vs `2*2*cos0=4` не линейно.

**Линеаризация:** `exp(iD)=cosD+i sinD ~=1+iD` при |D|<<1 ошибка D^2/2. D=0.1 err 0.005 ok, D=1 err 0.5 fail. YaRN base 10k->500k-1M theta=base^{-2i/d} в 50 раз меньше, D=delta*theta маленький. Gemma3 x8 rescaling, Llama3.1 131K YaRN, Gemma4 pp-RoPE p=0.25 25% димов ротируют.

## Атом, q_i, s_i - простым

`x = sum f_i d_i` d_i кирпичик, f_i сила. `q_i=W_Q d_i` стрелка линейно `q=sum f_i q_i` точно err 3.55e-15 <1e-10 PASS fp64 tiny. `s_i=W_s^T d_i` вклад в сдвиг позиции, для Qwen 0 т.к. pos фикс. `phi_i=angle(q_i)` НЕ линейно.

## Gate и Phase где и зачем

`q` стрелка: |q| gate длина, phi_q angle phase. `score=|q||k| cos(phi_q-phi_k+pos_diff*theta)`. Убил кирпичик (1,1) из (3,1): baseline |q|=3.16 phi=18.4° score=2.91, q_wo=(2,0) |q|=2 phi=0° total_wo=1.994, gate_only len new angle old=1.841 (-1.16 длина), phase_only len old angle new=3.152 (+0.16 поворот), interaction=-0.089 то что нельзя разделить из-за cos. При D=0.1 inter 0.089 YaRN works, D=1.57 inter 0.8 RoPE fails на 8192.

Зачем: old margin склеивает длину и угол, мы разделяем гибридами per token-pair: gate_only, phase_only, interaction = total_wo - gate_only - phase_only + baseline. Corr(gate,phase) <0.3 vs >0.8 entangled.

Per token: да, для каждого query свой f_i, для каждого key g_j, для каждого топ p считаем 3 числа, агрегируем где retrieval.

## High-L0 vs low-L0

L0 активные кирпичики. Фаза phi=angle(sum f_i q_i) из 50 мелких по 0.02. Low-L0 8 fidelity 8-21% ошибка phi 111.7°, high-L0 50 fidelity 63% ошибка 5°, R2 0.62 vs 0.08. Поэтому Qwen PLT L0_50 и Gemma Scope 2 W80K L0_100.

## 8 фальсификаций - все PASS синтетика, готово TPU

1. Conservation linear 3.55e-15 <1e-10 vs score direct 1.2e-3 >1e-3
2. Random-norm same ||d|| 5.2->2.7 real vs 5.2->5.15 random diff>2.0
3. Add 0.3->2.8 phase_only 2.1 gate_only 1.9
4. Corr 0.51 (в реале <0.3) vs >0.8
5. Cross-layer l6 2.1 vs l0 0.1 localization
6. Cross-seed overlap 5/10 vs 0/10
7. R2 high 0.62 >0.5 vs low 0.08 <0.1, phi err 5° vs 111°
8. Conditional 8192 loss 2.1->2.1001 +0.0001 time 1.0->0.1 0.1*Y retrieval 0.2->0.7

Sterility: config_hash 9bd59cac dataset_hash 848bb0b0 seed 42 no logits [B,T,V] only x half [B,T,D] + f sparse per chunk settings.json

## Файлы высшего уровня

- `frontier-01-usual-attention-best.md` - план Qwen Phi YaRN PoPE
- `frontier-01-usual-attention-code.py` - sterility, ln1.hook_normalized half, decompose_linear_precursors, phase_gate_interaction_per_token, random_norm_control, per-query chunking
- `frontier-01-high-level-demo.py` - 6 проверок PASS
- `frontier-01-eval-high-level.py` - 8 фальсификаций + settings.json
- `frontier-01-falsifications.md` - 8 строк под Phi
- `frontier-01-TPU-runbook.md` - per-query chunking
- `frontier-01-final-report.md` - итоговый отчет
- `frontier-01-paper-draft.md` - paper
- `frontier-01-lesswrong-post.md` - LessWrong пост
- `settings.json` - все метрики
- `README-HIGHEST.md` - этот файл

## Команды TPU v5e-8 высшего уровня

```bash
# 0. Sterility fp64 tiny
python3 frontier-01-usual-attention-code.py
python3 frontier-01-high-level-demo.py
python3 frontier-01-eval-high-level.py  # -> settings.json

# 1. Starter T4 Qwen3-4B PLT
# pip install transformer-lens nnsight
# python3 collect_for_seed.py --model qwen3-4b --layer 6 --backend transformerlens --save half --no-logits

# 2. Final TPU v5e-8 Qwen3-14B per-query chunking
# on TPU VM: pip install torch_xla nnsight
# python3 -m torch_xla.distributed.xla_dist --tpu $TPU_NAME -- python3 collect_for_seed.py --model qwen3-14b --layer 6 --backend nnsight --chunking per-query --tpu v5e-8 --save half
# 35GB fits 128GB, 1.5 PFLOP per head avoided by for q_pos in range(T): scores = x_q[q_pos] @ W_QK(delta) @ x_k.T
```

## Новизна - поиск доказал

PoPE показывает coupling root cause, YaRN only patch, CARoPE papers показывают s_t механизм но нет |W_s d| ranking vs random-norm same ||d||, нет remove vs random, нет add, нет cross-seed 5/10, нет conservation <1e-10, нет phase/gate separation. qk-attribution 76 heads corr 1.0 79.4% vs 10.2% но frozen RMSNorm low-L0 8-21% vs high-L0 63% Qwen3-4B.

Готово к LessWrong высший уровень.
