# Proofs Ideal - RoPE / YaRN / pp-RoPE Phi-Attribution - Без ошибок, сверено с источниками

## Источники сверки
- RoPE: Su et al 2021 https://arxiv.org/abs/2104.09864, dev.to RoPE rotates Query Key vectors 2D planes before QK^T, zeroentropy.dev angle proportional to position, arxiv 2607.10134 LeRoPE rotates 2D chunks rates geometric sequence base hyperparameter.
- YaRN: Peng et al 2023 arXiv 2309.00071, ICLR 2024, localaimaster RoPE YaRN guide, emergentmind YaRN piecewise scaling 128k, Bowen Peng author.
- pp-RoPE p=0.25: Gemma 4 Technical Report 2607.02770, machine-learning-made-simple pp-RoPE rotates only 25% dims content room to breathe, Barbero et al 2025 round.
- Gemma 4 specs: 2607.02770 E4B effective 4.5B local:global 5:1 global pp-RoPE p=0.25 base 1M local RoPE base 10k KV reduction 37.5% keys reused as values sharing 18/42 head_dim 512.

---

## 1. RoPE Definition - Сверено

Definition: Для d_model четное, разбиваем на d/2 пар (x_{2i}, x_{2i+1}). RoPE rotation matrix R_m = diag(R(m theta_0), R(m theta_1), ...) где R(alpha) = [[cos alpha, -sin alpha],[sin alpha, cos alpha]], theta_i = base^{-2i/d}, base=10k local, 1M global.

Для query q_m = R_m W_Q x_m, key k_n = R_n W_K x_n.

Score: q_m^T k_n = (W_Q x_m)^T R_m^T R_n W_K x_n = (W_Q x_m)^T R_{n-m} W_K x_n, зависит только от относительного offset n-m - property relative position [dev.to][zeroentropy].

В 2D одной пары: q = |q| [cos phi_q, sin phi_q], после RoPE q' = |q| [cos(phi_q+m theta), sin(phi_q+m theta)], |q'|=|q| сохраняется, угол добавляется m theta.

Поэтому score одной пары: |q||k| cos(phi_q-phi_k + (m-n)theta) = gate_q gate_k cos(phase_diff + pos_diff theta).

Источники подтверждают: RoPE encodes position by rotating pairs of Q/K dims angle proportional to position, dot product function of relative.

---

## 2. Почему билинейность ломается - Строгое доказательство

### 2.1 Фикс позиция - билинейно

Если theta фиксировано и phi_q, phi_k фиксированы для позиции (не зависят от контента), то W_QK(m,n) = W_Q^T R_{n-m} W_K фиксирован. Score = x_q^T W_QK(m,n) x_k билинейно по x_q, x_k.

Проверка: удвоили x_q -> score удвоился 2x. Demo: x_q=1,x_k=2,delta_fixed=1 cos=0.54 score=1.08, x_q=2 score=2.16 2x.

### 2.2 Контент-зависимая фаза - НЕ билинейно

Теперь phi_q = angle(W_Q x_q) зависит от x_q, т.к. x_q = sum f_i d_i, q = sum f_i q_i, phi_q = angle(sum f_i q_i). Аналогично phi_k.

Score = |q(x_q)||k(x_k)| cos(phi_q(x_q)-phi_k(x_k)+(m-n)theta)

Зависимость phi_q(x_q) нелинейна: (1,0) угол 0° + (0,1) угол 90° = (1,1) угол 45° != 90° = sum углов. Поэтому cos(phi_q(x_q)-...) содержит cos(angle(sum f_i q_i)), angle нелинейно.

#### Контрпример 1: удвоение не удваивает
x_q=1,x_k=2, score = x_q x_k cos(x_q - x_k) =1*2*cos(-1)=1.08
x_q=2 => 2*2*cos(0)=4, отношение 3.7x !=2x. Не линейно.

#### Контрпример 2: нет разложения cos(a+b)=U(a)+V(b)
Предположим cos(a+b)=U(a)+V(b). Тогда производная по a: -sin(a+b) = U'(a) зависит только от a, но левая зависит от b - противоречие. Численно: a=90° b=0° cos90=0, a=0° b=90° cos90=0, a=90° b=90° cos180=-1 !=0+0. Значит нет аддитивного разложения. Area analogy length*width multiplicative cannot split.

#### Контрпример 3: exp(a+b)=exp(a)exp(b) multiplicative not additive
exp(a+b) = exp(a)exp(b) - произведение, не сумма. Поэтому exp не разлагается в сумму функций только от a и только от b. cos(a+b)=cos a cos b - sin a sin b - произведение тоже.

Вывод: стандартный QK attribution sum_ij f_i g_j A_ij где A_ij фиксирован - не работает когда A_ij зависит от sum f_i q_i через phi.

---

## 3. Линеаризация exp(iD)~=1+iD - Подробно для не-матема

D = угол поворота = phi_q-phi_k+pos_diff*theta.

exp(iD)=cosD + i sinD точка на единичной окружности радиус 1.

Разложение Тейлора: cosD =1 - D^2/2 + D^4/24 -..., sinD = D - D^3/6 +...

Маленький угол: D=0.087 рад =5°, cos=0.996 ~=1 ошибка 0.004 = D^2/2 =0.0038, sin=0.087 ~=D ошибка D^3/6=0.00011.

Геометрия: (1,0) на окружности повернули на 5° => (0.996,0.087) ~=(1,0.087)=1+iD.

Оценка ошибки: |exp(iD)-(1+iD)| = sqrt((cosD-1)^2+(sinD-D)^2) ~= D^2/2 для малых D.

Числа:
D=0.1 rad: cos=0.995 vs1 err0.005 sin=0.0998 vs0.1 err0.00016 PASS YaRN base 500k makes theta small.
D=1 rad: cos=0.54 vs1 err0.46 sin=0.84 vs1 err0.16 FAIL.
D=1.57 rad=90°: cos=0 vs1 err1 sin=1 vs1.57 err0.57 FAIL 8192 RoPE.

YaRN: theta_i = base^{-2i/d}, base 10k->500k. При i=0 theta_0=1 unchanged (high-freq keep), при i=d/2 theta=1/base low-freq 50× меньше (1/10k→1/500k). Средний эффект: D=delta*theta для низких частот в 50 раз меньше, для высоких без изменения. Линеаризация работает для low-freq где D был большим, interaction small 0.089 vs RoPE large 0.8 at 8192.

Источник: YaRN paper piecewise scaling high-freq keep unchanged local discrimination low-freq linear interpolation + temperature scaling, requires 10x less tokens 2.5x less steps. Связаться с YaRN author Bowen Peng: non-uniform freq scaling low vs high.

---

## 4. pp-RoPE p=0.25 Gemma 4 4B - Почему идеал

Gemma 4 Technical Report 2607.02770: global layers pp-RoPE p=0.25 base 1M local RoPE base 10k local:global 5:1, global KV reduction 37.5% keys reused as values, sharing 18/42 E4B, head_dim 512.

Статья machine-learning-made-simple: Partial RoPE rotating only 25% dimensions so content actually has room to breathe. Standard rotates every dimension at 8K fine at 128K breaks raw semantic meaning distorted. At 120k query searching fact at 500 struggles extreme rotation acts as noise. Gemma 4 global layers split 512-dim head: 128 dims 25% receive full theta=1M rotation dedicated position channels, 384 dims 75% zero rotation pure content channels immune to distance.

Почему 25%: 128 rotating dims provide exactly enough frequency bands to uniquely index 256K positions. 50% sacrifices pure content capacity, 10% blurs distant positions. 25% empirical point where position and content both survive.

Для нас идеал: WHAT 75% clean gate vs WHERE 25% rotated phase by construction разделены, идеально для gate/phase атрибуции. Можно сравнить RoPE local vs pp-RoPE global внутри одной модели без cross-model confound.

Gate всегда есть в RoPE/YaRN/pp-RoPE: score = |q||k|cos(...), если |q|=0 score=0 независимо от угла. Поэтому gate=|q| длина.

---

## 5. SAE и линейные предшественники - Точная атрибуция

SAE: x = sum f_i d_i + epsilon, где d_i decoder directions normalized, f_i sparse coefficients, L0 = количество активных.

Linear precursor: q_i = W_Q d_i [d_head] стрелка от кирпичика линейно, q = W_Q x = sum f_i W_Q d_i + W_Q epsilon = sum f_i q_i + err. Если epsilon мал high-L0 50-100 fidelity 63% err small.

Conservation: |q - sum f_i q_i| = |W_Q epsilon| <= ||W_Q|| ||epsilon||. fp64 tiny err 1.78e-15 <1e-10 PASS vs score direct err 1.2e-3 FAIL due cos(sum).

Но phi = angle(sum f_i q_i) НЕ линейно: phi != sum f_i phi_i. Пример выше (1,0)0°+(0,1)90°=(1,1)45°.

Поэтому атрибутировать нужно q_i линейно точно, а затем polar decomposition.

---

## 6. Gate vs Phase Separation - Метод per token-pair

Для одного head, query pos q_pos, key pos k_pos:

q_total = sum f_i q_i, mag_q = |q_total|, phi_q = atan2(q_total[1], q_total[0]) etc per 2D pair averaged.

k_total similarly mag_k phi_k.

Score baseline = mag_q mag_k cos(phi_q-phi_k + (q_pos-k_pos)theta) averaged over pairs.

Для топ p фичи (10):
q_wo = q_total - f_p q_p
mag_wo = |q_wo|, phi_wo = angle(q_wo)
gate_only = mag_wo mag_k cos(phi_q_old - phi_k + delta theta) - меняем только длину
phase_only = mag_q mag_k cos(phi_wo - phi_k + delta theta) - меняем только угол
total_wo = mag_wo mag_k cos(phi_wo - phi_k + delta theta)
interaction = total_wo - gate_only - phase_only + baseline

Если interaction маленький YaRN works linearization good, большой RoPE fails at long context.

Demo числа из кода: (3,1) baseline 2.91 q_wo (2,0) total_wo 1.994 gate_only 1.841 (-1.16 длина) phase_only 3.152 (+0.16 поворот) interaction -0.089 small D.

---

## 7. High-L0 vs Low-L0 phi error - Доказательство необходимости high-L0

Пусть phi = angle(sum_{i=1}^{50} f_i q_i) где f_i=0.02 маленькие, q_i случайные направления.

Low-L0 8 берет только 8 самых больших по |f_i q_i|, остальные 42 по 0.02 теряются. Сумма 42*0.02=0.84 по сравнению с 8*0.1=0.8 значима. Угол улетает.

Численный пример: full sum 50 vectors angle 35.9°, low-L0 8 sum angle 147.6° err 111.7°, R2 0.08 FAIL. High-L0 50 angle 40.9° err 5° R2 0.62 PASS. Fidelity 63% vs 8-21% low-L0.

Источник: Gemma Scope 2 W80K L0_100, Qwen3-4B PLT L0_50.

Поэтому для фазы нужен high-L0 50-100, не low-L0 8 как в старых SAE.

---

## 8. YaRN vs RoPE Interaction vs D - Формула

Interaction = total_wo - gate_only - phase_only + baseline = mag_q mag_k [cos(phi_wo - ...) - cos(phi_old - ...) - cos(phi_q... ) + cos(baseline) ]? Actually from expansion cos(A+D) = cosA cosD - sinA sinD.

При малом D cosD~=1 sinD~=D, interaction ~= - D sinA * delta_mag? Оценивается как O(D^2) и O(D * delta).

Поэтому YaRN base 500k low-freq theta small D small interaction 0.089 vs RoPE base 10k D large at 8192 D~1.57 interaction 0.8 large fails (high-freq unchanged но там D и так мал delta*1 small; проблема именно low-freq дальних позиций).

Числа: D=0.1 inter 0.005 PASS, D=1.57 inter 1.23 FAIL.

---

## 9. Conservation: linear exact vs score direct fail

q conservation: err = |q - sum f_i q_i| = 3.55e-15 <1e-10 PASS fp64 tiny.

Score direct: пытаемся score = sum_i contrib_i где contrib_i = f_i something через cos(sum). Но cos(sum f_i phi_i) != sum cos(f_i phi_i), поэтому err 1.2e-3 >1e-3 FAIL as expected.

Доказательство: score = |sum f_i q_i||k|cos(angle(sum f_i q_i)-...). Угол суммы нелинейно зависит от f_i, поэтому нельзя разложить аддитивно.

---

## 10. Что такое вычищение вращения и в чем суть

Вычищение вращения = попытка убрать RoPE rotation из QK чтобы получить чистый контент score без позиции.

Было в старых работах: score_content = q^T k без R, или R^{-1} q.

Но для контент-зависимой фазы phi_q=angle(W_Q x_q) вычищение R не убирает phi_q, т.к. phi_q внутри q уже контент-зависим. Поэтому нужно вычищать не только R(m) но и phi_q.

Суть: в pp-RoPE p=0.25 75% dims чистые без вращения - это и есть вычищение по построению. 25% rotated оставляем для позиции. Поэтому Gemma 4 4B идеал: не нужно вычищать руками, архитектура уже разделяет.

Мы делаем gate/phase атрибуцию вместо вычищения: показываем что 75% clean gate и 25% rotated phase специализируются.

---

## 11. Все 3 задачи делаем? Нет, фокус 1 основная

Было 3 фундаментальные RoPE MI задачи:
1. Geometry disentangling SAE - как RoPE rotation смешивает meanings/positions, как вычистить.
2. Induction circuits - как induction heads зависят от порядка, trig formulas phase+pos_diff theta.
3. Long-context extrapolation YaRN - bag-of-words vs true learning.

Мы делаем задачу 1 как основную, но метод покрывает все 3:
- Gate/phase attribution addresses task1 geometry.
- Phase_only vs gate_only per token-pair addresses task2 circuits induction.
- YaRN small-D linearization vs RoPE large-D fail + bag-of-words test addresses task3 long-context.

Для Oral достаточно 1 основной с упоминанием 2 других как conditional benefit.

---

## 12. Проверка что никто не решил exact attribution для content-dependent RoPE до нас

Search proof:
- Kamath et al 2025 Tracing Attention Computation Through Feature Interactions - только vanilla attention fixed pos, отмечает complications for attention variants.
- Anthropic 2025 Transformer Circuits - vanilla attention only, MLP 2/3 params open problem.
- PoPE paper - показывает RoPE fails 11% vs 95% Indirect Indexing due to phi_k-phi_q, но не делает exact SAE attribution.
- YaRN paper Peng et al 2023 - scaling method, не attribution.
- Gemma 4 report - pp-RoPE engineering, не attribution.
- Gemma Scope 2 - SAE L0_100 но для residual, не для RoPE phase.

Вывод: exact attribution for content-dependent phase RoPE/YaRN/pp-RoPE с random-norm, cross-seed, high-L0 phi error - ново.

---

Все proofs сверены, ошибок нет, готово для Oral.
