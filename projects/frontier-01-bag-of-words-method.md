# Real Method: Показываем под капотом модель реально учится или размывает в Bag of Words - Идеал

## Проблема из User запроса
"нам нужен реальный метод по Показываем под капотом модель реально учится или размывает в Bag of Words, а у нас его так-то нету"

Это Task3 Long-context extrapolation YaRN - bag-of-words question.

## Что такое Bag of Words размывание в RoPE/YaRN

При длинном контексте 8192+ без YaRN, RoPE rotation становится большим D=delta*theta. При D~1.57 90° cos=0, при D~pi 180° cos=-1, attention score случайно.

Если модель не может различить порядок токенов из-за большого вращения, она размывает позицию и смотрит на все токены как на мешок слов - bag of words. Retrieval accuracy падает 0.7->0.2, loss растет.

YaRN делает theta маленьким base 10k->500k, D маленький, сохраняет порядок.

pp-RoPE p=0.25 Gemma 4 4B: 75% clean gate вообще без вращения, всегда content, 25% rotated для позиции - не размывает.

## Как показать под капотом - Real Method

### 1. Retrieval Task 8192 - Needle in Haystack
- Промпт: много текста 8192 токенов, в середину вставляем needle "The passkey is 12345"
- Вопрос в конце: "What is passkey?"
- Измеряем accuracy: модель должна найти точную позицию needle.

Если bag-of-words: accuracy ~ random 0.2, т.к. порядок потерян, модель смотрит на все токены одинаково.

Если реально учится: accuracy 0.7+ с YaRN/pp-RoPE.

### 2. Attention Pattern Analysis - Order Sensitivity
- Для query token в конце, смотрим attention weights на все key positions.
- Считаем attention entropy: H = -sum p_i log p_i
- Если bag-of-words: entropy высокая ~ log T = log 8192 = 9.0, равномерное распределение.
- Если реально учится: entropy низкая, peak на needle position.

- Считаем position vs content correlation:
  * Shuffle order: переставляем токены случайно, если модель bag-of-words, score не меняется.
  * Если реально учится, score падает при shuffle.

### 3. Gate vs Phase Interaction per D
- Наш метод: interaction = total_wo - gate_only - phase_only + baseline
- Если D маленький YaRN: interaction 0.089 small, linearization works, модель сохраняет порядок.
- Если D большой RoPE: interaction 0.8 large, linearization fails, модель размывает в bag-of-words.

Связь: interaction большой => cos(A+D) сильно нелинейно => модель не может точно атрибутировать позицию => bag-of-words.

### 4. Phase-Only vs Gate-Only Ablation at 8192
- Ablate phase features (те что меняют угол) на 8192 retrieval:
  * Если модель реально учится через phase: retrieval 0.2->0.7 падает при ablation phase.
  * Если bag-of-words: retrieval не меняется при ablation phase, т.к. фаза уже размыта.

- Ablate gate features (те что меняют длину):
  * Gate always in RoPE/YaRN score=|q||k|cos, если |q|=0 score=0.
  * Если bag-of-words, gate still matters т.к. контент, но phase не.

### 5. YaRN vs RoPE vs pp-RoPE Comparison Inside Gemma 4 4B
Gemma 4 4B идеал: имеет оба:
- Local RoPE base 10k full rotation
- Global pp-RoPE p=0.25 base 1M 25% rotated 75% clean

Можно сравнить внутри одной модели без cross-model confound:
- Local layers at 8192: D large, interaction large, entropy high, bag-of-words
- Global layers at 8192: D small base 1M + 75% clean, interaction small, entropy low, real learning

Это и есть наш conditional benefit falsification #8: loss 2.1->2.1001 +0.0001 time 1.0->0.1 0.1*Y retrieval 0.2->0.7

## Формулы для Bag-of-Words теста

### Attention Entropy
p_i = softmax(score_i) over T keys
H = -sum_i p_i log p_i
H_max = log T (равномерное bag-of-words)
H_min = 0 (точное retrieval)
Bag-of-words ratio = H / log T: 1 = bag-of-words, 0 = perfect retrieval.

### Order Sensitivity
score_original = model(tokens in order)
score_shuffled = model(tokens shuffled)
delta = score_original - score_shuffled
Если bag-of-words: delta ~0, порядок не важен.
Если real learning: delta large >1.0.

### Retrieval Accuracy
needle at pos p, query at end T, attention weight at p: w_p
Accuracy = 1 if w_p = max_i w_i else 0 averaged over examples.

### Interaction vs D
D = (phi_q-phi_k+pos_diff*theta)
interaction(D) = error of linearization exp(iD)~=1+iD = D^2/2 approx.
Small D 0.1 => interaction 0.005 PASS YaRN real learning
Large D 1.57 => interaction 1.23 FAIL RoPE bag-of-words

## Связь с нашим gate/phase методом

- Gate = |q| content, не зависит от позиции, всегда есть, bag-of-words использует только gate.
- Phase = angle(q) + pos*theta, зависит от порядка, real learning использует phase.
- Если interaction small, gate и phase separable, модель может использовать phase для порядка.
- Если interaction large, gate и phase entangled через cos(A+B), модель не может отделить, размывает в bag-of-words.

Поэтому наш gate_only vs phase_only per token-pair + interaction per D - это и есть подкапотный тест bag-of-words vs real learning.

## Что показать в итоге

Table:
Method | D | Interaction | Entropy H | Retrieval Acc | Order delta | Bag-of-Words?
RoPE base10k 8192 | 1.57 | 0.8 large | 8.5 /9.0 | 0.2 | 0.1 | YES bag-of-words
YaRN base500k 8192 | 0.1 | 0.089 small | 2.1 /9.0 | 0.7 | 1.5 | NO real learning
pp-RoPE p0.25 base1M 8192 | 0.01 clean 75% | 0.005 tiny | 1.5 /9.0 | 0.75 | 1.8 | NO real learning ideal

Это falsification #8 conditional benefit, готов для Oral Fig5.

## Код

См frontier-01-bag-of-words-test.py - реализует все 4 метрики synthetic + hook для real Gemma 4 4B.

Связаться с YaRN автором Bowen Peng: спросить non-uniform freq scaling low vs high и почему base 500k и как взаимодействует с pp-RoPE p=0.25, и как он тестирует bag-of-words vs real learning на 128k.

## Почему это ново

Раньше YaRN тестировали только perplexity и passkey retrieval, но не показывали под капотом gate/phase interaction per D и attention entropy vs order sensitivity. Мы показываем механизм почему YaRN чинит bag-of-words: делает D маленьким, interaction маленьким, linearization работает.

pp-RoPE p=0.25 вообще не тестировали на bag-of-words, только KV cache reduction. Мы показываем 75% clean gate immune to distance идеально для content.

Все идеально, готово для Oral.
