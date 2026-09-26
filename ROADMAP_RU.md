# Персональная программа подготовки к frontier AI research

**Версия:** 2026-08-23  
**Профиль нагрузки:** 20–30 часов в неделю  
**Текущий практический уровень:** самостоятельная реализация и обучение небольшого Transformer в PyTorch  
**Доступный compute:** Kaggle / Colab; использование только в рамках правил платформ и выделенных квот  
**Главная специализация:** mechanistic interpretability  
**Поддерживающая специализация:** LLM systems — inference, GPU performance, distributed training  
**Язык обучения:** объяснения и учебные документы на русском; английскими остаются технические термины, названия papers/libraries/API, code и formulas  

### Адаптация после первого ответа, 2026-08-23

Первый ответ на A1 подтвердил общую идею KV cache, но выявил пробелы в shapes и точной causal semantics. Расширенный diagnostic остановлен: продолжать его при большом количестве незнакомых терминов было бы измерением exposure, а не useful mastery. Текущий режим — один короткий guided block, четыре checkpoint-вопроса и переход дальше только после mastery gate.

---

## 0. Честная формулировка цели

Цель «стать ведущим учёным Anthropic» нельзя гарантировать учебным планом. Это результат нескольких лет сильных исследований, инженерной глубины, опубликованных или публично проверяемых результатов, научного вкуса, сотрудничества и способности формировать собственную research agenda.

Реалистичная цель этой программы:

1. За **12 месяцев** построить исследовательскую базу, сильный публичный portfolio и выполнить как минимум одну качественную replication/extension работу.
2. За **18–24 месяца** довести одну или две работы до уровня публичного preprint, workshop, TMLR или conference submission — только если результаты выдержат quality gates.
3. За **3–5+ лет** сформировать track record, необходимый не просто для входа в frontier lab, а для научного лидерства: несколько значимых работ, собственная agenda, влияние на инструменты/методы и способность вести других исследователей.

Мы оптимизируем не «количество пройденных тем», а вероятность стать человеком, который может:

- поставить важный и проверяемый research question;
- превратить его в чистый эксперимент;
- написать надёжный код и измерить систему;
- отличить свидетельство от красивой истории;
- найти альтернативные объяснения и провести controls;
- публично изложить как положительный, так и null result;
- воспроизвести и затем расширить работу сильной команды.

---

## 1. Аудит истории обучения

### 1.1. Что уже выглядит сильным

История показывает хорошую conceptual карту по:

- tensor roles и shapes в Transformer;
- autoregressive decode и базовому KV cache;
- RoPE и SwiGLU;
- linear attention, recurrent state, Mamba-2, DeltaNet, Gated DeltaNet и KDA;
- Attention Residuals;
- MLA absorption и compressed token-level cache;
- DSA и indexer/main-attention separation;
- MoE routing, dispatch, shared experts и loss-free balancing.

Особенно полезна уже сформулированная дисциплина:

1. Откуда tensor появился?
2. Каков его shape?
3. Что он представляет?
4. Хранится, читается, преобразуется или временный?
5. Что сломается, если его убрать?

И для speedup:

1. Какой tensor больше не materialize-ится?
2. Какое чтение памяти устранено?
3. Математика exact или approximate?
4. Какой новый bottleneck или failure mode появился?

Эти вопросы сохраняются как обязательная часть каждой темы.

### 1.2. Что пока нельзя считать доказанным mastery

История в основном фиксирует уровень **«могу объяснить после разбора»**. Она почти не содержит evidence для следующих уровней:

- вывести механизм с нуля без конспекта;
- реализовать его по спецификации;
- написать adversarial и property tests;
- найти ошибку в чужой реализации;
- предсказать profiler trace до запуска;
- сравнить альтернативы на одинаковом workload;
- перенести принцип на незнакомую систему;
- сформулировать и проверить новый research hypothesis.

Поэтому знание не засчитывается как mastered только потому, что объяснение кажется понятным.

### 1.3. Конкретный вердикт по новым темам

- **Continuous batching:** в истории присутствует только как название в списке дальнейших production optimizations. Считать не изученным.
- **Paged KV cache / PagedAttention:** базовый contiguous KV cache понятен, но block tables, allocation, fragmentation, copy-on-write, eviction, preemption и kernel indirection не разобраны. Считать не изученным.
- **DualPipe:** отсутствует. Перед ним нужны collectives, DP/TP/PP/EP, GPipe, 1F1B, pipeline bubbles, backward decomposition, communication-computation overlap и MoE all-to-all.
- **Mechanistic interpretability:** указана как долгосрочный интерес, но в истории нет causal interventions, TransformerLens, induction/IOI circuits, probes, superposition experiments, SAE evaluation или attribution graphs. Это главный пробел относительно выбранной специализации.

### 1.4. Критические пробелы, которые надо закрыть

1. **Research method:** hypotheses, controls, confounders, uncertainty, multi-seed analysis, ablations, null results.
2. **Statistics:** effect size, confidence interval, resampling, multiple comparisons, calibration.
3. **Systems foundations:** memory hierarchy, arithmetic intensity, roofline, profiling, queueing, OS paging, collectives.
4. **Engineering:** tests, typing, CI, reproducible environments, experiment tracking, clean repository design.
5. **Scientific communication:** short memo, paper figure, methods section, limitations, oral defense.
6. **Causal interpretability:** correlation/probe success is not mechanism; interventions and negative controls are required.

---

## 2. Целевая competency model

Текущая вакансия Anthropic Research Scientist, Interpretability описывает работу как reverse engineering алгоритмов, выученных в весах; robust experiments на toy и large models; features/circuits; infrastructure и visualization; публичную коммуникацию. От кандидата ожидают strong research track record, хотя бы некоторую работу по interpretability, комфорт с messy experimental science и сочетание science + engineering.

Официальный материал Anthropic «So You Want to Work in Mechanistic Interpretability?» рекомендует:

- реально обучить модель и работать с loss curves/hyperparameters;
- читать foundational papers;
- сделать маленький публичный interpretability experiment;
- работать с TransformerLens, NNSight, Neuronpedia и другими open tools;
- освоить mature codebases, testing, types и CI/CD;
- получить опыт visualization и distributed compute;
- писать публичные research reports;
- участвовать в community и collaborative research.

Из этого следует T-shaped профиль.

### Горизонталь

- mathematics and statistics;
- language modeling and optimization;
- software/research engineering;
- GPU and distributed systems;
- experiment design;
- research writing and collaboration.

### Главная вертикаль

- mechanistic interpretability;
- representations and circuits;
- causal interventions;
- superposition and sparse coding;
- scalable circuit tracing;
- faithfulness, completeness and interpretability evaluation;
- safety-relevant applications.

### Supporting vertical

- efficient activation extraction;
- serving and batching;
- KV memory management;
- GPU kernels;
- distributed training and inference;
- research infrastructure.

Это не отвлечение от interpretability. Anthropic отдельно ищет инженеров, способных строить instrumented forward/backward passes, activation extraction, steering infrastructure и устранять scaling bottlenecks. Systems depth делает interpretability-исследователя сильнее.

---

## 3. Что брать у «лучших в мире», а что не брать

Нет надёжного исследования, из которого можно причинно вывести: «ведущие AI-учёные учились именно так, поэтому этот путь оптимален». Публичные биографии и интервью страдают survivor bias и не дают counterfactual.

Поэтому программа использует три более надёжных источника:

1. **Learning science:** retrieval practice, distributed practice, feedback и mastery gates.
2. **Публичные требования frontier labs:** какие outputs и skills реально требуются сейчас.
3. **Структура лучших открытых curricula и codebases:** Stanford CS336, ARENA, CMU Deep Learning Systems, GPU MODE, TransformerLens, vLLM, SGLang, DeepSeek и Megatron.

Мы не используем миф «10 000 часов гарантируют expertise». Deliberate practice полезна как структура — конкретная слабость, целевое упражнение, немедленный feedback, повторная попытка, adaptive difficulty — но исследования не подтверждают, что один объём практики объясняет весь elite performance.

---

## 4. Основной learning loop для каждой темы

Каждая тема проходит девять стадий. Нельзя пропускать implementation и verification только потому, что conceptual часть понятна.

### Stage 0 — Cold pre-test

До чтения:

- 5–10 closed-book вопросов;
- одна shape/derivation задача;
- один prediction: что будет bottleneck и почему;
- confidence по каждому ответу от 0 до 100%.

Цель — обнаружить реальный prior, а не создать ощущение незнания.

### Stage 1 — Problem first

Сначала формулируется проблема:

- какая workload/model assumption;
- какой baseline;
- какая метрика неудовлетворительна;
- что невозможно или дорого в baseline.

### Stage 2 — Minimal mechanism

Одно объяснение без production деталей:

- state;
- transition;
- invariants;
- tensor shapes;
- complexity;
- exact/approximate status.

### Stage 3 — Derivation

Closed-book вывод ключевых формул, memory/FLOP estimates или scheduling invariants. Если вывод нельзя восстановить, тема ещё не понята.

### Stage 4 — Worked example, затем faded example

1. Один полностью разобранный пример.
2. Второй — с пропущенными шагами.
3. Третий — новая конфигурация без подсказок.

Interleaving начинается после первоначального blocked practice, а не до появления базовой схемы.

### Stage 5 — Minimal implementation

Реализация с нуля, максимально маленькая, но с:

- reference implementation;
- unit tests;
- randomized/property tests;
- explicit invariants;
- deterministic seed;
- failure cases.

### Stage 6 — Production code archaeology

Изучается не весь repository, а один end-to-end trace:

`request/input → state transition → allocation/cache → kernel/model call → output/free`

Результат чтения:

- call graph;
- state/tensor table;
- три invariants;
- один hidden assumption;
- один test, который должен ловить нарушение;
- различие между paper design и current implementation.

### Stage 7 — Measurement or causal experiment

До запуска записываются predicted results. После запуска:

- correctness first;
- profiler/metrics second;
- ablations;
- negative control;
- uncertainty;
- альтернативные объяснения.

### Stage 8 — Teach and defend

Ученик должен:

- объяснить тему за 5 минут;
- ответить на adversarial questions;
- написать 1–2 страницы;
- назвать ограничения и условия, при которых метод проигрывает.

### Stage 9 — Delayed retrieval

Повторные closed-book проверки ориентировочно на:

- D+1;
- D+7;
- D+21;
- D+60.

Это рабочая heuristic, а не единственно доказанный оптимальный интервал. Если recall ниже 60%, выполняется reteach и интервал сокращается. Если выше 90% два раза подряд, следующий интервал увеличивается.

---

## 5. Система тестирования

### 5.1. После каждой учебной сессии — 10–15 минут

Без заметок:

- 3 ключевых claims;
- 2 shapes/invariants;
- 1 failure mode;
- 1 вопрос, на который пока нет ответа.

### 5.1.1. Правило ранней остановки diagnostic

Если ученик:

- набирает меньше 60% в первом связанном блоке;
- не знает prerequisites для следующих вопросов;
- не может точно описать shapes/state;
- начинает угадывать незнакомую терминологию,

то diagnostic немедленно останавливается. Вместо продолжения выдаётся короткий remedial lesson с worked example и checkpoint на 3–5 вопросов. Большой diagnostic возвращается только после базового remediation. Это предотвращает бессмысленную перегрузку и позволяет измерять transfer, а не знакомство со словами.

### 5.2. Еженедельный тест — 45–75 минут

- 20% retrieval;
- 20% derivation/shapes;
- 25% coding/debugging kata;
- 20% experiment design;
- 15% explain/critique.

### 5.3. Module exam — каждые 4 недели

Продолжительность 2.5–4 часа:

1. closed-book theory;
2. derivation;
3. implementation или debugging;
4. unfamiliar production-code trace;
5. design of experiment;
6. 10–15 minute oral defense.

### 5.4. Quarterly qualifying project — каждые 12 недель

Обязательные компоненты:

- paper or claim reproduction;
- one meaningful extension or stress test;
- preregistered hypotheses;
- baselines and controls;
- reproducible code;
- report in conference style;
- 20-minute talk;
- adversarial review.

### 5.5. Mastery rubric

| Компонент | Вес |
|---|---:|
| Conceptual accuracy | 15 |
| Derivations, shapes, invariants | 15 |
| Correct implementation and tests | 20 |
| Code reading and systems reasoning | 10 |
| Experimental design and statistics | 20 |
| Transfer to a novel case | 10 |
| Scientific communication | 10 |

Pass:

- общий результат не ниже **80/100**;
- ни один раздел не ниже **60%**;
- implementation correctness обязателен;
- критическая causal claim не засчитывается без control/intervention.

Результат 80% не означает «тема закончена навсегда»: delayed test может понизить status.

### 5.6. Error ledger

Каждая ошибка получает категорию:

- `concept`;
- `math`;
- `shape/state`;
- `implementation`;
- `measurement`;
- `causal inference`;
- `reading`;
- `communication`;
- `carelessness`.

Следующее упражнение выбирается из наиболее частой категории, а не из самой интересной новой темы.

---

## 6. Недельный бюджет: базовая конфигурация 24 часа

| Работа | Часы |
|---|---:|
| Build / experiments | 9 |
| Mechanistic interpretability theory and labs | 5 |
| Systems / GPU / distributed foundations | 4 |
| Primary papers and production code | 2.5 |
| Retrieval tests and error correction | 2 |
| Research writing / weekly review | 1.5 |

Рекомендуемая структура:

- 4 deep-work дня по 4–5 часов;
- 1 день integration/test/write-up на 4 часа;
- 1 короткий spaced-retrieval block;
- минимум один полный день без тяжёлой исследовательской работы.

Stretch work выполняется только после основного deliverable и теста. Нельзя компенсировать отсутствие эксперимента дополнительными лекциями.

---

## 7. Первые 12 недель: детальный план

Если начать 2026-08-24, этот блок заканчивается примерно 2026-11-15. При другом старте сохраняются номера недель, а не даты.

### Week 0 — Adaptive baseline и foundational remediation

**Systems:** короткие micro-diagnostics по KV cache, shapes, prefill/decode и memory; следующий блок открывается только после mastery предыдущего.  
**Interpretability:** vocabulary map и короткие checks по causal claims без требования знать ещё не изученные tools.  
**Artifact:** карта baseline, error ledger и завершённые remedial lessons.  
**Gate:** каждый checkpoint выполняется без AI и конспекта; при результате ниже 60% diagnostic останавливается и включается guided remediation. Расширенный `diagnostic-00.md` переносится на момент, когда его prerequisites действительно изучены.

### Week 1 — Instrumentable Transformer

**Systems:** FLOPs, memory traffic, parameter/activation/KV accounting; prefill vs decode.  
**Interpretability:** residual stream, hooks, activation cache, logits.  
**Build:** небольшой Transformer с named hook points и exact shape assertions.  
**Test:** сопоставить собственные activations с reference PyTorch path.

### Week 2 — Serving workload and direct attribution

**Systems:** arrival traces, request lifecycle, TTFT, ITL/TBT, E2E latency, throughput, serving capacity, percentiles.  
**Interpretability:** logit lens, direct logit attribution, limitations of linear decompositions.  
**Build:** deterministic workload generator и metrics module.  
**Research habit:** predictions before every benchmark.

### Week 3 — Static, dynamic and continuous batching

**Systems:** request-level vs iteration-level scheduling, admission, completion, variable sequence lengths.  
**Primary source:** Orca.  
**Interpretability:** induction-head signatures and activation patterns.  
**Build:** CPU discrete-event scheduler simulator; static and continuous baselines.  
**Test:** hand-trace arrivals and verify exact scheduler state after each iteration.

### Week 4 — Continuous batching implementation

**Systems:** prefill/decode coexistence, token budgets, fairness, starvation, preemption.  
**Interpretability:** partial induction-head replication on a tiny/open model.  
**Build:** mini inference loop with requests entering/leaving between decode iterations.  
**Module Exam A:** concepts + implementation + oral defense.

### Week 5 — Paging and causal interventions

**Systems:** contiguous allocation, internal/external fragmentation, logical/physical blocks, free lists, reference counts, copy-on-write.  
**Interpretability:** activation patching; clean/corrupted pairs; necessity vs sufficiency.  
**Build:** CPU KV block allocator with invariants and randomized state-machine tests.  
**Control:** prove no double-free, leaked reference or aliased write after COW.

### Week 6 — PagedAttention

**Systems:** block tables, paged read/write, block-size trade-off, kernel indirection, sharing and eviction.  
**Primary source:** PagedAttention/vLLM paper.  
**Interpretability:** path patching and negative controls.  
**Build:** toy PyTorch PagedAttention exactness implementation.  
**Gate:** outputs/gradients where applicable match contiguous reference within declared tolerance; fragmentation benchmark is reproducible.

### Week 7 — vLLM code archaeology and IOI

**Systems code:** trace current vLLM scheduler → KV manager → block pool → block table/model runner.  
**Interpretability:** reproduce a subset of the IOI circuit workflow.  
**Artifact:** code map, state table, tests-to-invariants map and one documented discrepancy between paper abstraction and current implementation.  
**Adversarial review:** identify what the IOI evidence does not prove.

### Week 8 — Chunked prefill and stall-free scheduling

**Systems:** prefill/decode arithmetic intensity, TBT SLO, hybrid batches, chunk size overhead, Sarathi-Serve.  
**Interpretability:** IOI robustness across prompt variants and controls.  
**Build:** token-budget scheduler and ablation over chunk size/workload.  
**Module Exam B:** mini-serving capstone v1.

### Week 9 — Prefix reuse and superposition

**Systems:** prefix caching, radix tree, cache-aware scheduling, eviction and starvation; SGLang paper/code.  
**Interpretability:** Toy Models of Superposition; phase changes and geometric structure.  
**Build:** CPU radix-prefix cache and toy superposition replication.  
**Test:** distinguish representation evidence from causal use.

### Week 10 — Distributed foundations and SAE basics

**Systems:** process groups, collectives, DP, TP, PP, SP, EP; latency/bandwidth cost models; GPipe and 1F1B.  
**Interpretability:** dictionary learning, SAE objective, sparsity/reconstruction trade-off, dead latents, feature splitting.  
**Build:** pipeline schedule simulator plus a tiny SAE on synthetic activations.  
**Gate:** analytically predicted bubble/memory trend must agree with simulator.

### Week 11 — ZeroBubble, DualPipe and SAE evaluation

**Systems:** separate backward-input and backward-weight work; ZeroBubble intuition; bidirectional stages; F/B communication overlap; MoE all-to-all; DualPipe.  
**Interpretability:** SAE evaluation beyond cherry-picked top activations: reconstruction, sparsity, intervention effects, stability, synthetic recovery.  
**Build:** discrete-event DualPipe simulator; compare against GPipe/1F1B under ideal and perturbed communication.  
**Constraint:** Kaggle/Colab results cannot support claims about multi-node IB/RDMA; those remain simulation/cost-model claims.

### Week 12 — Qualifying project 1

Два связанных artifacts:

1. **MiniServe:** continuous batching + paged KV allocator + chunked prefill simulator/mini runtime.
2. **Causal interpretation report:** induction/IOI or toy-superposition replication with at least one failed or weakened claim under a control.

Обязательные outputs:

- reproducible repository;
- test suite;
- benchmark manifest;
- 4–6 page report;
- one systems diagram;
- one causal evidence diagram;
- 20-minute talk;
- written red-team review;
- 60-day retention questions.

---

## 8. Как изучать Continuous Batching до frontier уровня

### Prerequisite graph

`autoregressive generation → prefill/decode → KV state → latency metrics → batching → scheduling → admission/preemption → memory manager → kernels`

### Questions that must be answerable

1. Почему request-level batching теряет capacity при разной длине outputs?
2. Чем dynamic batching отличается от iteration-level continuous batching?
3. Какие operations легко batch-ятся, а какие зависят от per-request state?
4. Что происходит, когда новый prefill встречается с ongoing decodes?
5. Как scheduler балансирует TTFT, TBT и throughput?
6. Когда preemption через recompute лучше swap?
7. Какие policies вызывают starvation?
8. Как memory budget ограничивает admission?
9. Почему большой batch может ухудшить tail latency?
10. Как workload distribution меняет победителя между policies?

### Source sequence

1. Orca — iteration-level scheduling и selective batching.
2. PagedAttention/vLLM — co-design scheduler + memory manager.
3. Sarathi-Serve — chunked prefill и stall-free schedules.
4. SGLang — prefix reuse, RadixAttention и cache-aware scheduling.
5. Current vLLM/SGLang code — paper-to-production delta.

### Required implementation ladder

1. Hand simulation.
2. Discrete-event CPU simulator.
3. PyTorch decode loop with variable active batch.
4. Memory-aware admission.
5. Chunked prefill.
6. Prefix reuse.
7. Trace-driven benchmark with p50/p95/p99.
8. One scheduler policy extension and adversarial workload.

### Mastery proof

Нужно не только объяснить scheduler, но и показать workload, где:

- continuous batching выигрывает;
- continuous batching проигрывает по конкретной latency metric;
- неправильный admission вызывает OOM или excessive preemption;
- fairness policy меняет tail behavior;
- результат сохраняется после repeated runs и корректной warm-up процедуры.

---

## 9. Как изучать Paged KV Cache / PagedAttention

### Prerequisite graph

`KV tensor geometry → bytes/token → OS paging → fragmentation → allocator → block table → COW/refcount → paged kernel → scheduler co-design`

### Required knowledge

- exact KV bytes per token/request for MHA, GQA/MQA and compressed caches;
- logical vs physical block;
- last-block internal waste;
- external fragmentation in contiguous allocation;
- block allocation/free invariants;
- prefix sharing and copy-on-write;
- eviction and reference counting;
- non-contiguous read overhead;
- block-size trade-off;
- relation to prefix cache and beam/parallel sampling;
- distinction between PagedAttention and RadixAttention.

### Required projects

1. Allocator with property tests.
2. Fragmentation simulator comparing max-length reservation, dynamic contiguous and paged allocation.
3. Exact toy paged attention against contiguous attention.
4. Prefix sharing + COW.
5. Scheduler integration.
6. Code trace in vLLM.
7. One design variant: adaptive block size, alternate eviction or cache-aware admission — evaluated honestly, including negative result.

### Mastery proof

Ученик получает unfamiliar allocator trace и должен:

- восстановить block table;
- найти invalid refcount or aliasing;
- вычислить waste;
- предсказать число admissible sequences;
- объяснить kernel implications;
- предложить test, который ловит дефект.

---

## 10. Как изучать DualPipe

DualPipe нельзя учить как diagram memorization.

### Prerequisite graph

`autograd dependency graph → distributed collectives → DP/TP/PP/EP → microbatches → GPipe → 1F1B → bubble/memory accounting → backward-input/backward-weight split → ZeroBubble → MoE all-to-all → overlap → bidirectional pipeline → DualPipe`

### Required questions

1. Что именно считается pipeline bubble?
2. Как число stages и microbatches влияет на utilization и activation memory?
3. Какие dependencies мешают произвольно менять порядок F/B/W?
4. Почему MoE expert parallelism создаёт heavy all-to-all?
5. Что значит «communication hidden by computation» и как это измерить?
6. Почему bidirectional schedule требует иной placement/parameter accounting?
7. При каких ratios compute/communication overlap перестаёт помогать?
8. Какие assumptions DeepSeek делает о topology и kernels?
9. Что можно подтвердить на 2 GPU, а что требует multi-node H800/IB-like среды?
10. Чем paper schedule отличается от current DualPipe/DualPipeV code?

### Required implementation ladder

1. Timeline на бумаге для GPipe и 1F1B.
2. Discrete-event simulator с dependency checks.
3. Добавление communication events и resource contention.
4. ZeroBubble-like split.
5. DualPipe schedule.
6. Sensitivity analysis по stages, microbatches, F/B/W cost и communication latency.
7. При доступном lawful 2-GPU runtime — small validation; без него claims остаются simulation-only.
8. Чтение official DualPipe, DeepEP и Megatron pipeline code.

### Mastery proof

- аналитическая формула и simulator согласуются в ideal case;
- perturbation обнаруживает regimes, где assumed overlap ломается;
- schedule не нарушает data dependencies;
- memory accounting включает parameters, optimizer state, activations и communication buffers;
- отчёт чётко отделяет measured, simulated и inferred claims.

---

## 11. Параллельный mathematics and statistics spine

Этот spine идёт 3–4 часа в неделю первые 9 месяцев и затем поддерживается задачами из проектов. Темы не изучаются отдельными бесконечными курсами: каждая привязана к текущему механизму или эксперименту.

### Linear algebra and geometry

- vector spaces, basis changes and projections;
- eigendecomposition and SVD;
- rank, null space and conditioning;
- low-rank approximation;
- orthogonality in high dimensions;
- tensor contractions and einsum reasoning;
- linear maps in residual-stream and feature-space analyses.

**Evidence:** closed-book derivation + numerical experiment + connection to an interpretability claim.

### Probability, information and statistics

- random variables, expectation, variance and covariance;
- likelihood, cross-entropy, entropy, KL divergence and mutual information;
- sampling distributions and central-limit reasoning;
- bootstrap confidence intervals;
- effect sizes, power and practical significance;
- multiple comparisons and selection bias;
- calibration and proper scoring rules;
- permutation/randomization tests where appropriate.

**Evidence:** analyze a real weekly experiment, not only solve textbook exercises.

### Optimization and numerics

- gradient descent, SGD, momentum and AdamW;
- initialization and signal propagation;
- loss landscapes, curvature and Hessian intuition;
- floating-point formats, overflow/underflow and stable softmax;
- mixed precision and loss scaling;
- gradient clipping, normalization and optimizer-state memory.

### Causal and experimental reasoning

- intervention vs observation;
- confounding and selection effects;
- necessity vs sufficiency;
- positive/negative controls;
- distribution shift;
- preregistration and stopping rules;
- exploratory vs confirmatory analysis.

### Systems mathematics

- asymptotic work vs actual latency;
- bytes/FLOPs and arithmetic intensity;
- roofline model;
- basic queueing and Little’s law;
- latency/throughput/tail distributions;
- communication cost models;
- pipeline utilization and bubble formulas.

### Assessment cadence

- Week 0 includes a focused baseline from Diagnostic 00.
- Every weekly test contains one math/statistics transfer problem.
- Every four-week exam contains one derivation and one interpretation of noisy experimental data.
- At Week 12, weak areas receive a dedicated four-week remediation block before adding advanced theory.

### Time allocation by phase

- **Weeks 1–12:** systems-intensive foundation — approximately 40% systems/building, 35% interpretability, 15% math/statistics, 10% testing/writing.
- **Months 4–12:** interpretability becomes 55–65%; systems remains 15–25% as enabling infrastructure; the rest is math, research method and writing.
- **Year 2:** allocation follows the strongest research question, but no project may drop reproducibility, statistics or code quality.

## 12. Mechanistic Interpretability — главная вертикаль

### Block A — Instrumentation and transformer algebra

- residual stream as communication channel;
- decomposition by head/MLP;
- hooks and activation caches;
- logits, unembedding and direct attribution;
- attention patterns vs computed information;
- gauge/basis and decomposition caveats.

**Project:** instrumentable small Transformer with hooks, caches and intervention API.

### Block B — Classic circuits

- induction heads;
- composition and virtual weights;
- IOI;
- activation patching;
- path patching;
- ablation types;
- clean/corrupted distribution design.

**Project:** replication plus a stress test that attempts to break a published qualitative story.

### Block C — Representations and probes

- linear probes;
- probe capacity and selectivity;
- decodability vs causal use;
- steering vectors;
- counterfactual interventions;
- out-of-distribution validation.

**Project:** show at least one case where a high-accuracy probe does not establish the claimed mechanism.

### Block D — Superposition

- toy models;
- sparse features;
- privileged/non-privileged bases;
- phase transitions;
- feature geometry;
- interference;
- attention-head superposition.

**Project:** reproduce a toy phase transition and test sensitivity to feature frequencies, importance and optimizer settings.

### Block E — Sparse autoencoders

- dictionary learning objective;
- L1, TopK, JumpReLU and gated variants;
- reconstruction and sparsity;
- dead latents;
- feature splitting/absorption;
- autointerpretability;
- synthetic ground truth;
- stability across seeds and widths;
- causal feature interventions.

**Project:** evaluation-first SAE study; no claim based only on attractive top activations.

### Block F — Circuit tracing and scalable methods

- transcoders/cross-layer transcoders;
- replacement models;
- local attribution graphs;
- error nodes and «dark matter»;
- sufficiency and completeness;
- intervention-based validation;
- global vs prompt-local circuits;
- qualitative analysis and dataset coverage.

**Project:** use open circuit-tracer on an open model, generate a mechanism hypothesis, then attempt to falsify it with perturbations and prompt distribution shifts.

### Block G — Safety relevance

- faithfulness of chain-of-thought;
- hallucination/refusal mechanisms;
- deception/sleeper-agent probes;
- monitoring and steering;
- failure of feature labels under narrow datasets;
- how interpretability evidence should and should not enter safety cases.

**Project:** pre-registered safety-relevant behavior study on an open model with careful release and claims discipline.

### Block H — 2025–2026 frontier directions

Только после освоения causal и measurement foundations:

- attribution graphs on longer/multi-turn behavior;
- attention computation through feature interactions;
- sparse mixtures of linear transforms and alternatives to one-vector features;
- turn-averaged or hierarchical SAE representations;
- persona/assistant axes and introspection claims;
- automated interpretation with explicit faithfulness and coverage evaluation.

Frontier updates изучаются как provisional research, а не как учебниковые факты. Для каждого результата отделяются mature paper, preliminary lab update, open-source demonstration и independently replicated evidence.

---

## 13. Production-code reading syllabus

Читать надо pinned commit, а не плавающий `main`; commit записывается в weekly log. Код не копируется вслепую и не запускается до inspection лицензии, dependencies и scripts.

### LLM systems

- **vLLM:** `vllm/v1/core/sched/scheduler.py`, `kv_cache_manager.py`, `kv_cache_coordinator.py`, `single_type_kv_cache_manager.py`, `block_pool.py`, worker block tables/model runner, соответствующие tests.
- **SGLang:** `python/sglang/srt/managers`, `mem_cache/radix_cache.py`, base prefix cache, memory pools, `layers/radix_attention.py`.
- **DeepSeek DualPipe:** official schedule implementation and profile data.
- **DeepEP:** dispatch/combine interfaces, normal vs low-latency paths, overlap hooks; не пытаться запускать H800/RDMA code в Colab.
- **Megatron-LM:** `megatron/core/pipeline_parallel/schedules.py` and tests.
- **GPU MODE:** profiling, CUDA checklist, Triton, FlashAttention, NCCL, SGLang/FlashInfer lectures.

### Mechanistic interpretability

- **TransformerLens:** hook points, activation cache, model loading and tests.
- **SAELens:** SAE architectures, training loops, activation stores, metrics and tests.
- **circuit-tracer:** graph construction, replacement model, attribution and intervention flow.
- **ARENA:** exercises are scaffolding, not source of final claims.

### Code-reading three-pass protocol

**Pass 1 — Map**

- repository purpose;
- entry points;
- modules;
- tests;
- state owners;
- public interfaces.

**Pass 2 — Trace**

- one request/activation end to end;
- call graph;
- data structures;
- tensor shapes;
- device transfers;
- allocation/free lifecycle.

**Pass 3 — Challenge**

- hidden assumptions;
- race/OOM/aliasing risks;
- degenerate workloads;
- missing assertions;
- one candidate test;
- one candidate issue/PR.

Каждая code-reading сессия заканчивается artifact, а не закладкой в браузере.

---

## 14. 24-месячная project ladder

### Months 0–3 — Foundation artifacts

- Instrumentable small Transformer.
- MiniServe scheduler + paged allocator.
- Induction/IOI or toy-superposition replication.
- DualPipe simulator.

**Gate:** reproducible repositories, tests, reports and oral defense.

### Months 4–6 — Causal interpretability and open-source engineering

- Complete ARENA interpretability core selectively.
- Add controls to a classic circuit replication.
- Work inside TransformerLens/SAELens/NNSight code.
- Submit at least one useful issue, documentation fix, test or PR.

**Gate:** external maintainer/researcher can run the artifact from README.

### Months 7–9 — SAE and scalable interpretation

- Train/evaluate SAEs on synthetic and small open-model activations.
- Compare at least two architectures or objectives.
- Measure stability, splitting, dead latents and intervention effects.
- Build efficient activation extraction pipeline under Colab/Kaggle constraints.

**Gate:** result survives alternative seeds, widths and at least one negative control.

### Months 10–12 — Research project 1

Preferred shape:

- replicate a recent mechanistic interpretability claim;
- test generalization to another model/task/distribution;
- identify a failure mode or condition of validity;
- release code, exact environment, reduced-cost reproduction path and report.

Possible outputs:

- serious blog/preprint;
- workshop submission;
- TMLR/MLRC reproducibility direction if contribution is strong enough.

Submission is not mandatory; quality gate is mandatory.

### Months 13–18 — Research project 2

Move from replication to a narrow original contribution:

- better causal evaluation for SAE/circuit claims;
- scalable experiment selection;
- efficient activation/circuit infrastructure;
- robustness of circuit explanations across prompts/models;
- relation between representation and causal use;
- measurement framework for faithfulness/completeness.

**Gate:** external expert red-team review before paper framing.

### Months 19–24 — Publication and collaboration

- deepen the strongest project;
- recruit collaborator/mentor where appropriate;
- reproduce all main tables from clean environment;
- create artifact appendix;
- write paper and limitations early;
- submit to venue chosen by contribution, not prestige fantasy.

Potential venues:

- mechanistic interpretability / ML: ICLR, ICML, NeurIPS, COLM, TMLR;
- systems contribution: MLSys; OSDI/SOSP/ASPLOS only for genuinely systems-level novelty and evaluation;
- workshops for focused early work, not as substitute for rigor.

Deadlines for 2027+ are inserted only after official calls are published.

---

## 15. Conference-grade research workflow

### 14.1. Question selection

Score each candidate 1–5:

- significance;
- falsifiability;
- tractability;
- compute feasibility;
- availability of baselines;
- measurement quality;
- novelty after literature search;
- safety/release considerations.

Reject projects that depend on inaccessible frontier activations or multi-node hardware unless a valid smaller-scale question remains.

### 14.2. Before experiments

Create:

- claim table;
- related-work matrix;
- preregistered primary hypothesis;
- alternative hypotheses;
- success/failure criteria;
- baselines;
- controls;
- compute budget;
- data/model/license manifest;
- stopping rule.

### 14.3. During experiments

- immutable configs;
- seeds logged;
- raw and derived results separated;
- plots generated from scripts;
- failed runs preserved with reason;
- hypotheses updated explicitly, never rewritten silently after seeing data;
- weekly research memo.

### 14.4. Before a claim

Ask:

1. Is the effect replicated?
2. Does it survive a simpler baseline?
3. Does it survive a negative control?
4. Is it causal, correlational or merely descriptive?
5. Is effect size meaningful?
6. Is uncertainty shown?
7. Can another person run a reduced artifact?
8. What evidence would falsify the story?
9. Does the title/abstract overstate scope?
10. Are limitations and nulls visible?

### 14.5. Artifact standard from day one

- environment lock;
- exact commands;
- hardware and runtime;
- dataset/model versions and licenses;
- configuration files;
- raw metrics;
- scripts for tables/figures;
- smoke test;
- reduced-cost reproduction path;
- claim-to-command map.

Это соответствует направлению NeurIPS reproducibility checklist и MLSys artifact evaluation, но применяется задолго до submission.

---

## 16. Работа с Kaggle и Colab

### Что реалистично

- small Transformer training;
- TransformerLens/SAELens experiments на небольших open models;
- toy superposition;
- scheduler/allocator simulations;
- Triton basics, если доступная GPU/runtime поддерживается;
- single-GPU profiling;
- краткие 2-GPU tests, если конкретная среда официально предоставляет их.

### Что нельзя честно заявлять

- multi-node network scaling;
- IB/RDMA behavior;
- production H100/H800 kernel throughput;
- reliable DualPipe speedup at DeepSeek scale;
- frontier-model interpretability generalization.

### Compute discipline

- соблюдать Terms of Service и квоты; не обходить ограничения множественными аккаунтами;
- checkpoint frequently;
- separate smoke, pilot and full run;
- log GPU model and software versions;
- use tiny deterministic tests before expensive runs;
- cache only legally reusable assets;
- maintain a compute ledger in GPU-hours and estimated cost.

---

## 17. Progress dashboard

Не считать главным показателем часы или число прочитанных papers.

### Weekly metrics

- retrieval score;
- delayed retention score;
- implementation tests passed;
- number of unresolved error-ledger items;
- one concrete artifact completed;
- one written research memo;
- prediction calibration: confidence vs correctness.

### Quarterly metrics

- replication fidelity;
- robustness under controls;
- external reproducibility;
- code quality;
- clarity of report;
- quality of research questions;
- useful public contribution;
- ability to explain null results.

### Year-1 success condition

Минимум:

- 4 substantial repositories/artifacts;
- 3 replication reports;
- 1 project with a meaningful extension;
- 1 external code/documentation contribution;
- 4 oral defenses;
- 1 public technical talk or long-form write-up;
- one conference-style artifact package, submitted only if scientifically ready.

---

## 18. Что делать сейчас — advanced project-first override, 2026-08-25

Предыдущая introductory systems sequence поставлена на паузу после новой калибровки уровня. Lesson 05A пройден; Lesson 05B не продолжать как основной трек. Systems foundations вернутся just in time или отдельным accelerated engineering sprint.

Текущий authoritative project spec:

`/home/user/ai-research-program/projects/frontier-01-nonlinear-qk-attribution.md`

Порядок:

1. Выполнить closed-book derivation standard feature-pair QK attribution.
2. Вывести exact partition `score_HLA - score_base` с отдельным phase×gate interaction term и additive biases.
3. Показать non-uniqueness назначения interaction одному mechanism.
4. Перейти от одного score к target-vs-foil `log(p_target/p_foil)`.
5. До кода сформулировать четыре falsification tests.
6. После review derivation получить небольшой interface scaffold и failing tests.
7. Самостоятельно реализовать fp64 tiny reference; затем переходить к real hooks/checkpoints.
8. Оценивать progress по conservation, intervention faithfulness, completeness, robustness и качеству research memo, а не по числу пройденных тем.

---

## 19. Основные источники

### Learning science

- Dunlosky et al., *Improving Students’ Learning With Effective Learning Techniques*: https://journals.sagepub.com/doi/abs/10.1177/1529100612453266
- Karpicke & Roediger, *The Critical Importance of Retrieval for Learning*: https://www.science.org/doi/10.1126/science.1152408
- Cepeda et al., distributed practice meta-analysis: https://www.yorku.ca/ncepeda/publications/CPVWR2006.html
- Carpenter et al., review of spacing and retrieval practice: https://www.nature.com/articles/s44159-022-00089-1
- Ericsson, deliberate practice clarification: https://pmc.ncbi.nlm.nih.gov/articles/PMC6824411/
- Macnamara & Maitra, replication/limits: https://royalsocietypublishing.org/doi/10.1098/rsos.190327

### Curriculum and target role

- Anthropic, *So You Want to Work in Mechanistic Interpretability?*: https://transformer-circuits.pub/2025/april-update/index.html#work
- Anthropic Research Scientist, Interpretability: https://job-boards.greenhouse.io/anthropic/jobs/4980427008
- Anthropic Research Engineer, Interpretability: https://job-boards.greenhouse.io/anthropic/jobs/4980430008
- Stanford CS336: https://cs336.stanford.edu/
- CS336 Assignment 1: https://github.com/stanford-cs336/assignment1-basics
- CS336 Assignment 2: https://github.com/stanford-cs336/assignment2-systems
- CMU Deep Learning Systems: https://dlsyscourse.org/
- ARENA Chapter 1: https://www.arena.education/chapter1
- GPU MODE: https://github.com/gpu-mode/lectures

### Inference and distributed systems

- Orca, OSDI 2022: https://www.usenix.org/conference/osdi22/presentation/yu
- PagedAttention/vLLM, SOSP 2023: https://arxiv.org/abs/2309.06180
- vLLM code: https://github.com/vllm-project/vllm
- Sarathi-Serve, OSDI 2024: https://www.usenix.org/conference/osdi24/presentation/agrawal
- SGLang, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf
- SGLang code: https://github.com/sgl-project/sglang
- DeepSeek-V3 report: https://arxiv.org/abs/2412.19437
- DualPipe code: https://github.com/deepseek-ai/DualPipe
- DeepEP code: https://github.com/deepseek-ai/DeepEP
- Megatron-LM: https://github.com/NVIDIA/Megatron-LM

### Mechanistic interpretability

- Anthropic Interpretability research: https://www.anthropic.com/research/team/interpretability
- Transformer Circuits Thread: https://transformer-circuits.pub/
- Toy Models of Superposition: https://transformer-circuits.pub/2022/toy_model/index.html
- Circuit Tracing methods: https://transformer-circuits.pub/2025/attribution-graphs/methods.html
- On the Biology of a Large Language Model: https://transformer-circuits.pub/2025/attribution-graphs/biology.html
- Anthropic Circuits Updates, May 2026: https://transformer-circuits.pub/2026/may-update/index.html
- Anthropic Circuits Updates, June 2026: https://transformer-circuits.pub/2026/june-update/index.html
- Anthropic open-source circuit tracing: https://www.anthropic.com/research/open-source-circuit-tracing
- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- SAELens: https://github.com/decoderesearch/SAELens
- circuit-tracer: https://github.com/safety-research/circuit-tracer

### Publication and reproducibility

- NeurIPS Paper Checklist: https://neurips.cc/public/guides/PaperChecklist
- MLSys research-paper/artifact policy: https://mlsys.org/Conferences/2026/CallForResearchPapers
