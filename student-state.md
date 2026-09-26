# Текущее состояние обучения

**Последнее обновление:** 2026-08-25

## Цель

Долгосрочная цель — вырасти до уровня frontier AI researcher с ориентацией на Anthropic. Главная специализация — mechanistic interpretability; поддерживающая — LLM systems.

## Возраст и ближайшая карьерная цель

- Ученик сообщил, что ему 14 лет и он учится в школе.
- Реалистичный текущий формат — удалённая работа; обычные университетские internships пока часто недоступны по возрасту или требованиям к статусу студента.
- Ближайшая цель — не формальная full-time должность, а сильный публичный research artifact, open-source contribution и удалённое mentorship/fellowship с согласием родителя или опекуна, где оно требуется.
- Проверен проект ученика HLA (`laplace-attention`) на зафиксированном commit `d67a11c`; выводы ниже не переносятся ни на какие другие работы.
- Рабочая карьерная гипотеза после аудита: специализация на пересечении mechanistic interpretability, architectures и research engineering при сохранении широких основ.

## Результат HLA portfolio audit

- Полный отчёт: `/home/user/reviews/laplace-attention-audit-ru.md`.
- Инженерный уровень проекта сильный: чистая identity initialization, shared init, extensive tests, config/data guards и causal-probe infrastructure.
- Ученик уточнил, что run реальный и FLOPs-matched: `loss_HLA = loss_base − 0.02`, то есть при соглашении `Δloss = loss_HLA − loss_base` сообщённый эффект равен `−0.02` и HLA лучше. Это принимается как сообщённый фактический результат; отдельный artifact audit сейчас не продолжать без запроса.
- Ученик отдельно уточнил, что завершённое сравнение было FLOPs-matched. Предыдущий вопрос аудита о способе compute matching сейчас не продолжать без отдельного запроса. Наблюдение о разнице active parameters остаётся отдельным control-вопросом, а не опровержением сообщённого результата.
- Поддерживается равенство raw forward и pre-clip backbone gradients, но не exact trainer-level first step при global clipping.
- Теоретические claims о строгом function-class inclusion и абсолютной NaN-safety требуют ослабления или нового доказательства.
- Абсолютные оценки: engineering `8/10`, scientific maturity `3/10`. Для mentor/youth-fellowship outreach проект силён после честного README и публикации artifact bundle; для paper ещё не готов.
- HLA-аудит поставлен на паузу по просьбе ученика; текущая работа возвращена к учебной программе.

## Реальный baseline, подтверждённый ответом

### KV cache

- Понимает общую идею: прошлые K/V нужны новым queries; прошлые Q обычно не cache-ятся.
- Пока не формулирует точно, что current `k_t` и `v_t` должны участвовать в attention текущего token, чтобы он мог смотреть на себя; production kernel может fuse-ить cache write и attention.
- Не закреплены shapes между prefill и one-token decode.
- Понимает общий риск неправильной causal mask как доступ к future, но не знает специфический cached-decode failure: при некоторых API/backend semantics current query может потерять доступ к valid past.

### Остальные области

Первый расширенный diagnostic был остановлен после A1. Никаких выводов о знаниях continuous batching, PagedAttention, DualPipe или mechanistic interpretability tools по отсутствующим ответам не делать.

## Текущий педагогический режим

1. Один механизм за раз.
2. Короткий cold pre-test.
3. Guided explanation на русском языке.
4. Tensor shapes и state transitions.
5. Worked example.
6. Checkpoint на 3–5 вопросов без подсказок.
7. Переход дальше только после mastery gate.
8. Delayed retrieval на D+1, D+7, D+21 и D+60.
9. Большие tests только после изучения prerequisites.

## Последний checkpoint

Результат по Lesson 01:

- `q_t` shape — правильная структура;
- current `k_t` ошибочно смешан с full cache;
- `K_cache_new` — правильная структура;
- `scores` ошибочно записан как prefill-like `[B,H,T,T]` вместо decode `[B,H,1,T_total]`;
- причины cache current K/V, отсутствия past Q и риска потери valid past объяснены правильно.

Статус: conceptual understanding хорошее; требуется corrective transfer по current tensors vs full cache.

## Переход после Lesson 02

Ученик сообщил, что получение `q_t` и production fusion понятны и попросил перейти дальше. Formal transfer-checkpoint Lesson 02 не был выполнен, поэтому тема имеет статус `understood, delayed verification pending`, а не окончательно `mastered`. Короткая retrieval-проверка будет встроена в следующий cumulative test.

## Результат Lesson 03 checkpoint

- Numerical memory calculation был пропущен как «лёгкий», поэтому memory accounting не подтверждён.
- В shapes перепутаны axes `H_q`, `H_kv`, `T` и `P`.
- `q_t`, `k_t`, full cache и scores пока не различаются устойчиво в GQA setting.
- GQA ошибочно интерпретирован как уменьшение dimension каждой KV head; правильная идея — уменьшение числа K/V heads при прежнем `head_dim`.
- Distinction GQA vs PagedAttention пока сформулирован только как «разные функции», без механизма.

Root cause: axis-value confusion и отсутствие mental model head sharing. Lesson 04 дал достаточную remediation для продолжения; numerical verification остаётся delayed.

## Результат Lesson 04

- Symbolic axis order для `q_t`, `k_t`, cache и scores восстановлен.
- Conceptual distinction сформулирован правильно: в ответе C3 ученик заполнял исходную конструкцию «не потому, что уменьшает `KV_head_size`, а потому, что уменьшает число KV heads».
- Предыдущая оценка ошибочно прочитала первую часть заполнения как самостоятельное утверждение. Эта оценка отозвана.
- Exact numerical shapes, group size и arithmetic не были предъявлены полностью; reduction factor был объяснён после попытки.
- Ученик сообщил, что остальное понял, и попросил продолжить.

Статус: `understood, delayed numerical verification pending`, не formal mastery. Проверить memory arithmetic в cumulative retrieval без блокировки текущей последовательности.

## Попытка Lesson 05

Ученик сообщил, что C1–C3 непонятны из-за количества английских слов. Ответы не интерпретировать как пробел в memory allocation: текущая форма урока измеряла владение смешанной терминологией. Lesson 05 приостановлен и разбит на более короткие русскоязычные механизмы.

## Результат Lesson 05A

Первый ответ на mastery gate:

- место с K/V: `7` — правильно;
- всё отданное место: `14` — правильно;
- пустое обещанное место: `7` — правильно;
- действительно свободное место: ответ `13` — неправильно; правильное значение `6`.

Причина ошибки: `20 − 7 = 13` посчитало всё место без данных, включая уже обещанные ячейки. Corrective transfer выполнен правильно на новом примере: `4` пустых обещанных и `7` действительно свободных.

Статус Lesson 05A: `mastered now, delayed retrieval pending`.

## Переоценка уровня и новый режим

Ученик сообщил, что уже знаком с induction heads, основными lens-методами, включая jLens, circuits и grokking, и хочет двигаться к top-level research, а не проходить introductory interpretability sequence. Это self-report; проверять его нужно не базовым экзаменом, а качеством derivation, implementation, causal controls и research artifact.

Педагогическая ошибка: после проблемы с английской терминологией remediation была чрезмерно упрощена и стала проверять базовый memory accounting вместо research-level transfer. Lesson 05B и вводную последовательность systems не продолжать как основной трек.

Новый режим:

```text
frontier research question
→ exact derivation
→ небольшой scaffold
→ самостоятельная реализация
→ causal experiment и controls
→ adversarial review
→ research memo
```

Основной текущий проект:

`projects/frontier-01-nonlinear-qk-attribution.md`

Тема: faithful/causal QK attribution для content-conditioned rotations и gates, с paired base/HLA models как контролируемым testbed. Systems, sparse methods, model diffing и statistics подключать just in time.

Первый gate: closed-book exact decomposition `score_HLA - score_base`, явный phase×gate interaction, переход к target-vs-foil log attention ratio и четыре falsification tests. После review дать interface scaffold и failing tests; полную реализацию за ученика не писать.

## Языковое правило

Объяснения и учебные файлы — на русском. Английскими остаются технические термины, имена papers/libraries/API, code, variables и formulas.
