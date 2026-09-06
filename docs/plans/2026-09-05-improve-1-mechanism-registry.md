# PLAN: Улучшение 1 — Реестр механизмов + state_dict-персистенция

- **Date:** 2026-09-05 · **Status:** draft (предложение, вне задачи green-suite) · **Приоритет:** P0
- **Оценка пользы:** −280 строк boilerplate; закрытие классов багов B10/B13/B14/B16; добавление
  нового механизма = 1 файл + 1 строка реестра вместо правок в 4 местах; чекпоинт-инвариант
  становится машино-проверяемым. Трудоёмкость: средняя (1-2 дня), риск средний (миграция без
  изменения формата чекпоинта).

## Проблема (доказательства)
- save_checkpoint/load_checkpoint (src/train.py:81-192, 193-410) — ~280 строк ручных `if hasattr(...)
  state[key] = ...clone()` / `copy_` блоков по 12 механизмам.
- B10: чекпоинты неполные — STA ref_cov/ref_eigenvalues/current_eigenvalues не сохраняются
  (train.py:150-155), WSD target_cov/target_Q теряются (train.py:135-139), SPC adapt_step, PUC
  running_entropy/overconfidence — тоже → resume с мусорным состоянием до 100 шагов.
- PUC/RDC имеют собственные checkpoint_dict/load_checkpoint — **никем не вызываются** (мёртвый
  параллельный механизм персистенции).
- B13/B14: MechanismBundle (src/mechanisms.py:406-408, 461-499) расходится с inline-путём jepa.py
  (.data vs live view Q; from_config теряет STA-параметры).
- B16: jspace._prev_jspace_vectors, jawp._prev_workspace_Q, wsr._lagged_gradient — plain attrs:
  не буферы, не в чекпоинтах, device-хрупко.

## Решение
1. `src/models/mechanism_registry.py`: `MECHANISMS: dict[str, MechanismSpec]`, где spec = фабрика,
   конфиг-ключи, список state-буферов.
2. Каждый механизм: `capture_state() -> dict[str, Tensor]` / `restore_state(dict)` (тривиально для
   register_buffer-модулей — напрямую state_dict()).
3. train.py save/load: цикл по реестру вместо 12 ручных блоков; plain attrs → register_buffer.
4. Тест-инвариант: roundtrip state_dict всех механизмов (все ключи чекпоинта восстанавливаются
   бит-в-бит) — генерируется из реестра, новые механизмы покрываются автоматически.
5. MechanismBundle.from_config генерируется из того же реестра (устранение дубля ~50 параметров).

## Acceptance criteria (бинарные)
- [ ] `pytest tests/ -q` зелёный; новое число passed ≥ старого (регресс-тесты реестра добавлены).
- [ ] Новый тест: для каждого механизма из реестра roundtrip save→corrupt→restore бит-в-бит (0 ключей потеряно).
- [ ] `git diff --stat` → train.py save/load сократился ≥200 строк.
- [ ] B10-координаты (train.py:135-155) исчезают: grep "wsd_target_cov" находит сохранение в реестре.
- [ ] ruff clean; README-секция «Adding a mechanism» добавлена.

## Риски
Миграция чекпоинт-формата — НЕ делать (ключи сохранить 1-в-1, иначе старые чекпоинты отваливаются);
покрыть тестом на загрузку legacy-чекпоинта.
