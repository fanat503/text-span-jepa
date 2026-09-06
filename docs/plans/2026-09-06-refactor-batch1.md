# PLAN: Refactor Batch 1 — Quick Wins

- **Date:** 2026-09-06 · **Status:** approved (поведение сохраняется, полный гейт обязателен)
- **Ветка:** test/ci-gate → PR #2 · **Конвейер:** implement → full gate → review → commit/push (C1-C5)
- **Источник:** analysis/ANALYSIS_REPORT.md + planner-отчёт (проверен по коду, все координаты сверены)

## Изменения
### C1 security (torchio + тесты + .gitignore)
- torchio.py:18-28 → except pickle.UnpicklingError (не Exception); FileNotFoundError не даёт ложного warning; legacy-pickle fallback сохранён (совместимость).
- train.py:1403 `float(np.mean(val_losses))` — устраняет np.float64 в best_val_loss (корень weights_only-fallback в e2e).
- Тесты на safe_torch_load: test_model.py:1198 (+weights_only=True), :1224 (False→True), test_training_e2e.py:121-124 → safe_torch_load.
- НОВЫЙ tests/test_torchio.py (3 теста: clean без warning; legacy-pickle fallback с warning; missing file без fallback).
- .gitignore: .env, .env.*, results/, checkpoints/, *.safetensors, *.ckpt, *.pkl.

### C2 train (B2+B1, B10)
- train.py:723 → `scaler = torch.amp.GradScaler("cuda", enabled=False)` (+комментарий; bf16 — единственный AMP-режим, loss scaling не нужен; pass-through: scale=identity, get_scale=1.0, step=optimizer.step, state_dict={} — чекпоинт-пламбинг сохранён; ветки 1111-1114/1122/1159-1163 НЕ трогаем).
- train.py:1385-1400 `_validate(..., current_step=0, total_steps=1)` + вызов :1313-1315 передаёт global_step/total_steps (B10: future-вес честный).
- train.py:1175-1177 comment-only правка.
- НОВЫЙ tests/test_grad_scaler.py (2 теста: spy на конструктор — enabled никогда не True [наивный is_enabled() НЕ дискриминирует на CPU — torch молча отключает]; pass-through инвариант scale(t) is t) + test_validate_forwards_current_step.

### C3 hygiene
- __all__: src/masks/__init__.py (1), src/models/__init__.py (18 точных имён с импортов), src/utils/__init__.py (5).
- pyproject: удалить "F401" из lint.ignore (после __all__ — ruff --select F401 молчит).
- dict[str, any] → dict[str, Any] ×9: mechanisms.py:511 (Any уже импортирован); cmc.py:262, gac.py:145, rdc.py:126,246,258, puc.py:125,264,275 (+ `from typing import Any` в эти 4 файла).

### C4 deps
- pyproject: torch>=2.3.0 (фактический пол: train.py:723), numpy>=1.24.0,<3.0, transformers>=4.30.0,<6.0 (5.16.1 в окружении зелёная — отклонение от audit-deps `<5` задокументировано), scipy удалить из eval, black>=24.0 в dev.
- requirements.txt → `-e .[dev,eval]`.

### C5 autofix (отдельный коммит, ПОСЛЕ всех ручных)
`ruff check --fix --select UP015,SIM910,SIM300,C416,D413 .` → `black .` → `ruff check --fix --select COM812 .` → `black .` (465 находок, 464 auto-fixable, все сайты проинспектированы planner'ом — семантических рисков нет; C416 вручную не трогаем — не в дефолтном select).

### Бонус (в C3 или отдельный docs-коммит)
- DoD green-suite (analysis/dod-green-suite.md) → docs/plans/2026-09-05-green-suite-dod.md (non-blocking #1 ревьюера: план ссылается на DoD, которого нет в репо).

## Acceptance criteria (бинарные)
1. `pytest tests/ -q` → exit 0, **686 passed / 0 skipped** (680 + 6 новых), 0 failed.
2. `ruff check .` → exit 0 (F401 удалён из ignore; `ruff check --select F401 .` → 0).
3. `black --check .` → exit 0.
4. Grep-гейты: `GradScaler(` в train.py — единственная :723 с enabled=False; `current_step=global_step` в вызове _validate; `float(np.mean` :1403; `except Exception` в torchio — пусто; `dict[str, any]` в src/ — пусто; `"F401"` в pyproject — пусто.
5. e2e resume тест — passed БЕЗ `UserWarning: safe_torch_load ... UnpicklingError`.
6. Коммит-схема C1→C5 (C5 последним), пуш в PR #2 после зелёного гейта.

## Handoff
- Вне Batch 1: B12 (датасет), B13 (DDP), B19 (диагностика), B3/B8 (чекпоинты), M2 (CI), trusted-флаг, удаление scaler-объекта (У1), TextSpanJEPApredictor rename.
