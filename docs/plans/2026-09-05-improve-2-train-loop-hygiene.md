# PLAN: Улучшение 2 — Гигиена training loop: eval-безопасность + гейт диагностики

- **Date:** 2026-09-05 · **Status:** draft (вне green-suite) · **Приоритет:** P0
- **Оценка пользы:** чинит испорченные epoch-чекпоинты (корректность экспериментов) и OOM-хрупкость
  (блокер scaling-программы 140M/300M); −99% стоимости диагностики. Трудоёмкость: низкая (полдня),
  риск низкий.

## Проблема (доказательства)
- B9 (валидация мутирует состояние):
  - TargetCentering.forward → update_center на валидационных батчах (collapse.py:88-95, jepa.py:617);
  - WSD: step=0 % sync_interval==0 → update_target_cov+resync на val (wsd.py:229-237);
  - JAWP: compute_loss делает active_k.fill_(current_k(0)) (jawp.py:324), а epoch-чекпоинт
    сохраняется ПОСЛЕ валидации (train.py:1335) → jawp_active_k в каждом чекпоинте = k_start;
  - jspace._prev_jspace_vectors перезаписывается val-базисом.
- B19 (OOM): jepa.py:847-856 — CollapseDiagnostics+JSpaceMetrics каждый шаг; при B=64,T=512 N=32768:
  linear/rbf CKA строят N×N ядра (~4.3 ГБ каждый), ~12 svdvals (32768×768), SVCCA, subspace_overlap,
  matrix_rank. Гарантированный OOM/слайдер на small-конфиге.

## Решение
1. TargetCentering.update_center — только при self.training (1 строка + тест).
2. WSD.update_target_cov/resync — только при self.training (или флаг is_training из jepa).
3. JAWP compute_loss: не fill_ active_k при not self.training.
4. Диагностика: `diag_every: int` в конфиг (дефолт = log_freq), вызов в jepa.py:847-856 за гейтом;
   CKA/SVCCA/внутренние SVD — сабсэмплинг до ≤512 строк (паттерн уже есть в _uniformity).
5. _validate (train.py:1385-1403): гарантировать model.eval() и передачу реального step.

## Acceptance criteria (бинарные)
- [ ] Новый тест: после _validate centering.center, wsd.target_cov, jawp.active_k бит-в-бит равны pre-val.
- [ ] Новый тест: diag-вычисления выполняются только на шагах, кратных diag_every (мок-счётчик).
- [ ] E2E-прогон small-конфига не растёт по памяти >X (замер torch.max_memory_allocated до/после).
- [ ] Полный pytest зелёный, ruff clean; ни один существующий ассерт не ослаблен.

## Риски
Изменение поведения диагностики (реже) — логировать в README; численные результаты val-лосса
не меняются (центр на val теперь не мутирует — это и есть фикс; val-лосс на первом шаге может
отличаться — задокументировать в CHANGELOG плана).
