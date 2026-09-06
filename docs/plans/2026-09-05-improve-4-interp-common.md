# PLAN: Улучшение 4 — src/interp/common.py + единый train_linear_probe (устранение leakage-класса)

- **Date:** 2026-09-05 · **Status:** draft (вне green-suite) · **Приоритет:** P1
- **Оценка пользы:** −500-800 строк дублей; закрытие data-leakage в 3 местах (влияет на выводы
  статьи); единая точка правды для всех метрик. Трудоёмкость: средняя (2-3 дня), риск низкий
  (библиотечная консолидация, API сохраняется через реэкспорты).

## Проблема (доказательства)
- **5 копий тренировочного цикла линейных проб**: layer_analysis.py:47-89 (80/20 сплит),
  probe_generalization.py:45-86 (**без сплита, best по train**), probe_generalization.py:203-237,
  probing_complexity.py:94-163 (**эталон**: сплит, wd, cosine LR, best-state restore),
  src/eval/probes.py:24-57 (**без сплита, eval на train loader'е**).
- **Leakage**: probe_generalization cross_dataset_generalization называет train accuracy
  «source_accuracy» (114-127); eval/probes.py LinearProbe и FutureTokenProbe (accuracy копится
  во время обучения, 78-103) — все reported accuracy это train-accuracy.
- **7 копий «CKA после flatten + min-N»**: layer_analysis.py:189-193,218-222; robustness.py:89-93;
  stability.py:58-63,236-240,303-307; ablation.py:426.
- **2 SAE с несовместимыми API**: sae.SparseAutoencoder (encode→3-tuple) vs workspace_validation.TopKSAE
  (2-tuple) — feature_composition.py:75-77 с TopKSAE молча развалится; SAETrainer.load не загружает
  веса (sae.py:271-283); TopKSAE не экспортирован из __init__.
- **Статистические баги**: Cliff's delta ×2 (statistical_tests.py:336-361), BH-p не монотонны
  (:281-284), BootstrapCI.compare непарный (:109-161), DCI informativeness не по докстрингу
  (disentanglement.py:91-92).
- Прочие дубли: extract_representations ×2 (compare.py:32-71, run_comparison.py:66-92), spearman ×3,
  bootstrap ×2, 8 мёртвых CollapseDiagnostics().

## Решение
1. `src/interp/common.py`: `extract_pooled_reps`, `train_linear_probe(reps, labels, *, val_fraction,
   wd, scheduler, patience, seed)` (эталон probing_complexity: обязательный val-сплит, early stopping,
   best-state restore), `pairwise_cka_2d`, `spearman`, `pearson`, `bootstrap_ci`.
2. Все 5 probe-циклов → вызов common.train_linear_probe ( leakage закрывается автоматически).
3. eval/probes.py: LinearProbe/FutureTokenProbe → два лоадера (train/eval-сплит по индексам чанков),
   FutureTokenProbe — nn.Module с eval-фазой после обучения.
4. Один TopK-SAE: sae.SparseAutoencoder + адаптер encode()→(features, indices) в workspace_validation;
   фикс SAETrainer.load; экспорт workspace_validation из __init__.
5. Статистика: Cliff's delta без ×2; BH step-down монотонизация; paired-режим в BootstrapCI.compare;
   DCI по Eastwood & Williams (mean R² перфакторной регрессии).
6. Удалить 8 мёртвых CollapseDiagnostics(), дубли predictability (causal_scrubbing.InterventionPredictabilityScorer).

## Acceptance criteria (бинарные)
- [ ] grep "def _train_probe" в src/ → 1 определение (в common.py); все потребители импортируют его.
- [ ] Новый тест-регрессия: LinearProbe на labels, случайных относительно reps, даёт val-accuracy ≈
      chance (>0.9·chance против train-accuracy >> chance) — доказательство отсутствия leakage.
- [ ] Число строк src/interp + src/eval уменьшилось ≥400 (wc -l до/после).
- [ ] Cliff's delta на синтетике ∈ [-1,1] (тест); BH-коррекция монотонна по p (тест).
- [ ] Полный pytest зелёный, ruff clean; публичные API сохранены через реэкспорты (обратная совместимость).

## Риски
Изменение reported-чисел (val вместо train accuracy) — это и есть цель, но сравнение со старыми
результатами требует перегонки; задокументировать в плане-отчёте.
