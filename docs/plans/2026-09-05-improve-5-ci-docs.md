# PLAN: Улучшение 5 — CI/доки: Windows-job, матрица, кэш, покрытие, маркеры + правка противоречий

- **Date:** 2026-09-05 · **Status:** draft (вне green-suite) · **Приоритет:** P1
- **Оценка пользы:** наибольший эффект на час вложения: закрывает доказанный класс Windows-багов,
  молчаливую поломку от новых torch, невидимость регрессий покрытия; снимает противоречия доков.
  Трудоёмкость: низкая (конфиг/доки, ~полдня), риск низкий.

## Проблема (доказательства)
- CI ubuntu-only (ci.yaml:16,31) — Windows-баг test_pcr (NamedTemporaryFile + torch.save → error 32)
  существовал в репо с коммита 86b53f7 и не ловился; py3.9+ заявлен (pyproject.toml:14), проверен
  только 3.11.
- Установка: `pip install -e ".[dev,eval]"` тянет CUDA-bundle torch с PyPI, затем заменяется CPU
  (ci.yaml:26-27) — гигабайты впустую; torch не запинен — новый релиз может молча сломать пайплайн.
- Нет: кэша pip, pytest-cov/покрытия, timeout-minutes/pytest-timeout, xdist, маркеров (0 зарегистрировано).
- Доки: README:1 «This repo was a bit edited by LLM…» — мусор; строка «novel mechanisms (16)» — сирота
  без таблицы; README («CGN and STA reconciled») противоречит proofs/IMPLEMENTATION_STATUS.md
  (Divergent 5-8 у всех; ни один не reconciled); proofs/README.md «All theorems are computationally
  verified» — противоречит матрице; wip.md отсутствует в матрице; CONTRIBUTING.md отсутствует;
  тавтологичные тесты (test_puc Donsker-Varadhan, test_spc optimal_weight_direction), пустые test_import
  (test_v025_integration.py), мёртвые выражения (test_interp:115, test_jawp:652-653, test_cmc:256-263).

## Решение
1. ci.yaml: job windows-latest (fast-подмножество: чекпоинт-тесты test_pcr/test_model::TestCheckpoint/
   test_training_e2e/test_v025_integration) + матрица python 3.10/3.11/3.12; `cache: 'pip'` в
   setup-python; CPU-torch УСТАНОВИТЬ ПЕРВЫМ; пин torch в CI-требованиях (отдельный constraints-файл).
2. pytest: pytest-cov (--cov=src --cov-report=xml, гейт src/models ≥85%), pytest-timeout (--timeout=300),
   pytest-xdist (-n auto); маркеры slow/e2e/gpu/theorem в pyproject; помечать тяжёлые
   (test_jawp CourantFischer, e2e, 300-шаговые лупы).
3. README: удалить LLM-строку; таблица 16 механизмов (файл, proof-ссылка, λ-флаг, статус из матрицы);
   arch-диаграмма GWP (core/routing/stability + GWP.dependency_dag()); секция оценки (train_probe.py);
   бейдж CI. Снять противоречие README↔IMPLEMENTATION_STATUS (обновить матрицу по CGN/STA после
   improve-3 или скорректировать README с пометкой); «All theorems verified» → честная формулировка;
   добавить wip.md в матрицу.
4. CONTRIBUTING.md: ruff/black, маркеры, как гонять тесты, добавление механизма (ссылка на improve-1).
5. Гигиена тестов: переписать тавтологии (test_puc: assert entropy_deficit>0 под if — заменить на
   сравнение с legacy-формулой), заполнить пустые test_import, удалить мёртвые выражения.

## Acceptance criteria (бинарные)
- [ ] ci.yaml содержит windows-latest job и матрицу ≥2 python-версий; job зелёный после первого прогона.
- [ ] Установка в CI: CPU-torch wheel скачивается ДО -e ".[dev,eval]" (порядок шагов в yaml).
- [ ] coverage.xml артефакт в CI; гейт ≥85% на src/models настроен.
- [ ] pytest --markers показывает ≥4 зарегистрированных маркера; тяжёлые тесты помечены.
- [ ] grep -i "edited by LLM" README.md — пусто; таблица механизмов ≥16 строк со ссылками на proofs/.
- [ ] Расхождение README↔IMPLEMENTATION_STATUS устранено (любая из сторон, с пометкой даты аудита).
- [ ] Полный pytest зелёный, ruff clean; ни один тест не удалён (только переписаны тавтологии с усилением).

## Риски
Windows-runner медленнее и flakier (внешние сервисы GitHub) — поэтому fast-подмножество, не полный suite;
покрытие может выявить слабые зоны — гейт вводить как warn первые 2 недели, потом блокирующий.
