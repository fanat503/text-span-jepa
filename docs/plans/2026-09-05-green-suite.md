# PLAN: Green Suite — все тесты зелёные без skip/xfail

- **Date:** 2026-09-05
- **Status:** done (DoD составлен; вердикт ревью — self-APPROVE с отклонением: агент-инфраструктура
  недоступна 6 ретраев × `model concurrency limit exceeded`; эскалация пользователю — принять self-review
  или потребовать повторный прогон @code-reviewer/@security-auditor/@test-writer при освобождении лимита)
- **Ветка:** `refactor/analysis` (без коммитов до явной команды пользователя)
- **Конвейер:** $spec-to-pr (штаб-режим, глубина важнее скорости)

## Scope
Два падения/скипа в tests/, оба фиксятся **только в тестовом коде**; `src/` не трогаем.

**Запреты:** ослаблять/удалять ассерты запрещено; минимальные диффы (один root cause = один дифф);
попутные баги — в «выявленное попутное», не в код; БЕЗ КОММИТА до явной команды.

## Отклонение от конвейера (документировано)
@architect недоступен (3× `model concurrency limit exceeded`) → план составлен дирижёром, улучшен до v2.
Root-cause расследование проведено 2 Explore-ресёрчерами **по протоколу @debugger**
(repro → timeline → гипотезы с trade-offs → root cause одним предложением → evidence log).

---

## Root causes (доказаны, file:line)

### F1. FAILED `tests/test_pcr.py::TestPCRIntegration::test_checkpoint_pcr_roundtrip`
- **Repro:** `pytest tests/test_pcr.py::TestPCRIntegration::test_checkpoint_pcr_roundtrip -x` →
  `RuntimeError: [enforce fail at inline_container.cc:745] . open file failed with error code: 32`
- **Timeline:** `tests/test_pcr.py:465` — `with tempfile.NamedTemporaryFile(suffix=".pth") as f:`
  (delete=True по умолчанию, handle ОТКРЫТ весь with-блок) → `:466` `save_checkpoint(f.name, ...)`
  → `src/train.py:190` `torch.save(state, path)` — C++ PyTorchStreamWriter **переоткрывает путь**,
  Windows запрещает reopen-on-write при открытом handle → ERROR_SHARING_VIOLATION (32).
  На Linux (CI, ubuntu-latest, ci.yaml:16,31) reopen легален → CI зелёный.
- **Root cause (одно предложение):** это **единственный тест в репо**, передающий `f.name` открытого
  `NamedTemporaryFile(delete=True)` в `save_checkpoint`, тогда как весь остальной репо использует
  `tmp_path` (test_model.py:1150, test_training_e2e.py:163,182), `TemporaryDirectory`
  (test_model.py:1920) или `NamedTemporaryFile(delete=False)`+close+unlink
  (test_v025_integration.py:579-599).
- **Дополнительно (v2):** найдена слабость самого теста — между `save` и `load` in-memory state НЕ
  портится, поэтому `assert allclose(workspace_Q, original_Q)` прошёл бы даже при no-op `load_checkpoint`
  (PCR-часть теста недискриминирующая). Контракт подтверждён: save пишет `pcr_workspace_Q` +
  `pcr_level_gates` (src/train.py:126-128), load восстанавливает оба (src/train.py:237-242) и возвращает
  `(epoch, global_step, ema_step, mask_step, extra_state)`.
- **Фикс (A+, минимальный и идиоматичный):** pytest-фикстура `tmp_path` + corrupt-before-load
  (приём CGN-теста, tests/test_training_e2e.py:197-199) + ненулевые счётчики шагов + ассерты
  восстановления Q, level_gates и возвращённых счётчиков.
- Отвергнутые: B (delete=False + unlink — больше boilerplate, строго хуже A);
  C (file-object в torch.save — требует seek(0), ломает единообразие, diagnostic-path в torchio);
  D (atomic-save hardening в src/train.py — библиотека корректна, это попутное, не фикс регрессии).

### F2. SKIPPED `tests/test_spc.py::TestSPCIntegration::test_spc_bfloat16`
- **Evidence:** `tests/test_spc.py:425-426` — `if not torch.cuda.is_available(): pytest.skip(...)`.
  Тест мёртв **везде**: CI ставит CPU-torch (ci.yaml:26-27), локальный torch 2.13.0+cpu.
  Это единственный условный skip во всём suite (grep по tests/: других нет; xfail/importorskip отсутствуют).
- **Тело device-агностично:** `SpectralPredictiveCoding(64, 8).cuda().bfloat16()`, forward на bf16,
  единственный ассерт `loss.item() >= 0` (:433). В `src/models/spc.py` нет CUDA-специфики;
  bf16-лосс накапливается в fp32 (spc.py:347). CPU bf16 подтверждён экспериментально
  (forward/backward/stiefel_retract/autocast — работают; loss=15.29 ≥ 0).
- **Root cause:** тест написан под Kaggle T4 и **ни в одной среде не исполняется ни разу** —
  его проверки мертвы; guard-условие активируется всегда.
- **Фикс [DECISION-SPC: РЕШЁН — ДА]:** убрать GPU-guard, `.cuda()`/`device="cuda"` → CPU;
  ассерты сохранены + ДОБАВлены (`not isnan`, + autocast-подпроверка, зеркалящая продакшн-путь
  src/train.py:1026-1030). Усиление: тест из неисполняемого становится живым.

---

## Изменения (минимальные диффы)

### D1: tests/test_pcr.py (F1) — v2, усилен
```diff
-    def test_checkpoint_pcr_roundtrip(self):
+    def test_checkpoint_pcr_roundtrip(self, tmp_path):
         """PCR state should survive checkpoint save/load."""
-        import tempfile
-
         from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
         from src.train import load_checkpoint, save_checkpoint
@@
         # Modify PCR state
         with torch.no_grad():
             model.pcr.workspace_Q.add_(torch.randn_like(model.pcr.workspace_Q) * 0.01)
         original_Q = model.pcr.workspace_Q.data.clone()
+        original_gates = [g.data.clone() for g in model.pcr.level_gates]
 
-        with tempfile.NamedTemporaryFile(suffix=".pth") as f:
-            save_checkpoint(f.name, model, optimizer, None, 0, 0, 0, 0, model_name="text_span_jepa")
-            load_checkpoint(f.name, model, optimizer, None, model_name="text_span_jepa")
+        ckpt_path = str(tmp_path / "pcr-ckpt.pth.tar")
+        save_checkpoint(ckpt_path, model, optimizer, None, 2, 137, 99, 55, model_name="text_span_jepa")
+
+        # Corrupt in-memory state AFTER save: load must restore from the checkpoint,
+        # otherwise this assert is a no-op tautology (state was never changed between save/load)
+        with torch.no_grad():
+            model.pcr.workspace_Q.add_(torch.randn_like(model.pcr.workspace_Q) * 0.5)
+            for g in model.pcr.level_gates:
+                g.data.add_(torch.randn_like(g.data) * 0.5)
+
+        loaded = load_checkpoint(ckpt_path, model, optimizer, None, model_name="text_span_jepa")
 
         assert torch.allclose(model.pcr.workspace_Q.data, original_Q, atol=1e-5)
+        for restored, saved in zip(model.pcr.level_gates, original_gates):
+            assert torch.allclose(restored.data, saved, atol=1e-5)
+        epoch, global_step, ema_step, mask_step, extra_state = loaded
+        assert (epoch, global_step, ema_step, mask_step) == (2, 137, 99, 55)
+        assert extra_state is None
```
Ассерты: 1 → 4 (allclose Q — существующий; level_gates, счётчики, extra — добавлены).
Канон репо: tests/test_training_e2e.py:163-182 (CGN roundtrip через tmp_path) + corrupt-before-load (:197-199).

### D2: tests/test_spc.py (F2) — [DECISION-SPC: ДА]
```diff
     def test_spc_bfloat16(self):
-        """SPC works with bfloat16 (Kaggle T4 compatibility)."""
-        if not torch.cuda.is_available():
-            pytest.skip("No GPU available for bfloat16 test")
+        """SPC works with bfloat16 (CPU bf16 mirrors the CUDA-only autocast path in train.py)."""
         from src.models.spc import SpectralPredictiveCoding
 
-        spc = SpectralPredictiveCoding(embed_dim=64, n_bands=8).cuda().bfloat16()
-        z_pred = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
-        z_target = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
+        spc = SpectralPredictiveCoding(embed_dim=64, n_bands=8).bfloat16()
+        z_pred = torch.randn(2, 64, dtype=torch.bfloat16)
+        z_target = torch.randn(2, 64, dtype=torch.bfloat16)
         loss, _info = spc(z_pred, z_target)
         assert loss.item() >= 0
+        assert not torch.isnan(loss)
+
+        spc_fp32 = SpectralPredictiveCoding(embed_dim=64, n_bands=8)
+        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
+            loss_ac, _info_ac = spc_fp32(z_pred.float(), z_target.float())
+        assert loss_ac.item() >= 0
```
Ассерты: 1 → 3 (усиление). Дифф не ослабляет ни одну проверку.

---

## Acceptance criteria (бинарные, машино-проверяемые)

- [ ] **AC1:** `python -m pytest tests/ -q` exit code 0; последняя строка вывода: `680 passed, 0 failed, 0 skipped`
      (0 skipped и 0 xfailed/xpassed — парсинг финальной строки).
- [ ] **AC2:** `python -m ruff check .` → `All checks passed!` (exit 0).
- [ ] **AC3:** `git diff --name-only` ⊆ {`tests/test_pcr.py`, `tests/test_spc.py`} — src/ не тронут.
- [ ] **AC4:** число `assert` в каждом изменённом тесте не уменьшилось
      (D1: 1→4; D2: 1→3; проверяемо подсчётом по diff).
- [ ] **AC5:** mutation-verdict на каждый фикс — **протокол revert-verify** (Stage 3): для каждого диффа
      временно откатить файл к HEAD, запустить точечный тест, зафиксировать вывод (fail/skip = дискриминирует),
      восстановить фикс. Выводы paste'ом в DoD.
- [ ] **AC6:** `pytest.skip`/`xfail` отсутствуют в изменённых файлах (grep).
- [ ] **AC7:** оба mutation-вывода зафиксированы в DoD-отчёте (D1: RuntimeError 32 на HEAD-версии;
      D2: `1 skipped` на HEAD-версии).

## Стадии

1. **Stage 2 — Implement:** D1 и D2 критерий за критерием, минимальными диффами.
2. **Stage 3 — Mutation protocol + Tester:** revert-verify обоих фиксов (AC5/AC7) + @test-writer с
   контекст-пакетом (PLAN + файлы) проверяет дискриминирующую силу каждого теста; вердикт
   «упал бы без фикса, потому что …». Декор — назад.
3. **Stage 4 — Гейт:** полный `pytest tests/ -q` + `ruff check .`, вывод paste'ом. Красное → следствие (@debugger-протокол) → фикс → гейт заново.
4. **Stage 5 — Dual-pass review:** @code-reviewer (Pass 1: каждый AC с доказательством; Pass 2: качество/регрессии/минимальность). REQUEST_CHANGES → фикс → гейт → ревью. Максимум 2 круга → эскалация.
5. **Security:** @security-auditor по диффу (дифф трогает torch.save/torch.load использование в тестах; FIX-FIRST блокирует).
6. **Docs:** публичное поведение не меняется (тесты only) → «docs: не требуется»; проверить README на упоминания GPU/Windows-требований.
7. **Stage 7 — DoD-отчёт** по шаблону; PLAN → `status: done`. БЕЗ КОММИТА — жду явную команду.

## Handoff
- **Статус:** все фиксы и гейт завершены (AC1-AC7 [x], 680 passed / 0 skipped, ruff clean, mutation
  revert-verified, DoD в analysis/dod-green-suite.md). Шаг 10 цели: 5 улучшений оформлены КАЖДОЕ
  отдельным планом: docs/plans/2026-09-05-improve-{1-mechanism-registry,2-train-loop-hygiene,
  3-mechanism-science-fixes,4-interp-common,5-ci-docs}.md (status: draft, вне этой задачи).
- **В процессе → ЗАБЛОКИРОВАНО инфраструктурой:** агентские ревью. @test-writer — ГОТОВ (вердикт:
  «2 из 2 дискриминирующих, декораций нет, ослаблений нет» + мутационный эксперимент). @code-reviewer
  (dual-pass) и @security-auditor — ~15 попыток за >1 часа (включая тест кулдауна 2.5 мин),
  все `model concurrency limit exceeded`; окно лимитёра не пускает суб-агентов (квота аккаунта
  исчерпана объёмом сессии).
- **Следующий шаг (эскалация пользователю, A/B):** A — принять DoD с self-APPROVE дирижёра
  (полные доказательства в analysis/dod-green-suite.md) + агентский вердикт test-writer; B —
  команда «повтори агент-ревью позже» (когда квота восстановится: @code-reviewer dual-pass +
  @security-auditor по тому же диффу, дифф не меняется). БЕЗ КОММИТА в обоих случаях.
  Блокировка распространяется и на оставшуюся большую задачу (train+config воркер, 8 аудит-агентов,
  рефакторинг-конвейер) — все они требуют суб-агентов.

## Выявленное попутное (НЕ чиним в этой задаче)
1. В CI нет Windows-job — класс багов «тесты падают только на Windows» не ловится вовсе (ci.yaml:16,31 ubuntu-only).
   Спека follow-up: отдельный план — job `windows-latest` (CPU-torch, python 3.10-3.12) на подмножество
   чекпоинт-тестов (test_pcr, test_model::TestCheckpoint, test_training_e2e, test_v025_integration).
2. `save_checkpoint` не атомарен (tmp+rename) — crash-safety follow-up.
3. Прочие находки анализа (src/interp дубли, eval/probes leakage и т.д.) — в ANALYSIS_REPORT отдельной задачи, сюда не мешаем.
