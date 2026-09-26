# SAFE COMMIT в GitHub Codespace — патч уже добавлен в репо

Ты добавил `*.patch` в репо и открыл Codespace (VS Code в браузере, терминал уже авторизован — `git push` работает без PAT, т.к. Codespace имеет токен GitHub).

Это максимально качественная и безопасная инструкция с проверкой всех тестов перед `commit`. Если хоть один FAIL — не коммить.

---

## 0. Где ты находишься

- Codespace: `text-span-jepa` уже склонирован в `/workspaces/text-span-jepa` (или `/workspaces/...`)
- Патч уже в репо: `projects/bundle/V33.patch` или `V33-full.bundle` или твой `*.patch` в корне. `git status` покажет его как `??` или `M`.
- Терминал: `Terminal → New Terminal` в VS Code Codespace

---

## 1. Pre-flight проверки патча (30 сек, обязательно)

```bash
cd /workspaces/text-span-jepa  # или `pwd` покажет где ты, если `ai-research-program` — адаптируй путь
pwd
ls -lh *.patch projects/bundle/*.patch 2>&1 | head -n 20
cat projects/bundle/V33.patch 2>&1 | head -n 30  # или твой patch

# Проверка что патч применяется чисто (dry-run, не меняет файлы)
git apply --check projects/bundle/V33.patch 2>&1 && echo "PATCH_CHECK_OK" || echo "PATCH_CHECK_FAIL"

# Если FAIL — не применяй, открой патч и посмотри `git diff` внутри:
# git apply --reject projects/bundle/V33.patch  # создаст *.rej файлы для ручной правки
```

---

## 2. Применить патч (выбери один способ)

**Способ A — `git apply` (самый безопасный, не создаёт комит автоматически, ты сам контролируешь):**
```bash
git status --porcelain -b
git diff --stat

git apply projects/bundle/V33.patch 2>&1 && echo "APPLY_OK"

git status --porcelain -b
git diff --stat
# Ожидаешь: изменённые файлы `M projects/...` — только те что в патче, нет `.pyc` / `fig_` в корне
```

**Способ B — `git am` (создаёт комит автоматически из патча с сообщением, быстрее):**
```bash
git am --3way projects/bundle/V33.patch 2>&1 && echo "AM_OK"
# Если конфликт: git am --abort, исправь руками, затем git am --continue
git log --oneline -2
```

**Способ C — bundle (если у тебя `.bundle` а не `.patch`):**
```bash
git bundle verify projects/bundle/V33-full.bundle && echo "BUNDLE_OK"
git fetch projects/bundle/V33-full.bundle main:from-bundle 2>&1 | head
git log --oneline from-bundle -2
git merge from-bundle --ff-only 2>&1 | head
# или git checkout -b pr-from-bundle from-bundle
```

**Для тебя (патч уже добавлен в репо и делаю в Codespace) — рекомендуется Способ A.**

---

## 3. Полная проверка всех тестов (60 сек, обязательно перед commit)

Запусти в корне репо Codespace:

```bash
cd /workspaces/text-span-jepa  # проверь pwd

# --- стерильность кода ---
python3 -m py_compile projects/frontier-01-cli-ideal.py projects/frontier-01-eval-numpy-ideal.py projects/frontier-01-graphs-ULTIMATE-V11.py projects/frontier-01-bilinearity-standalone-DEMO-FOR-USER.py projects/frontier-01-bag-of-words-test.py && echo "PY_OK"
# Ожидаешь: PY_OK

grep -r "/home/" projects/*.py 2>&1 | wc -l
# Ожидаешь: 0

grep -r "TODO\|FIXME" projects/*.py 2>&1 | grep -v "guidelines" | wc -l
# Ожидаешь: 0 в CODE

cat projects/settings-ideal.json 2>&1 | grep -E "config_hash|dataset_hash"
# Ожидаешь: "config_hash": "9bd59cac", "dataset_hash": "848bb0b0"

# --- функциональные тесты ---
python3 projects/frontier-01-bilinearity-standalone-DEMO-FOR-USER.py 2>&1 | tail -n 6
# Ожидаешь: conservation 3.55e-15 <1e-10 PASS, 3.7× vs 2× FAIL, 0+0 != -1, 45° !=90°

python3 projects/frontier-01-bag-of-words-test.py 2>&1 | grep -E "ratio|PASS|FAIL"
# Ожидаешь: RoPE ratio 1.00 FAIL, YaRN ratio 0.23 PASS, pp-RoPE 0.17 PASS

python3 projects/frontier-01-graphs-ULTIMATE-V11.py 2>&1 | tail -n 6
# Ожидаешь: All 8 figures saved ... 200 dpi dark_background #111111
ls -lh projects/fig_*.png 2>&1 | awk '{print $9, $5}'
# Ожидаешь: каждый 162K-296K >150K (289K, 290K, 285K, 168K, 213K, 185K, 162K, 273K)

python3 projects/frontier-01-eval-numpy-ideal.py 2>&1 | grep -E "PASS|FAIL|risk_mitigation|figures_actual" | head -n 20
# Ожидаешь: все 8 falsifications + BoW PASS + risk_mitigation + figures_actual

python3 projects/frontier-01-cli-ideal.py --mode all 2>&1 | tail -n 20
# Ожидаешь: ALL DONE IDEAL + каждый PNG PASS

# --- размер репо ---
du -sh . 2>&1
# Ожидаешь: 7.1M-7.5M <128M
find . -name "*.png" -type f 2>&1 | wc -l
# Ожидаешь: 8

# --- git связность (в Codespace push работает без PAT) ---
git status --porcelain -b
git remote -v
git ls-remote origin HEAD 2>&1 | head
# Ожидаешь: e4d758c... HEAD
```

Если любой `FAIL` / `>150K` не PASS / `PY_OK` нет — **СТОП**, чини файл, не коммить.

---

## 4. SAFE COMMIT в Codespace (патч уже применён)

```bash
cd /workspaces/text-span-jepa

# 1. Посмотри что изменилось после apply
git status --porcelain -b
git diff --stat
# Ожидаешь: только файлы из патча, нет случайных `__pycache__` / `*.pyc` / `fig_*.png` в корне

# 2. Добавь ТОЛЬКО нужные файлы (не `git add .` вслепую если есть мусор)
git add projects/bundle/V33.patch projects/bundle/V33-full.bundle  # пример если патч это файл который ты хочешь закоммитить
# или если патч уже применён и изменил код:
git add -A
# или точечно:
git add projects/frontier-01-cli-ideal.py projects/frontier-01-eval-numpy-ideal.py projects/frontier-01-graphs-ULTIMATE-V11.py projects/frontier-01-proofs-ideal.md projects/settings.json

# 3. Проверь что в индексе (staging)
git diff --cached --stat
# Ожидаешь: только твои изменения, которые проверил в разделе 3

# 4. Сделай комит (локальное фото с hash, лежит только в Codespace)
git config user.name  # должен быть твой GitHub user, в Codespace уже настроен
git config user.email
# если пусто:
git config --global user.name "fanat503"
git config --global user.email "fanat503@users.noreply.github.com"

git commit -m "V36 safe Codespace: apply patch + checks PASS py_compile+3.55e-15+3.7×+sigma2 0.23+8 PNG>150K 9bd59cac"

# 5. Проверь комит
git log --oneline -2
git show --stat HEAD
git status --porcelain -b  # должен быть clean, ahead 1

# 6. Dry-run push (без отправки, проверка прав)
git push --dry-run origin main 2>&1 | head
# Ожидаешь: `To https://github.com/fanat503/text-span-jepa.git` без `fatal` / `403`
# В Codespace `fatal: could not read Username` НЕ будет — токен уже встроен

# 7. Отправь на GitHub (в Codespace не нужен PAT в URL)
git push -u origin main
# Успех: `e4d758c..a3c438e  main -> main`  (hash твоего нового коммита)

# 8. Проверка после push
git log --oneline -3
git status --porcelain -b  # clean, up to date with origin/main
git ls-remote origin HEAD 2>&1 | head  # должен показать твой новый hash
# На GitHub web: https://github.com/fanat503/text-span-jepa/commits/main — видишь свой commit
```

---

## 5. PR вместо пуша в main (самый безопасный для Codespace, рекомендуется)

Если боишься ломать `main`, делай PR:

```bash
cd /workspaces/text-span-jepa
git checkout -b pr-V36-safe-patch
git apply projects/bundle/V33.patch && echo "APPLY_OK"
# ... проверки из раздела 3 ...
git add -A && git commit -m "pr: V36 safe patch + checks PASS" && git push -u origin pr-V36-safe-patch

# Создай PR (в Codespace `gh` уже установлен):
gh pr create --title "V36 safe Codespace: patch + 8 PNG >150K 3.55e-15 PASS" --body "Checks: py_compile, graphs 162K-296K, bow 0.23, conservation 3.55e-15, du 7.2M <128M" --base main --head pr-V36-safe-patch

# Или Web UI: зайди https://github.com/fanat503/text-span-jepa/pull/new/pr-V36-safe-patch → Create pull request → Merge pull request → Confirm merge → Delete branch
```

---

## 6. Что делать если патч не применяется

```bash
git apply --check projects/bundle/V33.patch  # FAIL
git apply --reject projects/bundle/V33.patch  # создаст *.rej
cat *.rej  # смотри конфликты
# Исправь руками файлы, затем:
git add -A && git commit -m "fix patch conflict"
# Для `git am`:
git am --abort
```

---

## 7. Чеклист после push

```bash
git log --oneline -3
git status --porcelain -b  # clean
git ls-remote origin HEAD  # новый hash
# Web: https://github.com/fanat503/text-span-jepa/commits/main
# Web: https://github.com/fanat503/text-span-jepa/actions — green если есть CI
```

Файлы для Codespace уже в репо: `projects/bundle/V33-full.bundle` (6K), `projects/bundle/V33.patch` (4.7K), `projects/COMMIT-SAFE-INSTRUCTION.md`, `projects/PUSH-INSTRUCTIONS-FASTEST.md`, этот файл `projects/COMMIT-SAFE-CODESPACE-PATCH.md`.

Если любой тест в разделе 3 был FAIL — не пушь, чини.
