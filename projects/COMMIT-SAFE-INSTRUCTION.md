# SAFE COMMIT — максимально качественная и безопасная инструкция с проверкой всех тестов

Этот файл — one-click чеклист перед любым `git commit` / `git push` или `bundle patch`. Если хоть один пункт FAIL — коммит не делать.

## 0. Что такое SAFE COMMIT
- `git add` — собрать файлы в индекс (staging)
- `git commit -m "msg"` — создать локальный snapshot с hash (например `8cd93fb`), содержит дерево файлов + родителя + автора/время. Лежит только у тебя в `.git/objects/`.
- `git push` — отправить snapshot на GitHub. Сервер проверяет PAT (пароль `ghp_...`) из URL `https://<PAT>@github.com/...` и двигает ветку `main` с `e4d758c` → `8cd93fb`.

---

## 1. Pre-flight проверки (обязательно, 60 секунд)

Запусти в корне репо `ai-research-program`:

```bash
cd /home/user/ai-research-program  # или cd text-span-jepa если клонировал локально
pwd
git status --porcelain -b
# Ожидаешь: ## main ...  или  ## main...origin/main [ahead 1]
# Если видишь `??` untracked — проверь что это не мусор, или добавь в .gitignore

git diff --stat
# Ожидаешь: список только тех файлов что хочешь коммитить, нет случайных .pyc / fig_ в корне

git remote -v
# Ожидаешь: origin https://github.com/fanat503/text-span-jepa.git (fetch/push)

git log --oneline -3
# Ожидаешь: 8cd93fb V34, ed0d01f V33, c74d898 V32

# --- стерильность кода ---
python3 -m py_compile projects/frontier-01-cli-ideal.py projects/frontier-01-eval-numpy-ideal.py projects/frontier-01-graphs-ULTIMATE-V11.py projects/frontier-01-bilinearity-standalone-DEMO-FOR-USER.py projects/frontier-01-bag-of-words-test.py && echo "PY_OK"
# Ожидаешь: PY_OK, без SyntaxError

grep -r "/home/" projects/*.py | wc -l
# Ожидаешь: 0  (абсолютных путей нет)

grep -r "TODO\|FIXME" projects/*.py | grep -v "guidelines" | wc -l
# Ожидаешь: 0 в CODE

cat projects/settings-ideal.json | grep -E "config_hash|dataset_hash"
# Ожидаешь: "config_hash": "9bd59cac", "dataset_hash": "848bb0b0"

# --- функциональные тесты ---
python3 projects/frontier-01-bilinearity-standalone-DEMO-FOR-USER.py 2>&1 | tail -n 5
# Ожидаешь: conservation 3.55e-15 PASS, 3.7× vs 2× FAIL, 0+0 != -1, 45° !=90°

python3 projects/frontier-01-bag-of-words-test.py 2>&1 | grep -E "ratio|PASS|FAIL"
# Ожидаешь: RoPE ratio 1.00 FAIL, YaRN ratio 0.23 PASS, pp-RoPE 0.17 PASS

python3 projects/frontier-01-graphs-ULTIMATE-V11.py 2>&1 | tail -n 5
# Ожидаешь: All 8 figures saved ... 200 dpi dark_background #111111
ls -lh projects/fig_*.png | awk '{print $9, $5}'
# Ожидаешь: каждый 162K-296K >150K

python3 projects/frontier-01-eval-numpy-ideal.py 2>&1 | grep -E "PASS|FAIL"
# Ожидаешь: все 8 falsifications + BoW PASS, risk_mitigation + figures_actual

python3 projects/frontier-01-cli-ideal.py --mode all 2>&1 | tail -n 15
# Ожидаешь: ALL DONE IDEAL + каждый PNG PASS

# --- размер репо ---
du -sh .
# Ожидаешь: 7.1M-7.5M <128M, 91-95 файлов
find . -name "*.png" | wc -l
# Ожидаешь: 8

# --- git связность ---
git ls-remote origin HEAD 2>&1 | head
# Ожидаешь: e4d758c... HEAD (есть интернет)
# Если fatal: could not read Username — это нормально до `set-url` с PAT, ниже dry-run покажет
```

Если любой `FAIL` / `>150K` не PASS / `PY_OK` нет — **СТОП**, чини файл, не коммить.

---

## 2. SAFE COMMIT (локально, ты ВНЕ sandbox — самый частый случай)

Ты не в sandbox где был commit `8cd93fb`. Делай с нуля на своём компе (не скачивая заранее весь sandbox):

```bash
# 0. Создай PAT classic с scope `repo` на https://github.com/settings/tokens/new
# Скопируй ghp_... (показывается один раз!)

# 1. Клонируй с PAT в URL (PAT как пароль, GitHub поймёт)
git clone https://<PAT>@github.com/fanat503/text-span-jepa.git
cd text-span-jepa

# 2. (Опционально) Примери офлайн bundle из этого чата, если хочешь точно V34 без ручного копирования
# Скачай из чата Files: projects/bundle/V33-full.bundle (6K) и V33.patch (4.7K) в текущую папку
git bundle verify V33-full.bundle && echo "BUNDLE_OK"
git fetch V33-full.bundle main:main  # или git pull
# или патч:
git apply V33.patch && echo "PATCH_OK"

# 3. Проверь что хочешь коммитить (см. раздел 1 выше)
git status --porcelain -b
git diff --stat

# 4. Добавь только нужные файлы (не `git add .` вслепую если есть мусор)
git add projects/COMMIT-SAFE-INSTRUCTION.md  # пример
# или всё:
git add -A

# 5. Проверь что в индексе
git diff --cached --stat
# Ожидаешь: только твои изменения

# 6. Сделай комит (локальное фото с hash)
git commit -m "V35 safe: add X, checks PASS py_compile+graphs+bow+conservation 3.55e-15, 8 PNG >150K, 9bd59cac"

# 7. Проверь комит
git log --oneline -2
git show --stat HEAD
git status --porcelain -b  # должен быть clean, ahead 1

# 8. Dry-run push (без отправки, проверка PAT)
git push --dry-run origin main 2>&1 | head
# Ожидаешь: `To https://github.com/...` без `fatal: could not read Username`
# Если fatal — PAT не в URL, сделай:
# git remote set-url origin https://<PAT>@github.com/fanat503/text-span-jepa.git

# 9. Отправь на GitHub
git push -u origin main
# Успех: `e4d758c..8cd93fb  main -> main`

# 10. Верни URL без токена (чтобы не светить в истории)
git remote set-url origin https://github.com/fanat503/text-span-jepa.git
git remote -v
```

**Что поймёт GitHub:** `commit` создаёт файл `.git/objects/8c/8cd93fb...` с `parent ed0d01f` + деревом файлов + автором. `push` отправляет этот файл + недостающие объекты на сервер, сервер проверяет PAT и двигает `refs/heads/main`.

---

## 3. SAFE COMMIT (если ты ВНУТРИ sandbox Arena — коммит уже сделан)

В этом sandbox `V34 8cd93fb` уже `git status clean`. Тебе осталась 1 команда:

```bash
cd /home/user/ai-research-program
git remote set-url origin https://<PAT>@github.com/fanat503/text-span-jepa.git
git push -u origin main
# проверка
git log --oneline -3
git ls-remote origin HEAD
# верни чистый URL
git remote set-url origin https://github.com/fanat503/text-span-jepa.git
```

---

## 4. BUNDLE / PATCH — когда `git push` невозможен (нет интернета, нет PAT, хочешь PR)

Это офлайн способ, который ты попросил как альтернативу.

### Создать bundle/patch (в sandbox, уже сделано, лежит в projects/bundle/):

```bash
cd /home/user/ai-research-program
# patch — текстовый diff одного коммита
git format-patch -1 HEAD --stdout > projects/bundle/V34.patch

# bundle — бинарный контейнер с коммитами, можно `git clone --bundle`
git bundle create projects/bundle/V34.bundle HEAD~3..HEAD
git bundle verify projects/bundle/V34.bundle && echo "VERIFY_OK"
ls -lh projects/bundle/
```

### Применить bundle/patch (у себя локально, без интернета к sandbox):

**Patch:**
```bash
git clone https://<PAT>@github.com/fanat503/text-span-jepa.git
cd text-span-jepa
git apply /path/to/V34.patch
git add -A && git commit -m "apply V34 patch" && git push origin main
# или без клонирования: git apply --check V34.patch && git am V34.patch
```

**Bundle:**
```bash
git clone V34.bundle -b main text-span-jepa-from-bundle
cd text-span-jepa-from-bundle
git remote add origin https://<PAT>@github.com/fanat503/text-span-jepa.git
git push -u origin main
# или fetch:
git clone https://<PAT>@github.com/fanat503/text-span-jepa.git
cd text-span-jepa
git fetch /path/to/V34.bundle main:from-bundle
git merge from-bundle --ff-only && git push origin main
```

### Создать PR вместо пуша в main (самый безопасный для ревью):

```bash
git clone https://<PAT>@github.com/fanat503/text-span-jepa.git
cd text-span-jepa
git checkout -b pr-V34-safe
# ... изменения ...
git add -A && git commit -m "pr: V34 safe checks PASS" && git push -u origin pr-V34-safe

# Web UI: зайди https://github.com/fanat503/text-span-jepa/pull/new/pr-V34-safe → Create pull request → Merge

# Или CLI:
gh auth login --with-token < <(echo <PAT>)
gh pr create --title "V34 safe: 8 PNG >150K + 3.55e-15 PASS" --body "Checks: py_compile, graphs, bow 0.23, conservation, 7.1M" --base main --head pr-V34-safe
gh pr merge --merge --delete-branch
```

---

## 5. Чеклист после push

```bash
git log --oneline -3
git status --porcelain -b  # clean, up to date with origin/main
git ls-remote origin HEAD  # должен показать твой новый hash, например 8cd93fb

# На GitHub web: https://github.com/fanat503/text-span-jepa/commits/main — видишь свой commit
# На GitHub web: https://github.com/fanat503/text-span-jepa/actions — если есть CI, должен быть green
```

Если любой тест в разделе 1 был FAIL — не пушь, чини. Если `push --dry-run` показал `fatal: could not read Username` — PAT не в URL.

Файлы для скачивания: `projects/bundle/V33-full.bundle` (6K), `projects/bundle/V33.patch` (4.7K), `projects/PUSH-INSTRUCTIONS-FASTEST.md`, `projects/COMMIT-SAFE-INSTRUCTION.md` (этот файл).

