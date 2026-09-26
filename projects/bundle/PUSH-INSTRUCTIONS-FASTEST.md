# PUSH — Самая быстрая инструкция (30 секунд, ничего не скачивая)

Ты уже сделал токен (PAT classic, scope `repo`). Коммит уже сделан в этом sandbox: `c74d898 V32`. Осталось только отправить.

## Вариант 1 — Самый быстрый (прямо из этого sandbox, без скачивания)

В терминале этого sandbox выполни **одну команду** (вставь свой PAT вместо `<PAT>`):

```bash
cd /home/user/ai-research-program
git remote set-url origin https://<PAT>@github.com/fanat503/text-span-jepa.git
git push -u origin main
# проверка
git log --oneline -3
git ls-remote origin HEAD
```

Что происходит: `git remote set-url` переписывает URL с `https://github.com/...` на `https://<PAT>@github.com/...`. GitHub понимает кто ты по PAT в URL, принимает объекты коммита `c74d898`, `dae0e88`, `5acc0e2` и обновляет ветку `main` на GitHub. Коммит уже есть в `.git` (hash `c74d898`), `push` просто отправляет его.

Проверка успеха: `git push` не выдаст `fatal: could not read Username`, а выдаст `To https://github.com/...  e4d758c..c74d898  main -> main`.

Если хочешь вернуть URL без токена после пуша (чтобы не светить):
```bash
git remote set-url origin https://github.com/fanat503/text-span-jepa.git
```

## Вариант 2 — Ты на своём компе и не хочешь клонировать весь репо

Если ты вообще ничего не хочешь скачивать, можешь пушить прямо из GitHub Web:

1. Скачай patch: в этом sandbox `/tmp/v32.patch` (18K) или `/tmp/v32-full.bundle` (167K) — кнопка Download в Files.
2. На локальном компе с git:
```bash
git clone https://<PAT>@github.com/fanat503/text-span-jepa.git
cd text-span-jepa
git apply /path/to/v32.patch   # или git bundle fetch /path/to/v32.bundle
git push origin main
```

## Вариант 3 — Через GitHub CLI (если есть `gh`)

```bash
gh auth login --with-token < /path/to/PAT.txt
gh repo sync
# или вручную как в Варианте 1
```

## Что уже готово в sandbox (не надо делать)

- `git config user.name` = `Vitebsk Research`, `user.email` = `research@vitebsk.by`
- `git add` 5 файлов, `git commit -m "V32 ..."` = `c74d898` — **уже выполнен**
- `git status` = clean, `du -sh` = 7.1M 91 files <128M, 8 PNG 162K-296K, py_compile PASS, hash `9bd59cac` `848bb0b0`
- `git remote -v` = `https://github.com/fanat503/text-span-jepa.git`
- `git ls-remote origin HEAD` = `e4d758c20447b1d248120f2589428c047aa0f63a` (есть интернет, читает)
- `git push --dry-run` = `fatal: could not read Username` — это нормально, нет PAT в URL, после `set-url` пройдёт.

## Почему GitHub поймёт коммит

Коммит — это объект в `.git/objects` с hash `c74d898`, содержит: дерево файлов, родитель `dae0e88`, автор, сообщение. `push` отправляет этот объект и все недостающие на сервер. Сервер проверяет PAT, принимает.

## Если PAT не работает

- Токен должен быть **classic**, scope `repo` (все галочки repo), не fine-grained.
- URL формат точно `https://<PAT>@github.com/fanat503/text-span-jepa.git` без пробелов, PAT 40 символов `ghp_...`
- Если ошибка `403` — токен истёк или без `repo` scope — сделай новый на https://github.com/settings/tokens/new

---

**Итого: одна команда и 10 секунд — и V32 на GitHub.**
