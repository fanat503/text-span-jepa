# text-span-jepa — распределённый и ускоренный тренинг

Цели: (1) TPU-конфиги, (2) настоящая multi-GPU параллелизация без
одновременной перезаписи чекпоинтов, (3) детерминированные тесты,
(4) честная версия при запуске.

## 1. Выбор устройства

`src/train.py:894` — `torch.device("cuda" if cuda.is_available() else "cpu")`.
Это неверно для TPU и ломает DDP.

| Backend | Переменная | Проверка |
|---|---|---|
| TPU / PyTorch-XLA | `PJRT_DEVICE` | `import pytorch_xla` |
| GPU (DDP) | — | `LOCAL_RANK` от `torchrun` |
| CPU | — | fallback |

`pytorch_xla` **не установлен** — TPU-код должен деградировать в понятную
ошибку, а не в `AttributeError` посреди обучения.

## 2. Multi-GPU: DDP (несколько GPU на одной машине)

`scripts/wikitext/train_ddp.sh` зовёт `torchrun`, но в `src/` **ноль**
DDP-механики. Пять отдельных фатальных проблемы:

1. Нет процес-группы и all-reduce — каждый ранг это независимая копия.
2. Все ранки садятся на `cuda:0` (нет `local_rank`).
3. `logging.folder` одинаков для всех → четыре ранка пишут
   `checkpoint-latest.pth.tar` неатомарно. **Любой чекпоинт от этого скрипта
   непригоден.**
4. Все ранки видят одинаковый порядок батчей и одинаковые маски
   (`seed_everything(42)` без ранга, у `make_dataloader` нет `sampler`).
5. Рабочее пространство расщеплено параметр/буфер: `jawp.workspace_Q` —
   Parameter (all-reduce, корректно), а `rdc.workspace_Q` и `wsd.target_Q` —
   **буферы**, мутируемые на месте. При `broadcast_buffers=True` поведение
   это «значение ранка 0 с задержкой в один шаг», а не усреднение. Молча,
   без падения.

Требования к реализации:

- `init_process_group(backend="nccl")` при `RANK`/`WORLD_SIZE` в окружении.
- Ранк 0 — только он пишет чекпоинты, логи и `train_log.csv`.
- `find_unused_parameters=True`. Это **не может быть одна константа**:
  ветки механизмов зависят от шага (`pcr.level_gates`, `spc.freq_basis` в графе
  только когда их ветка сработала), а `train.py:1127-1146` делает **отдельный
  второй `backward()`** для GAC.
- Сид = `base_seed + rank`, иначе все ранки делают одно и то же.
- `DistributedSampler` — либо добавить параметр `sampler` в
  `make_dataloader`, либо выбирать сэмплер в трейнере по `RANK`.
- Буферы: либо `broadcast_buffers=False` с явной синхронизацией рабочих
  пространств, либо признать, что они ранго-зависимы, и задокументировать.
  Молча разъезжающиеся `Q` портят рабочее пространство.

## 3. TPU: конфиги под несколько TPU

`requirements.txt` содержит только `-e .[dev,eval]` — XLA нет даже списком.

Требования:

- `PJRT_DEVICE`, `XLA_USE_SPMD` для SPMD-режима.
- `torch_xla.core.xla_model.get_device_spec()` для числа конвееров
  (SPMD: один конвейер на чип).
- Батчи должны делиться на число конвейеров; `drop_last=True` в иначе
  последний конвейер получает короткий батч и SPMD падает.
- `torch_xla.utils.metrics.report_green_metrics()` для отчётности.
- Устройство в `train.py:894` должно уважать XLA.
- `GradScaler` сейчас жёстко `enabled=False` и привязан к `"cuda"`
  (`train.py:733`) — для bf16 на TPU нужен отдельный путь.

## 4. Конфиги под распределённый режим

Сейчас масштабная лестница смешивает размер модели с размером батча
(эффективный батч 512/512/256/256 — две группы по две точки, а не четыре
точки). Нужны отдельные конфиги, где при фиксированной модели варьируется
только число устройств, иначе распределённый запуск ничего не измеряет.

## 5. Детерминированные тесты

Ни в одном `tests/test_*.py` нет `conftest.py`. 12 из 21 файла вообще не
сеют случайность. `AGENTS.md` утверждает, что тесты детерминированы, и ничто
это не обеспечивает. Нужен autouse-фикстура в `tests/conftest.py`:
`torch.use_deterministic_algorithms` (там, где поддерживается),
фиксированные сиды, `torch.set_num_threads(1)`, и запрет на сеть/GPU.

## 6. Детерминированная диагностика

`seed_everything(seed)` всегда вызывается с `deterministic=False` →
`cudnn.benchmark = True`. Ни конфиг, ни CLI не могут потребовать
детерминизм. Проверено: 1 против 8 потоков, один seed и одни данные, 10 шагов —
шаги 0-2 совпадают, **с шага 3 расходятся**, финальный `qkv`-вес не
побитово равен. На 4-ядерном ноутбуке запуск не воспроизводим на
16-ядерном сервере.

## 7. Устройство-честная версия при старте

Требуется при каждом запуске печатать: имя устройства, число конвейеров,
мировую размерность, ранг, `find_unused_parameters`, разрешённые
`torch.randperm`-устройства, `weights_only` путь загрузки чекпоинта, живые
`rdc`/`puc` `checkpoint_dict()` (которые `train.py` сейчас игнорирует в пользу
ручного неполного списка).

## Границы владения файлами

| Файл | Владелец |
|---|---|
| `src/train.py` | **integration-orchestrator** |
| `src/utils/distributed.py` | **integration-orchestrator** (новый) |
| `src/utils/tpu.py` | **integration-orchestrator** (новый) |
| `scripts/*` | **integration-orchestrator** |
| `src/models/ddp_compat.py` | **integration-orchestrator** (новый) |
| `tests/conftest.py` | **integration-orchestrator** (новый) |
| `pyproject.toml`, `requirements.txt` | **integration-orchestrator** |
| `defaults.yaml` | **integration-orchestrator** |
| `.github/workflows/*.yml` | **integration-orchestrator** |

Остальные агенты не трогают эти файлы, иначе всё схлопнется в один конфликт.
