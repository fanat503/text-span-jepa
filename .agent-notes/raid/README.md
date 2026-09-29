# RAID scout — TASK-26

Read-only scouting card. No file outside `.agent-notes/raid/` was created or
modified. No training was run; the only Python executed was a single short
`-c` probe on reduced-vocab models (plus a two-point linear fit in vocab to get
exact 50304 counts without allocating 250M real parameters).

## Areas covered

1. **`baselines/`** — `mlm_baseline.py`, `data2vec_baseline.py`, plus
   `src/train.py:458-531` (`create_model`), `:534-564` (`compute_loss`),
   `:567-665` (`get_param_groups`), `:668-682` (`do_ema_update`), and the five
   baseline configs in `config/wikitext/` and `config/kaggle/`.
2. **`scripts/`** — all 8 files read in full, against every `--fname` target
   they invoke and against `run_comparison.py`'s argparse.
3. **`config/scaling/devices/`** — all 4 rungs, `defaults.yaml`, and
   `tests/test_config_system.py::TestFixedSizeDistributedFamily`
   (`LADDER_CONSTANTS` / `LADDER_CONSTANTS` / `_effective_batch`).
4. The card's two "also worth a look" items: `src/datasets/kaggle.py`,
   `src/train.py:_build_data_pipeline`, `config/tinystories/`,
   `config/kaggle/`.

## Seeds written — 10 findings + 2 negative results

| file | benefit | one line |
|---|---|---|
| seed-1 | 4 | the "identical model capacity / identical compute" guarantee in `mlm_baseline.py:5-7` is false in both directions |
| seed-2 | 4 | JEPA gets **3.89x** more prediction targets than the MLM arm (mask budget 0.35 vs 0.15, epochs 50 vs 30) |
| seed-3 | 4 | `model.drop_rate` is silently discarded for data2vec; three arms, three regularisation regimes |
| seed-4 | 3 | `run_experiment.sh compare_data2vec` pairs a 640/10 JEPA against a 768/12 data2vec checkpoint |
| seed-5 | 4 | all three `config/kaggle/*.yaml` arms declare the same `logging.folder` and overwrite each other |
| seed-6 | 4 | `scripts/fineweb/train.sh` announces FineWeb-Edu, ignores its shard arg, trains WikiText-103 |
| seed-7 | 4 | `config/scaling/devices/` is certified by 7 tests over `_meta.devices`, which no code reads |
| seed-8 | 2 | `dev8_bs64.yaml` is the only rung that does not declare its own `data.batch_size` |
| seed-9 | 4 | the TinyStories config silently **downloads** WikiText-103 over the network; its documented diagnosis is wrong |
| seed-10 | 2 | `data2vec_baseline.py` diverges from the fairseq file its header cites in 3 places |
| seed-none-baselines-truncation | — | the top-K truncation warning is benign; k = min(8, depth) and only fires at depth 6, where "top 8" == "all 6" |
| seed-none-scripts-hyperparams | — | **zero** hyperparameter overrides in `scripts/`; all 8 files pass only `--fname` |

## Highest-benefit finding

**seed-1 — the fair-comparison guarantee in `baselines/mlm_baseline.py:5-7` is
false, and it is false in both directions.** Measured:

```
MLM   640/10  (config/wikitext/mlm_wikitext_small.yaml) : 113,953,280  total, all trainable
JEPA  640/10  (config/scaling/small_100m.yaml:20-21)    : 170,706,561  total / 88,947,841 trainable
                                                  -> JEPA total 1.498x the baseline
                                                  -> the baseline's TRAINABLE count is 1.281x JEPA's
```

The header claims "identical model capacity (encoder params match exactly),
identical compute (same FLOPs per forward pass), only the training objective
differs". Only the parenthetical holds, and only for the encoder: JEPA carries
a frozen `target_encoder` deepcopy (81,958,720 params at 640/10) plus predictor
and decoder, while the baseline carries one `mlm_head` of 32,194,560.

The compute half fails for a mechanical reason. `mlm_baseline.py:77` projects
**all** `B*T` positions and only gathers masked rows at `:100`; `jepa.py:716-718`
applies its decoder to masked rows only. At the matched micro-batch of 64x512
over a 50304 vocab in fp32 that is **6.14 GiB** of dense logits for the baseline
against **2.15 GiB** for JEPA — a 2.86x activation-memory penalty that is not a
property of the objective but of where the gather sits. data2vec does not have
this problem (`data2vec_baseline.py:134` gathers first), so the defect is
specific to the MLM control.

Why this is the top finding: the false claim is the *load-bearing sentence* of
the file that exists to be the fair control, it is stated as a guarantee rather
than an intention, and it is invisible to the suite — `tests/test_model.py`
exercises both baselines at `embed_dim=32, depth=2, vocab_size=100`, where the
ratio is invisible. Two other seeds (2 and 3) make the same comparison worse and
are independent of it; fixing seed-1's prose without fixing seed-2 and seed-3
would document a fairness claim that is still untrue.

## Not re-reported

Read first and excluded as already-known: DDP non-functional + `train_ddp.sh`
checkpoint corruption (R8/P6), `run_comparison`'s dead checkpoint format and
random token ids (I11), `get_num_params()` under-reporting (P1), thread-count
sensitivity (R7), `src/interp/` unseeded RNG (R11/I16). The devices-family seed
(seed-7) is *not* a re-report of R8/P6: those cover the script and the absence
of DDP machinery; seed-7 covers a config family whose own guard tests certify an
invariant over a key no code reads, which is a distinct falsifiable claim.

The wave-1 config findings (V1 `ema_tau_end: 1.0`, V3 the missing leave-one-out
row, V5/P7 the confounded capacity ladder) appear to have been **fixed since
that document was written** — `config/scaling/small_100m.yaml` no longer sets
`ema_tau_end`, and its header documents the pre-fix state. Those findings docs
are stale on that point; worth telling the coordinator.
