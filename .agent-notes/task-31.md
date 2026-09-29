# TASK-31 — one `logging.folder` per run

status: **DONE**

---

## The question the card asked me to answer before fixing anything

> "If the three kaggle arms turn out to be variants of one experiment that
> genuinely must share output, say so with evidence and propose the
> alternative — do not assume either way."

**Verdict: they are NOT variants of one experiment. Each gets its own folder.**
No alternative proposed, because the evidence points one way. Five independent
pieces of evidence, all from shipped files:

1. **Three different model classes.** `meta.model_name` is
   `text_span_jepa_kaggle` / `mlm_kaggle` / `data2vec_kaggle`. `create_model`
   (`src/train.py:458-529`) normalises those to `text_span_jepa` / `mlm` /
   `data2vec` and dispatches to `TextSpanJEPA` (:466), `MLMBaseline` (:474),
   `Data2VecTextBaseline` (:498). Three model families, not three settings of
   one model.
2. **The configs label themselves as three arms.** `_meta.arm` is
   `standard_jepa` / `baseline_mlm` / `baseline_data2vec`; `_meta.config` is
   `textspanjepa_kaggle` / `mlm_kaggle` / `data2vec_kaggle`. "Arm" is the
   repo's own word for a compared condition, not for a variant.
3. **What the three share is the CONTROL, not the experiment.** The headers say
   it explicitly: `mlm_kaggle.yaml:4-6` "the hardware profile follows
   config/kaggle/textspanjepa_kaggle.yaml", `data2vec_kaggle.yaml:4-6` the
   same. Shared = micro-batch 32 x accumulation 16 = 512 sequences on one T4.
   That is the fixed-hardware comparison setup — which is the reason the three
   runs must stay *separable*, not the reason they must share a directory.
4. **Sharing a folder makes the family's purpose impossible.**
   `scripts/run_experiment.sh:53-69` is the repo's own head-to-head reader:
   `--jepa_ckpt "${CKPT_DIR}/jepa/best.pt" --baseline_ckpt
   "${CKPT_DIR}/mlm/best.pt" --baseline_type mlm`, and again with `data2vec`.
   Three files, three directories. If the Kaggle arms share one folder, only
   the last arm's `best.pt` survives and that comparison cannot be run at all.
5. **No precedent anywhere in the repo for sharing.** The closest existing
   analogue of "variants of one experiment" is the leave-one-out ablation
   grid — and `no_<mech>.yaml` / `<mech>_on.yaml`, the most tightly paired
   files in the tree, each get their OWN folder (`output/ablations/no_cgn/`,
   `output/ablations/cgn_on/`). Same-model siblings likewise: the four
   `mlm_wikitext_*.yaml` rungs have four folders. Not one of the 62 shipped
   configs shares a folder with a sibling.

So the "sessions are ephemeral so it does not matter" refutation fails for the
reason the scout gave — it is a deviation from the repo's own uniform
convention — and the substantive refutation fails too: these are three arms of
a comparison, and a comparison needs three sets of results.

---

## Independent reproduction of the scout's census (BEFORE the fix)

I did not take the scout's number on trust; I re-merged all 62 configs over
`defaults.yaml` with the shipped `src.train._deep_merge`:

```
62 configs -> 60 distinct logging.folder values
configs that do not override logging.folder: []
COLLISIONS: 1
  '/kaggle/working/output/' x3
    config/kaggle/data2vec_kaggle.yaml
    config/kaggle/mlm_kaggle.yaml
    config/kaggle/textspanjepa_kaggle.yaml
```

Exactly as claimed. This is the whole collision surface of the repo: there is
no OTHER collision, so there is nothing else to report and nothing else I
touched.

---

## Files changed

| file | change |
|---|---|
| `config/kaggle/textspanjepa_kaggle.yaml` | `logging.folder` + a 5-line why-comment + guard pointer |
| `config/kaggle/mlm_kaggle.yaml` | same |
| `config/kaggle/data2vec_kaggle.yaml` | same |
| `tests/test_config_system.py` | **added** section 9 `TestOneOutputDirPerRun` (3 tests) + docstring entry 8. Nothing existing changed, removed, weakened, skipped or xfailed. |

`defaults.yaml` untouched (forbidden, and also unnecessary — every config
overrides `logging.folder` anyway). No other `config/**` touched. No `src/**`
touched.

## Resolved folder per arm

| arm | `logging.folder` after the merge |
|---|---|
| `config/kaggle/textspanjepa_kaggle.yaml` | `/kaggle/working/output/kaggle/textspanjepa/` |
| `config/kaggle/mlm_kaggle.yaml` | `/kaggle/working/output/kaggle/mlm/` |
| `config/kaggle/data2vec_kaggle.yaml` | `/kaggle/working/output/kaggle/data2vec/` |

Naming rationale: the repo's convention is `<output root>/<config group>/<arm>/`
— `output/ablations/<arm>/`, `output/wikitext/<arm>/`, `output/scaling/<rung>/`,
`output/tinystories/<arm>/`, `output/scaling/devices/<rung>/`. These files live
in `config/kaggle/`, so `output/kaggle/<arm>/` is the same pattern with the
Kaggle session root substituted for the repo root. The `<arm>` is the model,
because in this family the model is the only thing that varies (all three are
base dimensions). Trailing slash, as in all 62 shipped folders.

---

## The guard

`tests/test_config_system.py`, section 9, class `TestOneOutputDirPerRun`:

- `test_every_config_declares_a_usable_output_folder` — the resolved
  `logging.folder` is a non-empty string (it goes straight to `os.makedirs()`
  and `os.path.join()`).
- `test_every_config_resolves_to_a_distinct_output_folder` — **the property the
  card asked for.** Walks `CONFIG_IDS` (all of `config/**/*.yaml` via
  `rglob`), deep-merges each over `defaults.yaml` with the real
  `src.train._deep_merge`, and asserts the resolved folders are pairwise
  distinct. The failure message names every colliding folder AND every
  config that points at it, so the next person who adds a colliding arm gets
  the culprit in the red output, not just a count.
- `test_the_three_kaggle_arms_get_one_directory_each` — **the three arms named
  explicitly**, as the card's `must pin` requires, asserting each resolves to
  its own directory.

Two design points worth flagging:

- The property is asserted over the **merged** config, not the raw YAML, so it
  also catches a future config that omits `logging.folder` and silently
  inherits `output/`.
- The guard grounds itself in shipped code rather than in my opinion:
  `src/train.py:1219` uses the folder verbatim and every artefact name under
  it (`train_log.csv` :1229, `checkpoint-latest.pth.tar` :1255, `best.pt`
  :1609, the `params-*.yaml` dump :1224) is a fixed literal, and
  `src/utils/distributed.py::log_dir_lock` (:536-595) documents the identical
  hazard for independent jobs and names the fix in its own error message:
  *"point logging.folder somewhere else or wait for it to finish"*.

---

## VERIFY (full paste)

Card's command, final state:

```
rt.py: C:\Users\...\Python310\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_config_system.py
rt.py: threads=1  total_budget=90s  slow_ok=True
============================= test session starts =============================
collected 584 items

tests\test_config_system.py ............................................ [  7%]
........................................................................ [ 19%]
...........................................................s............ [ 32%]
........................................................................ [ 44%]
........................................................................ [ 56%]
........................................................................ [ 69%]
........ss..........ssssssss..........................................ss [ 81%]
..........ssssssss...................................................... [ 93%]
....................................                                     [100%]

======================= 563 passed, 21 skipped in 8.04s =======================
```

No skip added, no test weakened. Proven by re-running the SAME command with my
changes stashed:

```
======================= 560 passed, 21 skipped in 7.93s =======================
```

`560 -> 563 passed, 21 -> 21 skipped`. Exactly +3 tests, all passing, zero
change to the skip count.

Lint and format on the touched file:

```
> & $PY -m ruff check tests/test_config_system.py
All checks passed!

> & $PY -m black --check tests/test_config_system.py
All done! ✨ 🍰 ✨
1 file would be left unchanged.
```

Post-fix census (same script, re-run):

```
62 configs -> 62 distinct logging.folder values
configs that do not override logging.folder: []
COLLISIONS: 0
```

---

## MUTATION VERDICT

**Mutate** — put two of the three arms back onto one shared folder
(`textspanjepa_kaggle.yaml` and `mlm_kaggle.yaml` both back to
`/kaggle/working/output/`):

```
============================= test session starts =============================
collected 584 items / 581 deselected / 3 selected

tests\test_config_system.py .FF                                          [100%]

=================================== FAILURES ===================================
_ TestOneOutputDirPerRun.test_every_config_resolves_to_a_distinct_output_folder _
tests\test_config_system.py:1232: in test_every_config_resolves_to_a_distinct_output_folder
    assert not collisions, (
E   AssertionError: 1 output folder(s) are shared by more than one config, so whichever config runs last overwrites train_log.csv, best.pt and checkpoint-latest.pth.tar of the others: '/kaggle/working/output/' <- config/kaggle/mlm_kaggle.yaml, config/kaggle/textspanjepa_kaggle.yaml
E   assert not {'/kaggle/working/output/': ['config/kaggle/mlm_kaggle.yaml', 'config/kaggle/textspanjepa_kaggle.yaml']}
__ TestOneOutputDirPerRun.test_the_three_kaggle_arms_get_one_directory_each ___
tests\test_config_system.py:1244: in test_the_three_kaggle_arms_get_one_directory_each
    assert got == KAGGLE_FOLDERS, (
E   AssertionError: the Kaggle arms resolve to ('/kaggle/working/output/', '/kaggle/working/output/', '/kaggle/working/output/kaggle/data2vec/'), expected ('/kaggle/working/output/kaggle/textspanjepa/', '/kaggle/working/output/kaggle/mlm/', '/kaggle/working/output/kaggle/data2vec/'). The three runs are the head-to-head comparison -- JEPA vs MLM vs data2vec on one T4, one corpus, one effective batch -- so a shared folder destroys the comparison it exists to produce. Update KAGGLE_FOLDERS in this file when the convention changes.
E   assert ('/kaggle/wor...le/data2vec/') == ('/kaggle/wor...le/data2vec/')
E
E     At index 0 diff: '/kaggle/working/output/' != '/kaggle/working/output/kaggle/textspanjepa/'
E     Use -v to get more diff
=========================== short test summary info ===========================
FAILED tests/test_config_system.py::TestOneOutputDirPerRun::test_every_config_resolves_to_a_distinct_output_folder
FAILED tests/test_config_system.py::TestOneOutputDirPerRun::test_the_three_kaggle_arms_get_one_directory_each
================= 2 failed, 1 passed, 581 deselected in 0.59s =================
```

The whole-tree guard went red AND named the two files I had just collided. The
arm-level guard caught the exact same edit from the other direction. The third
test (`..._declares_a_usable_output_folder`) correctly stayed green — the
mutation was a *collision*, not a malformed value, and a good guard is precise
about which property broke.

**Restore** — both folders back to their own, same command:

```
rt.py: C:\Users\...\Python310\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_config_system.py
rt.py: threads=1  total_budget=90s  slow_ok=True
============================= test session starts =============================
collected 584 items

tests\test_config_system.py ............................................ [  7%]
........................................................................ [ 19%]
...........................................................s............ [ 32%]
........................................................................ [ 44%]
........................................................................ [ 56%]
........................................................................ [ 69%]
........ss..........ssssssss..........................................ss [ 81%]
..........ssssssss...................................................... [ 93%]
....................................                                     [100%]

======================= 563 passed, 21 skipped in 8.04s =======================
```

Also recorded earlier, from the TDD side: the guard was written BEFORE any
config was touched and was red on the pre-fix tree for exactly the three kaggle
arms (`62 configs -> 60 distinct`, one collision), which is the state it was
designed to catch.

---

## DIFF-STAT

```
 config/kaggle/data2vec_kaggle.yaml     | 10 +++-
 config/kaggle/mlm_kaggle.yaml          | 10 +++-
 config/kaggle/textspanjepa_kaggle.yaml | 11 ++++-
 tests/test_config_system.py            | 87 ++++++++++++++++++++++++++++++++++
 4 files changed, 112 insertions(+), 6 deletions(-)
```

6 of the deletions are the three old `folder:` lines; the other 3 are the three
`# Guard:` comment lines I extended. No behaviour outside `logging.folder`
changed.

---

## не_сделано / риски

**Not done, deliberately:**

- **No other collision exists, so there was nothing else to report.** Verified
  by measurement, not assumption: 62 configs, 60 distinct folders before,
  62 distinct after. If a future scout claims a second collision, it is wrong;
  `test_every_config_resolves_to_a_distinct_output_folder` would be red.
- **Did NOT touch `src/train.py:1224`**, which hard-codes
  `params-text-span-jepa.yaml` as the config-dump filename, so an MLM or
  data2vec run writes a file *named after JEPA*. That is a real
  correctness-of-record defect (the scout flagged it, item 3), but it is a
  `src/` file I do not own, it is independent of the folder fix, and the card
  put it out of scope. **Separate card.** It is now *less* harmful than before
  — with one folder per arm, at least `params-*.yaml` sits next to the
  `best.pt` it describes instead of being overwritten by the next arm — but
  the wrong name remains.
- **Did NOT touch `scripts/kaggle/train.sh`.** It passes no `--output_dir`,
  which is correct and is why the config is the right place to fix this.
- **Did NOT add a `--output_dir`-style escape hatch or any `src/` change.**
  Out of scope and forbidden.
- **No `.agent-notes/`, no `TASKS.md` status edit, no `proofs/` edit** —
  not mine, and no mechanism's Verified/Divergent/Gaps state moved.

**Risks / things a reviewer should know:**

1. **The new whole-tree guard is a new convention, not just a new fix.** It
   will turn a future agent's suite red if they add a config that shares a
   folder on purpose (e.g. a per-seed or per-rank subdirectory family). That
   is the intended behaviour — but it is a *policy* the repo now has and did
   not have, so it should be a conscious accept, not a surprise. The closest
   legitimate worry: a future DDP config family that wants one folder per
   rank written by rank suffix. `src/utils/distributed.py` currently handles
   ranks by `is_main_process` gating, not by folder, so nothing today needs
   that. If it is ever needed, the guard needs a documented exemption in the
   same style as `_allowed_repeats` in `TestDeltaPurity` — a second exemption
   class, decided deliberately, not quietly.
2. **`KAGGLE_FOLDERS` pins three literal strings.** That is a deliberate trade:
   the card asked for the arms to be *named*, and a pin is the only way a
   rename is caught. The failure message says exactly what to update. If the
   kaggle folder convention is ever changed deliberately, this constant and
   the three YAMLs must move together.
3. **The change is untested at runtime and stays that way.** I did not, and
   per the card must not, run training. Nothing verifies that
   `os.makedirs("/kaggle/working/output/kaggle/textspanjepa/")` succeeds on a
   real Kaggle session. It should — it is a deeper path under a directory
   that already exists — but it is an assumption, not a measurement. Kaggle
   sessions get 20 GB of `/kaggle/working`; three arms of a 140M-param
   checkpoint set fit comfortably, so disk is not a new risk.
4. **A behavioural consequence worth stating plainly:** anyone who previously
   trained the three Kaggle arms into one directory will now find three
   subdirectories. There is no migration or back-compat shim, and there should
   not be — old `best.pt` files at `/kaggle/working/output/` are already
   ambiguous about which arm produced them, which was the point of the fix.
   Anyone pointing at the old flat path by hand needs to update.
