# CONTROL SCOUT — wave 1 raid seeds

Branch: `agent/wave-1` @ `c356369`. Read-only on `src/`, `tests/`, `config/`,
`docs/`, `scripts/`, `baselines/`. No training run. No whole-suite pytest.
Python used: five short `python -c` probes on tiny tensors (encoder at
V=64/D=16/L=2, V=100/D=32/L=2, V=37/D=11/L=3, plus `TiedTokenDecoder` at
D=640 and one analytic fit) and one config-merge script. Nothing was written
outside this file.

Scope note: the brief says "the six with benefit >= 4" and then lists **seven**
(1, 2, 3, 5, 6, 7, 9). All seven are covered.

| seed | benefit claimed | verdict | one-line reason |
|---|---|---|---|
| 1 | 4 | **SURVIVES** | every number reproduced independently; the header's "identical model capacity" is false, and false in the direction that *favours the baseline* |
| 2 | 4 | **REFUTED** | uniform repo-wide protocol choice, disclosed in 8/8 wikitext baseline headers and explicitly escalated to a human; the card is an unauthorised experiment |
| 3 | 4 | **SURVIVES (narrowed)** | the two dead `drop_rate: 0.1` lines are real; the "three regularisation regimes" framing is defensible per-method practice, and the card's "forward the params" branch must be rejected |
| 5 | 4 | **SURVIVES** | measured: the *only* `logging.folder` collision in all 62 configs is exactly these three arms; every other config has a unique folder |
| 6 | 4 | **SURVIVES** | 8-line stub, dead `$1`, zero FineWeb references repo-wide; it trains WikiText-103 |
| 7 | 4 | **REFUTED** | the limitation is disclosed 4x in the YAMLs and once in the test docstring, and the runtime consequence is verbatim the already-known R8/P6; the seed inverts its own evidence |
| 9 | 4 | **SURVIVES (narrowed)** | the config's documented diagnosis is wrong (it downloads, it does not read the directory) and the fallback is silent; the `data.dataset` selector is a feature request, not a fix |

---

## PER SEED

### seed-1 — SURVIVES

claim as stated: `baselines/mlm_baseline.py:5-7` states a "Fair comparison
guarantee: identical model capacity (encoder params match exactly), identical
compute (same FLOPs per forward pass), only the training objective differs" —
both quantitative halves are false.

what I checked:
- `baselines/mlm_baseline.py` in full (122 lines), header at :5-7, class
  docstring at :19-34 repeating the claim at :21-22.
- `src/models/encoder.py:36-201` (Attention / MLP / Block / Encoder).
- `src/models/decoder.py` `TiedTokenDecoder` (the JEPA output head).
- `src/models/jepa.py:362,399,712-720` (decoder construction and the
  masked-row-only decode at :715-718).
- `config/wikitext/mlm_wikitext_small.yaml:38-42` and
  `config/wikitext/textspanjepa_wikitext_small.yaml:32-37` (the compared pair;
  both 640/10/10 — this pair is what `run_experiment.sh:53-60` `compare` runs).
- `tests/test_model.py` (baseline construction sizes).
- `config/scaling/small_100m.yaml:18-25` (the repo's own documented counts).

what I found — I reproduced every number, and the seed is right:

1. The capacity counts are **exactly** right. Closed form for one encoder is
   `V*D + T*D + L*((4+2r)D^2 + (r+9)D) + 2D`. Verified against a real
   construction at three small shapes (exact match at all three) and evaluated
   at the shipped rung:

   ```
   encoder(640/10, V=50304, T=512)        =  81,758,720
   MLM total (encoder + 640*50304 head)   = 113,953,280   (all trainable)
   88,947,841 + 81,758,720                 = 170,706,561   <- config/scaling/small_100m.yaml:20
   JEPA total / MLM total                  = 1.498x
   MLM trainable / JEPA trainable          = 1.281x
   ```

   The cross-check that matters: the seed's frozen-encoder figure is the
   arithmetic complement of the number the repo *itself* documents. It holds.

   **Correction to the seed: "81,958,720" is a transposition typo. The correct
   figure is 81,758,720.** A card must not carry the wrong number.

2. The seed's *framing* is weaker than its evidence in one respect, and that
   strengthens the finding. The seed leads with "JEPA total is 1.498x the
   baseline", which reads as *JEPA being bloated*. It is not. The JEPA excess is
   a frozen `target_encoder` deepcopy — 81,758,720 params carrying no gradient
   path — plus a 1,639,680-param decoder. The meaningful number is **trainable**
   capacity, and there **the baseline is the larger model by 1.281x**. The
   header's implication of a matched comparison is false in the direction that
   flatters the control. That is the more interesting asymmetry and the seed
   under-sells it.

3. The seed is **wrong about why** the two models differ, in a way that matters.
   It attributes the gap to "JEPA carries a target_encoder plus predictor and
   decoder; the baseline carries one `mlm_head` of 32,194,560", treating the
   `mlm_head` as the smaller of two extras. It is the *larger*: JEPA's
   `TiedTokenDecoder` is `Linear(D,2D) -> GELU -> Linear(2D,D) + LayerNorm` and
   gets its vocab projection for free by **reusing `encoder.token_embedding.weight`**
   (`decoder.py`: `logits = F.linear(x, token_embedding_weight)`). Measured:

   ```
   TiedTokenDecoder(embed_dim=640, bias=False) = 1,639,680 params
       shapes: [(1280,640), (640,1280), (640,), (640,)]   <- no (640, 50304) matrix
   MLMBaseline.mlm_head                          = 32,194,560 params, separate + untied
   ```

   So the baseline carries a full untied, trainable vocab head that JEPA does
   not have at all. The real statement is: *the baseline has 32.2M more trainable
   parameters than JEPA, entirely in its output head, and its capacity advantage
   is invisible in the config file.* That is a stronger and cleaner claim than
   the one the seed makes.

4. The compute half is **true and mechanically explained**, exactly as claimed.
   `mlm_baseline.py:77` projects all `B*T` positions and `:100` gathers;
   `jepa.py:715-718` decodes masked rows only; `data2vec_baseline.py:133-135`
   gathers before its head, so data2vec is unaffected. Reproduced:

   ```
   fp32, micro-batch 64x512, V=50304:
     MLM dense (B,T,V)              = 6.14 GiB
     JEPA masked-only at 0.35       = 2.15 GiB     ratio 2.857x
   ```

   and confirmed the ratio is exactly `T / num_masked`, i.e. it is a property of
   *where the gather sits*, not of the objective. One thing the seed under-states:
   the boolean index at `:100` makes the backward pass allocate a second dense
   `(B,T,V)` gradient, so the peak penalty is **worse** than 2.86x, not better.
   (That last point is analysis of standard `index_select` backward semantics,
   not a measurement — I did not run it.)

5. Not already guarded. `tests/test_model.py` builds both baselines at
   `vocab_size=1000, embed_dim=64, depth=2`, where the head is 64k of a ~100k
   model and the logits penalty is 32x smaller than at the shipped rung. The
   ratio is invisible to the suite, as claimed.

6. Every refutation attempt failed:
   - *"the parenthetical is the real claim and it holds"* — true of
     `encoder params match exactly`, but it **contradicts the headline it
     qualifies**: "identical model capacity" and "encoder params match" are
     different claims, and only one is true. An internally inconsistent
     guarantee sentence is a defect in itself.
   - *"the 1.498x is an artefact of counting a frozen copy"* — true, and it
     makes the finding worse, not better (see 2).
   - *"it's a docstring, nobody reads it"* — it is stated as a **guarantee**,
     not an intention, in the header of the file whose entire purpose is to be
     the fair control, and it is repeated in the class docstring at `:21-22`.
   - *"already handled"* — nothing handles it; there is no comment, no test, no
     caveat anywhere in `baselines/` or the MLM config.

verdict: **SURVIVES.** I could not refute it. The claim is true, the numbers are
reproducible, the refutation attempts strengthen rather than weaken it.

if survives, the smallest fix:
goal — replace the unverifiable "fair comparison guarantee" with a measured one.
files — `baselines/mlm_baseline.py` (header + class docstring, **comments
only**), `config/wikitext/mlm_wikitext_small.yaml` (header), one new
`tests/test_baseline_fairness.py`.
statement to write — the encoder is identical (81,758,720 at 640/10); the arms
differ in the head: MLM carries a separate untied 32,194,560-param `mlm_head`,
JEPA's `TiedTokenDecoder` is 1,639,680 and reuses the token embedding; JEPA
total/trainable are 170,706,561 / 88,947,841 vs MLM 113,953,280 / 113,953,280.
The baseline therefore has **1.28x the trainable capacity**; the baseline also
materialises a dense `(B,T,V)` logit tensor where JEPA decodes masked rows only.
verify — a test that builds both at the same encoder dims and asserts the
*documented* ratios (not equality), plus an assertion that `MLMBaseline` never
produces a `(B,T,V)` tensor when a mask is supplied. Do **not** change
`MLMBaseline.forward` in the same commit: moving the gather before the head is a
behaviour change and needs its own card.

---

### seed-2 — REFUTED

claim as stated: the JEPA-vs-MLM comparison is not token-budget matched; the
JEPA arm gets 3.89x more prediction targets (mask budget 0.35 vs 0.15, epochs
50 vs 30).

what I checked:
- `src/masks/span.py:50-67` — `target_num_masked = int(seq_len * current_mask_ratio)`,
  so `mask_ratio` is the fraction of *tokens* supervised, independent of span
  length. Confirms the seed's mechanism.
- Every baseline and JEPA config under `config/wikitext/` and `config/kaggle/`
  (12 files), extracting `model_name` / `mask_ratio` / `span_length_range` /
  `epochs` after the defaults merge.
- The token-budget disclosure sentence in the header of all 8
  `config/wikitext/*` baseline configs.
- `defaults.yaml:65,274` (`mask_ratio: 0.35`, `epochs: 50`).

what I found — the arithmetic is right, the defect framing is not.

1. The 3.89x is **reproduced exactly**:

   ```
   JEPA      : 50 epochs x 0.35 mask = 17.5 masked-token-epochs
   MLM       : 30 epochs x 0.15 mask =  4.5                        -> 3.89x
   data2vec  : 50 epochs x 0.15 mask =  7.5   (inherits epochs; no `epochs:` key)
   ```

2. **It is not a per-file accident — it is a uniform, deliberate protocol
   choice.** All 8 baseline configs set `mask_ratio: 0.15` and
   `span_length_range: [1, 1]`; no JEPA config overrides either. A repo-wide
   convention applied to 8/8 baseline files is a decision, not a slip. And it
   is the *method-appropriate* value: BERT's 15% single-token masking is the
   published protocol; asking an MLM to mask 35% of tokens with `[1,1]` spans
   would be the deviation, not the reverse.

3. **It is already disclosed, in every file a reader would open.** All 8
   `config/wikitext/*` baseline headers carry, verbatim:

   > "`epochs` differs between columns (JEPA 50, MLM 30, data2vec 50). The three
   > methods therefore do not see the same token budget. That is a pre-existing
   > protocol difference, not a config defect, and reconciling it is an
   > experiment decision rather than a schema one"

   The repo raised this, named it, classified it, and routed it to a human. The
   seed's own evidence says "the config authors knew". A finding whose premise
   is *the authors knew, wrote it down, and escalated it* does not describe an
   undiscovered defect.

4. **The proposed card is the problem.** "Produce the mask-budget- and
   epoch-matched control arm the comparison needs" means inventing two new
   experiment configs and running an experiment nobody has authorised, to test a
   hypothesis nobody has stated. That is `AGENTS.md` rule 2 (no speculative work)
   and rule 4 (YAGNI) in one line. The right disposition is the one the config
   header already prescribes: a human experiment decision.

5. The one true residue, which is not a card: the disclosure names the **epoch**
   term (1.67x) and omits the **mask** term (2.33x), which is the larger half. A
   one-sentence addition to the existing disclosure would close it. The three
   `config/kaggle/*.yaml` headers disclose "schedule follows the <column>
   config" by reference rather than in full, which is weaker but not absent.

verdict: **REFUTED** (already disclosed, uniform by design, and already escalated
to a human as an experiment decision).

if refuted, the alternative explanation: the repo treats the mask budget as a
per-method hyperparameter and the epoch count as a per-method schedule, states
so in every baseline config header, and deliberately declines to reconcile them
in a config file because reconciling them is a scientific decision requiring
authority the config system does not have. The seed re-derives a number the repo
has already decided not to standardise, and proposes to answer it with new
configs.

---

### seed-3 — SURVIVES (narrowed)

claim as stated: `model.drop_rate` is silently discarded for the data2vec arm —
`create_model` passes it, `Data2VecTextBaseline.__init__` swallows it in
`**kwargs`, so the three arms train under three different unintended
regularisation regimes, and two shipped configs declare a value that does
nothing.

what I checked:
- `baselines/data2vec_baseline.py:39-56` (signature), `:70-77` (encoder build).
- `baselines/mlm_baseline.py:36-58` for contrast.
- `src/train.py:473-515` (`create_model`, the `mlm` and `data2vec` branches).
- `src/models/encoder.py:162-164, 184-200` (encoder dropout params, `dpr`).
- `defaults.yaml:75-79`; `config/kaggle/data2vec_kaggle.yaml:34`;
  `config/wikitext/data2vec_wikitext_train.yaml:39`;
  `config/kaggle/mlm_kaggle.yaml:31`.
- `tests/test_config_system.py` `LADDER_CONSTANTS` (:218-246) and
  `TestNoDeadKeys`.

what I found — the mechanism is confirmed, the framing is half wrong.

1. **Confirmed by probe.** Constructing `Data2VecTextBaseline(drop_rate=0.9, ...)`:

   ```
   'drop_rate' in inspect.signature(Data2VecTextBaseline.__init__).parameters -> False
   **kwargs present                                                          -> True
   encoder.blocks[0].mlp.drop.p after passing drop_rate=0.9                  -> 0.0
   ```

   `src/train.py:507` passes `drop_rate=model_cfg.get("drop_rate", 0.0)`; the
   encoder at `data2vec_baseline.py:70-77` is built without it. The value is
   discarded with no warning. Silent no-op, confirmed.

2. **Two configs declare a value that does nothing**, and it is a *non-default*
   value, so the config is making a statement the code ignores:
   - `config/kaggle/data2vec_kaggle.yaml:34` -> `drop_rate: 0.1`
   - `config/wikitext/data2vec_wikitext_train.yaml:39` -> `drop_rate: 0.1`
   (`defaults.yaml:75` is `0.0`, so `0.1` is a deliberate override.)

3. **The guard suite is blind, confirmed.** `LADDER_CONSTANTS` pins
   `model.drop_path_rate` and omits `model.drop_rate` and `model.attn_drop_rate`
   entirely. `DEAD_KEYS` is the hard-coded 2-tuple
   `("optimization.ema", "logging.write_tag")` — the dead-key test checks those
   two names, it does not scan for keys that are read-but-not-forwarded. So the
   one key that is set *and* discarded is in no fixed-key list.

4. **The "three unintended regularisation regimes" half of the claim is
   refuted, and I am killing it.** The actual regimes are:

   | arm | drop | attn_drop | drop_path |
   |---|---|---|---|
   | JEPA | 0.0 | 0.0 | 0.1 |
   | MLM | 0.1 | 0.0 | 0.0 |
   | data2vec | 0.0 | 0.0 | 0.0 |

   That is **standard per-method practice**, not a confound: BERT-family models
   use dropout, ViT/GPT-family JEPA models use stochastic depth. Nobody would
   file "the MLM baseline uses dropout and the JEPA baseline uses drop-path" as a
   bug. The asymmetry is defensible. Only the *silently discarded declared value*
   is a defect.

5. Not stale, not already handled, nothing warns.

verdict: **SURVIVES, narrowed** to: *two shipped configs declare `drop_rate: 0.1`
for a model that discards it, with no warning.* The "three regimes" framing is
refuted and should not appear in the card.

if survives, the smallest fix:
goal — remove the false statement from the configs. Do **not** forward the
parameter in the same commit: the seed itself notes that forwarding it *changes
baseline results*, and `AGENTS.md` forbids unrequested behaviour changes. Two
branches, land the safe one first.
- **Branch A (safe, recommended, 2 lines):** delete `model.drop_rate: 0.1` from
  `config/kaggle/data2vec_kaggle.yaml` and
  `config/wikitext/data2vec_wikitext_train.yaml`, or replace it with a comment
  saying the key is not forwarded. Zero behavioural change.
- **Branch B (separate card, behaviour-changing):** add `drop_rate` /
  `attn_drop_rate` / `drop_path_rate` to both baselines and forward them.
  Requires re-running the baseline arms; must not be bundled.
files — Branch A: the two configs. Branch B: `baselines/data2vec_baseline.py`,
`baselines/mlm_baseline.py`, `src/train.py:473-515`.
verify — Branch A: a test asserting no baseline config sets a `model.*` key that
`create_model` does not forward to that baseline's constructor (this generalises
and would have caught it). Branch B: assert
`encoder.blocks[0].mlp.drop.p == pytest.approx(configured drop_rate)` for each
baseline built through `create_model`. Add `model.drop_rate` and
`model.attn_drop_rate` to `LADDER_CONSTANTS` either way.

---

### seed-5 — SURVIVES

claim as stated: the three `config/kaggle/*.yaml` arms — the one place where
JEPA, MLM and data2vec share hardware, effective batch and corpus — all declare
the same `logging.folder` and overwrite each other's checkpoints and logs.

what I checked:
- All three kaggle configs in full; `logging.folder` lines confirmed at
  `textspanjepa_kaggle.yaml:55`, `mlm_kaggle.yaml:42`, `data2vec_kaggle.yaml:44`
  (the seed's line numbers are off by one on two of the three; the values are
  right).
- `src/train.py:1190-1202` (`log_dir` used verbatim, no subdirectory appended),
  `:1228` (`checkpoint-latest`), `:1581` (`best.pt`), `:1611`
  (`checkpoint-ep{N}`), `:1634-1643` (`keep_last_epoch_ckpts` pruning operates on
  `log_dir`), `:1789-1790` (`--output_dir` override), `:895-899` (resume gate).
- `scripts/kaggle/train.sh` (passes no `--output_dir`),
  `scripts/run_experiment.sh:27-50` (`--output_dir` only on the wikitext arms).
- **All 62 configs under `config/`, merged over `defaults.yaml`.**

what I found — this is the cleanest finding in the batch, and I found stronger
evidence than the seed did.

1. `log_dir = log_cfg.get("folder", "output/")` is used **verbatim**; no
   per-config or per-model subdirectory is appended. Every artefact name is a
   fixed literal. `scripts/kaggle/train.sh` passes no `--output_dir`, and
   `run_experiment.sh` only passes it for the wikitext arms. Confirmed.

2. **The decisive measurement the seed did not make.** I merged all 62 configs
   over `defaults.yaml` and counted `logging.folder` collisions:

   ```
   62 configs -> 60 distinct logging.folder values
   THE ONLY COLLISION IN THE ENTIRE REPO:
     '/kaggle/working/output/'  x3
       config/kaggle/data2vec_kaggle.yaml
       config/kaggle/mlm_kaggle.yaml
       config/kaggle/textspanjepa_kaggle.yaml
   configs that do not override logging.folder: none
   ```

   Every other config in the repo has a unique output directory, including all
   four scaling rungs, all four device rungs, all ablations and all 8 wikitext
   configs. **These three files are the sole exception to the repo's own
   convention.** That kills the best available refutation — "on Kaggle each
   session has a fresh `/kaggle/working/`, so it doesn't matter" is not a
   property of Kaggle the repo is relying on; it is a deviation from a rule the
   other 59 configs follow. It also means the seed's proposed guard test is
   clean today: it passes for all 59 and fails for exactly these 3.

3. **The provenance loss is worse than "logs get overwritten".**
   `src/train.py:1197` hard-codes `params-text-span-jepa.yaml`, so an MLM run
   writes a file *named after JEPA* containing the MLM config. After the three
   arms run in one session, the surviving `params-*.yaml` names the wrong model,
   so `params-*.yaml` can no longer be used to identify which arm produced a
   surviving `best.pt`. That is a correctness-of-record problem, not a tidiness
   one.

4. **It is not the known R8/P6 finding.** R8/P6 is N *ranks of one run* writing
   the same files. This is three *sequential runs of three different models*,
   with a different fix (per-config folder vs per-rank folder). The scout's
   distinction is correct and I confirmed it against
   `docs/plans/2026-09-27-wave1-audit-findings.md:326-334` and
   `...-perf.md:134-161`.

5. Refutation attempts that failed:
   - *"no auto-resume, so nothing corrupts a checkpoint"* — true
     (`defaults.yaml:52 load_checkpoint: false`, and `train.py:898-899` returns
     early), but irrelevant: the arms still overwrite `best.pt`,
     `train_log.csv`, `checkpoint-latest.pth.tar`, `checkpoint-ep*.pth.tar` and
     the config dump. And if `keep_last_epoch_ckpts` is ever set, arm B prunes
     arm A's epoch checkpoints via the `log_dir` scan at `:1634-1643`.
   - *"Kaggle sessions are ephemeral"* — answered above by the 59-config
     convention.
   - *"`--output_dir` already exists"* — it does, and it is the obvious escape
     hatch, but nothing in the repo uses it for these arms and the configs
     cannot express it.

verdict: **SURVIVES.** The refutation attempts failed; the collision is isolated,
measurable, and a deviation from the repo's own uniform convention.

if survives, the smallest fix:
goal — give each kaggle arm its own output directory.
files — the three `config/kaggle/*.yaml`, `logging.folder` only (3 lines).
verify — a test in `tests/test_config_system.py` that collects `logging.folder`
after the merge for every config and asserts global uniqueness. It is green for
59 configs today and red for exactly these 3, so the failure list is the
evidence. Land this test and the 3 lines in the same commit.
Separately, and **not** in the same commit: derive the `params-*.yaml` filename
at `src/train.py:1197` from `_meta.config` / `meta.model_name` instead of
hard-coding JEPA's name. That is a `src/` change and belongs to its own card.

---

### seed-6 — SURVIVES

claim as stated: `scripts/fineweb/train.sh` announces "Text-Span JEPA —
FineWeb-Edu (N shards)", accepts a shard count, ignores it entirely, and trains
WikiText-103.

what I checked:
- `scripts/fineweb/train.sh` in full (8 lines, confirmed).
- `config/scaling/small_100m.yaml` in full — sets no `data.root_path`.
- `defaults.yaml:64` — `root_path: data/wikitext-103`.
- `scripts/wikitext/train_small.sh`, `scripts/tinystories/train.sh`,
  `scripts/kaggle/train.sh` for contrast.
- Repo-wide grep for `fineweb|fine_web|fine-web` (case-insensitive) across
  `*.py, *.md, *.yaml, *.yml, *.sh`, excluding `.git/`, `output/`,
  `__pycache__`.
- `config/` top-level directory listing.

what I found — confirmed, with one minor overstatement in the seed.

1. The file is 8 lines. `NUM_SHARDS=${1:-1}` (line 6) is referenced exactly
   once, inside the `echo` on line 7, and never again. Line 8 is
   `python -m src.train --fname config/scaling/small_100m.yaml`. `set -euo pipefail`
   cannot catch an unused variable — `pipefail` is about pipelines, and there is
   no `nounset` violation because the variable *is* used.

2. `config/scaling/small_100m.yaml` has no `data:` block, so the run inherits
   `defaults.yaml:64 root_path: data/wikitext-103`, and
   `src/datasets/kaggle.py:85` prints `Loaded WikiText-103 train: N tokens`.
   Confirmed.

3. **Repo-wide grep returns 2 hits, both inside this file** (line 2 comment, line
   7 echo) plus the RAID notes themselves. `config/` contains only `ablations`,
   `kaggle`, `scaling`, `tinystories`, `wikitext` — no `config/fineweb/`.
   `src/datasets/` exposes only `kaggle.py` (`load_wikitext103`, the dead
   `load_bookcorpus`) and `__init__.py`; no FineWeb loader. Confirmed.

4. **Minor overstatement, recorded so the card does not inherit it:** the seed
   says the script is "indistinguishable, in every observable respect, from
   `scripts/wikitext/train_small.sh`". Not quite — they write to different
   `logging.folder`s (`output/scaling/small-100m/` vs
   `output/wikitext/small-100m/`) and `train_small.sh` accepts a config override
   as `$1` while this one misuses `$1` as a shard count. Both train WikiText-103
   at 640/10/10 with JAWP only, which is the substance of the point.

5. Refutation attempts that failed:
   - *"it's an obvious placeholder everyone knows is a stub"* — nothing in the
     file says so. It contains no `TODO`, no "not implemented", no `exit 1`. It
     *asserts* FineWeb-Edu. A stub that lies is worse than no stub, because
     `bash scripts/fineweb/train.sh` is a plausible thing for a user to run and
     it will happily train for hours on the wrong corpus and exit 0.
   - *"config/scaling/small_100m.yaml was meant to gain a `data.root_path`"* —
     its header is a pure capacity-rung header ("Capacity rung 2 of 4") with no
     corpus mention. No evidence of a pending FineWeb config.
   - *"a shell linter would catch it"* — no shell linter is configured; the
     repo's linters are `ruff` and `black`, neither of which reads `.sh`.

verdict: **SURVIVES.** Cheap, isolated, one-file, and the harm (a user runs a
script labelled FineWeb-Edu and silently gets WikiText-103) is concrete.

if survives, the smallest fix:
goal — delete the stub, or make it fail loudly. FineWeb is not a shipped
experiment: there is no config, no loader, and no dataset code. Inventing them
is exactly the speculative work `AGENTS.md` rule 2 forbids.
files — `scripts/fineweb/train.sh` and the `scripts/fineweb/` directory.
recommendation — **delete**. If the directory must survive as a marker,
replace the body with `echo "FineWeb-Edu support is not implemented" >&2; exit 1`.
verify — a test in `tests/test_config_system.py` that scans every
`scripts/**/*.sh`, extracts `--fname <path>`, asserts the path resolves under
`config/`, and asserts every shell variable assigned at the top of a script is
referenced at least once *outside its own assignment and its own echo*. The
second half is the generalisable guard and is what would have caught this; the
naive "referenced later" version would not, because `${NUM_SHARDS}` is
referenced later, in the echo. Do not bundle with other script changes.

---

### seed-7 — REFUTED

claim as stated: `config/scaling/devices/` passes seven guard tests that certify
a weak-scaling invariant whose device factor, `_meta.devices`, is read by
nothing outside the test file; the family is documentation with a test suite
attached, and no launcher can produce the device counts it claims to sweep over.

what I checked:
- All 4 rungs of `config/scaling/devices/` in full.
- `tests/test_config_system.py:802-894` — `TestFixedSizeDistributedFamily`, all
  7 tests, including both class and method docstrings and every assertion
  message.
- `tests/test_config_system.py:218-246` (`LADDER_CONSTANTS`), `:471-505, 598`
  (the `_meta.` exemption in the dead-key and path checks).
- Repo-wide grep for `_meta.devices` readers.
- `docs/plans/2026-09-27-wave1-audit-findings.md:326-334` (R8) and
  `...-perf.md:134-161` (P6).

what I found — the seed is factually accurate and rhetorically inverted.

1. **Every claim about the reader count is true.** `_meta.devices` has exactly
   two readers, `tests/test_config_system.py:864` and `:875`, both inside
   `TestFixedSizeDistributedFamily`. No `src/` code, no CLI flag, no script reads
   it. And the `_meta.` subtree is exempt from both the dead-key check
   (`:471,479,505`) and the unknown-key path check (`:598`), so the one key that
   would be flagged is the one key that is skipped. Confirmed.

2. **But the limitation is disclosed — five times — in the repo's own words, and
   the seed quotes the disclosure as if it were the evidence of deception.** All
   four YAML headers, verbatim and identical:

   > "Device count is a LAUNCHER argument, not a config key: the TPU/DDP side of
   > this repo is not built yet and this file deliberately does not invent a flag
   > for it. Invoke with whatever your launcher takes, e.g.
   > `torchrun --nproc_per_node=N -m src.train --fname ...`"

   And the test class docstring at `:803-810`, which is the sentence that matters:

   > "Device count is a launcher flag (the TPU/DDP side is owned elsewhere), so
   > the only device-dependent quantity the config system can express is the
   > per-device micro-batch. `data.batch_size` is therefore the sweep axis and
   > `optimization.grad_accum_steps` is pinned, holding the global effective
   > batch ... identical."

   That is an accurate, self-limiting description of exactly what the tests
   assert. The seed's charge that the tests "certify an unreachable invariant"
   requires the docstring not to exist.

3. **The one test whose name could overreach states its scope in its own
   docstring.** `test_global_effective_batch_is_held_fixed` (`:870-871`) reads
   `"""micro-batch x devices x accumulation is the token budget per step."""` —
   an *arithmetic* statement about the config family, and it is true as
   arithmetic. It does not assert anything about what a launcher does. And
   `test_device_count_is_recorded_and_increases`'s assertion message at `:861-863`
   gives the reason for a key with no reader: "the device count is a launcher
   argument, so the only way the sweep is reproducible is if the config says
   what it is meant to be run on". That is a *deliberate declaration of
   provenance*, which is what `_meta` is for.

4. **The runtime consequence is the already-known R8/P6, verbatim.** The seed's
   consequence paragraph — "produces 8 independent single-device replicas, each
   stepping on a 64-sequence batch, with gradients never summed and all 8
   writing to the same `output/scaling/devices/dev8-bs64/`" — is a restatement
   of R8 ("N ranks therefore run identical independent loops with the same seed
   and the same `logging.folder`, all writing the same checkpoint filenames")
   and P6 ("No process group, no all-reduce. Each rank is a fully independent
   replica. Effective batch is 4x B with gradients never summed" + "Checkpoint
   corruption. `logging.folder` is identical across ranks"). The scout lists R8/P6
   in its own "Not re-reported" section and then re-derives them in seed-7's
   consequence paragraph. The claim that this "is *not* a re-report of R8/P6
   because ... it is a distinct falsifiable claim" does not hold: the falsifiable
   claim that is *false* (the sweep can be launched as advertised) is R8's.

5. **The proposed fix would be a net loss.** Option (a), "delete
   `config/scaling/devices/` and `TestFixedSizeDistributedFamily` until DDP
   exists", deletes a correctly-arithmetic, honestly-labelled forward-looking
   artefact in order to hide a gap that R8 already records. That trades an
   accurate record of intent for the appearance of tidiness — the opposite of
   this repo's stated values. Option (b) asks for a rewrite of a docstring that
   is already accurate.

verdict: **REFUTED** (already disclosed in the artefacts the seed cites, and the
substantive consequence is a duplicate of the already-known R8/P6).

if refuted, the alternative explanation: the devices family is an
intent-declaration, not an experiment. It records "when the launcher exists, run
this shape on N devices with this per-device micro-batch" and it checks that the
four rungs are mutually consistent and consistent with the reference shape. Both
the configs and the test class say in plain words that DDP is not built. The only
real defect — that `torchrun` does not do what the header's example implies — is
R8, already filed, already verified, and already has an owner-shaped fix.

---

### seed-9 — SURVIVES (narrowed)

claim as stated: `config/tinystories/textspanjepa_tinystories.yaml` does not
error and does not read the TinyStories directory; it silently falls through to a
network call that downloads WikiText-103 and trains on it for 20 epochs. The
config's own "KNOWN GAP" comment describes a different failure mode than the one
that happens.

what I checked:
- `config/tinystories/textspanjepa_tinystories.yaml` in full (header :10-14,
  `_meta.note` :25, `data.root_path` :33, `optimization.epochs` :63).
- `src/train.py:724-790` (`_build_data_pipeline`) — the only dataset call in the
  trainer, at `:742-747`, and its validation twin at `:753-758`.
- `src/datasets/kaggle.py:46-89` (`load_wikitext103`), specifically the
  `os.walk` at `:62-67` and the bare `except Exception:` at `:69-79`.
- `pyproject.toml:14-20` — dependency list.
- `tests/` — every reference to `load_wikitext103` / `load_dataset` /
  `_build_data_pipeline`.
- `tests/test_config_system.py` `TestEveryConfigRuns`.

what I found — the mechanism is confirmed; the scope needs trimming.

1. **Confirmed: there is no corpus selector.** `src/train.py:742-747` calls
   `load_wikitext103` and passes `data.root_path` as `data_dir`. That is the only
   dataset call in the trainer. `data.root_path: data/tinystories` reaches
   `os.walk("data/tinystories")`, which yields nothing, so `filepath is None`.

2. **Confirmed: the fallback is a real, reachable network call, and it is
   inside a bare `except`.** `kaggle.py:70-79` does
   `load_dataset("wikitext", "wikitext-103-raw-v1", split=split)`. The
   refutation I expected — "maybe `datasets` is not installed, so the
   `ImportError` converts it to a `FileNotFoundError` and it fails loudly" —
   **fails**: `pyproject.toml:18` declares `datasets>=2.12.0` as a hard runtime
   dependency. The download path is live in any correctly installed environment.

3. **Confirmed: fully silent on success, mislabelled on failure.** On the happy
   path there is no log line, no warning, nothing between "Loading dataset..." and
   `Loaded WikiText-103 train: 103,227,021 tokens`. On the offline path the
   `except Exception` converts a connectivity error into
   `FileNotFoundError("Could not find WikiText-103 train data in data/tinystories.
   Add the wikitext-103 dataset to your Kaggle notebook.")` — a Kaggle-specific
   instruction for what is actually a network problem, on a config that is not a
   Kaggle config.

4. **Confirmed: the documented diagnosis is wrong, and wrong in the
   triage-relevant direction.** The header says:

   > "this config will load WikiText-103 text out of the TinyStories directory"

   `os.walk` on that directory yields nothing, the directory is never opened, and
   the run reaches the network. "Loads WikiText-103 out of the TinyStories
   directory" describes a stale-cache symptom; the actual behaviour is
   "downloads WikiText-103 from the internet". A maintainer triaging this reads
   the comment, concludes the run read the wrong *local* files, and fixes
   `root_path` — when the first-order problem is an unannounced network
   dependency. `_meta.note` at `:25` repeats the same wrong mechanism
   ("known src/ gap -- the trainer hard-codes load_wikitext103"), so the error is
   duplicated in the machine-readable place too.

5. **Confirmed: invisible to the guard suite.** Every test that touches the data
   path monkeypatches `load_wikitext103` wholesale
   (`test_training_e2e.py:131`, `test_checkpoint_fidelity.py:674,693,711,721`), so
   the real `os.walk` -> fallback path is never executed. And
   `TestEveryConfigRuns::test_deep_merge_then_validate` only calls
   `TextSpanJEPAConfig.validate()` for JEPA configs or checks
   `embed_dim % num_heads` for baselines — it never inspects `data.root_path`.
   The TinyStories config is green while pointing at a directory the loader
   cannot read.

6. The seed says the config "does not error". That is **conditional on network
   availability** and should be stated as such: online it does not error, offline
   it errors with the wrong message. Both are defects; only the first is
   silent.

verdict: **SURVIVES, narrowed.** The surviving defect is two-part and both parts
are small: (i) the header's mechanism claim is wrong in both the human-readable
and `_meta` copies, and (ii) the network fallback in `load_wikitext103` is
silent on success and mislabels connectivity failures as Kaggle dataset
absences. The seed's part (a) — adding a `data.dataset` corpus selector to
`src/train.py` — is a **feature request, not a defect fix**, and should not be in
the card (see KILLS).

if survives, the smallest fix:
goal — stop the silent network reach-out, and correct the two wrong diagnoses.
files — `src/datasets/kaggle.py` `load_wikitext103` only (the `except` block at
`:69-79`); `config/tinystories/textspanjepa_tinystories.yaml` (header `:10-14`
and `_meta.note` `:25`).
behaviour — when the on-disk walk misses, log at ERROR naming the directory that
was searched, then raise unless a new explicit `data.allow_network_fallback: true`
is set. Defaulting the new key to `false` is a behaviour change for anyone
relying on the download, so that is the one line needing a human decision — flag
it, do not assume it.
verify — a test that monkeypatches `datasets.load_dataset` and asserts a config
whose `data.root_path` contains no `wiki.*.tokens` raises rather than reaching
it; plus a second asserting the error message names the searched directory and
does not mention Kaggle. `tests/test_training_e2e.py` and
`tests/test_config_system.py` must stay green (they monkeypatch the loader, so
they should be unaffected).
Out of scope for this card: the `data.dataset` selector.

---

## KILLS

These should NOT become cards. Reasons recorded here for
`docs/decisions.md`.

1. **seed-2 (benefit 4) — the JEPA-vs-MLM token-budget asymmetry.** The 3.89x
   arithmetic is correct and I reproduced it. But the asymmetry is a *uniform,
   repo-wide protocol choice* — all 8 baseline configs set `mask_ratio: 0.15` /
   `span_length_range: [1,1]`, no JEPA config overrides either — it is the
   method-appropriate value for each objective, and it is **already disclosed in
   every baseline config header** with an explicit classification ("a pre-existing
   protocol difference, not a config defect") and an explicit routing to a human
   ("reconciling it is an experiment decision rather than a schema one"). The
   proposed card invents two new experiment configs to answer a question nobody
   has asked. That is `AGENTS.md` rule 2 (no speculative work) and rule 4 (YAGNI).
   *Residue worth keeping, not worth a card:* the disclosure names the epoch term
   (1.67x) and omits the larger mask term (2.33x). One sentence, if anyone is
   already editing those headers.

2. **seed-7 (benefit 4) — the `config/scaling/devices/` family and its guard
   tests.** The reader-count fact is true, but the limitation is disclosed four
   times in the YAML headers ("Device count is a LAUNCHER argument ... the TPU/DDP
   side of this repo is not built yet and this file deliberately does not invent a
   flag for it") and once more in the test class docstring, which describes
   precisely what the tests assert. The substantive consequence — that
   `torchrun --nproc_per_node=N` yields N independent replicas with gradients
   never summed, all writing the same checkpoint filenames — is a verbatim
   restatement of the already-known, already-VERIFIED **R8** and **P6**. The seed
   quotes the disclaimer as proof of deception, which inverts its own evidence.
   The proposed fix (delete the family) would remove an honest record of intent
   to make the repo look tidier while R8 stays filed. Not a card.

3. **seed-3's "three unintended regularisation regimes"** — refuted as a
   sub-claim and must not appear in the seed-3 card. JEPA uses drop-path, MLM
   uses dropout, data2vec uses neither. That is standard per-method practice
   (BERT-family models use dropout; ViT/GPT-family JEPA models use stochastic
   depth), not a confound. Only the two silently-discarded `drop_rate: 0.1`
   declarations are defects, and only the *delete-the-inert-key* branch is
   in-scope; forwarding the parameter changes baseline results and is a
   separate behaviour-changing card.

4. **seed-9's "add a `data.dataset` corpus selector"** — out of scope for a
   defect card. It is a feature request (make the corpus explicit in
   `src/train.py`) smuggled in as part (a) of a fix. The defect is the silent
   network fallback and the two wrong diagnoses. Land those; file the selector
   as its own card if the human wants it.

5. **seed-6's "indistinguishable from `scripts/wikitext/train_small.sh`"** — a
   minor overstatement to be dropped from any card. The two differ in
   `logging.folder` and in what `$1` means. The substance (both train
   WikiText-103 at 640/10/10) holds. Do not carry the absolute claim.

6. **seed-1's "81,958,720"** — a transposition typo for **81,758,720**. The
   correct figure is exactly `170,706,561 - 88,947,841`, i.e. one encoder
   deepcopy. Any card quoting 81,958,720 is quoting a number that does not
   reconcile with the config's own documented total.

7. **seed-1's framing of the 1.498x as a JEPA excess** — misleading and should
   be corrected in the card. The JEPA total is inflated by a *frozen* target
   encoder that carries no gradient path. The meaningful asymmetry is that the
   **baseline carries 1.28x the trainable capacity**, entirely in a separate,
   untied 32,194,560-param `mlm_head` that the JEPA does not have at all
   (JEPA's `TiedTokenDecoder` is 1,639,680 params and reuses the token
   embedding). The corrected framing is a stronger finding and the one a
   reviewer should be given.

## NOTES FOR THE COORDINATOR

- The brief says "the six with benefit >= 4" and then lists seven. All seven
  are covered above; no seed was skipped for that reason.
- `src/models/cmc.py` was **mid-write and contained 15,225 NUL bytes** while I
  was running probes, which made `import src.models` fail with
  `ValueError: source code string cannot contain null bytes`. It is `git status`
  dirty. Another agent is editing it concurrently. I did not touch it, but a
  wave barrier that runs the suite will fail until that write lands.
- I did **not** re-report the stale wave-1 config findings (V1/V3/V5) as
  instructed. I did read `docs/plans/2026-09-27-wave1-audit-findings*.md` to
  check seed-5 and seed-7 for duplication against R8/P6/V10, which is the only
  reason I opened them.
- Net result: **4 of 7 survive** (1, 3, 5, 6), **2 of 7 are killed** (2, 7), and
  seeds 3 and 9 survive only in narrowed form. The three strongest survivors
  (5, 1, 3) are all cheap: 3 config lines, one docstring, 2 config lines.
