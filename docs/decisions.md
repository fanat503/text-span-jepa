# decisions

Problem → options → why this one. Alternatives that lost, and findings
killed by later measurement, are recorded too, with the reason.

Format per entry: what was decided, what the alternatives were, why the
rejected ones lost, and what would reverse the decision.

---

> ## Reading the numbers in this file
>
> This is a **log**, so the numbers in an entry are the numbers that were true
> when the decision was taken, and they stay that way. They are not restated as
> current. Two entries carry counts that have since moved and are flagged here
> rather than edited:
>
> | entry | number as written | current |
> |---|---|---|
> | D-001 | "The local full suite is 1541 tests" | over 2,000 tests collected locally; a full CI run is recorded at commit `9732de2`: 1910 passed / 21 skipped / 1 xfailed |
> | D-002 | "~82s", "1483 of the 1541 assertions" | no wall clock is quoted: the machine is shared with its owner |
>
> The **decisions** in both entries are unaffected: neither one is about how many
> tests there are, only about where the number is measured from. Updated
> 2026-10-08.

---

## D-001 The full-suite baseline comes from CI, not from a local run

**Problem.** The project needs a baseline, and the rules require acceptance by
pasted command output. The local full suite is over 2,000 tests and saturates all 6
cores on a machine whose owner is using it to play games.

**Options.** (a) run it locally; (b) take the number from the GitHub Actions
`pytest` job; (c) skip the baseline.

**Decided.** (b). The CI runner is free, off-machine, and produces the same
evidence — same suite, same assertions — with a longer traceback. The local box
stays free.

**Why the others lost.** (a) spends the owner's CPU to obtain a number that
already exists for free, and would do so again on every gate. (c) is
unacceptable: without a baseline there is no regression signal at all.

**Reverses if** the repo gains a test that behaves differently on Linux CI and
Windows (spawn vs fork). That case is real and currently unverified — see
D-012.

---

## D-002 `gate.sh` default is the fast gate, `--full` is opt-in

**Problem.** The definition of done requires the gate to be green in the main
tree *and* in a clean clone. If the gate means the whole suite, that is ~82s of
CPU per invocation, several times during development.

**Options.** (a) gate = full suite; (b) gate = lint + format + fast contract
tests on one thread, with `--full` available; (c) gate = lint only.

**Decided.** (b). The fast gate is ~45s on a single thread and covers ruff,
black, and 1483 of the 1541 assertions — the config grammar, the checkpoint
security and fidelity contracts, the training-state guards, all twelve
mechanisms plus sterility, and the four interpretability contracts. The
remainder is the long tail, and its authoritative signal is the required CI
check.

**Why the others lost.** (a) makes the owner's gaming this project's problem.
(c) would let a correctness regression through, which is the exact class of
defect this project exists to remove.

**Cost accepted.** A regression confined to the long tail is caught by CI, one
to three minutes later, and not by the local gate. That is a deliberate trade:
slower detection in exchange for not stealing CPU.

---

## D-003 `TrainingStateGuard` lives in `src/models/_state_guard.py`

**Problem.** The shared state guard was first placed inside `wsd.py` because no
dedicated module existed yet, and the other seven mechanism modules imported it
from there. A shared primitive had grown inside one of its consumers -- a
placement error, not a layering decision.

**Options.** (a) leave it in `wsd.py`; (b) move it to `_state_guard.py`; (c) move
it to a `utils/` module.

**Decided.** (b). It guards mechanism state, so it belongs beside the mechanisms,
and eight modules depend on it.

**Why the others lost.** (a) means a guard that all mechanisms depend on lives
inside one of the mechanisms; deleting or refactoring `wsd.py` would silently
break seven modules. (c) puts model-training policy outside the model package,
where nothing would look for it.

---

## D-004 No shebang in `tools/rt.py`

**Problem.** CI lint failed with `EXE001 Shebang is present but file is not
executable` on Linux while local ruff was green.

**Options.** (a) set the executable bit; (b) drop the shebang; (c) add `EXE001`
to the ruff ignore list.

**Decided.** (b). The file is invoked as `python tools/rt.py`, so the shebang was
never used.

**Why the others lost.** (a) means chmod +x, which Windows git does not record
reliably — the failure would come back. (c) is exactly the "make the linter stop
complaining" move that the project's own rules forbid, and it would hide the
same class of problem for the next person.

**Also.** The commit that added a Windows CI job was reverted, because
`.github/workflows/*` is off-limits. The net diff on that file is zero. The
commit and its revert both remain in the branch history: rewriting the history
of an already-pushed branch is more dangerous than leaving a visible
commit/revert pair, and the file is byte-identical to `main`.

---

## D-005 `defaults.yaml` now describes the full 12-mechanism GWP model

**Problem.** Before this work `defaults.yaml` had `use_jawp: true` and every
other mechanism false, so an unqualified run trained a JAWP-only model. The
ablation grid could not be expressed against a coherent reference.

**Options.** (a) keep JAWP-only as the default; (b) make the full GWP model the
reference; (c) add an include mechanism to the config loader.

**Decided.** (b), with the shape declared once in `defaults.yaml` and inherited
by everything else.

**Why the others lost.** (a) leaves the ablation table with no single reference
model to ablate against. (c) is the cleaner engineering answer but changes the
merge contract in `src/train.py`, which is out of scope for the change that
adopted (b); (c) is recorded here as the right long-term move.

**This is the single most reviewable decision in this repository.** It changes what
an unqualified `python -m src.train` trains. A reviewer who disagrees should say
so before anything is trained, because every number in the ablation table is
relative to it.

---

## D-006 `ema_tau_end` is 0.9999, declared once, in `defaults.yaml`

**Problem.** Eighteen shipped configs set `ema_tau_end: 1.0`. That fails
`validate()`, and worse, `EMATauSchedule.step()` returns exactly `tau_end`, so
`update_target_encoder` became `mul_(1.0).add_(q, alpha=0.0)` — the target
encoder froze. In self-distillation the target encoder is the training signal,
so this is a broken run, not a rounding difference.

**Options.** (a) 0.9999; (b) 0.999; (c) treat `1.0` as an intended frozen
teacher and relax `validate()`.

**Decided.** (a), for three reasons: it is what the repo already used in
`defaults.yaml` and in `wsr_on`/`no_wsr`; it is the I-JEPA/DINOv2 standard
endpoint, the same lineage `src/utils/schedulers.py:66` cites; and it makes
`0.996 -> 0.9999` a ramp with real dynamic range, whereas `EMATauSchedule`'s
own docstring says constant tau is suboptimal.

**Why (c) lost.** A genuinely frozen teacher needs a different mechanism — a
momentum queue, or a pretrained teacher checkpoint. Shipping it as the silent
value `1.0` hides that behind a validation error. If a frozen teacher is
wanted, it should be its own named config with its own proof.

**Guard against regressing to (c).** A test zeroes the target, fills the online
encoder with 1.0, calls `update_target_encoder(tau)`, and asserts every target
parameter equals `1 - tau = 1e-4`. At `tau = 1.0` that value is 0.0 and the
assertion fails, so the freeze is caught numerically and not by a bound on a
number.

**Open → CLOSED 2026-10-08 (verification, no code change).** As written
this item said `src/train.py:836` read `ema_tau_end` with a fallback of `1.0`, so
a `--no_defaults` run froze the target encoder anyway. **That is no longer the
code**: the fallback is now `0.9999` and it lives at `src/train.py:848`
(`tau_end=model_cfg.get("ema_tau_end", 0.9999)`). A `--no_defaults` run therefore
lands on the same EMA endpoint as a merged one, which is what this entry wanted
when it was escalated. The decision stands; only the open item is closed.

---

## D-007 `no_jawp` activates 10 mechanisms, not 11

**Problem.** The leave-one-out row for the root mechanism cannot be 11. The code
constructs WSD only under `if config.use_wsd and use_jawp:`, so there is no
configuration in which WSD is on and JAWP is off.

**Options.** (a) add `use_wsd: false` to `no_jawp` so the count reads 11;
(b) leave the row as 10 and pin the consequence; (c) change the code so WSD can
run without JAWP.

**Decided.** (b). `no_jawp` flips exactly one flag — `use_jawp` — and activates
ten. A test names the reason.

**Why (a) lost.** It would make the table read 11 while the experiment is a
two-variable delta, which is precisely the defect this work removed from the
other nine files. Choosing the count over the truth reintroduces the bug.

**Why (c) lost.** WSD's proof is a drift bound against JAWP's workspace, so
running it without one is a different mechanism, not a configuration.

**Consequence for the paper.** If the "root and all four dependents removed" row
is wanted, it is a different experiment and needs its own name.

---

## D-009 `main` is a base only; all work lives in branches

**Problem.** During setup a 97-file change was committed directly to local
`main` before this rule existed. That commit is not on the remote, so nothing was
published, but the local branch was doing double duty.

**Options.** (a) leave it; (b) reset local `main` to `origin/main` and keep the
work in the feature branch.

**Decided.** (b). Local `main` now equals `origin/main` and the work is
reachable from `audit-hardening` / PR #10.

**Why (a) lost.** A `main` that already carries unreviewed work makes "main is
read-only" unenforceable by convention alone.

---

## D-010 One `xfail` exists and is waiting on a human decision

**Problem.** The project rules forbid skip, xfail, `--no-verify`, and weakening
tests. One xfail is present in the branch:
`TestWSRSamNoSilentSubstitute::test_retraction_preserves_column_orientation`.

**Context.** `wsr_mode=sam` selects column signs from a block that is not
triangular for a `(D, k)` matrix with `D > k`, so the retraction flips
individual columns: for a random orthonormal Q with `rho=0.05` the retracted Q
lands 4.0 away in Frobenius norm instead of about 0.05. The test is real and
currently unfixable without deciding what `wsr_mode=sam` should compute, which
is a research decision rather than a bug fix.

**Status.** Left in place, uncommented, unskipped, and reported. Not touched
silently, which is what the rules require.

**Needs:** a decision from the owner. The options are (a) fix the sign
selection and un-xfail, (b) redefine `sam` to match what the code does, or (c)
remove the mode and fail loudly if anyone selects it.

---

## D-011 Findings killed by their own controls are recorded, not silently dropped

**Decided.** When a later measurement contradicts a finding, the finding is kept
in this file marked *killed*, with the measurement that killed it. Editing a
finding away makes the record look more settled than it is.

**Worked example, and the reason the rule exists.** The CKA values
`0.96 / 0.89 / 0.49` at `N = 32/100/800` were reported as a defect and "fixed"
with the unbiased HSIC of Kornblith et al. (2019, eq. 4). The fix turned out to
be a numerical no-op, and the reported value is the true value of Kornblith CKA
at `D >> N` — not a bug. Killed, not fixed; the estimator in
`src/utils/cka_metrics.py` documents its own form so the number cannot be
re-quoted from this file.

---

## D-012 Windows-specific behaviour is unverified

**Problem.** `.github/workflows/*` is off-limits, so no Windows job was added.
The Linux CI job cannot see defects that only appear under the `spawn` start
method, which is exactly how a lambda `worker_init_fn` shipped in a config that
cannot start on a Windows host while CI stayed green.

**Current state.** The lambda is fixed and `tools/rt.py` runs on Windows, so the
local gate exercises the Windows path. But the local gate does not run the
whole suite, and CI does not run on Windows.

**Open risk.** A test whose outcome depends on `spawn` vs `fork` could fail on
Windows and be invisible in CI. Not currently known to exist; not ruled out.

**Reverses if** someone can add a Windows job, or the owner accepts a local
`--full` gate run on Windows as the compensating control.

---

## D-013 GWP is 12 modules and 16 numbered capabilities; say which

**Problem.** Four files stated four different totals. `mechanisms.py`'s header
and `GWP.N_MECHANISMS` said 16; `MechanismBundle.ALL_MECHANISMS` had 12
entries; `proofs/IMPLEMENTATION_STATUS.md` said 13 and carried 12 rows;
`proofs/README.md` said 13 and carried 11; `README.md` asserted a bare
"novel mechanisms (16)"; and `tests/test_model.py` had a test *named*
`test_mechanism_bundle_counts_16` that asserts `active == 12`. A NeurIPS
reviewer reads the first paragraph.

Measured, not inferred: `ALL_MECHANISMS` = 12 (`mechanisms.py:100`),
`N_MECHANISMS` = 16 (`mechanisms.py:682`), the header numbers exactly 16
contiguous items (`mechanisms.py:37-56`), `defaults.yaml` carries 12
`use_<mech>` keys, `config/ablations/` carries 12 leave-one-out `no_<mech>.yaml`
files, and `proofs/` carries 13 proof documents.

**Options.** (a) "12 mechanisms" everywhere; (b) "16 mechanisms" everywhere;
(c) state both, with the mapping, and never use either unqualified.

**Decided.** (c). The convention is: **12 when counting modules, 16 when
counting numbered capabilities, never either unqualified.** It is written out
in full, with the 16→12 mapping table, in `proofs/README.md`.

**Why (a) lost.** `GWP.N_MECHANISMS = 16` is pinned by
`tests/test_model.py::TestGWPFramk::test_gwp_import`. Adopting a bare "12"
makes that constant a lie and forces a code change outside this entry's scope.
**Why (b) lost.** It is the current state of the README and it is the one a
reviewer reads as an overstatement: only 12 things can be constructed, toggled,
ablated or counted by `active_mechanisms()`.

**What the extra four are.** Not modules. WIP (#2), Spectral Gap (#3),
Grassmann Optimization (#4) and Predictive Rank (#5) are **methods of
`JAWPModule`** — line numbers re-derived 2026-10-08, because the ones originally
written here had all drifted:
`detect_workspace_dimension` (`src/models/jawp.py:552`),
`workspace_information_preservation` (`jawp.py:659`),
`grassmann_retract` (`jawp.py:893`), `principal_angles` (`jawp.py:955`),
`subspace_distance` (`jawp.py:1005`), `predictive_rank_loss` (`jawp.py:1166`).
They cannot be turned on or off as units: `use_wip` does not exist anywhere in
the repo.

**Consequence recorded, not fixed.** Predictive Rank (#5) *is* a trained loss
term — `jepa.py:858` adds `lambda_predictive_rank * loss_pred_rank`, default
`0.0` (`defaults.yaml:198`), with an on-arm at
`config/ablations/predictive_rank_on.yaml` — yet it
has no `ALL_MECHANISMS` entry, so `active_mechanisms()`,
`mechanism_groups()`, `dependency_dag()` and `GWP.summary()` do not see it and
report `Core: ['jawp']`. All three sites are under `src/**`, out of scope here,
so the gap is documented rather than closed. (The `jepa.py:810`
line number written here had drifted; `defaults.yaml:140` had too.)

**What now pins the truth.**
`tests/test_config_system.py::TestAblationGridComplete::test_all_mechanisms_length_is_stable`
already asserts `len(ALL_MECHANISMS) == 12` and names this file,
`GWP.N_MECHANISMS` and the mechanisms.py header in its own failure message;
`test_gwp_import` asserts `N_MECHANISMS == 16`. So *if the code changes*, one
of those two tests fails and the message names the four sites to update. That
is a better guard than any prose count, and it already existed.

**Reverses if** someone adds a mechanism module (then `ALL_MECHANISMS` goes to
13 and the 16 no longer decomposes 12 + 4 JAWP methods), or promotes one of
the four JAWP methods to a module (then the gap closes and "16 capabilities,
12 modules" needs rewording), or gives Predictive Rank an `ALL_MECHANISMS`
entry and `use_*` flag (then it becomes a 13th module and the visibility gap
above is moot).

---

## D-014 The SSL baseline arms declare their view construction; the rows are not renamed

**Problem.** BYOL, Barlow Twins, SimSiam and VICReg are defined over two
augmented views of the same input. This repository hands all four arms the
trainer's `(masked, clean)` pair — one view span-masked, one untouched — so the
pair is nested, not two independent augmentation draws. A row labelled "BYOL"
that is not BYOL-as-published is exactly what a reviewer calls an unfair
comparison, and the fix is not to rename the row.

**Options.** (a) rename the rows; (b) wire a genuine two-augmentation pipeline
(a text augmentation policy per paper, an independent second draw per step) and
re-measure every collapse threshold in `tests/test_ssl_baselines.py`, which were
all measured under the masked/clean pair; (c) keep the arms and state the
divergence at the point of use.

**Decided.** (c). Each arm's `compute_loss` docstring and each ablation config
declares the construction; `TestTheTwoViewsAreNotThePublishedAugmentationPair`
checks the declaration against the trainer's real tensors rather than trusting
prose.

**Why (b) lost.** It is a science change, not a documentation pass: it changes
what every SSL arm optimises and voids every threshold measured here. It
belongs with the owner's training plan.

**Reverses if** a two-augmentation pipeline is wired in — then the arms become
the published methods and the docstrings, configs and thresholds must all move
in the same commit.
