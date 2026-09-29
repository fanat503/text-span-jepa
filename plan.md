# plan.md — the campaign's on-disk brain

Everything needed to continue lives in files, not in anyone's context. This file
is the index; the files it points at hold the detail.

**Why this file exists.** A session can be compacted, restarted, or handed to a
different agent. The campaign must not depend on any single model's recall. If
you are reading this for the first time and the state below is unfamiliar, trust
the files over any memory of a previous conversation.

## Where things are

| Need | File |
|---|---|
| Remaining work, one card each | `TASKS.md` |
| Closed-tick log | `.agent-notes/consolidation.md` |
| Current tick's strongest findings | `.agent-notes/brief.md` |
| Group leader memory | `.agent-notes/groups/*/board.md` |
| Decisions + rejected alternatives | `docs/decisions.md` |
| Worker evidence | `.agent-notes/task-NN.md` |
| Scouting seeds | `.agent-notes/raid/seed-N.md` |
| The whole gate in one command | `.agent-notes/gate.sh` |
| Audit findings behind the cards | `docs/plans/2026-09-27-wave1-audit-findings*.md` |
| Project rules + CPU constraint | `AGENTS.md` |
| Integration base state | `campaign/integration` |

## Progress ledger

| Metric | Value |
|---|---|
| Branch | `agent/wave-1`, gate green |
| Integration base | `campaign/integration` (forked from `audit-hardening`, 1541 tests) |
| `main` | read-only, equals `origin/main`, 686 tests |
| Ticks closed | 0 (tick 1 merged; verifier not yet run) |
| Cards | 26 total — 3 warmup, 9 arena, 14 solo |
| Cards closed | 7 (TASK-01, 02, 03, 04, 07, 13, 17) |
| Test count | 686 at `origin/main` → 1541 on the integration base |
| Parallelism `X` | 8 |

## Hard constraints

These are not preferences. They are the reason several otherwise-sensible
actions are forbidden here.

1. **Never run training.** The owner games on this 6-core box. `tools/rt.py` and
   the safety policy both refuse it.
2. **Tests only via `tools/rt.py`**: one thread, cumulative budget, no
   whole-suite run. The authoritative whole-suite signal is the required CI
   check, which is free and off-machine.
3. **The owner's CPU is the scarce resource, not tokens.** Prefer more agents on
   cheap work over fewer on heavy local verification.
4. **`main` is a base.** Merges go to `agent/wave-N`, then a PR.
5. **Push only on the owner's word.**

## Things known to be unresolved

Carried here so a context reset does not lose them.

- **Three fixes shipped in tick 1 with no detector.** TASK-01 (`conftest.py`),
  TASK-02 (`cmc.py`), TASK-07 (`run_comparison.py`). Each worker reported it
  themselves. The verifier must decide whether that is acceptable and write the
  test if it is not.
- **A possible resume regression introduced by TASK-02.** The new private
  generator's stream position is module state, not a registered buffer, so
  `_capture_rng_state` cannot save it. Resume was exact before, because CMC
  shared the checkpointed global stream. This campaign exists because resume
  diverged 5.8%, so it is the highest-priority thing to confirm or refute.
- **One `xfail` still stands**, in `tests/test_training_state_guards.py`, for
  `wsr_mode=sam`. The rules forbid xfail. It awaits the owner's decision; see
  `docs/decisions.md` D-010 and cards TASK-09/10.
- **The RAID produced 10 seeds, 6 with benefit >= 4.** They are unassigned. The
  highest is `seed-1`: `baselines/mlm_baseline.py` claims "identical model
  capacity / identical compute" and is false in both directions — measured
  JEPA total 1.498x the baseline, and the baseline's *trainable* count 1.281x
  JEPA's. A paper whose control is not a control is rejected at review, not at
  rebuttal.
- **Config findings V1/V3/V5 in the audit docs are stale.** They were fixed
  during setup. Do not re-open them; re-derive from `docs/decisions.md`.

## How to continue

Read `TASKS.md`, the tail of `.agent-notes/consolidation.md`, and
`.agent-notes/brief.md`. Run `bash .agent-notes/gate.sh` and confirm green. Then
take the **lowest-numbered unfinished card**, not a new one.

The full recovery procedure, including which rules do not relax, is in
`~/.config/opencode/CAMPAIGN-RESUME.md`, which opencode re-reads automatically
after any context compaction.
