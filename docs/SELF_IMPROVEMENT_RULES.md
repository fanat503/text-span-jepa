
Self-improvement rules derived from a paper that measured harness design.

Source: "An Empirical Study of Harness Design for Coding Agents"
arXiv:2609.20804, 17 Sep 2026 (UMass Amherst, Emory, UNC, Zoom).

Not a JEPA paper. It ablates a coding-agent harness - planning, action space,
context management - across 176 matched settings (4 models × SWE-Bench
Verified + Terminal-Bench 2.1 × five context-management tiers × four window
budgets, plus planning/action-space ablations at the T4/128k default). The
reason it is in this repo is that I am the harness it studies, and the session
that produced this campaign failed in ways the paper predicted precisely.

Each rule below is a measured finding from the paper, followed by the failure
it explains, followed by the rule. A rule with no failure behind it is
folklore and is not written down here.

---

## R1 — Durable state goes to disk before it lives in context

**Measured (paper, finding 1):** the value of context management grows as the
window tightens, and nearly all of its benefit is one thing - preventing
context-overflow failures. The managed–T0 success gap is 35.7 pp at 32k and
2.7 pp at 128k on SWE-Bench (15.9 / 5.5 between), and it tracks the T0
overflow rate falling from 78.7% to 8.7% (61.0% → 12.1% on Terminal-Bench).
Every managed tier overflows on exactly zero tasks at every budget. Context
management "extends execution trajectories without substantially altering
agent behavior" - it does not make the agent smarter, it stops the run from
dying.

**The failure it explains:** this session was interrupted by server restarts
and context compactions at least three times. Everything that lived only in
my context died with it - including a worker dispatch whose whole state was
"unknown" afterwards, and a half-written rules document (this file was itself
found as a 705-byte header with every rule below it missing). Nothing was
lost the second time because the recovery path was: read the repo state from
git, not from memory.

**Rule:** before a turn goes long, put its durable state in files (plan,
measurement scripts, notes) so that a compaction or restart costs a re-read,
not a re-derive. Never end a turn with the next step existing only in my own
context. Treat every summarization event as if the last undischarged plan
will be gone - because it was.

## R2 — Cheap machinery first; machinery nobody invokes is dead weight

**Measured (paper, finding 2):** staging rule-based elision before LLM
summarization (T4) gives the best accuracy–cost trade-off - comparable
success with the lowest mean cost in 7 of 8 model–benchmark panels, lowest
mean peak-context ratio at all four budgets. The expensive part it avoids
(LLM summarization calls) is invoked less than by any summarization-only
tier at every budget. Meanwhile the LOSSLESS part of the machinery -
recall_event, external storage - was almost never used: 36 of 64 T2/T4
settings (56.3%) never call it, median invocation rate zero, equal-weight
mean 0.540 calls/task at 32k falling to 0.007 at 128k, and T2 beats T1 by
−0.36 pp on average (+0.40 SWE-Bench, −1.12 Terminal-Bench).

**The failure it explains:** the campaign repeatedly built heavyweight
machinery whose invocation nobody measured. The clearest case is the CMC
gate: a `min_overlap_ratio: 0.2` threshold shipped with no measurement of
whether the gate ever OPENED - the metrics audit later found it effectively
shut at the shipped ratios, and wave 16's fix was to emit `cmc_gate_ratio` /
`cmc_gate_open` / `cmc_skip_reason` precisely so the invocation question
becomes observable. The same pattern: elaborate multi-paragraph worker
prompts whose structure the workers demonstrably did not follow (they kept
returning artifacts into forbidden paths, which is why `.agent-notes/` ended
up in `.gitignore`).

**Rule:** for every mechanism (mine or the model's), measure the invocation
rate before trusting the effect: how many times did the gate open, the
branch run, the rule fire? Prefer cheap deterministic interventions that
prevent failure outright over expensive intelligent ones that "add
machinery" - and kill machinery with a measured ~zero invocation count.

## R3 — Planning is a scaffold with a measured lifetime, not a ledger to maintain

**Measured (paper, finding 3):** planning is an accuracy scaffold for the
weakest model and a cost saver for the strongest. For Nemotron-3 30B,
disabling planning collapses the median SWE-Bench trajectory from 40 to 5
turns, 68.6% of runs terminate without an edit (58.4% still in Localize);
with planning, 27.8% / 10.4%. For the two strongest models, planning mostly
deletes redundant post-edit verification: cost −30% / −32% on SWE-Bench for
success-rate deltas of −2.0 / −0.4 pp. Planning "changes where trajectories
stop."

**The failure it explains:** the campaign maintained a growing task-ledger
system (TASKS.md, plan.md, per-group boards, consolidation notes) through
many waves - and the end-state cleanup deleted ALL of it (115 bookkeeping
files) as reviewer-visible pollution. The ledger's lifetime value was the
scheduling during the wave; its maintenance cost kept being paid long after
the scheduling was done, and it was paid in the currency a reviewer reads.

**Rule:** plans are scaffolds with a deletion date; write the minimum that
makes the next step unambiguous, execute, and remove the scaffold when the
work product is in. Verification artifacts (tests, commit messages, measured
verdicts) are work product - task ledgers are not. Do not pay for ledger
maintenance after its scheduling value is spent.

## R4 — Use the interface whose failure modes are already measured

**Measured (paper, finding 4):** the action space changes the granularity at
which code is written, and the right interface is model-dependent. Bash-only
shortens Terminal-Bench median trajectories (47 → 31 actions) while raising
the Write-code share (16% → 27%) and cutting re-patches (3.3 → 0.4, 4.6 →
1.5) - capable models bundle several operations into one command. But for
bash-weak models, bash-only is fatal: 66% of Nemotron-3 30B Terminal-Bench
trajectories die on out-of-interface emissions, average trajectory 71 → 15
turns.

**The failure it explains:** on this machine, freeform PowerShell one-liners
kept failing on measured, repeated quirks - the `ForEach-_-object` mangling
(at least twice), `&` redirection parse errors, quote mangling in `gh --jq`,
encoding breakage on non-ASCII output - while the same operations via a
Python script (written with the `write` tool, run with a full interpreter
path) worked every time.

**Rule:** after an interface fails twice the same way, it is measured, not
unlucky: switch to the interface whose failure modes are benign. Here that
means write tool + Python subprocess scripts over interactive PowerShell
idioms, and typed file tools (`write`/`edit`) over shell heredocs. For
bash-capable models, prefer the batched coarse action; for an interface the
model cannot drive reliably, prefer the structured one even at more turns.

## R5 — End turns on milestones, not on the first friction

**Measured (paper, finding):** context management extends execution
trajectories without substantially altering agent behavior - phase ordering
and proportions stay comparable at matched turns; the effect is purely that
runs survive longer (median SWE-Bench length 20–30 turns unmanaged at 32k →
50–180 managed).

**The failure it explains:** the user's complaint "перестань заканчивать
ответ, когда почти ничего не сделал" - turns ended on the first tool error
or after one small fix, when the correct behavior was to continue: an
interrupted answer is a trajectory that stopped where friction happened,
not where work was done.

**Rule:** a turn ends on a natural milestone (a merged PR, a verified fix, a
written report), not on the first exception. Friction inside a turn is
handled by the next tool call, not by ending the turn. The invariant from
the paper: extending the trajectory does not change the quality of each
step - it changes how far the work gets.

## R6 — One run per cell is a direction, not a result

**Measured (paper, Limitations):** each setting ran once per task;
Terminal-Bench has only 89 tasks, so many Terminal-Bench contrasts do not
reach significance under the paired McNemar test, and the authors state
their conclusions there rest on consistent directions across models and
budgets rather than on individually significant cells.

**The failure it explains:** the campaign's most dangerous moments were
claims formed from one measurement - the "CMC is dead" verdict that a later
audit had to retract (the threshold compares against the primary mask
fraction `p`, not `p²`), and the "179-nat gap" being initially attributed
without isolating which of the two softplus paths produced it.

**Rule:** a single measurement licenses a hypothesis, never a claim. A claim
needs either a significant paired test or a consistent direction across
every cell of a small grid (seeds × configs). For the paper itself: no
mechanism-effect statement from a single seed - state the direction, the
number of cells that agree, and the test.

## R7 — Read the decision record before reconciling an instruction

**Measured (paper, harness structure):** the components that are HELD FIXED
(safety gates, post-edit diagnostics, stuck detection) are what make the
varied components' effects interpretable - "the harness follows a ReAct loop
with the execution loop fixed." Unfixed substrates make every comparison
confounded.

**The failure it explains:** instructions arrived repeatedly as a standing
prompt that restated completed work (delete branches, read the paper, fix
metrics) without a record of what was already done. Twice this produced
re-verification of closed work and once it nearly produced a wrong action -
deleting work-tracking files whose removal would have destroyed the only
granular history, which was prevented by an explicit user decision that
superseded the standing instruction.

**Rule:** the newest recorded decision wins over any standing instruction;
when they conflict, say so out loud instead of silently following either.
Concretely: before acting on a standing directive, check the repo's decision
record and the git state, and report completed items as completed rather
than re-executing them.
