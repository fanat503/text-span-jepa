# Audit contract: proof claim vs. implementation assumptions

Read this whole file before doing anything. It is the full contract.

## Why this audit exists

`AGENTS.md` (repo root) says: **proofs are DESIGN DOCS, not specs.** Never
"fix code to match the proof" without an explicit human decision. So the
deliverable is NOT a set of repaired proofs. The deliverable is an honest
separation, per mechanism, of three different things:

- **(a)** what the proof document in `proofs/<mech>.md` actually claims,
- **(b)** whether `src/models/<mech>.py` satisfies the proof's **ASSUMPTIONS**,
- **(c)** if not, exactly which assumption fails and the practical consequence.

A theorem whose hypotheses do not hold in the implementation is a **DIVERGENCE**.
Naming one is the job. Repairing one is not — escalate instead.

## Your assigned mechanisms

You are given 3 mechanism names, e.g. `jawp wsd wsr`. The 12 modules in
`MechanismBundle.ALL_MECHANISMS` (`src/models/mechanisms.py`) are:

```
jawp                          (Core)
cgn swip pcr spc              (Routing)
wsd cmc gac sta puc rdc wsr   (Stability)
```

For EACH assigned mechanism:

1. Read `proofs/<mech>.md` **and** `src/models/<mech>.py` **together**. Then
   read `proofs/IMPLEMENTATION_STATUS.md` for that mechanism's known state,
   and `proofs/README.md` for the 12-vs-16 convention.
2. Also read `defaults.yaml` and the matching `config/ablations/*_<mech>*.yaml`
   only if the proof's assumption depends on a config value or default.
3. For EACH numbered theorem/claim in the proof: list its **assumptions**, then
   mark each **HOLDS** or **FAILS** with a `file.py:NNN` citation from the code
   (or a `yaml:NNN` citation). Cite path and line. NEVER quote full source.
4. Record the **practical consequence**: what does a reader get wrong if they
   trust a number this mechanism prints?

## Output format — return this, nothing else

For each assigned mechanism, in markdown:

```
### <mech>
CLAIM: <one or two sentences: the headline theorem as the proof states it>
ASSUMPTIONS:
- A1: <assumption> -> HOLDS (<file:line>) | FAILS (<file:line>)
- A2: ...
DIVERGENCES:
- D1: <assumption that fails> — <evidence: path:line>. CONSEQUENCE: <what a
  reader mis-believes if they trust the printed number>
WHAT HOLDS UP:
- <the parts of the proof the code does satisfy>
OPEN / COULD NOT DETERMINE:
- <what you could not settle, and exactly why>
```

Rules for honesty:

- If you cannot verify something, put it under OPEN / COULD NOT DETERMINE with
  the reason. Do not guess. A confident wrong line number is worse than an
  admitted gap.
- Do NOT re-derive the Verified/Divergent/Gaps counts in
  `proofs/IMPLEMENTATION_STATUS.md`. They are a dated audit artifact. You may
  *disagree* with a headline; if so, say so explicitly under DIVERGENCES with
  your evidence, but do not produce new counts.
- Do NOT edit any file. Read-only.
- Do NOT run the full test suite. Targeted single-file pytest is acceptable if
  you need to settle a factual question, but no training.
- Prefer a cited code line over an inference. "The proof says X; at
  `src/models/foo.py:200` the code computes Y" is the unit of value.

## Deliverable from you

The structured report above, in your final response. Nothing else.