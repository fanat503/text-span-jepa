# TASK-02 — CMC uses the generator it is handed

- **status**: done
- **branch**: `agent/task-02` (worktree `C:\dev\wt-02`), commit `87168bd`, not pushed
- **files**: `src/models/cmc.py` (only file touched)

## Two corrections to the card's premises (measured, not argued)

1. **The call site is `src/train.py:1299`, not `:1073`** in this worktree. It
   passes `seq_len`, `batch_size`, `mask_ratio`, `device` and nothing else, so
   it takes the `rng=None` path. It does not pass `rng=None` explicitly.
2. **`rng` was not dead code.** Measured on the *pre-fix* file: a supplied
   `torch.Generator` did advance, and the global stream stayed byte-identical —
   `torch.randint(..., generator=rng)` honours it. What was broken is the
   `rng=None` path, which is the only one production uses: the global CPU RNG
   stream was consumed, `bytes_differing=2489` of 5056. So the defect is
   (a) not "the parameter is ignored" but (b) "the only path anyone takes
   ignores stream ownership", and the card's stated acceptance check
   ("supplied generator advances, global untouched") **already passed before
   the fix** — see mutation-verdict.

## What changed

`generate_second_mask` keeps its signature (only a new trailing `seed=None`)
and its mask construction. Randomness is now resolved by `_mask_rng`:

| input | stream used | global torch RNG |
|---|---|---|
| `rng=g` | the caller's CPU generator, and only it | untouched |
| `seed=s` | private generator from `s` (repeatable) | untouched |
| neither | process-private `_default_mask_rng()` | untouched |

`_default_mask_rng()` is built once, from `int(torch.initial_seed()) +
_MASK_RNG_SALT`. `torch.initial_seed()` is a *query* of the master seed
`src/utils/seed.py::seed_everything` set — it consumes no stream state, so the
default path is deterministic per master seed and never touches the global
stream, not even once. `rng` **and** `seed` together raise `ValueError` rather
than silently picking one. `src/utils/seed.py` is deliberately *not* imported:
it only offers `seed_everything`, which sets global seeds, and calling that from
a mechanism module would clobber the caller's streams. The idiom used instead
(private CPU `torch.Generator`) is the one already in `src/eval/probes.py:64`.

### `fork_rng` default was tried and rejected (approach 1 of 2)

Isolating with `torch.random.fork_rng(devices=[])` keeps the global state
untouched *and* keeps resume reproducibility, but it made two consecutive
default calls return the **identical** mask (the global position had not
moved). That breaks this function's actual contract — a *second, different*
mask — and the pre-fix baseline measured `identical masks: False`. The private
stream is the better trade.

## verify

```
$ & $PY tools\rt.py tests/test_cmc.py
rt.py: C:\Users\...\python.exe -m pytest -q --no-header -p no:cacheprovider tests/test_cmc.py
rt.py: threads=1  total_budget=90s  slow_ok=False
============================= test session starts =============================
collected 33 items

tests\test_cmc.py .................................                      [100%]

============================= 33 passed in 2.81s =============================
```

Throwaway check (kept in the temp dir, **not committed**; `tests/**` is not
mine). "bytes_differing" is out of the 5056 bytes of the CPU RNG state tensor.

```
=== 1. SUPPLIED torch.Generator is the only source of randomness
  that generator : checksum 318543 -> 330037  bytes_differing=2493  byte_identical=False
  global torch   : checksum 317411 -> 317411  bytes_differing=0  byte_identical=True
  -> generator advanced AND global byte-identical: True

=== 2. DEFAULT rng=None (the src/train.py:1299 call path) must not consume global
  global torch   : checksum 317411 -> 317411  bytes_differing=0  byte_identical=True
  -> global byte-identical: True  (PRE-FIX: False, 2489 of 5056 bytes changed)
  mask shape/dtype/sum: (4, 32) torch.int64 53

=== 3. SEEDED determinism: two calls, same seed -> identical draws
  seed=7 twice -> identical masks : True
  mask sums (7, 7, 8)             : 55 55 54
  seed=7 differs from seed=8      : True
  global torch                    : checksum 317411 -> 317411  bytes_differing=0  byte_identical=True

=== 4. DEFAULT path is still a random generator: consecutive calls differ
  two default calls identical     : False (must be False)
  mask sums                       : 58 56

=== 5. rng + seed together is a clear error, not a silent pick
  ValueError: pass either rng (a caller-owned generator whose state you advance)
  or seed (a base seed for a private generator), not both: the two would
  silently fight over the same draw.

=== 6. MUTATION: the pre-fix draw, verbatim (generator=None, no private stream)
  global torch   : checksum 317411 -> 317874  bytes_differing=2492  byte_identical=False

=== 7. ROBUSTNESS of the fixed default stream over 200 master seeds
  mask ratio over 200 master seeds: min=0.348 max=0.386 mean=0.365
  tests/test_cmc.py asserts 0.15 < ratio < 0.55 -> True
  identical masks between consecutive master seeds: 0 / 199
```

**Global-RNG state, before and after, as required:** section 2 —
`checksum 317411 -> 317411, bytes_differing=0, byte_identical=True` (pre-fix:
`bytes_differing=2489`). Section 1 — the supplied generator moved
(`318543 -> 330037`, 2493 bytes) while the global stream stayed at
`317411` with 0 bytes differing.

Determinism (card step 3, "two calls with the same seed give identical
draws"): section 3, `seed=7` twice → `True`, and `seed=8` differs.

Lint/format: `ruff check` → `All checks passed!`; `black --check` → `1 file
would be left unchanged.`

## diff-stat

```
$ git diff --stat campaign/integration...HEAD
 src/models/cmc.py | 134 ++++++++++++++++++++++++++++++++++++++++++++++++------
 1 file changed, 121 insertions(+), 13 deletions(-)
```

## mutation-verdict

Reverted the default path to the global RNG (`yield None` in `_mask_rng`,
byte-for-byte the pre-fix behaviour) and re-ran:

```
$ & $PY tools\rt.py tests/test_cmc.py        # WITH the mutation applied
33 passed in 2.96s
```

**Nothing in the repo detects the revert.** `tests/test_cmc.py` is 33/33 green
with the defect back in place, because no test in it inspects RNG state at all.
The card's suggested check — "a supplied generator advances and the global
stream is untouched" — is likewise blind: it passes on the pre-fix file too
(section 1 of the pre-fix baseline: `True`). The only detector is the
`rng=None` property, and the only thing that measures it today is my
throwaway script, which is not committed. That is a real gap, not something to
paper over: **the fix ships without an in-repo test**, because `tests/**` is
forbidden on this card. Whoever owns `tests/test_cmc.py` should add, as a
property and not a magic number:

```python
def test_default_path_does_not_consume_global_rng(self):
    """rng=None must not move the process-global torch RNG."""
    before = torch.get_rng_state().clone()
    CrossMaskConsistency.generate_second_mask(
        seq_len=32, batch_size=4, mask_ratio=0.35, device=torch.device("cpu"))
    assert torch.equal(before, torch.get_rng_state())


def test_supplied_generator_advances_and_global_is_untouched(self):
    g = torch.Generator(device="cpu"); g.manual_seed(1234)
    g_before, gl_before = g.get_state().clone(), torch.get_rng_state().clone()
    CrossMaskConsistency.generate_second_mask(
        seq_len=32, batch_size=4, mask_ratio=0.35, device=torch.device("cpu"), rng=g)
    assert not torch.equal(g_before, g.get_state())
    assert torch.equal(gl_before, torch.get_rng_state())


def test_seed_reproduces_mask(self):
    kw = dict(seq_len=32, batch_size=4, mask_ratio=0.35, device=torch.device("cpu"))
    a = CrossMaskConsistency.generate_second_mask(seed=7, **kw)
    b = CrossMaskConsistency.generate_second_mask(seed=7, **kw)
    assert torch.equal(a, b)
```

## не_сделано

- **`src/train.py` resume gap — REPORTED, NOT FIXED (not my file).** The
  private stream's position is Python module state, not a registered buffer, so
  it is absent from `state_dict` and `_capture_rng_state` (`src/train.py:95`)
  cannot save it. A resumed run therefore re-enters CMC's mask stream part-way
  instead of replaying the uninterrupted sequence. Pre-fix, resume was exact
  *because* CMC shared the checkpointed global stream. To close it, the owner of
  `src/train.py` should either pass its own generator and checkpoint it, or
  save/restore the module singleton — minimal patch, in `_capture_rng_state`
  (around `:114`) and `_restore_rng_state` (around `:137`):

  ```python
  from src.models import cmc as _cmc_mod
  if _cmc_mod._mask_rng_default is not None:
      state["cmc_generator"] = _cmc_mod._mask_rng_default.get_state()
  # ...and in _restore_rng_state:
  if state.get("cmc_generator") is not None and _cmc_mod._mask_rng_default is not None:
      _cmc_mod._mask_rng_default.set_state(state["cmc_generator"])
  ```

  `_capture_rng_state` already documents CMC's second mask in its docstring
  (`src/train.py:99-100`), so that line is now stale and should be corrected
  in the same change.
- **No call-site change is required for the fix to work** — `train.py:1299`
  keeps working untouched (it takes the new default path).
- **`proofs/IMPLEMENTATION_STATUS.md` not updated.** Checked: the CMC row
  (`4 Verified / 6 Divergent / 2 Gaps`, "stability theorem misstated;
  reuse_encoder path is a stub") and `proofs/cmc.md` (zero mentions of seed,
  rng or reproducibility) are both unaffected — this change touches mask
  provenance, not the loss, `info` dict, or any theorem input. The row does not
  move between Verified / Divergent / Gaps. That file is not mine anyway.

## риски

- **Cross-run determinism of the default stream (accepted, visible).** With
  the same master seed, two runs now get the *same* CMC second-mask sequence;
  pre-fix they differed. The stream still advances within a run (section 4), so
  the mechanism still sees fresh masks — verified over 200 master seeds, all
  ratios in [0.348, 0.386] and 0/199 identical masks, so `test_mask_ratio_
  approximate` has a wide margin either way. Runs that used to differ only by
  this accident are now identical given the same seed, which is the repo's
  stated goal but is a behaviour change. Mitigation: pass `rng=`/`seed=` from
  config (`meta.seed`-derived) — the parameter now exists for it.
- **`torch.initial_seed()` is process-wide state read once, at first use.** A
  `torch.manual_seed()` issued *after* the first CMC draw does not move CMC's
  stream. Intentional (re-seeding it would reintroduce the coupling), but it
  means the stream is fixed at first use, not per seed change.
- **Module-level mutable singleton.** Lazy init is not lock-guarded; a race
  would discard one generator and lose entropy, not corrupt state. `nn.Module`
  `state_dict`/`strict=True` loading is unaffected because nothing was
  registered — deliberately: a new buffer would break strict loading of every
  pre-existing checkpoint.
- **Only `tests/test_cmc.py` was run** (33 tests, 2.8s of a 90s budget). No
  training was invoked. Other suites that construct `CrossMaskConsistency` are
  unaffected by an RNG-provenance change, but I have not executed them.
