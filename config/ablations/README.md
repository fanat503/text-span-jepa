# `config/ablations/` — the mechanism table

Every file here is a **delta on `defaults.yaml`**. `src/train.py` deep-merges
the file over `defaults.yaml` and there is no include mechanism, so anything
more than one experiment needs is declared once, in `defaults.yaml`. That is
why no ablation restates the architecture, the schedule, or the effective
batch: a copied defaults block keeps whatever value it had the day it was
copied, and the experiment silently stops matching its own description.

## The reference

`defaults.yaml` **is** the reference model and the reference mechanism set:

| | |
|---|---|
| architecture | ViT-Base: `embed_dim 768`, `encoder_depth 12`, `num_heads 12`, predictor `384 x 6` |
| mechanisms | all 12 of `MechanismBundle.ALL_MECHANISMS`, each with loss weight `0.01` |
| schedule | 50 epochs, `lr 1e-3`, effective batch `64 x 8 = 512` sequences = 262,144 tokens/step |
| EMA | `tau 0.996 -> 0.9999` cosine |

Rejected alternative: the `640 / 10 / 10` shape that ten of the pre-fix
ablations used. That shape is a *rung of the scaling ladder*
(`config/scaling/small_100m.yaml`), not an independent architecture, and
adopting it would have made every ablation row incomparable with the other 39
shipped configs that train 768/12/12.

Verified 2026-10-08 (TASK-41): all **40** files in this directory resolve to
`embed_dim 768 / encoder_depth 12` after the deep merge — there is no second
shape left in the grid. (This file said "29", which counted neither the current
40 nor the pre-fix 30.)

## The leave-one-out grid

For every mechanism `m` in `MechanismBundle.ALL_MECHANISMS` (12 of them):

| file | active set | `model:` keys |
|---|---|---|
| `no_<m>.yaml` | all 12 **minus** what `use_<m>: false` removes | `use_<m>: false` (+ `lambda_<m>: 0.0` where the mechanism has a weight) |
| `<m>_on.yaml` | all 12 | `use_<m>: true`, `lambda_<m>: 0.1` |

Each pair therefore differs by **exactly one mechanism**, which is the whole
point: a row that drops two mechanisms measures their sum, not either one.

The on-arm's `lambda_<m>: 0.1` is ten times the `0.01` grid weight the full
model runs every mechanism at, so the file is a genuine delta rather than a
restatement. The weight is inert in `no_<m>.yaml`, which is why the pair still
isolates one mechanism.

`use_<m>: true` in `<m>_on.yaml` restates an all-on default and is one of two
documented exceptions to the no-redundant-restatements rule
(`TestDeltaPurity::test_no_unnecessary_restatements`, budget 17). The on-arm of
a leave-one-out table *is* the full model, so the file would otherwise carry no
model delta at all; restating the flag it is named for makes the row provably
the on-arm by inspection, and
`TestLeaveOneOut::test_on_arm_is_the_full_model` proves the resolved active set
really is all twelve. The second exception is the five shape keys in
`config/scaling/base_140m.yaml`, which are that rung's experiment variable.

### JAWP is special

JAWP is the **root** of the dependency DAG (`MechanismBundle.dependency_dag`).
WSD, SWIP, RDC and WSR all read the JAWP workspace, and
`TextSpanJEPA.__init__` builds `self.wsd` only under
`if config.use_wsd and config.use_jawp:`. There is no configuration in which
WSD is on and JAWP is off — the code forbids it.

That makes `no_jawp.yaml` the one row where "turn one flag off" removes two
mechanisms: it activates **10**, not 11. Rather than hide that behind a typed-in
number, the tests *measure* it:

```python
def _downstream(mech):     # tests/test_config_system.py
    """Mechanisms that stop being constructed when `use_<mech>` goes false."""
    ...MechanismBundle.from_config(full_model_with_use_mech_false)...
```

* `no_<m>.yaml` must equal the full set minus exactly `_downstream(m)`.
* The pair must differ by exactly `_downstream(m)`.
* `_downstream(m) == {m}` for all eleven non-root mechanisms.
* `_downstream("jawp") == {"jawp", "wsd"}` — pinned explicitly, so if the guard
  in `TextSpanJEPA.__init__` ever changes, this test goes red instead of the
  table quietly becoming wrong.

`no_jawp.yaml` therefore flips exactly one flag. The dependent mechanisms are
constructed but compute nothing, which is the honest reading of "JAWP off": the
architecture is intact, the workspace is not. A "and the dependents are gone
too" row is a different experiment and would need its own name; it is not
`no_jawp.yaml`.

The tests read the active set back from
`MechanismBundle.active_mechanisms()` — the runtime truth — never from the
`use_*` flags, so the WSD guard above is *honoured* rather than assumed.

## The other arms

| file | what it varies |
|---|---|
| `none.yaml` | all twelve off, SIGReg as the only collapse preventer |
| `all_core.yaml` | nothing — it *is* the full model, named and given an output folder |
| `sigreg_only.yaml` | JAWP kept, VICReg variance/covariance terms replaced by SIGReg |
| `no_future_loss.yaml` | `lambda_future 0`, no warmup — span loss only |
| `no_decoder_loss.yaml` | `lambda_decoder 0` — purely latent prediction |
| `jawp_alpha_0.yaml` / `jawp_high_alpha.yaml` | `jawk_alpha` at 0.0 and 1.0 around the reference 0.1 |
| `jawp_k_fixed.yaml` | `jawk_k_start = jawk_k_end = 76 = D // 10`, no rank curriculum |
| `jawp_random_init.yaml` | `jawk_init: random` instead of `identity` |
| `predictive_rank_on.yaml` | `lambda_predictive_rank 0 -> 0.01` |
| `cgn_spc.yaml`, `jawp_swip.yaml`, `puc_rdc.yaml`, `sta_gac.yaml`, `sta_wsr.yaml` | pairwise interaction arms, both members at 10x the grid weight |
| `jawp_pcr.yaml` | `pcr_warmup_steps 1000 -> 0` (PCR has no loss weight to sweep) |

`lambda_swip: 0.0` appears next to `use_swip: false` in several files because
`tests/test_swip.py` (owned elsewhere) asserts that any config naming a
mechanism must also name its weight. It is inert today — with the flag off the
module is never constructed — and it stops a later edit that flips the flag
back from silently enabling SWIP at the grid weight. Only SWIP has such a test
today; the generalisation belongs to that test's owner.

`meta: {}` and `optimization: {}` appear in eight files because
`tests/test_v025_integration.py` (owned elsewhere) asserts those sections
exist. An empty mapping is a no-op in `src.train._deep_merge`, so declaring
them states "this arm inherits the reference metadata and schedule" rather than
leaving the reader to guess.

## Reproducing the table

```powershell
$PY = "C:\Users\Илья\AppData\Local\Programs\Python\Python310\python.exe"

# the row set, in table order
& $PY -m src.train --fname config/ablations/none.yaml
& $PY -m src.train --fname config/ablations/no_jawp.yaml
# ... one no_<mech>.yaml per mechanism ...

# the contracts that keep the table honest
& $PY tools\rt.py tests/test_config_system.py
```
