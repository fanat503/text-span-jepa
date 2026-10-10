# QK-normalisation — agent notes

Branch `agent/qk-norm`, worktree `C:\dev\wt-qk`, base commit `9584b5f`.
Backlog item: `docs/RESEARCH_BACKLOG.md` item 1 (there is no `docs/ESEARCH_BACKLOG.md`;
"ESEARCH" is a truncation of "RESEARCH" — the file that exists is the one I read).

Paper, verbatim: RoboJEPA arXiv 2610.10515 — *"We found QK-normalization to be of
particular importance for training stability when scaling the dataset size and the
model size."*

Files I own and touched: `src/models/encoder.py`, `src/models/predictor.py`,
`tests/test_qk_norm.py` (new). Nothing else. `jepa.py`, `mechanisms.py`, `train.py`,
`defaults.yaml` and `config/` are untouched.

---

## 1. What was already there

Checked before writing anything.

- **Norm machinery present:** `nn.LayerNorm` only, used pre-norm in both the encoder
  (`Block.norm1`/`norm2`, `norm_layer=partial(nn.LayerNorm, eps=1e-6)`) and the
  predictor (`PredictorBlock.norm1`/`norm2`, `predictor_norm`). No RMSNorm anywhere.
- **Positional scheme:** learned absolute `pos_embedding` in the encoder; learned
  `predictor_pos_embed` in the predictor.
- **RoPE:** none. `grep -rn "RMSNorm|rms_norm|RoPE|rope|rotary|rotate_half" src/`
  returns only the English word "property" in docstrings — zero hits in code.
- **QK-norm itself:** absent from both attention paths.

**So QK-norm is genuinely absent, not a duplicate of the existing LayerNorm.** The
distinction matters and is not subtle: the pre-norm LayerNorm bounds the *block
input*, which is why an end-to-end "scale the input" test proves nothing here. It
does not bound the attention logits, which grow with `||W_q x||·||W_k x||` — i.e.
with the projection *weights*, which is the thing that grows during a long run and
saturates the softmax. That is the mechanism QK-norm removes.

## 2. What I added

`QKNorm` (`src/models/encoder.py`), a **parameter-free** per-head RMS over the
head dimension, applied to `q` and `k` separately:

```python
q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + self.eps)
k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + self.eps)
```

Wired into `Attention.forward` (encoder) and `PredictorBlock._attend_qk_norm`
(predictor), both gated on `use_qk_norm=False`.

Deliberate choices:

- **No learnable gain.** A per-channel gain `g` would still be scale-invariant
  (g does not depend on the token), but it would add parameters and break this
  repo's parameter-matched-ablation standard. As written, **the two arms have
  identical parameter counts and identical state_dicts**, so the ablation isolates
  the normalisation and nothing else. `test_on_and_off_arms_are_parameter_matched`
  pins it. This is the main reason I would argue against a reviewer asking for a
  learnable gain here.
- **`self.qk_norm = None` when off**, not `nn.Identity()`. Keeps the disabled
  state_dict byte-identical, so every existing checkpoint still loads and the off
  arm is literally the pre-change module.
- **The predictor keeps `nn.MultiheadAttention` as the parameter holder** and
  re-does the projection by hand on the QK-norm arm only. `in_proj_weight` /
  `in_proj_bias` / `out_proj` names are preserved because `_fix_init_weight`
  rescales `attn.out_proj.weight` and every saved predictor checkpoint is keyed on
  them. The disabled arm is still the original single `self.attn(h, h, h, ...)`
  call — `test_off_arm_output_is_bit_identical_to_the_pre_qk_norm_recipe` asserts
  `rtol=0, atol=0` against a model built before the change.
- **`_attend` was split out of `Attention.forward`.** The scale-invariance
  property is a property of the q/k the dot product sees, and it is unobservable
  through `forward`. Purely for that reason; `forward` is otherwise unchanged.

## 3. The RoPE ordering — what I actually did

The brief requires `q = rms_norm(q); k = rms_norm(k);` **then** RoPE, and warns
that the reverse is wrong in a way shapes cannot detect. That reasoning is correct
and I have written it into the `QKNorm` docstring as a standing constraint.

**This repository has no RoPE.** Positions are a learned additive embedding applied
to the token embedding, i.e. *before* the qkv projection, so by the time q and k
exist there is no rotation anywhere in the path. The ordering requirement is
satisfied vacuously. I did **not** add RoPE: it is a separate backlog item, it is
on the same document's *does-not-transfer* list (no views, sequence positions
only), and adding it would be scope creep into another worker's file.

The order I used, concretely: project → **split heads → RMS-normalise q and k
per head → dot product**. If RoPE is ever added, it goes *between* the
normalisation and the dot product, and `test_what_reaches_the_dot_product_is_the_normalised_tensor`
plus `test_predictor_forward_normalises_before_the_dot_product` will fail loudly if
anyone reorders it.

## 4. The property test (not a shape test)

The headline assertions are on the **logits**, not the output:

- `test_logits_are_invariant_to_qk_scale` — scaling q and k by c leaves
  `(q·k)·scale` unchanged (c ∈ 0.5, 2, 8).
- `test_without_qk_norm_the_same_logits_move_by_c_squared` — the negative control.
  Without it, the test above could pass because the tensors are small rather than
  because the normalisation works. It asserts the off arm moves by exactly c².
- `test_bound_holds_under_qk_norm` — after normalisation `‖q‖=‖k‖=√head_dim`, so
  Cauchy–Schwarz gives `|logit| ≤ √head_dim`, an absolute ceiling that does not
  move with activation magnitude. Checked at activation scales 1, 1e3, 1e5.
  `test_bound_is_violated_without_qk_norm` shows the off arm blows it.
- `test_attention_output_is_invariant_to_qk_scale` — invariance survives to the
  block's return value.

## 5. Mutation verdict

All five mutations applied, run, and reverted. Each is a plausible wrong
implementation; each goes red.

| # | Mutation | Tests red |
|---|---|---|
| M1 | revert normalisation in `Attention.forward` (`q, k = q, k`) | **4** |
| M2 | normalise the wrong axis (token axis, `mean(-2)`) | **6** |
| M3 | revert normalisation in the predictor path | **3** |
| M4 | half-normalise — q only, k forgotten | **2** |
| M5 | normalise q and v, leaving k raw | **2** |

Restored state: `75 passed` (37 new + 38 pre-existing baseline).

**Two defects the mutation run exposed in my own work**, both worth recording
because both would have shipped a green suite that proved nothing:

1. **Stale `__pycache__` produced a false mutation verdict.** The mutation string
   `q, k = self.qk_norm(q, k)` → `q, _ = self.qk_norm(q, q)` is *byte-identical in
   length* to the original, so within one mtime tick Python reused stale bytecode
   and my "mutation" silently did not execute. My first mutation table was
   therefore unreliable and I re-ran every mutation **inline with an explicit
   `Select-String` confirming the mutated line was on disk**. Any future mutation
   run in this repo should verify the file, not trust the exit code.
2. **My hook-based test was self-defeating.** `register_forward_hook` stays live,
   so the later `attn.qk_norm(q, k)` call used to build the reference re-fired the
   hook and overwrote the capture — the assertion degenerated to `x == x` and
   passed against a deliberately broken encoder. Fixed by removing the handle in a
   `finally`. Documented in `_captured`'s docstring so nobody re-adds the leak.

Also worth noting: my first attempt at M1 caught only **1** test, because most
invariance tests called `attn.qk_norm(...)` directly and therefore never pinned
that `forward` *applies* it. That is why `TestAppliedInForward` exists and asserts
against a hand-written reference.

## 6. Config knob — REQUIRED FOLLOW-UP, NOT DONE (out of my file boundary)

`use_qk_norm` is implemented and threaded through `TextSpanJEPLEncoder`,
`TextSpanJEPApredictor`, `Attention`, `Block` and `PredictorBlock`. It is **not yet
reachable from YAML**, because the two files that would have to carry it are not
mine. Verified, not assumed:

```
>>> TextSpanJEPAConfig(use_qk_norm=True).use_qk_norm
AttributeError ... hasattr -> False
```

This is exactly the `data.mask_ratio` defect class that `defaults.yaml:123-142`
documents at length — a declared-but-unread key is worse than no key. **Do not add
the defaults.yaml line without the jepa.py line.**

Three edits, in this order:

1. **`defaults.yaml`**, in the `# ── Encoder: the reference shape ──` block
   (lines 146–157), next to `qkv_bias` / `drop_path_rate` / `gradient_checkpointing`:
   ```yaml
   use_qk_norm: false
   ```
   **Not** among the `use_<mech>` GWP flags at lines 195+: QK-norm is not a GWP
   mechanism (no module, no `proofs/qk_norm.md`, no `compute_loss`), and putting it
   there would imply it belongs to `MechanismBundle.ALL_MECHANISMS`, which
   `test_all_mechanisms_length_is_stable` pins at 12. Placing it with the
   architecture flags is both accurate and keeps the 12/16 convention intact.

2. **`src/models/jepa.py`**, `TextSpanJEPAConfig.__init__` beside `qkv_bias`
   (line ~51): `self.use_qk_norm = kwargs.get("use_qk_norm", False)`, and pass
   `use_qk_norm=config.use_qk_norm` to both the encoder (line ~375) and the
   predictor (line ~393).

3. **`config/ablations/qk_norm_on.yaml`**: `model: {use_qk_norm: true}` and
   nothing else.

**Default is `false`, deliberately.** Flipping it to `true` would silently change
every shipped config and invalidate every recorded number in the repo. Opt-in keeps
the ablation honest and the history reproducible.

On the ablation-grid guard: the existing one is keyed on `ALL_MECHANISMS`
(`test_config_system.py:331`, `PURITY_BUDGET = len(ALL_MECHANISMS) + len(LADDER_VARYING)`),
so it **will not** cover `qk_norm_on.yaml` and adding it does **not** change
`PURITY_BUDGET` — `qk_norm_on.yaml` sets a value different from the default, so
`_repeats` does not flag it and the per-file purity test passes. But that also
means the arm is currently unguarded. `tests/test_config_system.py` is not mine;
the owner should add a guard asserting the arm exists and is a one-key delta.

**Not verified by execution.** I did not create the config files even temporarily —
other workers are live and `CONFIG_IDS` is a live `rglob`, so a transient file could
have been swept into someone else's concurrent run. The reasoning above is from the
guard source, not from a test run. Whoever lands these three edits must run
`pytest tests/test_config_system.py` to confirm.

## 7. Риски / risks

1. **The knob does not work yet.** Highest-severity item, section 6. Nothing
   experimental ships until those three edits land together.
2. **`eps` puts a floor under scale invariance.** RMS is `x·rsqrt(mean(x²)+eps)`,
   invariant under `x→cx` only while `c²·mean(x²) ≫ eps`. Below that the
   normalisation stops compensating and the logit shrinks toward zero — benign
   (that drives the softmax toward *uniform*, never toward saturation, so the
   stability claim holds unconditionally), but "invariant" is the wrong word
   there. Pinned by `test_invariance_degrades_below_the_eps_floor`.
3. **The bound is √head_dim ≈ 5.7 for head_dim=32, which is still large enough to
   saturate a softmax.** QK-norm removes the *unbounded* growth; it does not
   guarantee a non-degenerate attention distribution. Anyone reading the arm's
   results as "attention is now well-conditioned" is over-reading it.
4. **Parameter-free is a choice, not a free win.** With no learnable gain the
   per-head logit scale is fixed by `head_dim` and cannot adapt down. ViT-22B /
   Gemma-2 style implementations that use a learnable gain can learn a smaller
   effective scale. If the arm shows too-peaked attention, that is the first
   thing to revisit — and it must then be re-parameter-matched.
5. **Pre-existing, not mine, but adjacent:** `Attention.forward` takes `qk_scale`
   and then uses `F.scaled_dot_product_attention`, which applies its own internal
   `1/√head_dim` and ignores `self.scale`. So a non-default `qk_scale` is silently
   discarded on the fused path. Untouched — out of scope, and flagging rather than
   fixing. Also `PredictorBlock.__init__` accepts `qkv_bias` and never forwards it
   to `nn.MultiheadAttention` (dead parameter). Both predate this change.
6. **No training evidence.** Everything above is verified on CPU properties only.
   "Training stability at scale" is the paper's claim, and it is not something
   these tests demonstrate — they demonstrate the logit-scale invariance and the
   bound that make the claim plausible. Say it that way in any writeup.