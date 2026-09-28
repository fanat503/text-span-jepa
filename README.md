This repo was a bit edited by LLM due to my English, but i really tried to do it maximally readable, and reviewed it a lot of times :)

text-span-jepa
==============

latent prediction at masked spans + future positions.

not token reconstruction. predict in latent space — that's the point of JEPA (LeCun). the encoder learns what matters because it shouldn't waste capacity on useless details.

something like twist: span masking forces the model to use broader context. future latent prediction gives it a reason to encode directionality.

---

setup
-----

```
pip install -r requirements.txt
pip install -e ".[dev,eval]"   # dev: pytest/ruff · eval: sklearn/scipy/matplotlib
```

python 3.9+, pytorch 2.0+

training
--------

```
# JEPA on WikiText-103 (~100M params)
python -m src.train --fname config/scaling/small_100m.yaml

# ablations: each toggles one mechanism against defaults.yaml
python -m src.train --fname config/ablations/swip_on.yaml

# baseline objectives share the same encoder/capacity
python -m src.train --fname config/wikitext/mlm_wikitext_small.yaml
python -m src.train --fname config/wikitext/data2vec_wikitext_train.yaml
```

configs live in `config/`:
- `scaling/` — xsmall_30m, small_100m, base_140m, large_300m
- `wikitext/`, `tinystories/`, `kaggle/` — dataset and variant for kaggle
- `ablations/` — one-mechanism on/off sweeps (deep-merged over defaults.yaml)

resume: set `meta.load_checkpoint: true` in the config. picks up from
`<logging.folder>/checkpoint-latest.pth.tar`. override output dir with
`--output_dir`; skip the defaults merge with `--no_defaults`.

operational notes
-----------------

- `logging.keep_last_epoch_ckpts: <K>` prunes older `checkpoint-ep{N}`
  files every epoch (default: keep everything).
- checkpoint loading tries `weights_only=True` first and falls back with
  a warning for legacy pickled files.
- the trainer warns about config keys absent from `defaults.yaml`
  (catches typos like `lamda_swip`) and `_meta.*` subtrees are exempt.
- theory status: `proofs/` are DESIGN documents with an audited
  implementation matrix in `proofs/IMPLEMENTATION_STATUS.md` — several
  theorems describe aspirational objects, not the shipped code. CGN and
  STA have been reconciled (code now matches the stated math); see the
  matrix for per-mechanism verdicts.


cite
----

```bibtex
@article{textspanjepa2026,
  title={Text-Span JEPA: Latent Predictive Learning for Language Representations},
  author={Text-Span JEPA Authors},
  year={2026}
}
```

license
-------

apache 2.0

novel mechanisms
----------------

GWP (Grassmann Workspace Prediction) ships **12 mechanism modules** in 3
groups, plus **4 numbered capabilities that are methods of the JAWP module, not
modules of their own** — 16 numbered capabilities in total. the counting
convention and its justification are in
[`proofs/README.md`](proofs/README.md); quote 12 or 16 only with which one you
mean.

core — workspace construction
- `jawp` — jacobian-aligned workspace prediction (courant-fischer optimality)

routing — information flow
- `cgn` — contextual gating network (partition of unity)
- `swip` — selective whitening with information preservation
- `pcr` — predictive cascade refinement (cascade capacity)
- `spc` — spectral predictive coding (info-proportional allocation)

stability — workspace integrity
- `wsd` — workspace-target synchronization drift (drift bound)
- `cmc` — cross-mask consistency (cauchy-schwarz stability)
- `gac` — gradient-allocated capacity (no dead zones)
- `sta` — spectral transport alignment (davis-kahan + wasserstein-1)
- `puc` — prediction uncertainty calibration (minimax)
- `rdc` — representation drift compensation (drift bound)
- `wsr` — workspace sharpness regularization (generalization)

the four extra numbered capabilities are methods of
[`src/models/jawp.py`](src/models/jawp.py), not modules:

- `#2` wip — `workspace_information_preservation`. no `use_wip` key exists
  anywhere; reachable only via `MechanismBundle.compute_capacity_bound`, a
  composite diagnostic. it is the only one of the four with its own proof
  ([`proofs/wip.md`](proofs/wip.md)), and that proof is **unaudited** — see
  [`proofs/IMPLEMENTATION_STATUS.md`](proofs/IMPLEMENTATION_STATUS.md).
- `#3` spectral gap — `detect_workspace_dimension`, selects the active rank.
- `#4` grassmann optimization — `grassmann_retract` / `principal_angles` /
  `subspace_distance`. holds the manifold constraint and reports diagnostics.
- `#5` predictive rank — `predictive_rank_loss`. this one **is** a trained loss
  term (`lambda_predictive_rank`, default `0.0`), but it has no
  `ALL_MECHANISMS` entry, so it is invisible to `active_mechanisms()` and
  `GWP.summary()`, which report `Core: ['jawp']`.

each mechanism addresses a specific failure mode of standard JEPA, which will be tested on different sizes of models
