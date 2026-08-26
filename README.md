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

python 3.9+, pytorch 2.0+. trains on wikitext-103 out of the box.

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
- `wikitext/`, `tinystories/`, `kaggle/` — dataset/model variants
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


the differences between text-span jepa, data2vec,  and MLM are best understood by reading their respective compute_loss() functions.

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

novel mechanisms (16)

each mechanism addresses a specific failure mode of standard JEPA, which I hope will help to large JEPA models
