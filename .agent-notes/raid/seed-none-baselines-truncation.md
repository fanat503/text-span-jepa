# seed-none-baselines-truncation
area: the question the card asked — "does `data2vec_baseline.py:121`'s truncation warning mean the baseline is weaker than intended?"

answer: **No. The truncation is benign, and the warning fires only where it cannot matter.** Negative result, recorded with evidence so it is not re-opened.

evidence:
  1. `src/models/encoder.py:268-284` — `get_intermediate_layers` appends exactly one tensor per `Block`, so `len(intermediates) == depth`. Not depth+1 (the embedding output is not in the list) and not depth-1.
  2. `baselines/data2vec_baseline.py:119-125` — `k = min(self.average_top_k_layers, len(intermediates))`, and the warning is emitted only when `len(intermediates) < self.average_top_k_layers`. `average_top_k_layers: 8` (`defaults.yaml:263`).
  3. Every shipped data2vec config runs a model deeper than 8: `data2vec_wikitext_small.yaml` -> 640/10 (`defaults.yaml` overridden by the matching `config/scaling/small_100m.yaml` shape), `data2vec_wikitext_train.yaml` and `data2vec_kaggle.yaml` -> 768/12 (`defaults.yaml:70-71`), `data2vec_wikitext_large.yaml` -> 1024/16, `data2vec_wikitext_xsmall.yaml` -> 384/6. So k = 8 for every rung except xsmall, where k = 6.
  4. When the warning does fire (depth 6 < 8), the code takes `intermediates[-6:]` — **all six available blocks**. That is not a degradation: it is exactly what data2vec does on a model shallower than 8, where "the top 8" and "all of them" coincide. The target therefore contains strictly more information than the un-truncated form would have contained had more layers existed.
  5. The averaging is also faithful in the sense that matters: `:130` takes the *last* k, i.e. the layers nearest the output, which is data2vec's top-K convention.

so: no seed is warranted. The correct disposition is a one-line comment at `data2vec_baseline.py:121` saying the truncation is expected on shallow models, or nothing at all.

what I checked and found clean in the same pass:
  - `head_layers: 2` is correctly forwarded by `src/train.py:514` and matches fairseq's default, despite the class default being 1.
  - The EMA anneal is wired correctly end-to-end: `do_ema_update` (`src/train.py:680-681`) calls `model.update_target_encoder()` for `model_name == "data2vec"`, the loss path is reached via the `regression_head` branch of `compute_loss` (`src/train.py:555-558`), and `get_param_groups` (`:643-662`) correctly excludes the frozen `target_encoder` from the optimizer.
  - `num_updates` is incremented only under `self.training` (`data2vec_baseline.py:167-168`), so validation does not advance the anneal — the same discipline the JEPA mechanisms are *missing* (see the known R4 finding), so the baseline is the correct side of this one.
  - The empty-mask guard is present and returns a graph-connected zero (`:137-143`), matching the MLM baseline's guard at `mlm_baseline.py:106-108`.
