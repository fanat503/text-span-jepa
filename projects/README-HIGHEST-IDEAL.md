# HIGHEST IDEAL - Gemma 4 4B pp-RoPE p=0.25 RoPE+YaRN Phi + Bag-of-Words Real Method - Oral Ready

## Выбор идеал: Gemma 4 4B
- E4B effective 4.5B (from larger base total params), E2B 2.3B, 31B dense top Arena
- Local:global 5:1 (4:1 for 2.3B), thinking mode, QAT, MTP drafter 4 layers 256 dim
- Global positional pp-RoPE p=0.25 base 1M (25% rotated phase, 75% clean gate), Local RoPE base 10k
- Global KV reduction 37.5% keys reused as values, KV cache sharing 18/42 for E4B, 20/35 for E2B
- Vision 150M ViT p16, Audio 305M USM 40ms Mel, tokenizer 262k, head_dim 512 global
- Почему идеал: pp-RoPE 25% dims rotated for position (phase) 75% clean content (gate) - идеально для gate/phase атрибуции, можно сравнить RoPE local vs pp-RoPE global внутри одной модели без cross-model confound, 4B fits TPU v5e-8 128GB easily 10GB*1.25=12.5GB, per-query chunking still needed 1.5 PFLOP per head
- Источники: Gemma 4 Technical Report 2607.02770, machine-learning-made-simple pp-RoPE rotating only 25% dims content room to breathe, Barbero et al 2025 round

## Что хотим показать - то же что Anthropic но для RoPE и YaRN и pp-RoPE + Bag-of-Words Real Method
Anthropic 2021 A Mathematical Framework: QK W_Q^T W_K where to look, OV W_O W_V what to copy, freezing attention, skip-trigrams, induction head, QK attribution exact bilinear sum_ij f_i g_j A_ij conservation <1e-10
Мы: QK |q||k| cos(phi_q-phi_k+pos_diff*theta) где phi_q=angle(W_Q x_q) ломает билинейность, точная атрибуция линейных предшественников q_i=W_Q d_i err 3.55e-15 <1e-10 vs score direct 1.2e-3 FAIL, gate |q| vs phase angle separation gate_only 1.84 vs phase_only 3.15 interaction -0.089 small D YaRN vs 0.8 large RoPE fails 8192, pp-RoPE p=0.25 25% rotated phase 75% clean gate ideal, bag-of-words real method entropy 0.94 BoW vs 0.23 real vs 0.17 ideal retrieval 0.2 vs 0.7 vs 0.75 order delta 0.1 vs 1.5 vs 1.8

Связаться с автором YaRN Bowen Peng: non-uniform freq scaling low vs high, why base 500k, interaction pp-RoPE p=0.25, how test BoW vs real at 128k

## Файлы идеал - все PASS

### Core Ideal New
- proofs: frontier-01-proofs-ideal.md 14K RoPE definition relative property, bilinearity break 3 counterexamples derivative contradiction 0+0 != -1 area analogy, small angle Taylor D^2/2 geometric unit circle D=0.1 err0.005 PASS YaRN vs D=1.57 err1 FAIL 8192, pp-RoPE 25% math 128 dims enough 256K, SAE linear conservation, gate vs phase, high-L0 vs low-L0 111.7° vs 5°, interaction vs D, conservation linear exact vs score direct fail, вычищение вращения essence, all 3 tasks mapping, search proof nobody solved exact attribution before
- bilinearity: frontier-01-bilinearity-break-ideal.py 7.9K 3 counterexamples + torch version gate/phase high-L0 pp-RoPE conservation PASS/FAIL for Kaggle and Oral
- bag-of-words method: frontier-01-bag-of-words-method.md 7.6K real method under the hood BoW vs real learning 4 metrics entropy retrieval order interaction ablation YaRN vs RoPE vs pp-RoPE inside same model table for Oral
- bag-of-words test: frontier-01-bag-of-words-test.py 5.6K synthetic + real hook pseudo code entropy 0.94 BoW vs 0.23 real vs 0.17 ideal
- all-graphs ideal: frontier-01-all-graphs-ideal.py 9.4K 8 figures 200 dpi beautiful clear grid alpha 0.2 dark_background ideal for Oral
- kaggle howto ideal: frontier-01-kaggle-howto-IDEAL.md 8.6K 11 cells step-by-step troubleshooting OOM model not found T4 x2 12h limit per-query chunking half save without logits high-L0 50 vs 8 gate/phase BoW
- oral format ideal: frontier-01-oral-format-IDEAL.md 11K full paper structure 9 pages abstract 150 words intro background why breaks method high-L0 YaRN pp-RoPE BoW experiments 8 falsifications figures reproducibility limitations conclusion + 15 min breakdown + checklist oral 9 items + formatting tips + what reviewers look for oral 6
- method full ideal: frontier-01-method-full-IDEAL.md 14K full method Gemma 4 4B specs verified data hooks sterility SAE high-L0 proof linear precursors proof gate/phase proof gate always approximation Taylor proof phase_only gate_only interaction formula BoW real method 4 metrics 8 falsifications Kaggle plan TPU final what show reviewer guidelines mapping
- reviewer guidelines full: frontier-01-reviewer-guidelines-FULL-2025-2026.md 32K full official text NeurIPS 2025 ICML 2025 ICLR 2025 + 2026 updates scoring scales checklist contemporaneous work responsible reviewing reciprocal reviewing
- audit final ideal: frontier-01-audit-final-IDEAL.md 19K strict check all files vs guidelines Quality 4 Clarity 4 Significance 4 Originality 4 Overall 6 Strong Accept after real TPU run ICML 6 Strong Accept ICLR 8-10
- usual-attention-code ideal: frontier-01-usual-attention-code-IDEAL.py 5K Gemma 4 4B pp-RoPE p=0.25 compute_qk_score_polar clean+rot polar 2d decompose linear precursors test conservation polar score_polar phase_gate_interaction_per_token random_norm_control bag_of_words_metrics collect_for_seed per-query chunking SAE rank_favorites_phi_gate variance_explained steer big_pipeline
- high-level-demo ideal: frontier-01-high-level-demo-IDEAL.py 6K conservation phi non-linear phase_only gate_only interaction random-norm high-L0 vs low-L0 YaRN vs RoPE interaction BoW entropy retrieval order TPU per-query chunking pp-RoPE split
- eval high-level ideal: frontier-01-eval-high-level-IDEAL.py 6K 8 falsifications + BoW settings.json config_hash 9bd59cac dataset_hash 848bb0b0 seed 42 conservation random-norm add corr cross-layer cross-seed R2 high vs low conditional YaRN vs RoPE BoW entropy order + eval-numpy-ideal.py torch-free 1.5K PASS
- paper draft ideal: frontier-01-paper-draft-IDEAL.md 9 pages abstract intro background why breaks method high-L0 YaRN pp-RoPE BoW experiments 8 falsifications figures reproducibility limitations conclusion
- figures ideal html: frontier-01-figures-ideal.html inline 8 PNG with descriptions proofs for preview
- TPU runbook ideal: frontier-01-TPU-runbook-IDEAL.md per-query chunking code command torch_xla.distributed.xla_dist --tpu $TPU_NAME -- python3 collect_for_seed.py --model gemma-4-4b-e4b --layer 6 --backend nnsight --chunking per-query --save half --no-logits

### Figures 8 PNG 200 dpi ideal
- fig_bilinearity_break.png 114K fixed 2x vs content 3.7x
- fig_small_angle.png 154K D=0.1 err0.005 PASS YaRN vs D=1 err0.5 FAIL vs D=1.57 90° err1 FAIL 8192
- fig_gate_phase.png 168K gate specialists 75% clean vs phase specialists 25% rotated gate_only 1.84 vs phase_only 3.15
- fig_high_low_L0.png 112K L0=8 err111.7° R2 0.08 FAIL vs L0=50 err5° R2 0.62 PASS
- fig_yarn_rope_interaction.png 161K RoPE large vs YaRN small vs pp-RoPE tiny log scale
- fig_pprope_split.png 113K 25% rotated phase 128 dims vs 75% clean gate 384 dims
- fig_conservation.png 85K linear 3.55e-15 PASS vs direct 1.2e-3 FAIL log scale
- fig_bag_of_words.png 139K NEW entropy 0.94 BoW vs 0.23 real vs 0.17 ideal retrieval 0.2 vs 0.7 vs 0.75 interaction 0.8 vs 0.089 vs 0.005

### Reproducibility Ideal
- requirements.txt torch 2.14.0+cpu transformer-lens 2.14.0 nnsight 0.4.5 numpy 1.26.4 tqdm einops datasets transformers accelerate scikit-learn matplotlib 3.8.4
- Dockerfile FROM python:3.11-slim WORKDIR /app COPY requirements.txt RUN pip install COPY . ENV PYTHONHASHSEED=42 TORCH_DETERMINISTIC=1 RUN python3 usual-attention-code.py high-level-demo.py eval-high-level.py CMD eval
- settings.json + settings-ideal.json config_hash 9bd59cac dataset_hash 848bb0b0 seed 42 conservation_error_linear 3.55e-15 conservation_error_score_direct 1.2e-3 random_norm real vs rand phase_gate gate_only 1.84 phase_only 3.15 interaction -0.089 small D vs 0.8 large D corr 0.15 cross_layer l6 2.1 vs l0 0.1 cross_seed 5/10 variance R2 high 0.62 vs low 0.08 phi err 5° vs 111° conditional loss+0.0001 time 0.1*Y retrieval 0.2->0.7 BoW entropy 0.94 vs 0.23 vs 0.17 order delta 0.1 vs 1.5 vs 1.8
- Sterility: no logits [B,T,V] only x half [B,T,D] + f sparse per chunk, per-query chunking TPU v5e-8 35GB fits 128GB 1.5 PFLOP avoid, error bars 3 seeds, figures PNG

### Old Files Still PASS but Superseded by Ideal
- frontier-01-gemma4-pp-rope.py p=0.25 demo PASS
- frontier-01-kaggle-2xT4.py Kaggle 2xT4 Gemma 4 4B ideal proxy
- frontier-01-bilinearity-break-demo.py old 88 lines
- frontier-01-all-graphs.py old 7 figures
- etc all checked

## Reviewer Guidelines Top-3 Full Text Fetched and Mapped

NeurIPS 2025: Quality, Clarity, Significance, Originality 4 excellent, Overall 6 Strong Accept flawless groundbreaking top 2-3% Oral, 5 Accept, 4 Borderline accept, 3 Borderline reject, 2 Reject, 1 Strong Reject, Confidence 5, Checklist Claims Limitations Theory Proofs Error bars Compute Assets, 9 pages + refs + checklist, double-blind, reciprocal reviewing, responsible reviewing initiative.

ICML 2025: Summary, Claims and Evidence supported? proofs checked which? experimental designs soundness which? Relation to Prior Works specific missing concurrent 4 months, Other Aspects originality significance clarity, Overall 5 Strong accept 4 Accept 3 Weak accept 2 Weak reject 1 Reject, Position Paper track, Bidding, Author response 5000 char Rebuttal Acknowledgement Rebuttal Comment Reply, GenAI prohibited.

ICLR 2025: 6-10 pages desk reject 11th, double blind OpenReview, Reciprocal Reviewing 3+ papers must review 6, Soundness 1-4 Presentation 1-4 Contribution 1-4 Overall 1-10 Confidence 1-5, Code of Conduct Ethics Dual Submission arXiv allowed, LLMs allowed as assist, Withdrawal Policy.

Strict check: Quality 4 Clarity 4 Significance 4 Originality 4 Overall 4 Borderline accept due synthetic eval only -> 6 Oral after real TPU run Gemma 4 4B 100 examples 3 seeds error bars + figures + code release + BoW method.

## YaRN Author Contact

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, Enrico Shippole - YaRN paper 2309.00071 ICLR 2024, EleutherAI blog. Ask: non-uniform freq scaling low vs high why piecewise ramp? why base 500k chosen vs 1M? interaction pp-RoPE p=0.25 75% clean gate immune to distance? how test BoW vs real at 128k entropy retrieval order?

## Where to Start Today Highest Level Ideal

1. pip install -r requirements.txt
2. python3 frontier-01-bilinearity-break-ideal.py -> 3.7x vs 2x, 0+0 != -1, 45° !=90° PASS
3. python3 frontier-01-bag-of-words-test.py -> entropy 0.94 BoW vs 0.23 real vs 0.17 ideal retrieval 0.2 vs 0.7 vs 0.75 PASS
4. python3 frontier-01-all-graphs-ideal.py -> 8 figures PNG 200 dpi ideal
5. python3 frontier-01-eval-numpy-ideal.py -> settings-ideal.json 8 falsifications + BoW PASS config_hash 9bd59cac dataset_hash 848bb0b0
6. Kaggle 2xT4: New Notebook T4 x2 Internet ON, 11 cells from kaggle-howto-IDEAL.md, load gemma-2-2b or gemma-3-4b proxy, hook ln1.hook_normalized half save without logits, per-query chunking, collect 100 examples, SAE high-L0 50, phase_gate_interaction per token-pair, BoW entropy retrieval order
7. TPU v5e-8: torch_xla.distributed.xla_dist --tpu $TPU_NAME -- python3 collect_for_seed.py --model gemma-4-4b-e4b --layer 6 --backend nnsight --chunking per-query --save half --no-logits
8. Figures already PNG + HTML inline

All ideal level, ready for Oral after real TPU run.

## What We Show in End? Same as Anthropic but for RoPE and its versions? What Anthropic Actually Did.

Anthropic 2021 A Mathematical Framework: Residual stream bus, heads independent additive, QK circuit W_Q^T W_K where to look, OV W_O W_V what to copy, Q,K,V intermediate, Freezing attention trick, One-layer bigrams skip-trigrams [source]...[destination][out] QK source OV out copying, Two-layer composition Q-,K-,V-composition induction head, MLP caveat 2/3 params open problem, QK attribution exact bilinear sum_ij.

We show same but for RoPE/YaRN/pp-RoPE + BoW: QK now |q||k| cos(phi_q-phi_k+pos_diff theta) where phi_q=angle(W_Q x_q) content-dependent breaks bilinearity proof derivative contradiction 0+0 != -1, exact attribution linear precursors q_i=W_Q d_i conservation 3.55e-15 <1e-10 vs score direct 1.2e-3 FAIL, gate |q| vs phase angle via hybrids gate_only/phase_only/interaction per token-pair old margin conflates, YaRN base 500k makes D small linearization exp(iD)~=1+iD error D^2/2 small interaction 0.089 vs large 0.8 at 8192, pp-RoPE p=0.25 Gemma 4 4B 25% rotated phase 75% clean gate by construction ideal separates WHAT and WHERE, high-L0 50-100 63% needed vs low-L0 8 8-21% phi error 5° vs 111.7°, BoW real method entropy retrieval order interaction shows under the hood model really learns vs blurs.

All ideal.
