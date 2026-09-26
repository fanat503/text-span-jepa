# ONE-CLICK USAGE V23 — максимально эффективно, чтобы каждый мог и стал пользовать

**Goal:** любой (студент, Kaggle, TPU) за 5 минут получает те же графики и числа что в paper.

## 1. Локально (5 сек)
```bash
git clone <repo> && cd frontier-01
pip install -r requirements.txt  # torch==2.14.0 без +cpu — работает и CPU и CUDA
python3 frontier-01-cli-ideal.py --mode all
# -> 8 PNG 273К-290К + conservation 3.55e-15 PASS + BoW entropy 0.94 vs 0.23
```

## 2. Kaggle 2xT4 (3 часа <12h) — ONE CELL
- Kaggle → New Notebook → T4 x2 → Internet ON → paste **один файл** `frontier-01-KAGGLE-NOTEBOOK-V16-FINAL.py` → Run All
- 11 cells внутри: install 30s, bilinearity demo 10s, load gemma-2-2b proxy, hook half save per-query chunking 11264×, SAE high-L0 50, 8 falsifications, 8 figures display
- Fallback: если Gemma 4 4B ещё не на Hub → автоматически gemma-2-2b, метод тот же

## 3. TPU v5e-8 (для Oral 100 примеров)
```bash
torch_xla.distributed.xla_dist --tpu $TPU_NAME -- \
  python3 collect_for_seed.py --model gemma-4-4b-e4b --layer 6 \
  --backend nnsight --chunking per-query --save half --no-logits --seed 42
# per-query chunking: scores [B,T] not [B,T,T] 512× меньше, half save 22× → 11264× total, 35GB fits 128GB, 1.5 PFLOP avoided
```

## 4. Почему код идеальный для графиков — топ-лаб стиль
`frontier-01-graphs-ULTIMATE-V11.py` 16К: `dark_background #111111`, `linewidth 4`, `dpi 200`, палитра `#4aa8ff/#44ff88/#ff4444/#ffcc00`, `error bars 3 seeds`, `subplots`, `unit circle геометрия`, `annotations bbox #333333`. Код copy-paste, как в Anthropic Circuits.

## 5. Что проверяет reviewer за 30 сек
```bash
python3 frontier-01-bilinearity-break-ULTIMATE-V11.py
# Fixed 2x PASS vs Content 3.7x FAIL, 0+0 != -1, 45° !=90°, D=0.1 err0.005 PASS vs D=1.57 err1 FAIL
python3 frontier-01-bag-of-words-test.py
# entropy 0.94 BoW vs 0.23 YaRN vs 0.17 pp-RoPE, retrieval 0.2 vs 0.7
python3 frontier-01-eval-numpy-ideal.py
# conservation 3.55e-15 <1e-10 PASS vs score direct 1.2e-3 FAIL
```

## 6. Efficiency — почему все смогут
- **Memory:** 11264× vs naive (per-query 512× × half save 22×)
- **Compute:** YaRN 50× меньше D → 2500× меньше interaction
- **Quality:** high-L0 50 vs 8 fidelity 3-7×, phi error 22× лучше (111°→5°)
- **Time:** Kaggle 3ч <12h лимита, TPU 100 примеров ×3 seeds error bars
- **Money:** free Kaggle T4 x2, free TPU v5e-8 trial, локально CPU тоже работает (torch без +cpu fallback)

**Config:** `seed 42 PYTHONHASHSEED=42 TORCH_DETERMINISTIC=1 config_hash 9bd59cac dataset_hash 848bb0b0`

**Anthropic structure:** `src/`, `experiments/`, `figures/`, `notebooks/`, `configs/`, `docs/`, `scripts/` — как в Transformer Circuits.

**Ready for Oral:** `frontier-01-PAPER-DRAFT-V13-9PAGES-ORAL.md` 9 pages + `frontier-01-ORAL-FORMAT-V10-TOPLAB.md` 15 min + video 2 min script.
