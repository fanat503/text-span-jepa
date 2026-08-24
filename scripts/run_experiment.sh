#!/bin/bash
# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0
#
# Experiment runner for Text-Span JEPA
#
# Usage:
#   bash scripts/run_experiment.sh train_jepa
#   bash scripts/run_experiment.sh train_mlm
#   bash scripts/run_experiment.sh train_data2vec
#   bash scripts/run_experiment.sh compare
#   bash scripts/run_experiment.sh validate
#   bash scripts/run_experiment.sh ablation
#   bash scripts/run_experiment.sh index

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="${REPO_ROOT}/checkpoints"
RESULT_DIR="${REPO_ROOT}/results"

mkdir -p "${CKPT_DIR}" "${RESULT_DIR}"

case "${1}" in
    train_jepa)
        echo "=== Training Text-Span JEPA ==="
        python -m src.train \
            --fname config/wikitext/textspanjepa_wikitext_small.yaml \
            --output_dir "${CKPT_DIR}/jepa"
        ;;

    train_jepa_base)
        echo "=== Training Text-Span JEPA (base) ==="
        python -m src.train \
            --fname config/wikitext/textspanjepa_wikitext_base.yaml \
            --output_dir "${CKPT_DIR}/jepa_base"
        ;;

    train_mlm)
        echo "=== Training MLM Baseline ==="
        python -m src.train \
            --fname config/wikitext/mlm_wikitext_small.yaml \
            --output_dir "${CKPT_DIR}/mlm"
        ;;

    train_data2vec)
        echo "=== Training data2vec Baseline ==="
        python -m src.train \
            --fname config/wikitext/data2vec_wikitext_train.yaml \
            --output_dir "${CKPT_DIR}/data2vec"
        ;;

    compare)
        echo "=== Running Full Comparison ==="
        python -m src.interp.run_comparison \
            --jepa_ckpt "${CKPT_DIR}/jepa/best.pt" \
            --baseline_ckpt "${CKPT_DIR}/mlm/best.pt" \
            --baseline_type mlm \
            --output "${RESULT_DIR}/jepa_vs_mlm/"
        ;;

    compare_data2vec)
        echo "=== Running JEPA vs data2vec Comparison ==="
        python -m src.interp.run_comparison \
            --jepa_ckpt "${CKPT_DIR}/jepa/best.pt" \
            --baseline_ckpt "${CKPT_DIR}/data2vec/best.pt" \
            --baseline_type data2vec \
            --output "${RESULT_DIR}/jepa_vs_data2vec/"
        ;;

    validate)
        echo "=== Validating Pipeline on Ground Truth ==="
        python -c "
from src.interp.ground_truth import GroundTruthValidation
v = GroundTruthValidation()
results = v.full_validation()
for name, r in results.items():
    if name.startswith('_'):
        continue
    status = 'PASS' if r.get('pipeline_valid') else 'FAIL'
    print(f'  {name}: {status}')
summary = results.get('_summary', {})
print(f'\nOverall: {summary.get(\"n_tests_passed\", \"?\")}/{summary.get(\"n_tests_total\", \"?\")} passed')
print(f'Pipeline reliable: {summary.get(\"pipeline_reliable\", False)}')
"
        ;;

    ablation)
        echo "=== Running Ablation Study ==="
        python -c "
import torch
from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
from src.interp.ablation import AblationStudy, ABLATION_CONFIGS, MODEL_SIZE_CONFIGS

# Quick ablation demo at tiny size
config = TextSpanJEPAConfig(
    vocab_size=1000, max_seq_len=32,
    embed_dim=64, encoder_depth=2, num_heads=4, mlp_ratio=2.0,
    predictor_embed_dim=32, predictor_depth=2,
    future_offsets=(1,), num_refine_steps=1)
model = TextSpanJEPA(config)

def train_fn(ablated_model, n_steps):
    losses = []
    opt = torch.optim.Adam(
        [p for p in ablated_model.parameters() if p.requires_grad], lr=1e-3)
    for step in range(n_steps):
        ids = torch.randint(0, 1000, (4, 32))
        mask = torch.zeros(4, 32, dtype=torch.long); mask[:, 5:8] = 1
        loss, info = ablated_model(ids, ids, mask)
        opt.zero_grad(); loss.backward(); opt.step()
        # Handle EMA skip
        if ablated_model.skip_ema_update():
            ablated_model.ablate_ema()
        losses.append(loss.item())
    return losses

study = AblationStudy(model, train_fn)
results = study.run_all(n_steps=50, ablations=['full', 'no_future_loss', 'no_vicreg', 'no_decoder'])
for name, r in results.items():
    if 'error' in r:
        print(f'  {name}: ERROR - {r[\"error\"]}')
    else:
        print(f'  {name}: final_loss={r[\"final_loss\"]:.4f} ({r[\"description\"]})')
"
        ;;

    scaling)
        echo "=== Running Scaling Ablations ==="
        python -c "
import torch
from src.models.jepa import TextSpanJEPA, TextSpanJEPAConfig
from src.interp.ablation import AblationStudy, MODEL_SIZE_CONFIGS

def train_fn(ablated_model, n_steps):
    losses = []
    opt = torch.optim.Adam(
        [p for p in ablated_model.parameters() if p.requires_grad], lr=1e-3)
    for step in range(n_steps):
        ids = torch.randint(0, 1000, (4, 32))
        mask = torch.zeros(4, 32, dtype=torch.long); mask[:, 5:8] = 1
        loss, info = ablated_model(ids, ids, mask)
        opt.zero_grad(); loss.backward(); opt.step()
        if ablated_model.skip_ema_update():
            ablated_model.ablate_ema()
        losses.append(loss.item())
    return losses

# Quick scaling demo with tiny + small sizes
config = TextSpanJEPAConfig(
    vocab_size=1000, max_seq_len=32,
    embed_dim=64, encoder_depth=2, num_heads=4, mlp_ratio=2.0,
    predictor_embed_dim=32, predictor_depth=2,
    future_offsets=(1,), num_refine_steps=1)
model = TextSpanJEPA(config)

study = AblationStudy(model, train_fn)
results = study.run_scaling_ablations(
    n_steps=20, model_sizes=['tiny'], ablations=['full', 'no_vicreg'])
for key, r in results.items():
    if 'error' in r:
        print(f'  {key}: ERROR')
    else:
        print(f'  {key}: params={r[\"n_params\"]:,} loss={r[\"final_loss\"]:.4f}')
"
        ;;

    index)
        echo "=== Computing Interpretability Index ==="
        python -c "
from src.interp.interpretability_index import InterpretabilityIndex
from src.interp.representation_geometry import RepresentationGeometry
from src.interp.ground_truth import SyntheticStructuredModel
import torch

synth = SyntheticStructuredModel(n_samples=200)
data = synth.generate()
reps = data['representations']

geom = RepresentationGeometry.compute_all(reps)
metrics = InterpretabilityIndex.from_collapse_diagnostics({}, {
    'effective_dimension': geom['effective_dimension'],
    'anisotropy': geom['anisotropy'],
    'effective_rank': 20.0,
    'sv_entropy': 0.7,
    'collapsed_dim_ratio': 0.1,
    'frac_monosemantic': 0.4,
})

idx = InterpretabilityIndex()
result = idx.compute(metrics)
print(f'Interpretability Index (synthetic): {result[\"interpretability_index\"]:.4f}')
print(f'Metrics used: {result[\"n_metrics_used\"]}')
"
        ;;

    *)
        echo "Usage: $0 {train_jepa|train_jepa_base|train_mlm|train_data2vec|compare|compare_data2vec|validate|ablation|scaling|index}"
        echo ""
        echo "  train_jepa       — Train Text-Span JEPA (small)"
        echo "  train_jepa_base  — Train Text-Span JEPA (base)"
        echo "  train_mlm        — Train MLM baseline"
        echo "  train_data2vec   — Train data2vec baseline"
        echo "  compare          — Run full JEPA vs MLM comparison"
        echo "  compare_data2vec — Run JEPA vs data2vec comparison"
        echo "  validate         — Validate pipeline on ground truth"
        echo "  ablation         — Run ablation study"
        echo "  scaling          — Run scaling ablations"
        echo "  index            — Compute Interpretability Index (demo)"
        exit 1
        ;;
esac
