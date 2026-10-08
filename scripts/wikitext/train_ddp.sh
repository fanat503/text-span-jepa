#!/bin/bash
# Text-Span JEPA — WikiText-103 multi-GPU (DDP)
# Copyright 2026 Slyatski Ilya
set -euo pipefail

NUM_GPUS=${1:-4}
CONFIG=${2:-config/wikitext/textspanjepa_wikitext_base.yaml}
echo "=== Text-Span JEPA — WikiText-103 DDP (${NUM_GPUS} GPUs) ==="
echo "Config: $CONFIG"
torchrun --nproc_per_node="${NUM_GPUS}" -m src.train --fname "$CONFIG"
