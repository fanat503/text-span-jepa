#!/bin/bash
# Text-Span JEPA — WikiText-103 large (single GPU, 40GB+ or A100)
# Copyright 2026 Slyatski Ilya
set -euo pipefail

CONFIG=${1:-config/scaling/large_300m.yaml}
echo "=== Text-Span JEPA — WikiText-103 large ==="
echo "Config: $CONFIG"
python -m src.train --fname "$CONFIG"
