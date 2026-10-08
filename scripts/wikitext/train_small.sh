#!/bin/bash
# Text-Span JEPA — WikiText-103 small (single GPU, 16GB+)
# Copyright 2026 Slyatski Ilya
set -euo pipefail

CONFIG=${1:-config/wikitext/textspanjepa_wikitext_small.yaml}
echo "=== Text-Span JEPA — WikiText-103 small ==="
echo "Config: $CONFIG"
python -m src.train --fname "$CONFIG"
