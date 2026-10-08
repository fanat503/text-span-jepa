#!/bin/bash
# Text-Span JEPA — WikiText-103 base (single GPU, 24GB+)
# Copyright 2026 Slyatski Ilya
set -euo pipefail

CONFIG=${1:-config/wikitext/textspanjepa_wikitext_base.yaml}
echo "=== Text-Span JEPA — WikiText-103 base ==="
echo "Config: $CONFIG"
python -m src.train --fname "$CONFIG"
