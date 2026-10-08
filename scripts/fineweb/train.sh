#!/bin/bash
# Text-Span JEPA — FineWeb-Edu
# Copyright 2026 Slyatski Ilya
set -euo pipefail

NUM_SHARDS=${1:-1}
echo "=== Text-Span JEPA — FineWeb-Edu (${NUM_SHARDS} shards) ==="
python -m src.train --fname config/scaling/small_100m.yaml
