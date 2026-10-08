#!/bin/bash
# Text-Span JEPA — Kaggle T4/P100 training
# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
set -euo pipefail

CONFIG=${1:-config/kaggle/textspanjepa_kaggle.yaml}
echo "=== Text-Span JEPA — Kaggle ==="
echo "Config: $CONFIG"
python -m src.train --fname "$CONFIG"
