#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-formation-submission.zip}"
cd "$ROOT"
rm -f "$OUT"
zip -r "$OUT" . \
  -x "*.git*" \
  -x "*/.git/*" \
  -x ".venv/*" \
  -x "venv/*" \
  -x "*__pycache__/*" \
  -x "*.pyc" \
  -x "multirun/*" \
  -x "outputs/*" \
  -x "wandb/*" \
  -x ".pytest_cache/*" \
  -x "*.pt" \
  -x "*.pth" \
  -x "formation-submission.zip"
echo "Wrote ${OUT}"
