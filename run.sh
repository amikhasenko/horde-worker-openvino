#!/usr/bin/env bash
# Launcher for the OpenVINO SD1.5 generation worker.
#
#   ./run.sh                    # uses ./.venv if present, else python3 on PATH
#   PYTHON=/path/to/python ./run.sh
#
# Run tools/convert_ov_assets.py and tools/add_model.py first: the worker needs the
# CLIP + safety-checker IR and at least one exported SD1.5 pipeline to exist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Keep the worker off any CUDA device: this backend is OpenVINO-only.
export CUDA_VISIBLE_DEVICES=""

# Cache Hugging Face downloads under the repo by default instead of scattering them
# across $HOME.
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf-cache}"

if [[ -n "${PYTHON:-}" ]]; then
    PY="$PYTHON"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PY="$SCRIPT_DIR/.venv/bin/python"
else
    PY="python3"
fi

exec "$PY" -s worker.py "$@"
