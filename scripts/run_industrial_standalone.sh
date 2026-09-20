#!/bin/bash
# Run LBP Industrial Disaster Evacuation Digital Twin in Standalone Mode
# No Docker or root privileges required!
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

PYTHON_BIN="/home/kitretsu/miniconda3/envs/ct_pipeline/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "======================================================================"
echo "  STARTING LBP INDUSTRIAL HAZMAT SCADA DIGITAL TWIN"
echo "  Using Python: $PYTHON_BIN"
echo "  URL: http://localhost:8080"
echo "======================================================================"

PYTHONPATH=. "$PYTHON_BIN" scripts/run_industrial_standalone.py
