#!/bin/bash
# Run LBP Industrial Disaster Evacuation Digital Twin in Standalone Mode
# No Docker or root privileges required!
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Python binary '$PYTHON_BIN' not found. Please install Python 3.10+." >&2
    exit 1
fi

echo "======================================================================"
echo "  STARTING LBP INDUSTRIAL HAZMAT SCADA DIGITAL TWIN"
echo "  Using Python: $PYTHON_BIN"
echo "  URL: http://localhost:8080"
echo "======================================================================"

PYTHONPATH=. "$PYTHON_BIN" scripts/run_industrial_standalone.py
