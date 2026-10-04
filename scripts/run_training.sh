#!/bin/bash
# Train the production four-state ST-TBA-GAT policy used by simulator/.
# Existing legacy checkpoints are preserved; a completed directional policy is
# automatically selected by the simulator on its next start.
set -e

PYTHON_EXEC="${PYTHON_EXEC:-/home/kitretsu/miniconda3/envs/ct_pipeline/bin/python}"
SAVE_DIR="${ST_TBA_GAT_SAVE_DIR:-checkpoints/directional}"

if [ -f "$PYTHON_EXEC" ]; then
    echo "=========================================================="
    echo "Starting four-state direct-signboard ST-TBA-GAT MAPPO training..."
    echo "Checkpoints: $SAVE_DIR"
    echo "=========================================================="
    PYTHONPATH=. "$PYTHON_EXEC" simulator/training/train_mappo.py --save-dir "$SAVE_DIR" "$@"
else
    echo "Python environment not found: $PYTHON_EXEC" >&2
    echo "Create the documented environment or set PYTHON_EXEC before running this script." >&2
    exit 1
fi
