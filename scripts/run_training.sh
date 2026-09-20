#!/bin/bash
# Start ST-TBA-GAT Industrial Evacuation Training & Benchmark
set -e

PYTHON_EXEC="/home/kitretsu/miniconda3/envs/ct_pipeline/bin/python"

if [ -f "$PYTHON_EXEC" ]; then
    echo "=========================================================="
    echo "Starting ST-TBA-GAT PPO Training (Python / CUDA)..."
    echo "=========================================================="
    PYTHONPATH=. "$PYTHON_EXEC" sim/services/training/src/train.py "$@"
else
    echo "Starting LBP Training via Docker Compose..."
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile train up -d
    echo "Training started. Monitor with: docker compose logs -f training"
fi
