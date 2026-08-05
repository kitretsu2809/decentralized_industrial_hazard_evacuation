#!/bin/bash
# Start training with GPU support
set -e
echo "Starting LBP MAPPO Training (GPU)..."
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile train up -d
echo ""
echo "Training started. Monitor with:"
echo "  docker compose logs -f training"
echo "  WandB dashboard (if configured)"
