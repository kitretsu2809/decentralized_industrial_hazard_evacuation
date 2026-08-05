#!/bin/bash
# LBP Evacuation System - One-command setup
set -e

echo "========================================"
echo "  LBP Evacuation System Setup"
echo "========================================"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed."
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
echo "[OK] Docker found: $(docker --version)"

# Check Docker Compose
if ! docker compose version &> /dev/null; then
    echo "ERROR: Docker Compose V2 not found."
    exit 1
fi
echo "[OK] Docker Compose found: $(docker compose version)"

# Check NVIDIA GPU (optional)
if command -v nvidia-smi &> /dev/null; then
    echo "[OK] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "[WARN] No NVIDIA GPU detected. Training will use CPU."
fi

# Create data directories
mkdir -p data/{fds_exports,floorplans,models}
mkdir -p sim/services/training/checkpoints

# Build Docker images
echo ""
echo "Building Docker images..."
docker compose build

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Quick start:"
echo "  docker compose up          # Start simulation + GUI"
echo "  Open http://localhost:8080  # View GUI"
echo ""
echo "Training:"
echo "  docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile train up"
