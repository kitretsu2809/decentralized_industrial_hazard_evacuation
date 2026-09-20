#!/bin/bash
# Start the simulation stack (without training)
set -e
echo "Starting Industrial Disaster Evacuation Simulation & Digital Twin..."
docker compose up -d --build
echo ""
echo "Services started. GUI available at http://localhost:8080"
echo "Use 'docker compose logs -f' to view logs"
echo "Use 'docker compose down' to stop"
