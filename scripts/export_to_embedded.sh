#!/bin/bash
# Export trained models for embedded deployment
set -e

MODEL_DIR="data/models"
EMBEDDED_DIR="embedded/router_node/models"

if [ ! -f "$MODEL_DIR/policy.onnx" ]; then
    echo "ERROR: No trained ONNX model found at $MODEL_DIR/policy.onnx"
    echo "Run training first, then export with:"
    echo "  python -m sim.services.training.src.train --export"
    exit 1
fi

mkdir -p "$EMBEDDED_DIR"
cp "$MODEL_DIR/policy.onnx" "$EMBEDDED_DIR/"
echo "[OK] Model exported to $EMBEDDED_DIR/policy.onnx"

# Generate node configs from building graph
echo "Generating node configs..."
python -c "
from sim.services.environment.src.floorplan_generator import generate_default_building
from core.graph.building_graph import BuildingGraph
import yaml, os

building = generate_default_building()
graph = BuildingGraph(building)

for node_id, node in building.all_nodes.items():
    if node.type.value in ('INTERSECTION', 'STAIRWELL'):
        config = {
            'node_id': node_id,
            'type': node.type.value,
            'floor': node.floor,
            'position': list(node.position),
            'neighbors': graph.get_neighbors(node_id),
        }
        path = f'embedded/router_node/config/{node_id}.yaml'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config, f)
        print(f'  [OK] Generated config for {node_id}')
"
echo "[OK] All node configs generated."
