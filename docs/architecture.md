# System Architecture

## System Overview
The LBP system operates on a 7-step pipeline:
1. **Sensing**: Edge devices (cameras, sensors) detect anomalies.
2. **Perception**: YOLOv8 extracts localized threat data (fire, debris, hostile entities).
3. **Graph Update**: The decentralized network updates the local and global `BuildingGraph`.
4. **Message Passing**: Hazard states propagate to neighbor nodes via Mesh networking.
5. **Policy Evaluation**: The GAT-based MAPPO agent at each router calculates optimal exit paths.
6. **Actuation**: Dynamic signage and sealed doors adapt to the new state.
7. **Crowd Movement**: Evacuees follow updated directions.

## Docker Services
The system is divided into multiple independent services:
- **Environment**: Simulates building physics, evacuee movement, and hazard propagation.
- **Perception**: Consumes video feeds and updates environment hazards via Redis.
- **Training**: Orchestrates RL environments (PettingZoo) and Ray workers for MAPPO.
- **GUI**: Web dashboard visualizing the graph and evacuation metrics.
- **Redis**: Central message broker for inter-process communication in sim mode.

## Core Module Design
The `core/` folder uses a Hardware Abstraction Layer (HAL) pattern to ensure code runs seamlessly in Python containers and on MicroPython/C++ embedded devices. It contains definitions for graph topologies (`types.py`), messaging schemas (`schemas.py`), and action spaces (`action_types.py`).

## Multi-Floor Graph Model
- `Node`: Represents physical locations (Rooms, Intersections, Stairwells).
- `Edge`: Paths connecting nodes. They can be horizontal (corridors) or vertical (stairs/elevators).
- **Cross-Floor Propagation**: The environment simulates physical phenomena like smoke rising through vertical edges faster than horizontal spread.

## MARL Policy Architecture (GAT + MAPPO)
Each router node operates as a partially observable agent. It uses a Graph Attention Network (GAT) to embed local topological data and neighbor hazard states. The Multi-Agent PPO (MAPPO) algorithm leverages centralized training and decentralized execution, ensuring edge nodes make near-optimal decisions with minimal compute (4GB GPU constraint during training, INT8 ONNX deployment).
