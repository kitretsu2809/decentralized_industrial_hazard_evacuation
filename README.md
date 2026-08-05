# LBP Decentralized Agentic Evacuation System

A scalable, intelligent evacuation system utilizing Multi-Agent Reinforcement Learning (MARL) for dynamic threat response and optimal crowd routing in multi-story buildings.

## Architecture

The system uses a dual-branch architecture:
- **`core/`**: Framework-agnostic portable logic that runs on both simulation hardware and edge devices (Jetson/ESP32).
- **`sim/`**: Simulation environment, training services, perception modules, and GUI.
- **`embedded/`**: PCB firmware and node logic for physical deployment.

```
+----------------+        +-----------------+        +------------------+
|   Perception   | -----> |   Environment   | -----> |     Training     |
| (YOLOv8 + CV)  |  Redis | (Building Sim)  |  Ray   | (MAPPO + GAT)    |
+----------------+        +-----------------+        +------------------+
                                  |
                                  v
                          +---------------+
                          |      GUI      |
                          | (React/Vite)  |
                          +---------------+
```

## Quick Start

1. Start the simulation stack (Environment + Perception + GUI):
   ```bash
   ./scripts/setup.sh
   ./scripts/run_sim.sh
   ```
2. Open the GUI at `http://localhost:8080`.

## Training

To run the MAPPO training with GPU support:
```bash
./scripts/run_training.sh
```

## Multi-floor Building Support

The system models multi-story buildings using a specialized `BuildingGraph`. Nodes include ROOM, CORRIDOR, INTERSECTION, STAIRWELL, ELEVATOR, and EXIT. Hazards such as fire and smoke can propagate across floors (e.g., smoke rising through stairwells), triggering dynamic re-routing strategies.

## Technology Stack

- **ML & RL**: PyTorch, Ray RLlib, PettingZoo
- **Simulation**: Python, Redis (Pub/Sub)
- **Frontend**: React, Vite, Canvas API
- **Deployment**: Docker, Docker Compose, ONNX (for embedded models)

## License
MIT
