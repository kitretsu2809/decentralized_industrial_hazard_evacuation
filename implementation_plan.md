# Decentralized Agentic Evacuation System — Implementation Plan

> **Project Summary**: A decentralized, AI-driven building evacuation system that uses edge-deployed Computer Vision (YOLO) and time-series models (1D-CNN) to detect hazards, maps them to a normalized Hazard Vector, and feeds that into a Multi-Agent Reinforcement Learning (MARL) policy engine using Graph Attention Networks (GAT). The system dynamically reroutes evacuees via smart signage, locks doors for containment, and operates over a serverless mesh network with no central point of failure.

---

## Source Documents Summary

| PDF | Contents |
|-----|----------|
| [TECH - Google Drive.pdf](file:///home/kitretsu/Desktop/LBP/TECH%20-%20Google%20Drive.pdf) | Core architecture, threat mapping matrix, 3 first-person scenarios (fire, wildlife, shooter), operational execution pipeline (7-step) |
| [TECH - Google Drive(1).pdf](file:///home/kitretsu/Desktop/LBP/TECH%20-%20Google%20Drive(1).pdf) | Software stack & toolchain — FDS, SUMO, NetworkX, PettingZoo, PyTorch Geometric, Ray RLlib/CleanRL, YOLO, OpenCV, TraCI, WandB |
| [TECH - Google Drive(2).pdf](file:///home/kitretsu/Desktop/LBP/TECH%20-%20Google%20Drive(2).pdf) | Real-world deployment case study for India's New Parliament (Sansad Bhawan) — hardware sizing, cost breakdown, DevOps/CI-CD |

---

## User Review Required

> [!IMPORTANT]
> **Simulation vs. Physical Deployment**: This plan focuses on the **software simulation** side (laptop-based, no physical microcontrollers), as described in PDF 2. The Sansad Bhawan deployment (PDF 3) is treated as a future scaling target. Please confirm this scope.

> [!IMPORTANT]
> **Training Compute**: MARL training with MAPPO on graph-structured environments can be GPU-intensive. Do you have access to a GPU (NVIDIA CUDA) or should we plan for CPU-only training with smaller models?

> [!WARNING]
> **NIST FDS**: Fire Dynamics Simulator requires a separate installation (Fortran-based) and significant domain expertise to create input files for a building floorplan. We can either (a) use pre-generated FDS CSV data, (b) create synthetic fire propagation data with a simplified Python model, or (c) install and run FDS. Please advise.

## Open Questions

1. **Building Floorplan**: Do you have a specific building floorplan (CAD/DXF/image) to model, or should we create a synthetic multi-story floorplan graph?
2. **Video Assets**: Do you have test videos (fire, debris, weapons) for the YOLO perception bridge, or should we source sample videos?
3. **YOLO Weights**: Should we use standard YOLOv8-nano pretrained weights (COCO) and fine-tune, or do you have custom-trained weights for threat classes (weapon, debris, fire)?
4. **RL Library Preference**: The docs mention both **Ray RLlib** and **CleanRL**. Ray RLlib is more feature-rich but heavier; CleanRL is lightweight and transparent. Which do you prefer?
5. **Dashboard**: PDF 3 mentions a "React/Mobile Dashboard for security forces." Should we include a web dashboard in this implementation, or defer it?

---

## Proposed Changes

The project will be structured as a Python monorepo with the following layout:

```
LBP/
├── README.md
├── pyproject.toml                 # uv/pip project config
├── requirements.txt
│
├── config/
│   ├── building_graph.yaml        # Node-edge graph definition
│   ├── fire_scenarios.yaml        # FDS-derived or synthetic fire configs
│   └── training_config.yaml       # Hyperparameters for MARL
│
├── src/
│   ├── __init__.py
│   │
│   ├── environment/               # Phase 1: Simulation Environment
│   │   ├── __init__.py
│   │   ├── building_graph.py      # NetworkX graph builder
│   │   ├── fire_model.py          # FDS data loader or synthetic fire propagation
│   │   ├── crowd_sim.py           # SUMO/TraCI crowd simulation interface
│   │   └── evac_env.py            # PettingZoo ParallelEnv wrapper
│   │
│   ├── perception/                # Phase 2: Perception Layer
│   │   ├── __init__.py
│   │   ├── yolo_detector.py       # YOLOv8 threat detection (fire/debris/weapon/animal)
│   │   ├── sensor_cnn.py          # 1D-CNN for MQ-2/temp time-series
│   │   ├── hazard_vector.py       # Threat → normalized H_i mapping
│   │   └── video_bridge.py        # Maps video feed → specific graph node
│   │
│   ├── policy/                    # Phase 3: MARL Policy Engine
│   │   ├── __init__.py
│   │   ├── gat_encoder.py         # Graph Attention Network (PyG)
│   │   ├── actor_critic.py        # Actor-Critic policy network
│   │   ├── mappo_trainer.py       # MAPPO training loop (CTDE)
│   │   └── onnx_export.py         # Export trained model to INT8 ONNX
│   │
│   ├── actuation/                 # Phase 4: Actuation / Output
│   │   ├── __init__.py
│   │   ├── signage_controller.py  # Simulated LED matrix sign commands
│   │   ├── door_controller.py     # Simulated maglock commands
│   │   └── audio_controller.py    # Simulated directional audio alerts
│   │
│   └── dashboard/                 # Phase 6: Visualization Dashboard
│       ├── __init__.py
│       └── metrics_logger.py      # WandB integration for training metrics
│
├── scripts/
│   ├── train.py                   # Main training entry point
│   ├── evaluate.py                # Evaluation & metrics
│   ├── run_scenario.py            # Run a specific scenario (fire/shooter/wildlife)
│   └── export_model.py            # ONNX quantization export
│
├── tests/
│   ├── test_building_graph.py
│   ├── test_hazard_vector.py
│   ├── test_evac_env.py
│   └── test_gat_encoder.py
│
├── data/
│   ├── floorplans/                # Building graph definitions
│   ├── fire_data/                 # FDS CSV exports or synthetic data
│   └── videos/                    # Test threat videos
│
└── docs/
    ├── architecture.md
    └── scenarios.md
```

---

### Phase 1 — Environment & Physics Simulation

> Build the digital twin: building graph + fire propagation + crowd simulation + RL environment wrapper.

#### [NEW] config/building_graph.yaml
- YAML definition of building nodes (rooms, corridors, intersections) and edges (connections with capacity/distance attributes).
- Supports multi-floor definitions with stairwell/elevator connections.

#### [NEW] src/environment/building_graph.py
- Loads `building_graph.yaml` and constructs a `networkx.Graph`.
- Each node has attributes: `type` (room/corridor/intersection/exit), `capacity`, `floor`, `position (x,y)`.
- Each edge has attributes: `distance`, `width`, `max_throughput`, `has_door`, `door_type`.
- Provides methods: `sever_edge()`, `get_neighbors()`, `shortest_path()`, `get_subgraph_by_floor()`.

#### [NEW] src/environment/fire_model.py
- **Option A (Full)**: Loads pre-computed NIST FDS CSV data (temperature, smoke density, CO levels) indexed by `(node_id, timestep)`.
- **Option B (Synthetic)**: Python-based fire propagation model using heat diffusion on the graph — fire spreads along edges based on material properties and distance, with configurable origin points and spread rates.
- Outputs a time-series of hazard scores per node: `H_fire(node, t)`.

#### [NEW] src/environment/crowd_sim.py
- Interface to SUMO via TraCI (Traffic Control Interface).
- Loads a SUMO network file generated from the building graph.
- Methods: `add_pedestrians()`, `step()`, `get_positions()`, `reroute_pedestrian()`, `get_congestion_at_node()`.
- Converts SUMO pedestrian positions to graph node occupancy counts.

#### [NEW] src/environment/evac_env.py
- PettingZoo `ParallelEnv` implementation.
- **Agents**: One per Router Node (intersection/decision point).
- **Observation space** per agent: local hazard vector `H_i`, neighbor hazard vectors `H_j`, local crowd count, neighbor crowd counts, edge states (open/severed/locked).
- **Action space** per agent: directional routing command per outgoing edge (ALLOW / REDIRECT / BLOCK) + door command (OPEN / LOCK).
- **Reward function**: `R = α * (evacuees_exited) - β * (casualties) - γ * (congestion_penalty) - δ * (time_penalty)`.
- Steps the fire model and SUMO crowd sim each environment tick.

---

### Phase 2 — Perception Layer (Simulated Hardware Input)

> Detect threats from video feeds and sensor data, output normalized hazard vectors.

#### [NEW] src/perception/yolo_detector.py
- Wraps `ultralytics.YOLO` with YOLOv8-nano model.
- Processes video frames from `cv2.VideoCapture(filepath)`.
- Detects classes: `fire`, `smoke`, `debris`, `weapon`, `person`, `animal`.
- Outputs: list of `(class_name, confidence, bbox)` per frame.
- Supports real-time and batch processing modes.

#### [NEW] src/perception/sensor_cnn.py
- 1D Convolutional Neural Network for time-series sensor data (MQ-2 gas, temperature, humidity).
- Input: sliding window of last N sensor readings.
- Output: binary anomaly classification + severity score.
- For simulation: generates synthetic sensor data correlated with fire_model output.

#### [NEW] src/perception/hazard_vector.py
- **Universal Threat Mapping** implementation (from the threat matrix in PDF 1).
- Maps perception outputs → normalized scalar `H_i ∈ [0, 1]`:
  - Fire/Smoke → `H` based on thermal expansion rate / smoke density.
  - Structural Collapse (debris) → `H = 1.0` (immediate edge severance).
  - Moving Animal → `H` tracks with animal position across nodes.
  - Off-Hours Intruder → triggers Trap Protocol flag.
  - Active Shooter / Weapon → triggers Lockdown Flag.
- Determines **graph action type**: `DYNAMIC_REROUTE`, `EDGE_SEVER`, `MOBILE_TRACK`, `TRAP_PROTOCOL`, `LOCKDOWN`.

#### [NEW] src/perception/video_bridge.py
- The "Bridge Script" from PDF 2.
- Maps a video file path → a specific graph node ID.
- As video plays frame-by-frame, runs YOLO detection → updates the target node's hazard vector in the PettingZoo environment.
- Supports multiple concurrent video feeds mapped to different nodes.

---

### Phase 3 — MARL Policy Engine

> The brain: Graph Attention Networks + MAPPO for decentralized routing decisions.

#### [NEW] src/policy/gat_encoder.py
- PyTorch Geometric `GATConv`-based encoder.
- Input: node feature vectors (hazard score, crowd count, edge states).
- Architecture: 2-layer GAT with multi-head attention (4 heads).
- Attention mechanism heavily penalizes data from high-`H` neighbors (as specified in PDF 1).
- Output: learned node embeddings for policy input.

#### [NEW] src/policy/actor_critic.py
- Actor-Critic architecture for each agent.
- **Actor**: Takes GAT-encoded embedding → outputs action probability distribution over routing commands.
- **Critic** (centralized, training only): Takes global state (all node embeddings) → outputs state value estimate.
- Implements CTDE (Centralized Training, Decentralized Execution) as specified in PDF 2.

#### [NEW] src/policy/mappo_trainer.py
- Multi-Agent Proximal Policy Optimization implementation.
- Training loop: collects rollouts from PettingZoo env → computes GAE advantages → updates actor/critic.
- Integrates with Ray RLlib or CleanRL (based on user preference).
- Supports curriculum learning: start with simple scenarios, increase complexity.
- Logs metrics to WandB: reward curves, survival rates, congestion, evacuation times.

#### [NEW] src/policy/onnx_export.py
- Exports trained PyTorch model to ONNX format.
- Applies INT8 quantization for edge deployment.
- Validates inference latency target: `< 10ms` per policy step.

---

### Phase 4 — Actuation Layer (Simulated)

> Simulated hardware outputs: LED signs, door locks, audio alerts.

#### [NEW] src/actuation/signage_controller.py
- Simulated RGB LED matrix controller.
- Takes routing actions from MARL policy → converts to directional arrow commands.
- States: green arrow (safe direction), red X (blocked), pulsing yellow (caution), LOCKDOWN text.

#### [NEW] src/actuation/door_controller.py
- Simulated magnetic lock controller.
- Supports: `OPEN`, `LOCK`, `EMERGENCY_SEAL` commands per door.
- Implements sector isolation for Lockdown Protocol and Trap Protocol.

#### [NEW] src/actuation/audio_controller.py
- Simulated directional speaker alerts.
- Pre-rendered alert tones: evacuation alarm, lockdown warning, directional voice guidance.

---

### Phase 5 — Integration & Scenario Runner

> Wire everything together and implement the 3 documented scenarios.

#### [NEW] scripts/run_scenario.py
- CLI entry point to run specific scenarios:
  - `--scenario fire` → Scenario A: Structural Collapse & Fire
  - `--scenario wildlife` → Scenario B: Mobile Wildlife Threat
  - `--scenario shooter` → Scenario C: Active Shooter Lockdown
- Loads building graph, initializes environment, connects video bridge, runs MARL policy.
- Opens SUMO-GUI for real-time 2D visualization.

#### [NEW] scripts/train.py
- Main MARL training script.
- Configurable via `config/training_config.yaml`.
- Supports checkpointing, resume, and multi-GPU training.

#### [NEW] scripts/evaluate.py
- Runs trained policy against held-out scenarios.
- Computes key metrics: evacuation time, survival rate, congestion index, path optimality.

---

### Phase 6 — Visualization & Metrics

> Track experiments and visualize the system in action.

#### [NEW] src/dashboard/metrics_logger.py
- WandB integration for logging:
  - Training: reward curves, loss, policy entropy.
  - Evaluation: survival rate, avg evacuation time, congestion peaks.
  - Per-scenario breakdown with video artifacts.

---

### Phase 7 — Documentation & Sansad Bhawan Case Study

> Document the system and include the Parliament House deployment analysis.

#### [NEW] docs/architecture.md
- System architecture documentation with diagrams.
- The 7-step operational pipeline.
- Threat mapping matrix.

#### [NEW] docs/scenarios.md
- Detailed walkthrough of all 3 scenarios with expected system behavior.

#### [NEW] README.md
- Project overview, setup instructions, quickstart guide.

---

## Verification Plan

### Automated Tests

```bash
# Unit tests for graph construction
pytest tests/test_building_graph.py -v

# Unit tests for hazard vector normalization
pytest tests/test_hazard_vector.py -v

# Integration test for PettingZoo environment
pytest tests/test_evac_env.py -v

# Test GAT encoder forward pass
pytest tests/test_gat_encoder.py -v
```

### Simulation Verification

1. **Smoke test**: Build a small 10-node graph, run environment for 100 steps without policy → verify crowd movement and fire spread.
2. **Random policy baseline**: Run random actions → log metrics as baseline.
3. **Trained policy**: Train MAPPO for N episodes → verify reward convergence and survival rate > random baseline.
4. **Scenario replay**: Run each of the 3 scenarios with trained policy → verify correct behavior (rerouting, tracking, lockdown).

### Perception Verification

1. **YOLO detection**: Feed sample fire/debris/weapon video → verify correct class detection with confidence > 0.5.
2. **Hazard mapping**: Verify threat class → hazard vector normalization produces correct `H` values and action types.
3. **Video bridge**: Verify video feed correctly updates target node's hazard in real-time.

---

## Implementation Order (Suggested)

| Order | Phase | Est. Effort | Dependencies |
|-------|-------|-------------|--------------|
| 1 | Phase 1: Environment (graph + fire + crowd + PettingZoo env) | 3-4 days | NetworkX, PettingZoo, SUMO |
| 2 | Phase 2: Perception (YOLO + hazard vector + bridge) | 2-3 days | Ultralytics, OpenCV |
| 3 | Phase 3: MARL Policy (GAT + MAPPO + training) | 4-5 days | PyTorch, PyG, RLlib/CleanRL |
| 4 | Phase 4: Actuation (simulated controllers) | 1 day | Phase 1 |
| 5 | Phase 5: Integration & scenarios | 2 days | Phases 1-4 |
| 6 | Phase 6: Viz & metrics | 1 day | WandB |
| 7 | Phase 7: Docs | 1 day | All |

**Total estimated effort: ~14-17 days**

---

## Key Dependencies

```
# Core
python >= 3.10
networkx >= 3.0
pettingzoo >= 1.24
torch >= 2.0
torch-geometric >= 2.4
ultralytics >= 8.0
opencv-python >= 4.8

# RL Training
ray[rllib] >= 2.7    # OR cleanrl
gymnasium >= 0.29

# Simulation
traci (from SUMO)    # Requires SUMO installation

# Logging
wandb >= 0.16

# Export
onnxruntime >= 1.16

# Testing
pytest >= 7.0
```
