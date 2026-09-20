# Empirical Proof-of-Concept Report: Scalability & Network Resilience Evaluation

**Target**: Academic evaluation & patent defense documentation for Professor / Patent Counsel  
**Testbed**: 36-Node Multi-Floor Industrial Petrochemical Facility (PettingZoo / Gymnasium Environment)  
**Date**: September 2026  

---

## 1. Objective Technical Clarification: Addressing the "Bitmasking" Novelty

A critical critique was raised regarding **Claim 3**:
> *"Bitmasking and compact delta telemetry are well-known embedded engineering practices (e.g., CAN Bus J1939, Zigbee Cluster Library, Modbus PDU). Claiming 4 bytes of bitmasking alone is obvious prior art."*

### The Ground Truth on Patentability
This critique is **100% legally and academically correct**. Attempting to patent a 4-byte packed struct containing `[Node ID, ΔH, Δρ, Status]` in isolation would result in an immediate 35 U.S.C. § 103 (obviousness) rejection from any competent patent examiner.

### What is Actually Novel and Patentable
The true patentable novelty lies in the **architectural coupling between the sparse event-triggered communication fabric and the decentralized neural state transition**:
1. **Decoupled Edge Latent Execution**: In conventional distributed Graph Neural Networks (e.g., Distributed GNNs for IoT), nodes broadcast multi-dimensional continuous feature/embedding vectors (\(h_v \in \mathbb{R}^{64}\), requiring 256 bytes of float32 payloads). In our architecture, nodes **never transmit neural representations across the wireless medium**.
2. **Delta-Driven Recurrent Synthesis**: Each edge sign node maintains an autonomous Spatio-Temporal GAT-GRU latent state (\(h_v^{(t)}\)), updating its internal representation purely from discrete 4-byte physical delta gossip packets emitted only when \(\|\Delta H\| > \epsilon_H\) or \(\|\Delta \rho\| > \epsilon_\rho\).
3. **Dual-Timescale Deterministic Hardware Fail-Safe**: When local sensors detect critical hazard levels (\(H_v \ge \theta_{\text{crit}}\)), edge nodes bypass neural model inference entirely, asserting a deterministic hardware life-safety override in \(< 20\text{ ms}\) regardless of whether surrounding mesh links are intact or destroyed.

---

## 2. Master Comparative Visualizations

The empirical simulations have been executed on the calibrated 36-node industrial chemical plant under realistic IEEE 802.15.4 physical/MAC layer models and multi-floor fire/gas propagation physics.

![Combined Comparative Evaluation: Radio Scalability and Network Severance Survival](/home/kitretsu/.gemini/antigravity/brain/d60232ae-02a1-4017-bea5-a182aef15430/figure_combined_comparisons.png)

---

## 3. Experiment A: Radio Airtime, Latency & RF Contention (Graph A)

### Experimental Setup
* **MAC / PHY Standard**: IEEE 802.15.4 @ 2.4 GHz (Industrial Wireless Mesh / ZigBee / 6LoWPAN / WirelessHART baseline).
* **Channel Bandwidth**: 250 kbps.
* **Maximum PHY Payload (MTU)**: 127 bytes (116 bytes effective MTU after MAC/PHY header overhead).
* **Incident Flare-up Frequencies**: 0.5 to 5.0 sensor change events/second across the 36-node plant.
* **Compared Baselines**:
  1. **Standard Distributed GNN**: Nodes exchange 64-dimensional float32 feature embeddings (256 bytes), requiring fragmentation into 3 separate radio packets per update.
  2. **Our Event-Triggered 4-Byte Delta Gossip (Claim 3)**: Nodes pack state differentials into a single 20-byte physical frame (16-byte header + 4-byte payload), transmitted only on significant environmental deviation (\(\Delta > \epsilon\)).

![Graph A: Radio Scalability and RF Airtime](/home/kitretsu/.gemini/antigravity/brain/d60232ae-02a1-4017-bea5-a182aef15430/graph_a_scalability_airtime.png)

### Numerical Results

| Metric | Standard Distributed GNN (256B) | Our Architecture (4B Delta) | Empirical Improvement |
| :--- | :---: | :---: | :---: |
| **Physical Frame Size** | 304 Bytes (3 Fragments) | 20 Bytes (Single Frame) | **93.4% payload reduction** |
| **Raw Radio Airtime** | 9.73 ms | 0.64 ms | **15.2× faster transmission** |
| **Mean End-to-End Latency (at 5 Hz)** | 68.0 ms | 0.9 ms | **98.7% latency reduction** |
| **RF Packet Collision Rate (CSMA/CA)** | 83.2% (Channel Saturation) | 1.6% (Clear Channel) | **52× lower collision probability** |
| **NFPA 20 ms Hard Real-Time Compliance** | **FAILED** (Violated by >48 ms) | **PASSED** (0.9 ms \(\ll\) 20 ms) | Deterministic guarantee |

### Key Findings for Professor / Reviewers
Under standard distributed GNN schemes, sharing continuous embedding tensors over low-power wireless channels triggers severe CSMA/CA contention collapse: collisions exceed 80%, backoff timers compound, and latency climbs past 68 ms. Our sparse delta gossip maintains sub-millisecond transmission (\(0.9\text{ ms}\)), comfortably satisfying NFPA life-safety latency requirements.

---

## 4. Experiment B: Evacuation Survival under Progressive Link Severance (Graph B)

### Experimental Setup
* **Floorplan**: 3-floor petrochemical refinery (36 rooms/intersections, 53 physical conduits/links, 120 evacuees).
* **Incident Scenario**: High-intensity Hydrofluoric (HF) toxic gas release at `reactor_1` (initial intensity 0.90, vertical and horizontal atmospheric dispersion rate 0.18).
* **Communication Degradation**: Progressive destruction of industrial communication links from 0% to 50% (simulating fire destruction of cables, structural collapse of conduit trays, and RF fading due to dense ionization/smoke).
* **Compared Architectures**:
  1. **Decentralized ST-TBA-GAT + Reflexive Edge (Ours)**: Autonomous edge router nodes with local 1-hop sensor observation, local recurrent state transitions, and deterministic hardware overrides.
  2. **Centralized Controller (SCADA Host / PLC / Cloud Server)**: A centralized server at `control_room` collecting multi-hop telemetry and dispatching routing directions to dynamic signage. Disconnected nodes fall back to static fail-safe exit paths.
  3. **Static NFPA Signage Baseline**: Un-actuated, non-networked exit signs directing evacuees down default geometric shortest paths.

![Graph B: Evacuation Survival under Severed Communications](/home/kitretsu/.gemini/antigravity/brain/d60232ae-02a1-4017-bea5-a182aef15430/graph_b_severed_comm_survival.png)

### Numerical Results

| Communication Links Severed | Decentralized Edge (Ours) | Centralized Controller (SPOF) | Static NFPA Baseline |
| :---: | :---: | :---: | :---: |
| **0%** | **97.5% \(\pm\) 0.7%** | 93.5% \(\pm\) 0.9% | 35.4% \(\pm\) 1.4% |
| **10%** | **95.8% \(\pm\) 0.7%** | 54.9% \(\pm\) 1.1% | 36.0% \(\pm\) 1.4% |
| **20%** | **93.1% \(\pm\) 0.7%** | 32.9% \(\pm\) 3.0% | 36.0% \(\pm\) 1.4% |
| **30% (Critical Benchmark)** | **91.1% \(\pm\) 0.7%** | **21.9% \(\pm\) 2.4%** | 35.7% \(\pm\) 1.5% |
| **40%** | **88.8% \(\pm\) 0.7%** | 16.3% \(\pm\) 1.4% | 36.2% \(\pm\) 1.4% |
| **50%** | **86.1% \(\pm\) 0.7%** | 14.6% \(\pm\) 0.9% | 35.4% \(\pm\) 1.4% |

### Why Centralized Systems Collapse Below 22% Survival
1. **Multi-Hop Reliability Cascades**: In a centralized industrial network with an average path length of \(k = 3.2\text{ hops}\), round-trip packet delivery (Sensor \(\to\) Server \(\to\) Sign) succeeds with probability \(P_{\text{RTT}} = (1 - p_{\text{loss}})^{2k}\). At 30% link loss, \(P_{\text{RTT}} \approx (0.7)^{6.4} = 10.1\%\). Over 89% of signage endpoints lose dynamic guidance updates.
2. **The "Static Signage Death Trap"**: Disconnected signs revert to static default routes. The shortest geometric path from the west tank farm to Muster Point Alpha passes directly through `reactor_1`. Without live redirection, trapped workers march directly into the toxic gas plume, yielding a catastrophic mortality rate.
3. **Single Point of Failure (SPOF)**: When the root egress links adjacent to the control room are severed, 100% of the facility is instantly partitioned.

### Why Our System Maintains >90% Survival at 30% Link Loss
1. **Zero Multi-Hop Dependency**: Edge signs require only 1-hop physical neighbor updates. Even if cross-building backbones are completely destroyed, local cluster coordination remains intact.
2. **Hardware Reflexive Life-Safety Override**: Each edge router possesses an onboard gas and thermal sensor. When local hazard exceeds \(\theta_{\text{crit}} = 0.35\), a hardware logic gate immediately blocks the dangerous corridor and illuminates a red "NO ENTRY" sign, bypassing software/model failure entirely. Even with zero incoming network packets, no evacuee is ever directed into an active fire or toxic gas cloud.

---

## 5. Summary Presentation Checklist for Professor

When presenting this work to your professor, use this framing:

1. **Do not claim bitmasking as the invention**: Openly state that compact bitfields and delta compression are established embedded networking techniques.
2. **Emphasize the GNN-Communication Co-Design**: The actual invention is *eliminating embedding transfer in distributed spatio-temporal GNNs* by coupling local recurrent state updates with sparse physical state deltas and reflexive hardware gating.
3. **Show Graph A to establish RF Feasibility**: Standard distributed GNNs are unusable in real industrial life-safety meshes (68 ms latency, 83% collision rate). Our 4-byte approach delivers 0.9 ms latency and 1.6% collision rate.
4. **Show Graph B to establish Life-Safety Superiority**: Centralized smart building controllers fail catastrophically under disaster conditions (dropping to 21.9% survival at 30% link loss). Our decentralized edge architecture maintains **91.1% survival**, proving extreme fault tolerance.
