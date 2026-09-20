# PATENT APPLICATION SPECIFICATION

**TITLE OF THE INVENTION**:  
**DECENTRALIZED SPATIO-TEMPORAL TOPOLOGY-BOUND GRAPH ATTENTION NETWORK (ST-TBA-GAT) FOR REAL-TIME DYNAMIC EVACUATION ROUTING AND LIFE-SAFETY CONTROL IN EDGE-COMPUTING MESH ENVIRONMENTS**

---

## 1. TECHNICAL FIELD
The present disclosure relates generally to cyber-physical systems (CPS), distributed multi-agent systems, deep reinforcement learning, and emergency egress routing. More particularly, the disclosure relates to edge-native decentralized multi-agent reinforcement learning architectures utilizing spatio-temporal graph attention networks with recurrent memory cells and fail-safe reflexive overrides for dynamic evacuation pathfinding under non-stationary hazard conditions.

---

## 2. BACKGROUND AND PRIOR ART
### 2.1 Deficiencies in Existing Evacuation Systems
Conventional building emergency management systems rely on static emergency exit signage or centrally controlled programmable logic controllers (PLCs). Under catastrophic conditions (such as fire propagation, toxic smoke dispersion, structural collapse, or adversarial threats), static egress signage frequently routes evacuees toward compromised corridors or blocked stairwells, increasing casualty rates.

Centralized computerized routing systems suffer from severe vulnerabilities:
1. **Single Point of Failure (SPOF)**: Central servers, switches, or cloud links fail early during structural fires or power outages.
2. **Bandwidth Saturation**: Transmitting raw sensor streams or high-dimensional graph states across communication backbones creates multi-second latency, exceeding safe egress response windows ($<100\text{ ms}$).
3. **Temporal Amnesia in Graph Neural Networks (GNNs)**: Traditional Spatial Graph Attention Networks (GATs) compute node representations purely from instantaneous spatial snapshots, failing to model temporal velocity, hazard expansion gradients, or directional smoke propagation through vertical shafts.
4. **Non-stationarity and Instability**: Standard Reinforcement Learning (RL) policies can oscillate or output suboptimal actions during out-of-distribution fire flare-ups, which is unacceptable in life-critical operations.

Accordingly, there exists a critical need for a decentralized, edge-native, spatio-temporal graph neural network architecture that operates in real-time under constrained bandwidth and compute budgets while ensuring deterministic life-safety guarantees.

---

## 3. SUMMARY OF THE INVENTION
The present invention addresses the aforementioned deficiencies by providing a **Decentralized Spatio-Temporal Topology-Bound Graph Attention Network (ST-TBA-GAT)** system and method. 

Key innovations include:
1. **Hybrid Spatio-Temporal Architecture**: A coupled Spatial Multi-Head Attention Encoder and Gated Recurrent Unit (GRU) cell deployed on edge router nodes, continuously modeling temporal derivatives of multi-floor hazard propagation (e.g., thermal plume stack effects) without central coordination.
2. **Fixed-Complexity Topology-Bound Attention (TBA)**: Graph attention aggregation restricted strictly to 1-hop physical topological neighbors ($\mathcal{N}_i$), maintaining bounded $\mathcal{O}(|\mathcal{N}_i|)$ compute complexity regardless of total building scale.
3. **Ultra-Low Bandwidth Delta-Bitmap Gossip Protocol**: Distributed state synchronization via a compact 4-byte hazard and density delta bitfield, achieving real-time consensus over low-power IEEE 802.15.4 / ESP-NOW / BLE mesh networks without cloud or central server connectivity.
4. **Deterministic Life-Safety Reflexive Layer**: A dual-timescale hybrid control architecture combining the neural policy with a hardware-level deterministic hazard override and hysteresis filter ($H_v > \theta_{\text{crit}}$), mathematically guaranteeing that evacuees are never routed toward lethal nodes regardless of neural network output.
5. **Ultra-Low-Power Edge Quantization**: Direct INT8 ONNX export compatible with edge microcontrollers and NPUs (e.g., Jetson Nano, ESP32-S3, Edge TPU) with $<15\text{ ms}$ inference latency and $<2\text{W}$ power draw.

---

## 4. BRIEF DESCRIPTION OF THE DRAWINGS
- **FIG. 1**: Overall cyber-physical architecture of the decentralized evacuation system showing edge router nodes, dynamic digital signage, and peer-to-peer mesh gossip.
- **FIG. 2**: Dual-timescale inference pipeline illustrating the parallel execution of the ST-TBA-GAT neural policy and the deterministic life-safety reflexive override.
- **FIG. 3**: Internal neural architecture of the ST-TBA-GAT encoder, highlighting the multi-head spatial graph attention mechanism cascaded into the recurrent GRU state transition cell.
- **FIG. 4**: Multi-floor vertical hazard propagation model demonstrating dynamic stack effect tracking across stairwells, elevators, and floor slabs.
- **FIG. 5**: 4-Byte Sparse Delta-Bitmap gossip packet format for resilient edge-mesh synchronization.

---

## 5. DETAILED DESCRIPTION OF PREFERRED EMBODIMENTS

### 5.1 System Architecture and Hardware Abstraction
The system comprises a plurality of physical edge computing router devices (e.g., RISC-V, ARM Cortex-M/A, or Nvidia Jetson nodes) physically distributed throughout a multi-story structure. Each edge node is topologically bound to a physical vertex $v \in \mathcal{V}$ representing a room, corridor segment, intersection, stairwell landing, or exit door.

Each node is interfaced with:
- Local environmental sensors (optical smoke, carbon monoxide, temperature, flame photodiode);
- Edge perception sensors (RGB-IR micro-cameras running quantized YOLOv8 for occupant count estimation);
- Dynamic digital signage actuators (bi-color LED directional matrix displays, dynamic emergency lighting);
- Peer-to-peer wireless transceiver (ESP-NOW, IEEE 802.15.4, or ad-hoc Wi-Fi mesh).

### 5.2 Mathematical Formulation of ST-TBA-GAT
Let $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ denote the topological graph of the facility. At each discrete time step $t$, node $i$ observes a feature vector $\mathbf{x}_i^{(t)} \in \mathbb{R}^F$, comprising:
\[
\mathbf{x}_i^{(t)} = \left[ H_i^{(t)}, \, \rho_i^{(t)}, \, C_i, \, \mathbf{p}_i \right]
\]
where $H_i^{(t)} \in [0, 1]$ is the local hazard severity index, $\rho_i^{(t)} \in [0, 1]$ is normalized crowd density, $C_i$ is physical throughput capacity, and $\mathbf{p}_i$ is structural egress proximity.

#### 5.2.1 Spatial Topology-Bound Attention
For each neighbor $j \in \mathcal{N}_i$, the spatial attention coefficient $\alpha_{ij}^{(t)}$ is computed via parameterized projection matrices $\mathbf{W} \in \mathbb{R}^{D \times F}$ and attention vector $\mathbf{a} \in \mathbb{R}^{2D}$:
\[
e_{ij}^{(t)} = \text{LeakyReLU}\left(\mathbf{a}^\top \left[ \mathbf{W}\mathbf{x}_i^{(t)} \,\|\, \mathbf{W}\mathbf{x}_j^{(t)} \right]\right)
\]
\[
\alpha_{ij}^{(t)} = \frac{\exp(e_{ij}^{(t)})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik}^{(t)})}
\]
The spatial context vector $\mathbf{s}_i^{(t)}$ is aggregated over the 1-hop topology:
\[
\mathbf{s}_i^{(t)} = \sigma\left( \sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(t)} \mathbf{W}\mathbf{x}_j^{(t)} \right)
\]

#### 5.2.2 Recurrent Temporal State Transition (GRU Cell)
To capture the velocity and acceleration of hazard plumes (such as smoke rising through vertical shafts according to the stack effect equation $\Delta P = C a h (1/T_o - 1/T_i)$), the spatial representation $\mathbf{s}_i^{(t)}$ is supplied to a recurrent Gated Recurrent Unit (GRU) cell maintaining persistent hidden state $\mathbf{h}_i^{(t)}$:
\[
\mathbf{r}_i^{(t)} = \sigma\left(\mathbf{W}_r \mathbf{s}_i^{(t)} + \mathbf{U}_r \mathbf{h}_i^{(t-1)} + \mathbf{b}_r\right)
\]
\[
\mathbf{z}_i^{(t)} = \sigma\left(\mathbf{W}_z \mathbf{s}_i^{(t)} + \mathbf{U}_z \mathbf{h}_i^{(t-1)} + \mathbf{b}_z\right)
\]
\[
\tilde{\mathbf{h}}_i^{(t)} = \tanh\left(\mathbf{W}_h \mathbf{s}_i^{(t)} + \mathbf{U}_h (\mathbf{r}_i^{(t)} \odot \mathbf{h}_i^{(t-1)}) + \mathbf{b}_h\right)
\]
\[
\mathbf{h}_i^{(t)} = (1 - \mathbf{z}_i^{(t)}) \odot \mathbf{h}_i^{(t-1)} + \mathbf{z}_i^{(t)} \odot \tilde{\mathbf{h}}_i^{(t)}
\]
The temporal state $\mathbf{h}_i^{(t)}$ encodes historical trajectory and enables predictive hazard avoidance prior to physical barrier breach.

### 5.3 Deterministic Life-Safety Reflexive Layer
The neural actor evaluates action logits $\pi_\theta(\mathbf{a}_i | \mathbf{h}_i^{(t)})$. However, to satisfy international building safety codes (e.g., NFPA 101 Life Safety Code), the output is mediated by a hardware reflexive filter:
\[
\tilde{\pi}(\mathbf{a}_i = j) = \begin{cases}
0, & \text{if } H_j^{(t)} \ge \theta_{\text{crit}} \text{ or Edge}(i, j) \text{ blocked} \\
\pi_\theta(\mathbf{a}_i = j), & \text{otherwise}
\end{cases}
\]
Followed by softmax re-normalization over non-hazardous candidate edges:
\[
P(\mathbf{a}_i = j) = \frac{\tilde{\pi}(\mathbf{a}_i = j)}{\sum_{k \in \mathcal{N}_i^{\text{safe}}} \tilde{\pi}(\mathbf{a}_i = k)}
\]
If all forward paths are compromised, the node triggers an emergency localized redirect beacon and flashes an audible and visual reverse-triage warning.

---

## 6. PATENT CLAIMS

### WE CLAIM:

**Claim 1 (Independent System Claim)**:  
A decentralized cyber-physical emergency evacuation system for multi-story buildings, comprising:
- a plurality of edge router nodes physically disposed at topological egress vertices across multiple floors of a building, each edge router node comprising a local processor, an environmental sensor interface, a peer-to-peer mesh transceiver, and a dynamic digital directional signage actuator;
- a memory storing instructions that, when executed by the local processor of each respective edge router node, configure the edge router node to execute a decentralized spatio-temporal topology-bound graph attention network (ST-TBA-GAT) policy comprising:
  - (a) a spatial attention encoder configured to compute dynamic attention weights over feature vectors of physically adjacent 1-hop neighbor nodes within a local topological graph $\mathcal{G}$;
  - (b) a recurrent temporal state transition engine comprising a gated recurrent unit (GRU) coupled to an output of the spatial attention encoder, configured to maintain a persistent hidden state tracking temporal hazard propagation velocities and crowd density fluxes across discrete simulation and real-world time steps;
  - (c) an actor policy network configured to map the persistent hidden state to an optimal evacuation routing action; and
  - (d) a deterministic life-safety reflexive controller configured to monitor adjacent neighbor hazard indices, evaluate said hazard indices against a predetermined critical safety threshold $\theta_{\text{crit}}$, and deterministically override and mask out actions designated by the actor policy network that route toward any node exceeding $\theta_{\text{crit}}$, thereby directly actuating the dynamic digital directional signage actuator to display real-time conflict-free egress paths.

**Claim 2 (Dependent Claim - Fixed Complexity TBA)**:  
The system of claim 1, wherein the spatial attention encoder evaluates attention coefficients strictly over 1-hop physical topological neighbors $\mathcal{N}_i$, such that computational complexity per node is $\mathcal{O}(|\mathcal{N}_i|)$, operating independently of the total number of vertices $|\mathcal{V}|$ in the building graph.

**Claim 3 (Dependent Claim - 4-Byte Sparse Delta Gossip Protocol)**:  
The system of claim 1, wherein each edge router node communicates with peer nodes over an asynchronous wireless mesh network using a 4-byte sparse delta-bitmap packet format comprising:
- an 8-bit node identifier field;
- an 8-bit quantized hazard intensity differential ($\Delta H$);
- an 8-bit quantized occupant density differential ($\Delta \rho$); and
- an 8-bit status and topological edge health bitmask;  
wherein transmissions are broadcast exclusively when local sensor state differentials exceed a non-zero transmission threshold $\epsilon_{\text{delta}}$, minimizing mesh packet collisions.

**Claim 4 (Dependent Claim - Multi-Floor Vertical Stack Propagation)**:  
The system of claim 1, wherein the local topological graph comprises vertical connector edges representing stairwells and elevator shafts, and wherein the recurrent temporal state transition engine dynamically learns thermal stack-effect coefficients modeling accelerated vertical toxic gas and thermal plume dispersion relative to horizontal corridor dispersion.

**Claim 5 (Dependent Claim - Hysteresis Signage Switching)**:  
The system of claim 1, wherein the dynamic digital directional signage actuator is controlled via a temporal hysteresis filter requiring a candidate path direction to remain optimal for a threshold duration $\tau_{\text{dwell}}$ before altering visual indicator arrows, thereby preventing rapid alternating directional signage that induces pedestrian indecision or crowd crushing.

**Claim 6 (Dependent Claim - Low-Power INT8 Edge Quantization)**:  
The system of claim 1, wherein the actor policy network and recurrent temporal state transition engine are quantized into an 8-bit integer (INT8) Open Neural Network Exchange (ONNX) runtime model configured for real-time edge execution with an inference latency of less than 20 milliseconds and a power dissipation of less than 5 Watts.

**Claim 7 (Dependent Claim - Decentralized Execution Centralized Training)**:  
The system of claim 1, wherein the ST-TBA-GAT policy is trained via Multi-Agent Proximal Policy Optimization (MAPPO) with Centralized Training and Decentralized Execution (CTDE), wherein during training a centralized critic network receives global state observations across all building floors, and during inference each edge router node executes solely its local actor network and recurrent temporal engine using strictly local and 1-hop neighbor observations.

**Claim 8 (Dependent Claim - Edge-Assisted Vision Perception)**:  
The system of claim 1, wherein each edge router node is coupled to a low-power edge vision sensor executing a quantized deep object detection model (YOLOv8-nano) to infer real-time occupant counts and pedestrian flow vectors, feeding the inferred counts into the local feature vector $\mathbf{x}_i^{(t)}$.

**Claim 9 (Dependent Claim - Anti-Bottleneck Cooperative Density Balancing)**:  
The system of claim 1, wherein the actor policy network is trained with an objective reward function penalizing spatial variance of occupant densities across adjacent parallel exit routes, causing the dynamic signage actuators of adjacent edge nodes to actively balance crowd throughput and prevent bottleneck congestion at primary fire exits.

**Claim 10 (Dependent Claim - Autonomous Fail-Safe Fallback)**:  
The system of claim 1, wherein upon detection of a complete transceiver communication loss lasting greater than a timeout period $T_{\text{fail}}$, each edge router node autonomously transitions to a local Dijkstra shortest-path fallback mode computed over the static topological graph weighted by local sensor hazard readings.

**Claim 11 (Independent Method Claim)**:  
A computer-implemented method for real-time dynamic evacuation routing in a multi-story building, comprising:
- receiving, by an edge computing router node, localized sensor readings and 1-hop peer node state updates via a peer-to-peer mesh network;
- computing, via a spatial attention encoder, dynamic attention weights over feature vectors of physically adjacent 1-hop neighbor nodes within a local topological graph;
- updating, via a gated recurrent unit (GRU), a persistent recurrent hidden state using an output of the spatial attention encoder to continuously track temporal hazard propagation velocities;
- generating, via an actor neural network, a candidate egress routing action based on the persistent recurrent hidden state;
- executing a deterministic reflexive safety check against adjacent neighbor hazard indices;
- overriding the candidate egress routing action if the candidate egress routing action directs evacuees toward a node having a hazard index exceeding a critical safety threshold; and
- actuating a dynamic digital display to project visual directional indicators guiding evacuees along safe egress paths.

**Claim 12 (Independent Non-Transitory Computer-Readable Medium Claim)**:  
A non-transitory computer-readable storage medium having stored thereon computer-executable instructions that, when executed by an edge computing processor, cause the edge computing processor to perform the method of Claim 11.

---

## 7. ABSTRACT
A decentralized cyber-physical evacuation system and method for multi-story buildings employing an edge-native Spatio-Temporal Topology-Bound Graph Attention Network (ST-TBA-GAT). Distributed edge router nodes situated at egress vertices compute localized spatial attention over 1-hop neighbors and evolve a persistent recurrent hidden state via a Gated Recurrent Unit (GRU) to track dynamic hazard velocities and crowd density fluxes in real time. Decisions from the neural actor network are mediated by a deterministic life-safety reflexive controller that overrides any routing toward hazardous nodes ($H > \theta_{\text{crit}}$). The system operates under extreme edge constraints using an INT8 ONNX architecture and 4-byte sparse delta mesh gossip, providing fail-safe, ultra-low latency ($<15\text{ ms}$) evacuation control without cloud dependency.
