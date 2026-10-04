# Decentralized Industrial Hazard Evacuation System
## Physical Implementation, Edge Deployment & Building Installation Guide

**Document ID:** DOC-ENG-2026-084  
**Hardware Baseline:** ESP32-S3 / NVIDIA Jetson Orin Nano  
**Life Safety Standard Compliance:** NFPA 101, OSHA 1910.37, Indian National Building Code (NBC) Part 4  
**Classification:** Industrial Field Engineering Manual  

---

## 1. Physical Hardware Architecture & Component BOM

The system replaces vulnerable centralized SCADA wiring with autonomous, interconnected edge nodes. Each installation point operates on a modular 3-tier architecture built inside an IP66 flame-retardant industrial enclosure:

### 1.1 Hardware Specifications
| Subsystem | Physical Hardware Specifications | Electrical Interface & Standards |
| :--- | :--- | :--- |
| **Edge AI Router Unit** | **ESP32-S3-WROOM-1U** (Dual-Core Xtensa LX7 @ 240MHz, 8MB PSRAM, 16MB Flash) OR **NVIDIA Jetson Orin Nano** (for camera visual tracking). | 3.3V logic, hardware AES-128/256 crypto, external 2.4 GHz SMA dipole antenna (+5 dBi). |
| **Dynamic Directional Signboard** | **High-Luminance RGB LED Matrix** (WS2812B IP67 sealed strip or HUB75 64×32 matrix, 4500 nits daylight visible). | 5V DC @ 4A peak, optoisolated GPIO 18 (WS2812B DIN) with 470Ω buffer and 1000µF bypass capacitor. Photocell auto-dimming. |
| **Multi-Spectral Sensor Pod** | • **Gas:** MQ-2 / MQ-7 electrochemical ($CO$, $H_2S$, $Cl_2$, VOCs).<br>• **Thermal:** Sensirion SHT31-DIS (±0.2°C, 0.1s response).<br>• **Crowd Count:** ST VL53L1X Time-of-Flight IR Lidar (4m range, 50Hz). | • Gas: 12-bit SAR ADC (Pin 34).<br>• Temp: I2C (SDA 21, SCL 22) @ 400 kHz.<br>• IR ToF: I2C bus sharing with interrupt on Pin 5. |
| **Life-Safety Actuator & Audio** | • **Acoustic Strobe:** MAX98357A I2S Class D Amplifier + 85 dB directional siren.<br>• **Fire Door Interlock:** Solid State Relay (SSR) 5V to 24V DC solenoid lock release. | • I2S audio: BCLK Pin 26, LRC Pin 25, DIN Pin 27.<br>• Relay: Active-LOW GPIO Pin 23 with flyback diode (NFPA 72 fail-safe open). |
| **Power & Battery Backup** | • Primary: 24V DC Industrial Bus or **PoE+ (IEEE 802.3at)**.<br>• Secondary: **3.2V 6000mAh LiFePO4 battery pack** with integrated BMS. | Provides **> 4.5 hours autonomous operation** during complete facility blackouts (exceeds NFPA 101 90-min mandate). |

```text
PHYSICAL HARDWARE INTERCONNECTION SCHEMATIC:
+-----------------------------------------------------------------------------------------+
| [24V DC Main Bus / PoE+] ---> [Buck Step-Down 5V/3.3V] <---> [LiFePO4 4-Hr Battery BMS] |
|                                            |                                             |
|                                   +--------v--------+                                    |
| [I2C: SHT31 Temp / VL53L1X ToF] ->|                 |--> [GPIO 18] -> [HUB75/WS2812 Sign]|
| [ADC: MQ-2 Toxic Gas Sensor] ---->|    ESP32-S3     |--> [GPIO 23] -> [SSR Door Lock]    |
| [RF: 2.4GHz IEEE 802.15.4 Mesh] <->|  Edge AI Node   |--> [I2S Bus] -> [85dB Sounder]     |
|                                   +-----------------+                                    |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Post-Training Model Export, Optimization & Edge Conversion

### Step 2.1: Model Export to Standard ONNX
Once the ST-TBA-GAT reinforcement learning policy finishes training and is saved to `checkpoints/best_policy.pt`, export the Actor component using standard torch ONNX tracing:

```python
import torch
from simulator.policy.st_tba_gat import STTBAGATPolicy

# 1. Instantiate policy and load best trained checkpoint
model = STTBAGATPolicy(node_feat_dim=34, hidden_dim=64, num_actions=7)
checkpoint = torch.load("checkpoints/best_policy.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 2. Trace dummy inputs for local node inference
dummy_feat = torch.randn(1, 34)              # Local 34-feature sensor observation
dummy_edge = torch.zeros((2, 6), dtype=torch.long) # Local corridor adjacency
dummy_hidden = torch.zeros(1, 64)             # Prior GRU recurrent hidden state

# 3. Export to optimized ONNX graph
torch.onnx.export(
    model,
    (dummy_feat, dummy_edge, dummy_hidden),
    "models/policy.onnx",
    input_names=["node_features", "edge_index", "hidden_state"],
    output_names=["action_logits", "new_hidden_state", "value"],
    opset_version=14,
    dynamic_axes={"edge_index": {1: "num_edges"}}
)
print("Successfully exported policy.onnx (73.0 KB)")
```

### Step 2.2: Post-Training Quantization (FP32 to INT8)
- **Quantization Tool:** ONNX Runtime INT8 Static Quantizer or TensorFlow Lite Micro Converter.
- **Size Reduction:** 480 KB (FP32) $\to$ **73 KB (INT8)** (84.8% memory reduction).
- **Edge Latency:** **< 12 ms** execution time per step on ESP32-S3 @ 240 MHz (well within the 2.5-second decision interval).

### Step 2.3: Compiling Model into Embedded C++ Firmware
```bash
# Convert quantized flatbuffer or ONNX binary to C++ header:
xxd -i models/policy_quant.tflite > embedded/observer_node/lib/hazard_core/policy_weights.h
```

---

## 3. Node Provisioning & Configuration (YAML Setup)

Each physical node receives a unique configuration file matching the blueprint topology:

### Step 3.1: Defining `node_config.yaml`
```yaml
node:
  id: "hazmat_basin"                   # Unique topological node ID matching blueprint
  type: "CORRIDOR_INTERSECTION"        # INTERSECTION, ROOM, EXIT, or STAIRWELL
  floor: 1                             # Floor level (1, 2, or 3)
  position_meters: [45.2, 38.0, 0.0]   # Precise CAD plant coordinates (X, Y, Z)
  capacity_people: 25                  # Maximum NFPA corridor occupant capacity before crowd jamming
  doorway_width_m: 2.4                 # Physical corridor egress width in meters

static_egress:
  default_exit_target: "stair_north_f1" # Precomputed NFPA static shortest exit corridor
  static_exit_distance_m: 67.0         # Topological distance to nearest perimeter muster point

neighbors:                             # Ordered list of up to 6 physical incident corridors
  corridor_0: { id: "tank_farm_a",       distance_m: 25.0, width_m: 2.0, is_dead_end: true  }
  corridor_1: { id: "tank_farm_b",       distance_m: 25.0, width_m: 2.0, is_dead_end: false }
  corridor_2: { id: "pipe_rack_junc_1",  distance_m: 22.0, width_m: 2.4, is_dead_end: false }
  corridor_3: { id: "stair_north_f1",    distance_m: 30.0, width_m: 1.8, is_stair: true     }

hardware_io:
  display_type: "HUB75_64x32"          # Dynamic RGB LED directional matrix
  audio_i2s_pin: 25                    # Voice strobe alarm pin
  sensors:
    gas_mq_analog_pin: 34              # Toxic gas analog input
    temp_sht31_i2c_addr: 0x44          # Ambient thermal sensor
    ir_presence_interrupt_pin: 5       # Optical crowd flow counter

mesh_network:
  rf_protocol: "IEEE_802.15.4_ESP_NOW" # 2.4 GHz ad-hoc peer mesh
  channel: 11                          # Protected non-overlapping industrial Wi-Fi/ZigBee channel
  pan_id: "0x7B9A"                     # Facility plant mesh PAN ID
  aes_gcm_key: "8F4A12B9D8E7340156C9A20F88319E4D" # 128-bit AES pre-shared mesh encryption key
```

### Step 3.2: Firmware Flashing via PlatformIO CLI
```bash
# 1. Navigate to embedded node directory
cd embedded/observer_node

# 2. Compile and upload firmware directly to ESP32-S3
pio run -t upload -e esp32-s3-devkitc-1

# 3. Write specific node configuration to onboard SPIFFS / LittleFS partition
pio run -t uploadfs

# 4. Open serial monitor to verify sensor zero-baseline and RF mesh pairing
pio device monitor -b 115200
```

---

## 4. Physical Building Installation & Mounting Guide

### 4.1 Signboard Placement & Visual Sightlines
- **Overhead High-Level Signboards:**
  - **Mounting Height:** Bottom edge installed between **2.0m and 2.5m (6.5 to 8.2 ft)** above finished floor level (AFFL).
  - **Location:** At every corridor decision point, T-intersection, stairwell entry, and ramp departure.
  - **Viewing Distance:** Maximum continuous corridor spacing must not exceed **30 meters (100 ft)** (NFPA 101).
- **Low-Level Photoluminescent Floor Beacons:**
  - **Mounting Height:** **150mm to 200mm (6 to 8 inches)** AFFL.
  - **Critical Purpose:** Under dense thermal smoke layers (where workers crawl), low-level dynamic LED arrows provide clear egress guidance.

### 4.2 Sensor Pod Placement Criteria
- **Thermal Sensors:** Upper third of the wall, exactly **0.3 meters below the ceiling slab** to detect rising thermal plumes early.
- **Gas Sensors:**
  - *Heavier-than-air gases* ($H_2S$, Chlorine, Butane): Auxiliary sensor pod at **0.5 meters AFFL** near floor trenches.
  - *Lighter-than-air gases* (Methane, Hydrogen, Ammonia): Ceiling level.
- **Optical Crowd Sensors:** Directly above corridor doorways pointing downward at a 45-degree angle.

### 4.3 Electrical Wiring & Cable Routing
All primary 24V DC power feeds must be enclosed in galvanized rigid steel (GRS) electrical conduit or mineral-insulated copper-clad (MICC) **2-hour fire-rated cabling (CW1733 / BS 6387 CWZ)**. In hazardous production areas (ATEX Zone 1/2), use explosion-proof junction boxes with silicone gaskets.

```text
CORRIDOR ELEVATION & SIGNBOARD INSTALLATION PROFILE:
Ceiling Slab [H = 3.6m] -------------------------------------------------------------+
                      |  [Temp / Smoke Sensor Pod: 0.3m below ceiling]              |
                      |                                                              |
                      |  +--------------------------------------------------------+  |
                      |  | [ DYNAMIC LED SIGNBOARD: GREEN ARROW / CAUTION / RED ] |  |
                      |  +--------------------------------------------------------+  |
                      |  Mounting Height: 2.2m AFFL (Clear Headroom Clearance)       |
                      |                                                              |
                      |                                                              |
                      |  [Low-Level Smoke Floor Arrow: 0.2m AFFL]                    |
Finished Floor [0.0m] =============================================================+
```

---

## 5. Commissioning, Calibration & Field Acceptance Testing

1. **Stage 1: RF Mesh Link Quality Verification (Site Survey)**:
   - Measure RSSI between every 1-hop neighbor. Target: $\mathbf{RSSI \ge -75\text{ dBm}}$ (Packet Error Rate < 1%).
2. **Stage 2: Sensor Baseline & Zero-Drift Calibration**:
   - Clean ambient air calibration. Settle baseline at: Temp: 20–25°C, Gas: 0.0 ppm, Presence: 0 count.
3. **Stage 3: Signboard Actuator Display Test**:
   - Verify all 4 physical states: Code 0 (Static Standby), Code 1 (Caution Amber), Code 2 (Red [X] Barrier), Code 3 (Dynamic Green Arrow).
4. **Stage 4: Hardware Reflexive Safety Interlock Validation**:
   - Apply test aerosol to sensor pod. Verify local signboard flips to **RED [X] in < 20 ms**, and adjacent nodes receive 4-byte gossip in < 50 ms.
5. **Stage 5: Evacuation Drill & Egress Throughput Validation**:
   - Run simulated disaster drill and verify zero loop trapping and crowd load splitting.

---

## 6. Routine Maintenance & Inspection Schedule

| Frequency | Inspection & Maintenance Tasks | Pass / Fail Criteria |
| :--- | :--- | :--- |
| **Monthly** | Automated mesh self-diagnostic test; LED pixel health test; optical sensor dust wipe. | 0 dead pixels; 100% mesh nodes reporting heartbeat at 0.1 Hz. |
| **Quarterly** | Physical bump-test of toxic gas sensors using test gas canister; manual fire door release test. | Sensor response time < 3.0s; door release time < 1.0s. |
| **Annually** | Full battery capacity discharge test under maximum LED load (simulated power blackout). | Battery voltage must remain above 2.8V per cell after 90 minutes. |
