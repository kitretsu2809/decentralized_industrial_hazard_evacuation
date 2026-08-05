# LBP Embedded Deployment

This directory contains the deployment code for the LBP Decentralized Agentic Evacuation System hardware nodes.

## Hardware Requirements
- **Router Node:** NVIDIA Jetson Nano or Orin Nano. Camera (CSI or USB).
- **Observer/Actuator Nodes:** ESP32-S3 boards.
- **Sensors:** DHT22 (temperature/humidity), MQ-2 (gas), IR ToF (presence).
- **Actuators:** Relay module (door lock), WS2812B/Hub75 (LED matrix signs), I2S Speaker (audio alerts).

## Wiring Diagrams (ESP32-S3)
- **Observer Node:**
  - MQ-2 (Analog Out) -> GPIO 34
  - DHT22 (Data) -> GPIO 4
  - IR ToF (SIG/DATA) -> GPIO 5
- **Actuator Node:**
  - WS2812B (Data) -> GPIO 18
  - Relay (IN) -> GPIO 23
  - I2S Speaker (BCLK, LRC, DIN) -> standard ESP32 I2S pins

## Firmware Flashing
### ESP32-S3 Nodes (PlatformIO)
1. Install VSCode and the PlatformIO extension.
2. Open `embedded/observer_node` or `embedded/actuator_node`.
3. Build and upload via USB:
   ```bash
   pio run -t upload
   ```

### Jetson Router Nodes
1. Flash SD card with JetPack.
2. Install Python dependencies and TensorRT if using YOLO acceleration.
3. Deploy the `embedded/router_node` directory to the device.
4. Run the main loop script.

## Network Configuration
Nodes communicate using **ESP-NOW** for low-latency peer-to-peer messaging without needing a central Wi-Fi access point, or a **Thread** mesh network.
See `embedded/router_node/config/network_config.yaml` for parameters.

## Node Configuration
Each Router node must have a specific `node_config.yaml` defining its location, ID, connected neighbors, and hardware types. See `embedded/router_node/config/node_config.yaml` for an example.

## Testing Procedure
1. Flash an observer node and verify sensor readings via serial.
2. Flash an actuator node and send mock commands via mesh to test relays/signs.
3. Start the router node with `--simulate` to verify YOLO and ONNX policy inference.
4. Connect all 3 node types and use a small flame/smoke source to trigger an evacuation alert.

## Troubleshooting
- **No Mesh Comms:** Verify all ESP32s and Jetson are on the same channel (e.g. Channel 1) and sharing the identical encryption key (if enabled).
- **Low YOLO FPS:** Ensure TensorRT is correctly installed on the Jetson Nano and that `use_tensorrt: true` is set.
