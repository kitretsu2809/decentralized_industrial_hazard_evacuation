"""
Router Node Main Loop - Jetson Nano/Orin Nano

This is the primary intelligence unit. It:
1. Captures frames from a camera
2. Runs YOLO threat detection
3. Computes hazard vectors
4. Runs ONNX policy inference
5. Actuates signage, doors, and audio
6. Broadcasts state to mesh neighbors
"""
import argparse
import logging
import time
import yaml
from pathlib import Path

# Stub imports assuming core logic exists in sys.path
from core.hazard.hazard_vector import combine_hazards
from yolo_edge import YOLOEdge
from mesh_client import MeshClient
from hal_impl.jetson_camera import JetsonCamera
from hal_impl.gpio_door import GPIODoor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PolicyMock:
    def __init__(self, path: str):
        self.path = path
    def infer(self, hazard, neighbor_data):
        # Mock policy action for skeleton
        class MockAction:
            def __init__(self):
                self.signs = {}
                self.doors = {}
        return MockAction()

class SignageMock:
    def set_direction(self, signs):
        pass

class AudioMock:
    def play_if_needed(self, action):
        pass

class RouterNode:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.node_id = self.config['node']['id']
        self.neighbors = self.config['neighbors']
        self.fps_target = self.config['hardware']['camera'].get('fps', 30)
        self.sleep_time = 1.0 / self.fps_target

        self.setup_hardware()
        
        model_path = self.config['model']['path']
        use_trt = self.config['model'].get('use_tensorrt', True)
        self.yolo = YOLOEdge(model_path=model_path, use_tensorrt=use_trt)
        self.policy = PolicyMock(path="policy.onnx")

    def setup_hardware(self):
        logger.info(f"[{self.node_id}] Initializing hardware...")
        self.camera = JetsonCamera()
        
        mesh_cfg = self.config['hardware']['mesh']
        self.mesh = MeshClient(node_id=self.node_id, protocol=mesh_cfg['protocol'])
        
        self.signage = SignageMock()
        self.door = GPIODoor(gpio_pin=self.config['hardware']['actuators']['door'].get('gpio_pin', 23))
        self.audio = AudioMock()
        logger.info(f"[{self.node_id}] Hardware initialized.")

    def compute_hazard(self, detections) -> float:
        # Mock computation combining visual threat with base
        if not detections:
            return 0.0
        return 0.5  # placeholder

    def run(self):
        logger.info(f"[{self.node_id}] Starting main loop...")
        try:
            while True:
                start_time = time.time()
                
                # 1. Capture Frame
                frame = self.camera.capture_frame(self.node_id)
                if frame is None:
                    time.sleep(0.01)
                    continue

                # 2. YOLO Threat Detection
                detections = self.yolo.detect(frame)
                
                # 3. Compute Hazard Vector
                H_local = self.compute_hazard(detections)
                
                # 4. Broadcast to Mesh
                crowd_count = sum(1 for d in detections if d.get('class_name') == 'person')
                self.mesh.broadcast({'hazard': H_local, 'crowd': crowd_count})
                
                # 5. Receive Neighbor Data
                neighbor_data = self.mesh.receive_all()
                
                # 6. Policy Inference
                action = self.policy.infer(H_local, neighbor_data)
                
                # 7-9. Actuation
                self.signage.set_direction(action.signs)
                self.door.set_state(action.doors)
                self.audio.play_if_needed(action)
                
                # 10. Frame Rate Maintenance
                elapsed = time.time() - start_time
                if elapsed < self.sleep_time:
                    time.sleep(self.sleep_time - elapsed)
                    
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        logger.info(f"[{self.node_id}] Shutting down node...")
        # Cleanup code here

def main():
    parser = argparse.ArgumentParser(description="LBP Router Node Main Loop")
    parser.add_argument('--config', type=str, default='config/node_config.yaml', help='Path to node config YAML')
    parser.add_argument('--simulate', action='store_true', help='Run in simulation mode')
    args = parser.parse_args()

    node = RouterNode(args.config)
    node.run()

if __name__ == "__main__":
    main()
