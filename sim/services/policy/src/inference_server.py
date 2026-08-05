import time
import logging
import random
import numpy as np
from typing import Dict, Tuple

from core.messaging.schemas import EnvironmentStateMsg, PolicyActionMsg
from core.policy.inference import PolicyInference
from .event_bus import PolicyEventBus

logger = logging.getLogger(__name__)

class PolicyInferenceServer:
    def __init__(self, model_path: str = None, redis_url: str = 'redis://localhost:6379'):
        self.model_path = model_path
        self.redis_url = redis_url
        self.event_bus = PolicyEventBus(redis_url)
        self.use_onnx = model_path is not None
        
        if self.use_onnx:
            self.policy_inference = PolicyInference(model_path=model_path)
            if not self.policy_inference.is_available():
                logger.warning(f"Failed to load ONNX model from {model_path}. Falling back to random policy.")
                self.use_onnx = False
        
        self.event_bus.subscribe_env_state(self._on_env_state)

    def start(self):
        logger.info(f"Starting Policy Inference Server connected to {self.redis_url}")
        self.event_bus.start_listening()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        logger.info("Stopping Policy Inference Server...")
        self.event_bus.close()

    def _extract_features(self, state: EnvironmentStateMsg) -> Tuple[np.ndarray, np.ndarray]:
        # Dummy feature extraction for illustration.
        # In a real implementation, this would map the graph structure from state.nodes to edge_index and node features.
        num_nodes = len(state.nodes)
        
        # 8 features: hazard, crowd, floor_norm, is_stairwell, is_exit, capacity_ratio, time_remaining, pad
        node_features = np.zeros((num_nodes, 8), dtype=np.float32)
        
        edge_index_list = []
        # simplified edge extraction
        
        # we return dummy for now
        edge_index = np.zeros((2, max(1, num_nodes)), dtype=np.int64)
        
        return node_features, edge_index

    def _random_policy(self, num_agents: int, max_actions: int = 6) -> Dict:
        actions = {}
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            # Random edge actions, door commands, vertical action
            actions[agent_id] = {
                "edge_actions": {f"edge_{j}": random.choice([0, 1, 2]) for j in range(max_actions)},
                "door_commands": {f"door_{j}": random.choice([0, 1]) for j in range(max_actions)},
                "vertical_action": random.choice([0, 1, 2, 3]),
                "protocol_mode": 0
            }
        return actions

    def _on_env_state(self, state: EnvironmentStateMsg):
        start_time = time.time()
        
        # 1. Parse EnvironmentStateMsg is already done by event bus
        # 2. Extract node features from state
        node_features, edge_index = self._extract_features(state)
        
        # 3. Run policy inference
        if self.use_onnx and self.policy_inference.is_available():
            # In a real implementation we would format inputs for the ONNX model properly
            # and map outputs to action dict
            # actions = self.policy_inference.infer(...)
            # For now fallback to dummy random
            num_agents = len(state.nodes) # Assuming one agent per node
            actions_dict = self._random_policy(num_agents)
        else:
            num_agents = len(state.nodes)
            actions_dict = self._random_policy(num_agents)
            
        # 4. Publish PolicyActionMsg
        action_msg = PolicyActionMsg(
            step=state.step,
            actions=actions_dict,
            timestamp=time.time()
        )
        
        self.event_bus.publish_action(action_msg)
        
        latency = (time.time() - start_time) * 1000.0
        if latency > 10.0:
            logger.warning(f"Inference latency exceeded 10ms: {latency:.2f}ms")

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    server = PolicyInferenceServer(redis_url=redis_url)
    server.start()
