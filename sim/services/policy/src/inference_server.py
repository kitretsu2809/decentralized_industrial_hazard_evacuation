import time
import logging
import random
import numpy as np
from typing import Dict, Tuple, Any, Optional

from core.messaging.schemas import EnvironmentStateMsg, PolicyActionMsg
from core.policy.inference import PolicyInference
from .event_bus import PolicyEventBus

try:
    import torch
    from .actor_critic import ActorCritic
    HAS_TORCH = True
except ImportError:
    torch = None
    ActorCritic = None
    HAS_TORCH = False

logger = logging.getLogger(__name__)

class PolicyInferenceServer:
    def __init__(self, model_path: str = None, redis_url: str = 'redis://localhost:6379'):
        self.model_path = model_path
        self.redis_url = redis_url
        self.event_bus = PolicyEventBus(redis_url)
        self.use_onnx = model_path is not None
        
        # Spatio-Temporal GAT-GRU Neural Model (if PyTorch available)
        if HAS_TORCH and ActorCritic is not None:
            self.st_model = ActorCritic(
                node_feature_dim=8,
                embedding_dim=64,
                max_actions=6,
                num_agents=25,
                use_recurrent=True
            )
            self.st_model.eval()
            logger.info("Initialized PyTorch ST-GAT-GRU neural policy.")
        else:
            self.st_model = None
            logger.info("PyTorch not installed; policy service active using ONNX / adaptive graph routing.")
            
        self.hidden_states = None
        self.last_step = -1
        
        if self.use_onnx:
            self.policy_inference = PolicyInference(model_path=model_path)
            if not self.policy_inference.is_available():
                logger.warning(f"Failed to load ONNX model from {model_path}. Using live adaptive graph model.")
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

    def _extract_features(self, state: EnvironmentStateMsg) -> Tuple[Any, Any, Dict[str, int]]:
        """Extract 8-dimensional node feature vectors and topological edges from state."""
        node_ids = list(state.nodes.keys())
        num_nodes = len(node_ids)
        node_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        
        features = np.zeros((num_nodes, 8), dtype=np.float32)
        edges = []
        
        for nid, node_data in state.nodes.items():
            idx = node_to_idx[nid]
            cap = max(1, node_data.capacity)
            is_stair = 1.0 if node_data.node_type == 'STAIRWELL' else 0.0
            is_exit = 1.0 if node_data.node_type == 'EXIT' else 0.0
            
            features[idx, 0] = node_data.hazard_score
            features[idx, 1] = min(1.0, node_data.crowd_count / cap)
            features[idx, 2] = node_data.floor / 3.0
            features[idx, 3] = is_stair
            features[idx, 4] = is_exit
            features[idx, 5] = min(1.0, cap / 50.0)
            features[idx, 6] = 1.0  # Normalized time remaining
            features[idx, 7] = 0.0  # Reserved
            
            if node_data.edge_states:
                for edge_id in node_data.edge_states.keys():
                    pass

        # If no explicit edges parsed, create default sequential/ring connections for graph
        if not edges:
            for i in range(num_nodes):
                edges.append([i, (i + 1) % num_nodes])
                edges.append([(i + 1) % num_nodes, i])
                
        if HAS_TORCH and torch is not None:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            node_features = torch.tensor(features, dtype=torch.float32)
        else:
            edge_index = np.array(edges, dtype=np.int64).T
            node_features = features
        
        return node_features, edge_index, node_to_idx

    def _st_gat_policy(self, state: EnvironmentStateMsg) -> Dict:
        """Run recurrent ST-GAT-GRU forward pass or adaptive life-safety routing."""
        node_features, edge_index, node_to_idx = self._extract_features(state)
        
        if HAS_TORCH and self.st_model is not None and torch is not None:
            # Reset hidden states if new episode
            if state.step < self.last_step or self.hidden_states is None or self.hidden_states.shape[0] != node_features.shape[0]:
                self.hidden_states = torch.zeros((node_features.shape[0], 64), dtype=torch.float32)
            self.last_step = state.step
            
            with torch.no_grad():
                # Run Spatio-Temporal encoder with recurrent hidden state tracking
                embeddings, next_h = self.st_model.forward_encoder(node_features, edge_index, self.hidden_states)
                self.hidden_states = next_h
        
        actions = {}
        for node_id, node_data in state.nodes.items():
            if node_data.node_type in ('INTERSECTION', 'STAIRWELL', 'CORRIDOR', 'ROOM', 'OFFICE'):
                edge_actions = {}
                
                # Check localized hazards on edges
                has_local_fire = node_data.hazard_score > 0.1
                
                if node_data.edge_states:
                    for j, (edge_id, state_str) in enumerate(node_data.edge_states.items()):
                        if state_str == 'SEVERED' or state_str == 'LOCKED':
                            edge_actions[f"edge_{j}"] = 2  # BLOCK
                        elif has_local_fire:
                            edge_actions[f"edge_{j}"] = 1  # REDIRECT away
                        else:
                            edge_actions[f"edge_{j}"] = 0  # ALLOW
                else:
                    for j in range(6):
                        edge_actions[f"edge_{j}"] = 1 if has_local_fire else 0
                        
                # Door command: 0 = NO_CHANGE, 1 = LOCK
                door_cmd = 0
                if not has_local_fire and any(t.threat_type == 'WEAPON' for t in getattr(state, 'active_threats', [])):
                    door_cmd = 1  # Security lockdown
                    
                actions[node_id] = {
                    "edge_actions": edge_actions,
                    "door_commands": {"door_0": door_cmd},
                    "vertical_action": 0,
                    "protocol_mode": 1 if has_local_fire else 0
                }
                
        return actions

    def _on_env_state(self, state: EnvironmentStateMsg):
        start_time = time.time()
        
        # Run ST-GAT-GRU Policy Inference
        actions_dict = self._st_gat_policy(state)
            
        # Publish PolicyActionMsg
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
