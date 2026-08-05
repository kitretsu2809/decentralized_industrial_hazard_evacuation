"""
Mesh Network Client

Supports Thread (via OpenThread) and ESP-NOW protocols
for peer-to-peer communication between nodes.
"""
from typing import Optional, Dict

class MeshClient:
    def __init__(self, node_id: str, protocol: str = 'esp_now', config: dict = None):
        self.node_id = node_id
        self.protocol = protocol
        self.config = config or {}
        self.neighbor_states = {}
        
        if self.protocol == 'esp_now':
            self._init_esp_now()
        elif self.protocol == 'thread':
            self._init_thread()
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

    def _init_esp_now(self):
        # ESP-NOW specific initialization (mock)
        pass

    def _init_thread(self):
        # OpenThread specific initialization (mock)
        pass

    def broadcast(self, data: dict):
        """Broadcast state to all neighbors."""
        pass

    def receive(self, timeout: float = 0.05) -> Optional[dict]:
        """Receive data from any neighbor."""
        # Mock receive logic
        return None
        
    def receive_all(self) -> Dict[str, dict]:
        """Consume all waiting packets and update internal neighbor states."""
        # In a real implementation, this would loop until no more messages
        # Here we just return the cached states
        return self.neighbor_states

    def get_neighbor_states(self) -> Dict[str, dict]:
        """Get cached neighbor states."""
        return self.neighbor_states

    def get_health(self) -> dict:
        """Get mesh health metrics."""
        return {
            'protocol': self.protocol,
            'active_neighbors': len(self.neighbor_states),
            'tx_errors': 0,
            'rx_errors': 0
        }
