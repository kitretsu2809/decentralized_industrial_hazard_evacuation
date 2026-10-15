"""
Mesh Network Client

Supports Thread (via OpenThread) and ESP-NOW protocols
for peer-to-peer communication between nodes.
"""
from typing import Optional, Dict, Union

from core.communication.hazard_packet import HazardPacket32, unpack_hazard_packet

class MeshClient:
    def __init__(self, node_id: str, protocol: str = 'esp_now', config: dict = None):
        self.node_id = node_id
        self.protocol = protocol
        self.config = config or {}
        self.neighbor_states = {}
        self.last_tx_payload: Optional[bytes] = None
        
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

    def broadcast(self, data: Union[HazardPacket32, bytes, bytearray]) -> bytes:
        """Validate and broadcast one canonical four-byte hazard PDU."""
        payload = data.to_bytes() if isinstance(data, HazardPacket32) else bytes(data)
        if len(payload) != 4:
            raise ValueError("Mesh hazard payload must be exactly four bytes")
        self.last_tx_payload = payload
        # Hardware adapters send payload; ESP-NOW/Thread metadata identifies sender.
        return payload

    def receive(self, timeout: float = 0.05) -> Optional[dict]:
        """Receive data from any neighbor."""
        # Mock receive logic
        return None
        
    def receive_all(self) -> Dict[str, dict]:
        """Consume all waiting packets and update internal neighbor states."""
        # In a real implementation, this would loop until no more messages
        # Here we just return the cached states
        return self.neighbor_states

    def ingest(self, sender_id: str, payload: bytes) -> dict:
        """Decode a received PDU; useful for hardware adapters and simulation."""
        decoded = unpack_hazard_packet(payload).public_dict()
        self.neighbor_states[sender_id] = decoded
        return decoded

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
