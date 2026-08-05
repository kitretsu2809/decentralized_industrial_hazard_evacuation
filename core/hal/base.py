from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np

class SignageHAL(ABC):
    """Abstract Hardware Abstraction Layer for digital signage."""

    @abstractmethod
    def set_direction(self, node_id: str, directions: Dict[str, str]) -> None:
        """Set arrow directions per outgoing edge."""
        pass

    @abstractmethod
    def set_lockdown_message(self, node_id: str, message: str) -> None:
        """Display lockdown text."""
        pass

    @abstractmethod
    def clear(self, node_id: str) -> None:
        """Clear all signs at node."""
        pass

    @abstractmethod
    def get_current_state(self, node_id: str) -> Dict[str, str]:
        """Get current sign state."""
        pass

class DoorHAL(ABC):
    """Abstract Hardware Abstraction Layer for smart doors."""

    @abstractmethod
    def lock(self, edge_id: str) -> bool:
        """Lock door, return success."""
        pass

    @abstractmethod
    def unlock(self, edge_id: str) -> bool:
        """Unlock door, return success."""
        pass

    @abstractmethod
    def emergency_seal(self, edge_id: str) -> bool:
        """Emergency seal, return success."""
        pass

    @abstractmethod
    def get_state(self, edge_id: str) -> str:
        """Get door state."""
        pass

    @abstractmethod
    def get_all_states(self) -> Dict[str, str]:
        """Get all door states."""
        pass

class AudioHAL(ABC):
    """Abstract Hardware Abstraction Layer for audio systems."""

    @abstractmethod
    def play_alert(self, node_id: str, alert_type: str) -> None:
        """Play alert (evacuate/lockdown/caution)."""
        pass

    @abstractmethod
    def stop_alert(self, node_id: str) -> None:
        """Stop playing alert."""
        pass

    @abstractmethod
    def set_volume(self, node_id: str, volume: float) -> None:
        """Set volume (0.0-1.0)."""
        pass

class SensorHAL(ABC):
    """Abstract Hardware Abstraction Layer for environmental sensors."""

    @abstractmethod
    def read_gas(self, node_id: str) -> float:
        """MQ-2 gas reading (ppm)."""
        pass

    @abstractmethod
    def read_temperature(self, node_id: str) -> float:
        """Temperature (°C)."""
        pass

    @abstractmethod
    def read_humidity(self, node_id: str) -> float:
        """Humidity (%)."""
        pass

    @abstractmethod
    def read_presence(self, node_id: str) -> int:
        """IR ToF presence count."""
        pass

class CameraHAL(ABC):
    """Abstract Hardware Abstraction Layer for cameras."""

    @abstractmethod
    def capture_frame(self, node_id: str) -> Optional[np.ndarray]:
        """Capture RGB frame or return None if failed."""
        pass

    @abstractmethod
    def is_available(self, node_id: str) -> bool:
        """Check if camera is available."""
        pass

    @abstractmethod
    def get_resolution(self, node_id: str) -> tuple:
        """Get camera resolution as (width, height)."""
        pass

class MeshHAL(ABC):
    """Abstract Hardware Abstraction Layer for mesh network communication."""

    @abstractmethod
    def broadcast(self, packet: bytes, ttl: int = 2) -> bool:
        """Broadcast packet to mesh network."""
        pass

    @abstractmethod
    def send_to(self, target_id: str, packet: bytes) -> bool:
        """Send packet to a specific node."""
        pass

    @abstractmethod
    def receive(self, timeout: float = 0.1) -> Optional[bytes]:
        """Receive packet from mesh network."""
        pass

    @abstractmethod
    def get_neighbors(self) -> List[str]:
        """Get list of discovered mesh neighbors."""
        pass

    @abstractmethod
    def get_mesh_health(self) -> Dict[str, Any]:
        """Get mesh status and health metrics."""
        pass
