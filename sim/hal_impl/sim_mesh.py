"""
Simulated mesh network controller.

Uses Redis pub/sub to simulate the Thread/ESP-NOW ad-hoc mesh network.
In a real deployment, this would use OpenThread or ESP-NOW protocols
for peer-to-peer broadcast over radio.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from core.hal.base import MeshHAL

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. SimMesh will operate in local-only mode.")


class SimMesh(MeshHAL):
    """Simulated mesh network using Redis pub/sub.

    Each node broadcasts its state on a Redis channel.
    Nodes subscribe to their neighbors' channels to receive updates.
    """

    def __init__(self, node_id: str, redis_url: str = "redis://localhost:6379"):
        """Initialize the simulated mesh network.

        Args:
            node_id: This node's unique identifier.
            redis_url: Redis connection URL.
        """
        self._node_id = node_id
        self._neighbors: List[str] = []
        self._receive_buffer: List[bytes] = []
        self._redis_client: Optional[Any] = None
        self._pubsub: Optional[Any] = None

        if REDIS_AVAILABLE:
            try:
                self._redis_client = redis.from_url(redis_url)
                self._pubsub = self._redis_client.pubsub()
                logger.info(f"[MESH] Node {node_id}: Connected to Redis mesh")
            except Exception as e:
                logger.error(f"[MESH] Node {node_id}: Redis connection failed: {e}")

    def set_neighbors(self, neighbor_ids: List[str]) -> None:
        """Configure which nodes are mesh neighbors.

        Args:
            neighbor_ids: List of neighbor node IDs to listen to.
        """
        self._neighbors = list(neighbor_ids)
        if self._pubsub:
            # Subscribe to each neighbor's broadcast channel
            for nid in neighbor_ids:
                channel = f"lbp:mesh:{nid}"
                self._pubsub.subscribe(channel)
            logger.debug(f"[MESH] Node {self._node_id}: Subscribed to {len(neighbor_ids)} neighbors")

    def broadcast(self, packet: bytes, ttl: int = 2) -> bool:
        """Broadcast a packet to all mesh neighbors.

        Args:
            packet: Binary data to broadcast.
            ttl: Time-to-live (hop count). Not used in simulation.

        Returns:
            True if broadcast was successful.
        """
        if not self._redis_client:
            return False

        try:
            channel = f"lbp:mesh:{self._node_id}"
            envelope = json.dumps({
                "sender": self._node_id,
                "ttl": ttl,
                "timestamp": time.time(),
                "data": packet.hex(),
            })
            self._redis_client.publish(channel, envelope)
            return True
        except Exception as e:
            logger.error(f"[MESH] Broadcast failed: {e}")
            return False

    def send_to(self, target_id: str, packet: bytes) -> bool:
        """Send a packet to a specific node.

        Args:
            target_id: Target node ID.
            packet: Binary data to send.

        Returns:
            True if send was successful.
        """
        if not self._redis_client:
            return False

        try:
            channel = f"lbp:mesh:direct:{target_id}"
            envelope = json.dumps({
                "sender": self._node_id,
                "timestamp": time.time(),
                "data": packet.hex(),
            })
            self._redis_client.publish(channel, envelope)
            return True
        except Exception as e:
            logger.error(f"[MESH] Send to {target_id} failed: {e}")
            return False

    def receive(self, timeout: float = 0.1) -> Optional[bytes]:
        """Receive a packet from the mesh.

        Args:
            timeout: Max time to wait for a packet (seconds).

        Returns:
            Received packet bytes, or None if no packet available.
        """
        if not self._pubsub:
            return None

        try:
            message = self._pubsub.get_message(timeout=timeout)
            if message and message["type"] == "message":
                envelope = json.loads(message["data"])
                return bytes.fromhex(envelope["data"])
        except Exception as e:
            logger.error(f"[MESH] Receive failed: {e}")

        return None

    def get_neighbors(self) -> List[str]:
        """Get list of configured mesh neighbors."""
        return list(self._neighbors)

    def get_mesh_health(self) -> Dict[str, Any]:
        """Get mesh network health status."""
        return {
            "node_id": self._node_id,
            "neighbors": len(self._neighbors),
            "redis_connected": self._redis_client is not None,
            "status": "healthy" if self._redis_client else "disconnected",
        }

    def close(self) -> None:
        """Close mesh connections."""
        if self._pubsub:
            self._pubsub.close()
        if self._redis_client:
            self._redis_client.close()
