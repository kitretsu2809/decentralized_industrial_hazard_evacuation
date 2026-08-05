"""
Simulated audio controller.

Logs audio alert events. In a real deployment, this would drive
I2S audio playback to directional speakers.
"""
from __future__ import annotations

import logging
from typing import Dict

from core.hal.base import AudioHAL

logger = logging.getLogger(__name__)


class SimAudio(AudioHAL):
    """In-memory simulation of directional speaker alerts."""

    VALID_ALERTS = {"evacuate", "lockdown", "caution", "all_clear"}

    def __init__(self):
        # node_id -> {alert_type, volume, playing}
        self._states: Dict[str, Dict] = {}

    def play_alert(self, node_id: str, alert_type: str) -> None:
        """Play an alert tone at a node.

        Args:
            node_id: The router node ID.
            alert_type: Type of alert ('evacuate', 'lockdown', 'caution', 'all_clear').
        """
        if alert_type not in self.VALID_ALERTS:
            logger.warning(f"[AUDIO] Unknown alert type: {alert_type}")
            return
        self._states[node_id] = {
            "alert_type": alert_type,
            "volume": self._states.get(node_id, {}).get("volume", 0.8),
            "playing": True,
        }
        logger.info(f"[AUDIO] Node {node_id}: Playing '{alert_type}'")

    def stop_alert(self, node_id: str) -> None:
        """Stop playing audio at a node."""
        if node_id in self._states:
            self._states[node_id]["playing"] = False
        logger.debug(f"[AUDIO] Node {node_id}: Stopped")

    def set_volume(self, node_id: str, volume: float) -> None:
        """Set audio volume at a node.

        Args:
            node_id: The router node ID.
            volume: Volume level (0.0 to 1.0).
        """
        volume = max(0.0, min(1.0, volume))
        if node_id not in self._states:
            self._states[node_id] = {"alert_type": None, "playing": False}
        self._states[node_id]["volume"] = volume

    def get_state(self, node_id: str) -> Dict:
        """Get current audio state for a node."""
        return self._states.get(node_id, {"alert_type": None, "volume": 0.8, "playing": False})
