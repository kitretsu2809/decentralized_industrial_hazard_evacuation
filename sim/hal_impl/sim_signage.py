"""
Simulated signage controller.

Tracks LED sign states in memory and publishes changes via logging.
In the simulation, the GUI reads sign states from the environment state
broadcast over Redis.
"""
from __future__ import annotations

import logging
from typing import Dict

from core.hal.base import SignageHAL

logger = logging.getLogger(__name__)


class SimSignage(SignageHAL):
    """In-memory simulation of RGB LED matrix directional signs."""

    def __init__(self):
        # node_id -> {edge_id: direction_string}
        self._states: Dict[str, Dict[str, str]] = {}
        self._lockdown_messages: Dict[str, str] = {}

    def set_direction(self, node_id: str, directions: Dict[str, str]) -> None:
        """Set directional arrows for a node's outgoing edges.

        Args:
            node_id: The router node ID.
            directions: Mapping of edge_id -> direction string
                        (e.g., 'ALLOW', 'REDIRECT', 'BLOCKED').
        """
        self._states[node_id] = dict(directions)
        logger.debug(f"[SIGN] Node {node_id}: {directions}")

    def set_lockdown_message(self, node_id: str, message: str) -> None:
        """Display a lockdown message on the sign.

        Args:
            node_id: The router node ID.
            message: Text to display (e.g., 'LOCKDOWN: ENTER NEAREST ROOM').
        """
        self._lockdown_messages[node_id] = message
        self._states[node_id] = {"lockdown": message}
        logger.info(f"[SIGN] Node {node_id} LOCKDOWN: {message}")

    def clear(self, node_id: str) -> None:
        """Clear all sign displays at a node."""
        self._states.pop(node_id, None)
        self._lockdown_messages.pop(node_id, None)
        logger.debug(f"[SIGN] Node {node_id}: CLEARED")

    def get_current_state(self, node_id: str) -> Dict[str, str]:
        """Get current sign state for a node."""
        return self._states.get(node_id, {})

    def get_all_states(self) -> Dict[str, Dict[str, str]]:
        """Get all sign states across all nodes."""
        return dict(self._states)
