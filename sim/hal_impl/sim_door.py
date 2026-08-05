"""
Simulated door controller.

Tracks magnetic lock states in memory. In a real deployment,
this would control GPIO pins connected to relay modules for maglocks.
"""
from __future__ import annotations

import logging
from typing import Dict

from core.hal.base import DoorHAL
from core.graph.types import EdgeState

logger = logging.getLogger(__name__)


class SimDoor(DoorHAL):
    """In-memory simulation of magnetic door locks."""

    def __init__(self):
        # edge_id -> state string
        self._states: Dict[str, str] = {}

    def lock(self, edge_id: str) -> bool:
        """Lock a door.

        Args:
            edge_id: The edge (corridor) with the door.

        Returns:
            True if lock was successful.
        """
        self._states[edge_id] = EdgeState.LOCKED.value
        logger.info(f"[DOOR] Edge {edge_id}: LOCKED")
        return True

    def unlock(self, edge_id: str) -> bool:
        """Unlock a door.

        Args:
            edge_id: The edge (corridor) with the door.

        Returns:
            True if unlock was successful.
        """
        self._states[edge_id] = EdgeState.OPEN.value
        logger.info(f"[DOOR] Edge {edge_id}: UNLOCKED")
        return True

    def emergency_seal(self, edge_id: str) -> bool:
        """Emergency seal a door (higher priority than lock).

        Args:
            edge_id: The edge (corridor) with the door.

        Returns:
            True if seal was successful.
        """
        self._states[edge_id] = EdgeState.EMERGENCY_SEALED.value
        logger.warning(f"[DOOR] Edge {edge_id}: EMERGENCY SEALED")
        return True

    def get_state(self, edge_id: str) -> str:
        """Get current state of a door."""
        return self._states.get(edge_id, EdgeState.OPEN.value)

    def get_all_states(self) -> Dict[str, str]:
        """Get all door states."""
        return dict(self._states)
