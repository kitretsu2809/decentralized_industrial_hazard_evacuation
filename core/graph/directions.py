"""Deterministic eight-direction slots for node corridors.

Policy actions, signboards, and the communication bitmap all use the same slot
order.  This removes the previous implicit dependency on edge insertion order.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DIRECTION_NAMES: Tuple[str, ...] = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
MAX_DIRECTIONS = len(DIRECTION_NAMES)
_COLLISION_OFFSETS = (0, 1, -1, 2, -2, 3, -3, 4)


def compass_direction_index(
    origin: Sequence[float],
    target: Sequence[float],
    floor_delta: int = 0,
) -> int:
    """Return the closest clockwise compass slot, with N at index zero.

    Co-located vertical links use N for upward travel and S for downward travel;
    collision resolution in :func:`assign_direction_slots` keeps every physical
    corridor in a unique one-bit slot.
    """
    dx = float(target[0]) - float(origin[0])
    dy = float(target[1]) - float(origin[1])
    if abs(dx) + abs(dy) < 1e-9:
        return 0 if floor_delta >= 0 else 4
    clockwise_degrees = math.degrees(math.atan2(dx, -dy)) % 360.0
    return int((clockwise_degrees + 22.5) // 45.0) % MAX_DIRECTIONS


def assign_direction_slots(
    node_id: str,
    neighbors: Iterable[str],
    positions: Dict[str, Sequence[float]],
    floors: Optional[Dict[str, int]] = None,
) -> Tuple[Optional[str], ...]:
    """Assign at most eight neighbors to stable compass-oriented slots.

    Multiple corridors can share the same coarse compass bearing.  They are
    placed in the nearest free adjacent slot, deterministically by neighbor id.
    """
    neighbor_list = sorted(set(neighbors))
    if len(neighbor_list) > MAX_DIRECTIONS:
        raise ValueError(
            f"Node {node_id!r} has {len(neighbor_list)} neighbors; "
            f"the explicit protocol supports at most {MAX_DIRECTIONS}."
        )
    if node_id not in positions:
        raise KeyError(f"Missing position for node {node_id!r}")

    floors = floors or {}
    preferred = []
    for neighbor in neighbor_list:
        if neighbor not in positions:
            raise KeyError(f"Missing position for neighbor {neighbor!r}")
        floor_delta = int(floors.get(neighbor, 0)) - int(floors.get(node_id, 0))
        slot = compass_direction_index(positions[node_id], positions[neighbor], floor_delta)
        preferred.append((slot, neighbor))

    slots: List[Optional[str]] = [None] * MAX_DIRECTIONS
    for preferred_slot, neighbor in sorted(preferred):
        for offset in _COLLISION_OFFSETS:
            candidate = (preferred_slot + offset) % MAX_DIRECTIONS
            if slots[candidate] is None:
                slots[candidate] = neighbor
                break
    return tuple(slots)


def connected_direction_mask(slots: Sequence[Optional[str]]) -> int:
    """Return an eight-bit mask where one means a corridor occupies the slot."""
    if len(slots) != MAX_DIRECTIONS:
        raise ValueError(f"Expected {MAX_DIRECTIONS} direction slots")
    return sum(1 << index for index, neighbor in enumerate(slots) if neighbor is not None)
