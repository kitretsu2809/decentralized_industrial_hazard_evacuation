"""Discrete routing actions for one or two illuminated compass directions.

The radio protocol's eight-bit direction field describes physical connectivity.
The policy action is separate: it selects no direction (standby), one direction,
or a pair of directions for learned flow splitting.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Iterable, Tuple

import numpy as np

from core.graph.directions import MAX_DIRECTIONS


@lru_cache(maxsize=None)
def routing_action_masks(max_directions: int = MAX_DIRECTIONS) -> Tuple[int, ...]:
    """Return action-index -> direction-bitmask in a stable order.

    Indices ``0..K-1`` retain the historical single-corridor meaning, index
    ``K`` is standby, and the remaining indices enumerate all two-corridor
    combinations.  For eight directions this produces 8 + 1 + C(8, 2) = 37
    categorical choices without exposing invalid three-or-more-arrow commands.
    """
    if max_directions < 1 or max_directions > MAX_DIRECTIONS:
        raise ValueError(f"max_directions must be in [1, {MAX_DIRECTIONS}]")
    singles = tuple(1 << index for index in range(max_directions))
    duals = tuple(
        (1 << first) | (1 << second)
        for first, second in combinations(range(max_directions), 2)
    )
    return singles + (0,) + duals


def num_routing_actions(max_directions: int = MAX_DIRECTIONS) -> int:
    return len(routing_action_masks(max_directions))


def standby_action(max_directions: int = MAX_DIRECTIONS) -> int:
    return max_directions


def action_to_direction_mask(
    action_index: int,
    max_directions: int = MAX_DIRECTIONS,
) -> int:
    masks = routing_action_masks(max_directions)
    index = int(action_index)
    if index < 0 or index >= len(masks):
        raise ValueError(f"Routing action {index} is outside [0, {len(masks) - 1}]")
    return masks[index]


def direction_mask_to_action(
    direction_mask: int,
    max_directions: int = MAX_DIRECTIONS,
) -> int:
    mask = int(direction_mask)
    try:
        return routing_action_masks(max_directions).index(mask)
    except ValueError as error:
        raise ValueError(
            "A routing action must select zero, one, or two valid directions"
        ) from error


def selected_directions(
    action_index: int,
    max_directions: int = MAX_DIRECTIONS,
) -> Tuple[int, ...]:
    mask = action_to_direction_mask(action_index, max_directions)
    return tuple(index for index in range(max_directions) if mask & (1 << index))


def valid_actions_for_direction_mask(
    available_direction_mask: int,
    max_directions: int = MAX_DIRECTIONS,
) -> np.ndarray:
    """Return valid actions using only the supplied physically safe directions."""
    full_mask = (1 << max_directions) - 1
    available = int(available_direction_mask) & full_mask
    return np.asarray(
        [selection & ~available == 0 for selection in routing_action_masks(max_directions)],
        dtype=bool,
    )


def direction_mask_from_indices(indices: Iterable[int]) -> int:
    mask = 0
    for index in indices:
        if index < 0 or index >= MAX_DIRECTIONS:
            raise ValueError(f"Direction index {index} is outside [0, {MAX_DIRECTIONS - 1}]")
        mask |= 1 << index
    return mask
