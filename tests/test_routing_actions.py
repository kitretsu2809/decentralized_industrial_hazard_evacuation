import unittest

import numpy as np

from core.policy.routing_actions import (
    action_to_direction_mask,
    direction_mask_to_action,
    num_routing_actions,
    routing_action_masks,
    selected_directions,
    standby_action,
    valid_actions_for_direction_mask,
)


class RoutingActionCodecTests(unittest.TestCase):
    def test_eight_directions_have_37_unique_actions(self):
        masks = routing_action_masks(8)
        self.assertEqual(num_routing_actions(8), 37)
        self.assertEqual(len(set(masks)), 37)
        self.assertEqual(masks[standby_action(8)], 0)
        self.assertTrue(all(mask.bit_count() <= 2 for mask in masks))

    def test_single_and_dual_actions_round_trip(self):
        for mask in ((1 << 2), (1 << 1) | (1 << 6), 0):
            action = direction_mask_to_action(mask, 8)
            self.assertEqual(action_to_direction_mask(action, 8), mask)
        self.assertEqual(selected_directions(direction_mask_to_action(0b10000010)), (1, 7))

    def test_validity_rejects_actions_containing_unavailable_direction(self):
        available = (1 << 1) | (1 << 4)
        valid = valid_actions_for_direction_mask(available, 8)
        allowed_masks = {
            action_to_direction_mask(index, 8)
            for index in np.flatnonzero(valid)
        }
        self.assertEqual(allowed_masks, {0, 1 << 1, 1 << 4, (1 << 1) | (1 << 4)})


if __name__ == "__main__":
    unittest.main()
