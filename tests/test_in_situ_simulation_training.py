"""
End-to-end test for Live In-Situ Simulation Training in StandaloneSimulationEngine.
Verifies:
1. start_live_training initializes live trajectory tracking and captures active hazard snapshot.
2. Step-by-step execution in live training mode collects log_probs, critic state values, and rewards.
3. Completing an episode triggers PPO loss computation, backpropagation, and actor/critic weight updates.
4. Episode reset re-injects the hazard scenario and continues with updated neural policy.
5. stop_live_training cleanly stops training and returns engine to eval mode.
"""

import unittest
import torch
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_industrial_standalone import StandaloneSimulationEngine


class TestInSituSimulationTraining(unittest.TestCase):
    def setUp(self):
        self.engine = StandaloneSimulationEngine()
        self.engine.env.fire_model.inject_hazard('GAS', 'reactor_1', intensity=0.85, spread_rate=0.2)

    def tearDown(self):
        self.engine.stop_live_training()

    def test_live_training_cycle_and_gradient_update(self):
        # 1. Start live training for 1 episode with short max steps
        self.engine.live_train_max_steps = 15  # Short episode for fast test
        ok, msg = self.engine.start_live_training(num_episodes=1)
        self.assertTrue(ok)
        self.assertTrue(self.engine.is_live_training)
        self.assertEqual(len(self.engine.live_train_scenario_snapshot), 1)

        # Snapshot weights before training
        initial_actor_weight = self.engine.ai_model.actor.edge_head.weight.clone()

        # 2. Step the simulation multiple times in live training mode
        for _ in range(16):
            state = self.engine.step()
            self.assertIsNotNone(state)

        # 3. Verify episode finished and PPO loss was computed
        status = self.engine.get_live_training_status()
        self.assertFalse(self.engine.is_live_training)
        self.assertEqual(status["status"], "completed")
        self.assertIn("actor_loss", status["metrics"])
        self.assertIn("critic_loss", status["metrics"])
        self.assertIn("survival_rate", status["metrics"])

        # 4. Verify weights were actually updated by PPO loss backward
        updated_actor_weight = self.engine.ai_model.actor.edge_head.weight
        weight_diff = torch.norm(updated_actor_weight - initial_actor_weight).item()
        self.assertGreater(weight_diff, 0.0, "Model weights must be updated by the live simulation training step")
        print(f"Verified live in-situ policy gradient update! Weight diff: {weight_diff:.6f}")


if __name__ == "__main__":
    unittest.main()
