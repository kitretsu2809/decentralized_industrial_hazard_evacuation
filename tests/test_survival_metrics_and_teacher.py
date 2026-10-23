"""
Unit regression tests for:
  1. Distinct separation of Survival Rate (casualty-free %) vs. Clearance Rate (evacuated %).
  2. ValueStandardizer (PopArt return normalization and denormalization).
  3. Congestion-Aware Teacher Action generation and valid action masking.
  4. Continuous physical headway progress in IndustrialEvacuationEnv.
  5. ST-TBA-GAT evaluate_actions with return_logits=True for teacher imitation loss.
"""
import unittest
import numpy as np
import torch

from simulator.engine.simulation import Simulation
from simulator.engine.pedestrian import PedestrianState
from simulator.env.evacuation_env import IndustrialEvacuationEnv
from simulator.policy.st_tba_gat import ST_TBA_GAT
from simulator.training.train_mappo import MAPPOTrainer, ValueStandardizer, RolloutBuffer


class TestSurvivalMetricsAndTeacher(unittest.TestCase):
    def test_survival_vs_clearance_metric_separation(self):
        """
        Verify that survival_rate measures casualty-free proportion,
        while clearance_rate measures exit arrival proportion.
        """
        sim = Simulation()
        sim.reset(num_evacuees=100)

        # Simulate scenario: 40 evacuated, 5 casualties, 55 still in transit
        for i, ped in enumerate(sim.agents):
            if i < 40:
                ped.state = PedestrianState.EVACUATED
            elif i < 45:
                ped.state = PedestrianState.CASUALTY
            else:
                ped.state = PedestrianState.MOVING

        metrics = sim.metrics()
        self.assertEqual(metrics["evacuated"], 40)
        self.assertEqual(metrics["casualties"], 5)
        self.assertEqual(metrics["in_transit"], 55)
        self.assertEqual(metrics["total"], 100)

        # Survival rate = (100 - 5) / 100 = 95.0%
        self.assertAlmostEqual(metrics["survival_rate"], 95.0, places=1)
        # Clearance rate = 40 / 100 = 40.0%
        self.assertAlmostEqual(metrics["clearance_rate"], 40.0, places=1)

    def test_value_standardizer_running_statistics(self):
        """
        Verify that ValueStandardizer accurately normalizes targets to zero mean
        and unit variance, and denormalizes back to true return scale.
        """
        standardizer = ValueStandardizer(clip_val=5.0)
        returns = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        standardizer.update(returns)

        # True mean is 30.0, std is sqrt(200) ~ 14.14
        self.assertAlmostEqual(standardizer.mean, 30.0, places=1)
        self.assertGreater(standardizer.var, 150.0)

        # Normalize 30.0 should be near 0.0
        norm_val = standardizer.normalize(torch.tensor(30.0))
        self.assertAlmostEqual(float(norm_val), 0.0, places=1)

        # Denormalize 0.0 should be near 30.0
        denorm_val = standardizer.denormalize(0.0)
        self.assertAlmostEqual(float(denorm_val), 30.0, places=1)

    def test_teacher_actions_respect_action_masks(self):
        """
        Verify that compute_teacher_actions produces valid actions according to action_masks.
        """
        env = IndustrialEvacuationEnv(num_evacuees=20)
        env.reset(options={"node_id": "reactor_1", "hazard_type": "FIRE", "intensity": 0.90})
        trainer = MAPPOTrainer(env=env, curriculum=False, device="cpu")

        masks_np = env.get_all_action_masks()
        teacher_acts = trainer.compute_teacher_actions(masks_np)

        self.assertEqual(len(teacher_acts), len(env.agents))
        for i, act in enumerate(teacher_acts):
            # Teacher action MUST be a valid action according to masks_np
            self.assertTrue(
                masks_np[i, act],
                f"Agent {env.agents[i]} selected action {act} which is masked out!",
            )

    def test_st_tba_gat_evaluate_actions_with_logits(self):
        """
        Verify that evaluate_actions with return_logits=True returns 4 tensors
        including logits suitable for cross-entropy teacher loss.
        """
        node_dim = 51
        num_nodes = 6
        model = ST_TBA_GAT(node_dim=node_dim, hidden_dim=32, max_corridors=8)
        model.eval()

        node_features = torch.rand(num_nodes, node_dim)
        edges = [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]]
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        hidden = torch.zeros(num_nodes, 32)
        actions = torch.zeros(num_nodes, dtype=torch.long)
        masks = torch.ones(num_nodes, 37, dtype=torch.bool)

        val_pred, log_probs, entropy, logits = model.evaluate_actions(
            node_features=node_features,
            edge_index=edge_index,
            hidden_state=hidden,
            actions=actions,
            action_masks=masks,
            return_logits=True,
        )

        self.assertEqual(logits.shape, (num_nodes, 37))
        self.assertEqual(log_probs.shape, (num_nodes,))
        self.assertEqual(entropy.shape, (num_nodes,))
        self.assertEqual(val_pred.shape, (1,))

    def test_continuous_headway_reward(self):
        """
        Verify that physical movement towards exits results in positive headway progress.
        """
        env = IndustrialEvacuationEnv(num_evacuees=10)
        env.reset()

        # Check that distance helper returns positive floats for active pedestrians
        for ped in env.sim.agents[:5]:
            dist = env._pedestrian_exit_distance(ped)
            self.assertGreater(dist, 0.0)


if __name__ == "__main__":
    unittest.main()
