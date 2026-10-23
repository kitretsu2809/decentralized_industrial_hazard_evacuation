import random
import tempfile
import unittest

import numpy as np
import torch

from simulator.policy.st_tba_gat import ST_TBA_GAT
from simulator.training.train_mappo import MAPPOTrainer, RolloutBuffer


class TestMAPPOTrainingLogic(unittest.TestCase):
    def _single_step_buffer(self, terminated: bool) -> RolloutBuffer:
        buffer = RolloutBuffer(num_agents=1, obs_dim=1, max_corridors=1, hidden_dim=1)
        buffer.add(
            node_features=np.zeros((1, 1), dtype=np.float32),
            hidden_states=np.zeros((1, 1), dtype=np.float32),
            actions=np.zeros(1, dtype=np.int64),
            action_log_probs=np.zeros(1, dtype=np.float32),
            action_masks=np.ones((1, 2), dtype=bool),
            reward=1.0,
            value=0.0,
            terminated=terminated,
        )
        return buffer

    def test_gae_bootstraps_truncation_but_not_terminal_state(self):
        truncated = self._single_step_buffer(terminated=False)
        terminal = self._single_step_buffer(terminated=True)

        truncated_returns, _ = truncated.compute_returns_and_advantages(
            last_value=10.0, gamma=0.99, gae_lambda=0.95
        )
        terminal_returns, _ = terminal.compute_returns_and_advantages(
            last_value=10.0, gamma=0.99, gae_lambda=0.95
        )

        self.assertAlmostEqual(float(truncated_returns[0]), 10.9, places=5)
        self.assertAlmostEqual(float(terminal_returns[0]), 1.0, places=5)

    def test_rollout_and_update_log_probs_match_without_weight_update(self):
        torch.manual_seed(11)
        model = ST_TBA_GAT(
            node_dim=51,
            hidden_dim=64,
            max_corridors=8,
            dropout=0.50,
        )
        model.eval()
        num_nodes = 8
        node_features = torch.rand(num_nodes, 51)
        node_features[:, 0] = 0.0
        node_features[:, 11::5] = 0.0
        edges = []
        for idx in range(num_nodes):
            edges.extend([[idx, (idx + 1) % num_nodes], [(idx + 1) % num_nodes, idx]])
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        hidden = torch.zeros(num_nodes, 64)
        masks = torch.ones(num_nodes, 37, dtype=torch.bool)

        with torch.no_grad():
            actions, rollout_log_probs, _, _ = model.act(
                node_features, edge_index, hidden, masks, deterministic=True
            )
            _, update_log_probs, _ = model.evaluate_actions(
                node_features, edge_index, hidden, actions, masks
            )

        self.assertTrue(torch.equal(rollout_log_probs, update_log_probs))

    def test_checkpoint_rank_prioritizes_curriculum_maturity(self):
        easy = {"curriculum_stage": 1, "survival_rate": 100.0, "reward": 50.0}
        hard = {"curriculum_stage": 4, "survival_rate": 55.0, "reward": -20.0}
        harder_better = {"curriculum_stage": 4, "survival_rate": 60.0, "reward": -30.0}

        self.assertGreater(
            MAPPOTrainer._checkpoint_rank(hard),
            MAPPOTrainer._checkpoint_rank(easy),
        )
        self.assertGreater(
            MAPPOTrainer._checkpoint_rank(harder_better),
            MAPPOTrainer._checkpoint_rank(hard),
        )

    def test_resume_restores_optimizer_state(self):
        trainer = MAPPOTrainer(
            min_evacuees=2,
            max_evacuees=2,
            curriculum=False,
            device="cpu",
        )
        trainer.optimizer.zero_grad()
        sum(parameter.sum() for parameter in trainer.policy.parameters()).backward()
        trainer.optimizer.step()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = f"{temp_dir}/resume.pt"
            trainer.policy.save_checkpoint(
                checkpoint,
                extra_meta={"episode": 7},
                optimizer_state_dict=trainer.optimizer.state_dict(),
            )
            resumed = MAPPOTrainer(
                min_evacuees=2,
                max_evacuees=2,
                curriculum=False,
                device="cpu",
                resume_checkpoint=checkpoint,
            )

        self.assertEqual(resumed.start_episode, 8)
        self.assertTrue(resumed.optimizer.state)
        self.assertFalse(resumed.policy.training)

    def test_training_episode_updates_actor_with_finite_diagnostics(self):
        random.seed(23)
        np.random.seed(23)
        torch.manual_seed(23)
        trainer = MAPPOTrainer(
            min_evacuees=6,
            max_evacuees=6,
            curriculum=False,
            num_episodes=1,
            ppo_epochs=1,
            device="cpu",
        )
        before = {
            name: value.detach().clone()
            for name, value in trainer.policy.actor_mlp.state_dict().items()
        }

        metrics = trainer.train_episode(1)

        actor_change = sum(
            (trainer.policy.actor_mlp.state_dict()[name] - value).abs().sum().item()
            for name, value in before.items()
        )
        self.assertGreater(actor_change, 0.0)
        for key in ("pi_loss", "v_loss", "entropy", "approx_kl", "clip_fraction"):
            self.assertTrue(np.isfinite(metrics[key]), key)


if __name__ == "__main__":
    unittest.main()
