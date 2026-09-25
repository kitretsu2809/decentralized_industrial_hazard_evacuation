import unittest
import numpy as np
from simulator.env.evacuation_env import IndustrialEvacuationEnv

class TestIndustrialEvacuationEnv(unittest.TestCase):
    def setUp(self):
        self.env = IndustrialEvacuationEnv(num_evacuees=25)

    def test_environment_initialization(self):
        self.assertEqual(len(self.env.agents), 36)
        self.assertEqual(self.env.MAX_CORRIDORS, 6)
        self.assertEqual(self.env.obs_dim, 34)

        for agent in self.env.agents:
            obs_space = self.env.observation_spaces[agent]
            act_space = self.env.action_spaces[agent]
            self.assertEqual(obs_space.shape, (34,))
            self.assertEqual(len(act_space.nvec), 6)

    def test_reset(self):
        obs, infos = self.env.reset(seed=123)
        self.assertEqual(len(obs), 36)
        self.assertEqual(len(infos), 36)

        for agent in self.env.agents:
            self.assertEqual(obs[agent].shape, (34,))
            self.assertIn("action_mask", infos[agent])
            mask = infos[agent]["action_mask"]
            self.assertEqual(len(mask), 6)

    def test_step_execution(self):
        obs, infos = self.env.reset(seed=456)
        actions = {
            agent: np.random.randint(0, 3, size=self.env.MAX_CORRIDORS)
            for agent in self.env.agents
        }
        next_obs, rewards, terminations, truncations, next_infos = self.env.step(actions)

        self.assertEqual(len(next_obs), 36)
        self.assertEqual(len(rewards), 36)
        self.assertEqual(len(terminations), 36)
        self.assertEqual(len(truncations), 36)

        # Team reward should be consistent across all router agents
        first_reward = rewards[self.env.agents[0]]
        for agent in self.env.agents:
            self.assertAlmostEqual(rewards[agent], first_reward, places=5)

    def test_action_masking(self):
        all_masks = self.env.get_all_action_masks()
        self.assertEqual(all_masks.shape, (36, 6))
        # Ensure every agent has at least 1 valid incident corridor
        self.assertTrue(np.all(all_masks.sum(axis=1) >= 1))

if __name__ == "__main__":
    unittest.main()
