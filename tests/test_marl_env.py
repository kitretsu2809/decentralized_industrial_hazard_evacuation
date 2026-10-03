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

    def test_hazard_model_laplacian_conservation(self):
        """Verifies that HazardModel discrete diffusion conserves mass in a closed network."""
        from simulator.engine.hazard_model import HazardModel, HazardType
        # 3-node linear chain without open exterior sinks
        adj = {
            "node_a": [("node_b", 10.0)],
            "node_b": [("node_a", 10.0), ("node_c", 10.0)],
            "node_c": [("node_b", 10.0)],
        }
        hm = HazardModel(adj, exits=[])
        hm.levels["node_a"] = 1.0
        hm.types["node_a"] = HazardType.GAS_RELEASE

        total_init = sum(hm.levels.values())
        # Disable decay for pure diffusion conservation test
        from simulator.engine.hazard_model import _PARAMS
        old_decay = _PARAMS[HazardType.GAS_RELEASE]["decay"]
        _PARAMS[HazardType.GAS_RELEASE]["decay"] = 0.0
        try:
            for _ in range(10):
                hm.step(dt=0.1)
            total_final = sum(hm.levels.values())
            self.assertAlmostEqual(total_init, total_final, places=4)
        finally:
            _PARAMS[HazardType.GAS_RELEASE]["decay"] = old_decay
        self.assertGreater(hm.levels["node_b"], 0.0)

    def test_pure_signboard_guidance(self):
        """Verifies pedestrian next waypoint is strictly determined by local signboard, not Dijkstra."""
        self.env.sim.set_policy_mode("marl")
        u = self.env.agents[0]
        nbrs = self.env.agent_neighbors[u]
        if len(nbrs) >= 2:
            # Set signboard u action: corridor 0 = BLOCK (2), corridor 1 = ALLOW (0)
            acts = np.zeros(6, dtype=np.int64)
            acts[0] = 2  # Block first corridor
            acts[1] = 0  # Allow second corridor
            self.env.sim.set_signboard_actions({u: acts})

            chosen = self.env.sim.get_next_waypoint(u, floor=1)
            # Must choose corridor 1 (nbrs[1]), NOT blocked corridor 0
            self.assertEqual(chosen, nbrs[1])

    def test_iso_13571_fed_accumulation(self):
        """Verifies cumulative FED dose accumulation and casualty trigger."""
        from simulator.engine.pedestrian import Pedestrian, PedestrianState
        ped = Pedestrian(id=999, x=10.0, y=10.0, current_node="test_node")
        self.env.sim.agents.append(ped)
        self.env.sim.hazard.levels["test_node"] = 0.50

        # Run 20 physics steps (2.0s): FED should increase by 0.50 * 2.0 / 30.0 = 0.0333
        for _ in range(20):
            self.env.sim._update_states()

        self.assertGreater(ped.fed, 0.03)
        self.assertNotEqual(ped.state, PedestrianState.CASUALTY)

        # Force FED >= 1.0
        ped.fed = 1.0
        self.env.sim._update_states()
        self.assertEqual(ped.state, PedestrianState.CASUALTY)


if __name__ == "__main__":
    unittest.main()
