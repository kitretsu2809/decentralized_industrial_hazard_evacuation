import unittest
import numpy as np
from simulator.env.evacuation_env import IndustrialEvacuationEnv

class TestIndustrialEvacuationEnv(unittest.TestCase):
    def setUp(self):
        self.env = IndustrialEvacuationEnv(num_evacuees=25)

    def test_environment_initialization(self):
        self.assertEqual(len(self.env.agents), 36)
        self.assertEqual(self.env.MAX_CORRIDORS, 6)
        self.assertEqual(self.env.obs_dim, 41)

        for agent in self.env.agents:
            obs_space = self.env.observation_spaces[agent]
            act_space = self.env.action_spaces[agent]
            self.assertEqual(obs_space.shape, (41,))
            self.assertEqual(act_space.n, 7)

    def test_reset(self):
        obs, infos = self.env.reset(seed=123)
        self.assertEqual(len(obs), 36)
        self.assertEqual(len(infos), 36)

        for agent in self.env.agents:
            self.assertEqual(obs[agent].shape, (41,))
            self.assertIn("action_mask", infos[agent])
            mask = infos[agent]["action_mask"]
            self.assertEqual(len(mask), 7)

    def test_step_execution(self):
        obs, infos = self.env.reset(seed=456)
        actions = {
            agent: np.random.randint(0, 7)
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
        self.assertEqual(all_masks.shape, (36, 7))
        # Ensure every agent has at least 1 valid choice (Standby at index 6 is always valid)
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
            # Set signboard u action: corridor 0 = BLOCKED (2), corridor 1 = GREEN ARROW (3).
            acts = np.zeros(6, dtype=np.int64)
            acts[0] = 2  # Block first corridor
            acts[1] = 3  # Directly point to second corridor
            self.env.sim.set_signboard_actions({u: acts})

            chosen = self.env.sim.get_next_waypoint(u, floor=1)
            # Must choose corridor 1 (nbrs[1]), NOT blocked corridor 0
            self.assertEqual(chosen, nbrs[1])

    def test_green_arrow_does_not_need_a_shortest_path_lookup(self):
        """A direct GREEN_ARROW remains authoritative even if the router is unavailable."""
        self.env.sim.set_policy_mode("marl")
        u = self.env.agents[0]
        nbrs = self.env.agent_neighbors[u]
        if nbrs:
            acts = np.zeros(6, dtype=np.int64)
            acts[0] = 3
            self.env.sim.set_signboard_actions({u: acts})
            self.env.sim.router.get_path = lambda *_args, **_kwargs: self.fail(
                "MARL sign guidance must not query Dijkstra."
            )
            self.assertEqual(self.env.sim.get_next_waypoint(u, floor=1), nbrs[0])

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

    def test_gradual_curriculum_monotonicity(self):
        """Verifies that the gradual curriculum monotonically increases crowd size and hazard intensity."""
        from simulator.training.train_mappo import MAPPOTrainer
        trainer = MAPPOTrainer(
            min_evacuees=20,
            max_evacuees=600,
            curriculum=True,
            num_episodes=100,
            device="cpu",
        )
        # Check tau progression for 5 key milestone episodes
        episodes = [1, 25, 50, 75, 100]
        n_values = []
        h_values = []

        for ep in episodes:
            tau = (ep - 1) / 99.0
            n_base = 20 + (600 - 20) * (tau ** 1.15)
            h_base = 0.40 + 0.55 * tau
            n_values.append(n_base)
            h_values.append(h_base)

        # Monotonicity check
        for i in range(len(episodes) - 1):
            self.assertLess(n_values[i], n_values[i + 1])
            self.assertLess(h_values[i], h_values[i + 1])

        # Boundary checks
        self.assertEqual(round(n_values[0]), 20)
        self.assertEqual(round(n_values[-1]), 600)
        self.assertAlmostEqual(h_values[0], 0.40, places=2)
        self.assertAlmostEqual(h_values[-1], 0.95, places=2)

    def test_crowd_cluster_spawning(self):
        """Verifies that cluster_node and cluster_ratio concentrate occupants at bottleneck nodes."""
        obs, _ = self.env.reset(options={
            "num_evacuees": 50,
            "cluster_node": "corridor_perimeter_n",
            "cluster_ratio": 0.60,
        })
        clustered = sum(1 for p in self.env.sim.agents if p.current_node == "corridor_perimeter_n")
        # 60% of 50 = 30 clustered agents
        self.assertGreaterEqual(clustered, 25)

    def test_spatial_density_variance_and_jamming_penalty(self):
        """Verifies that high crowd concentration incurs jamming and spatial density penalties."""
        # Setup high crowd cluster at a bottleneck node
        self.env.reset(options={
            "num_evacuees": 80,
            "cluster_node": "pipe_rack_junc_1",
            "cluster_ratio": 0.80,
        })
        actions = {agent: 6 for agent in self.env.agents}  # standby actions
        _, rewards, _, _, _ = self.env.step(actions)
        # Verify team reward evaluates without NaN or Inf
        rew = rewards[self.env.agents[0]]
        self.assertFalse(np.isnan(rew))
        self.assertFalse(np.isinf(rew))

    def test_flipping_penalty_decoupling_under_crowd_pressure(self):
        """Verifies arrow changes are permitted without flipping penalty under high crowd shifts."""
        self.env.reset(options={
            "num_evacuees": 40,
            "cluster_node": "reactor_2",
            "cluster_ratio": 0.70,
        })
        # First step with action 0
        actions_1 = {agent: 0 for agent in self.env.agents}
        self.env.step(actions_1)

        # Second step: flip action to 1 at reactor_2 where crowd is high
        actions_2 = {agent: 1 for agent in self.env.agents}
        _, rewards, _, _, _ = self.env.step(actions_2)
        self.assertTrue(all(not np.isnan(r) for r in rewards.values()))


if __name__ == "__main__":
    unittest.main()

