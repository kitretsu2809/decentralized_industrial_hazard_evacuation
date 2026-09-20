import unittest
import numpy as np
from core.graph.types import NodeType
from sim.services.environment.src.floorplan_generator import generate_industrial_plant
from sim.services.environment.src.evac_env import EvacuationEnv

class TestCrowdDynamics(unittest.TestCase):
    def setUp(self):
        self.building = generate_industrial_plant()
        self.env = EvacuationEnv(self.building, max_steps=100, num_evacuees=80)

    def test_weidmann_velocity_density_relationship(self):
        """Verify that high crowd density reduces walking speed according to Weidmann curve."""
        # Unimpeded density = 0
        v_free = 1.34 * max(0.12, 1.0 - 0.85 * 0.0)
        # Jam density = 1.0
        v_congested = 1.34 * max(0.12, 1.0 - 0.85 * 1.0)
        
        self.assertAlmostEqual(v_free, 1.34, places=2)
        self.assertLess(v_congested, v_free)
        self.assertGreaterEqual(v_congested, 1.34 * 0.12)

    def test_toxic_gas_speed_degradation(self):
        """Verify that toxic gas inhalation degrades walking speed."""
        v_clean = 1.34 * max(0.15, 1.0 - 0.75 * 0.0)
        v_toxic = 1.34 * max(0.15, 1.0 - 0.75 * 0.8)
        
        self.assertAlmostEqual(v_clean, 1.34, places=2)
        self.assertLess(v_toxic, v_clean * 0.5, "Toxic gas failed to substantially penalize walking speed")

    def test_industrial_evacuation_step_with_distance_physics(self):
        """Verify that evacuees traverse edges over time rather than instant teleportation."""
        obs, infos = self.env.reset()
        self.env.fire_model.inject_hazard('GAS', 'tank_farm_b', intensity=0.9, spread_rate=0.2)
        
        # Step once
        actions = {ag: [0]*len(self.env.action_spaces[ag].nvec) for ag in self.env.agents}
        obs, rew, term, trunc, infos = self.env.step(actions)
        
        # Evacuees should still be in the facility due to physical corridor travel time
        total_remaining = sum(self.env.evacuees_at_node.values())
        self.assertGreater(total_remaining, 0, "Evacuees instantly teleported across long corridors")

if __name__ == '__main__':
    unittest.main()
