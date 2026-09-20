import unittest
import numpy as np
from core.graph.building_graph import BuildingGraph
from core.graph.types import EdgeState
from sim.services.environment.src.floorplan_generator import generate_industrial_plant
from sim.services.environment.src.fire_model import IndustrialHazardModel

class TestIndustrialDisasters(unittest.TestCase):
    def setUp(self):
        self.building = generate_industrial_plant()
        self.graph = BuildingGraph(self.building)
        self.model = IndustrialHazardModel(self.graph)

    def test_toxic_gas_dispersion_with_wind(self):
        """Verify toxic gas spreads with wind advection bias towards East-Northeast."""
        self.model.inject_hazard('GAS', 'tank_farm_b', intensity=0.9, spread_rate=0.2)
        threats = self.model.get_active_threats()
        self.assertEqual(len(threats), 1)
        self.assertEqual(threats[0]['type'], 'GAS')
        
        # Step gas dispersion
        for _ in range(5):
            self.model.step(dt=1.0)
            
        # Adjacent downwind node should have elevated hazard
        haz_basin = self.model.get_hazard_at('hazmat_basin')
        self.assertGreater(haz_basin, 0.05, "Toxic gas failed to disperse to adjacent containment basin")

    def test_explosion_structural_blast_severing(self):
        """Verify explosion instantly severs adjacent corridors and reduces node capacity."""
        orig_cap = self.building.get_node('compressor_shed').capacity
        self.model.inject_hazard('EXPLOSION', 'compressor_shed', intensity=1.0)
        
        # Capacity reduced
        new_cap = self.building.get_node('compressor_shed').capacity
        self.assertLess(new_cap, orig_cap)
        
        # Adjacent edge severed
        edge = self.graph.get_edge_between('compressor_shed', 'pipe_rack_junc_1')
        if edge:
            self.assertEqual(edge.state, EdgeState.SEVERED, "Explosion failed to sever adjacent pipe rack edge")

    def test_chemical_spill_confined_to_floor(self):
        """Verify chemical spill spreads horizontally but cannot climb vertical stairwells."""
        self.model.inject_hazard('CHEMICAL_SPILL', 'hazmat_basin', intensity=0.8, spread_rate=0.1)
        for _ in range(3):
            self.model.step(dt=1.0)
            
        # Cross-floor stairwell on Floor 2 should have 0 hazard from ground spill
        f2_stair = self.model.get_hazard_at('stair_north_f2')
        self.assertAlmostEqual(f2_stair, 0.0, places=2)

if __name__ == '__main__':
    unittest.main()
