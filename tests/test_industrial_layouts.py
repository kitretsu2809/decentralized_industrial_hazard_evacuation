import unittest
from collections import deque

from core.graph.directions import MAX_DIRECTIONS
from sim.services.environment.src.floorplan_generator import (
    BUILDING_LAYOUTS,
    generate_industrial_layout,
)
from simulator.engine.simulation import Simulation


class IndustrialLayoutTests(unittest.TestCase):
    def test_catalog_contains_three_distinct_industrial_layouts(self):
        self.assertEqual(
            set(BUILDING_LAYOUTS),
            {"petrochemical", "hexagonal_cell", "u_shaped_hall"},
        )
        node_sets = [set(generate_industrial_layout(key).all_nodes) for key in BUILDING_LAYOUTS]
        self.assertEqual(len({frozenset(nodes) for nodes in node_sets}), 3)

    def test_every_layout_has_redundant_reachable_egress_and_valid_mesh_degree(self):
        for layout_id in BUILDING_LAYOUTS:
            with self.subTest(layout=layout_id):
                building = generate_industrial_layout(layout_id)
                nodes = building.all_nodes
                exits = set(building.all_exits)
                self.assertGreaterEqual(len(exits), 2)

                adjacency = {node_id: set() for node_id in nodes}
                for edge in building.all_edges:
                    self.assertIn(edge.source, nodes)
                    self.assertIn(edge.target, nodes)
                    self.assertGreater(edge.distance, 0)
                    self.assertGreater(edge.width, 0)
                    adjacency[edge.source].add(edge.target)
                    adjacency[edge.target].add(edge.source)

                self.assertLessEqual(max(map(len, adjacency.values())), MAX_DIRECTIONS)
                reachable = set(exits)
                queue = deque(exits)
                while queue:
                    current = queue.popleft()
                    for neighbour in adjacency[current] - reachable:
                        reachable.add(neighbour)
                        queue.append(neighbour)
                self.assertEqual(reachable, set(nodes))

    def test_hexagonal_layout_has_six_spoke_central_cell(self):
        building = generate_industrial_layout("hexagonal_cell")
        neighbours = set()
        for edge in building.all_edges:
            if edge.source == "robotic_assembly_cell":
                neighbours.add(edge.target)
            elif edge.target == "robotic_assembly_cell":
                neighbours.add(edge.source)
        self.assertEqual(len(neighbours), 6)

    def test_simulation_can_switch_topology_without_stale_nodes(self):
        simulation = Simulation()
        old_nodes = set(simulation.node_positions)

        self.assertTrue(simulation.switch_layout("hexagonal_cell", num_evacuees=45))
        self.assertEqual(simulation.layout_id, "hexagonal_cell")
        self.assertEqual(simulation.num_evacuees, 45)
        self.assertIn("robotic_assembly_cell", simulation.node_positions)
        self.assertFalse(old_nodes & set(simulation.node_positions))
        self.assertEqual(set(simulation._mesh_inboxes), set(simulation.node_positions))
        self.assertEqual(simulation.building_dict()["layout_id"], "hexagonal_cell")

        self.assertTrue(simulation.switch_layout("u_shaped_hall"))
        self.assertIn("main_assembly_line", simulation.node_positions)
        self.assertFalse(simulation.switch_layout("unknown_layout"))
        self.assertEqual(simulation.layout_id, "u_shaped_hall")


if __name__ == "__main__":
    unittest.main()
