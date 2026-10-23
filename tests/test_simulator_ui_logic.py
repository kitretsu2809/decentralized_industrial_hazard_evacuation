import re
import unittest
from pathlib import Path

from simulator.engine.simulation import Simulation


ROOT = Path(__file__).resolve().parents[1]


class SimulatorOperationalLogicTests(unittest.TestCase):
    def test_active_hazard_prevents_alarm_reset(self):
        sim = Simulation()
        target = next(node for node in sim.node_positions if node not in sim.exits)

        self.assertTrue(sim.inject_disaster(target, "FIRE", 0.8))
        self.assertTrue(sim.alarm_active)
        self.assertFalse(sim.reset_alarm())
        self.assertTrue(sim.alarm_active)

        sim.reset()
        self.assertFalse(sim.alarm_active)
        self.assertFalse(any(level >= 0.05 for level in sim.hazard.levels.values()))

    def test_unready_ai_policy_falls_back_instead_of_using_random_weights(self):
        sim = Simulation()
        sim.policy_model = object()
        sim.policy_ready = False

        self.assertFalse(sim.set_policy_mode("marl"))
        self.assertEqual(sim.policy_mode, "dijkstra")

    def test_state_exposes_policy_assurance_and_population_is_clamped(self):
        sim = Simulation()
        sim.reset(num_evacuees=900)
        self.assertEqual(sim.num_evacuees, 400)
        sim.reset(num_evacuees=-10)
        self.assertEqual(sim.num_evacuees, 5)

        policy = sim.state_dict()["policy"]
        self.assertEqual(policy["mode"], sim.policy_mode)
        self.assertIn("ready", policy)
        self.assertIn("checkpoint", policy)
        self.assertIn("load_error", policy)


class SimulatorUIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (ROOT / "simulator/static/index.html").read_text(encoding="utf-8")
        cls.javascript = (ROOT / "simulator/static/simulator.js").read_text(encoding="utf-8")
        cls.styles = (ROOT / "simulator/static/style.css").read_text(encoding="utf-8")
        cls.tour = (ROOT / "simulator/static/tour_controller.js").read_text(encoding="utf-8")

    def test_control_ids_are_unique_and_present(self):
        ids = re.findall(r'\bid="([^"]+)"', self.html)
        self.assertEqual(len(ids), len(set(ids)), "HTML control IDs must be unique")
        required = {
            "sim-canvas", "btn-alarm", "btn-play", "btn-pause", "btn-reset",
            "btn-inject", "sign-policy-select", "active-incidents", "event-log",
            "policy-assurance", "facility-status", "floor-tabs",
            "camera-node-select", "camera-video-input", "btn-camera-upload",
            "camera-preview", "vector-hazard", "vector-crowd",
            "building-layout-select",
        }
        self.assertTrue(required.issubset(ids))

    def test_production_console_has_truthful_routing_options(self):
        self.assertNotIn('value="random"', self.html)
        self.assertIn('value="static"', self.html)
        self.assertNotIn("randomExplorationCache", self.javascript)
        self.assertIn('send({action: "set_policy_mode", mode: e.target.value})', self.javascript)

    def test_guided_demo_is_offline_capable_and_switches_real_modes(self):
        self.assertNotIn("cdn.jsdelivr.net", self.tour)
        self.assertIn('window.setRoutingMode("dijkstra")', self.tour)
        self.assertIn('window.setRoutingMode("marl")', self.tour)

    def test_perception_console_exposes_real_model_and_vector_status(self):
        self.assertIn("Node camera perception", self.html)
        self.assertIn("/api/perception/feeds", self.javascript)
        self.assertIn("updatePerceptionPanel", self.javascript)
        self.assertIn("visual_confidence", self.javascript)
        self.assertIn("sensor_anomaly", self.javascript)
        self.assertIn('id="packet-hex"', self.html)
        self.assertIn('id="packet-bits"', self.html)
        self.assertIn('id="packet-directions"', self.html)

    def test_map_and_camera_node_selection_share_one_node_context(self):
        self.assertIn("function syncNodeSelectors(nodeId)", self.javascript)
        self.assertIn("syncNodeSelectors(signHit.u.id)", self.javascript)
        self.assertIn("syncNodeSelectors(n.id)", self.javascript)
        self.assertIn("const activeFeedId = (perception.node_feeds || {})[nodeId]", self.javascript)

    def test_workspace_is_split_into_persistent_operational_screens(self):
        for screen in ("operations", "scenario", "perception"):
            self.assertIn(f'data-screen-target="{screen}"', self.html)
        self.assertIn("setWorkspaceScreen", self.javascript)
        self.assertIn('data-screen-section="perception"', self.html)
        self.assertIn("Video-to-routing data flow", self.html)

    def test_facility_layout_selector_switches_real_backend_topology(self):
        for layout_id in ("petrochemical", "hexagonal_cell", "u_shaped_hall"):
            self.assertIn(f'value="{layout_id}"', self.html)
        self.assertIn('action: "switch_layout"', self.javascript)
        self.assertIn("populateLayoutSelect", self.javascript)

    def test_desktop_rails_scroll_without_expanding_the_viewport_grid(self):
        rail_rule = re.search(r"\.rail\s*\{(?P<body>.*?)\}", self.styles, re.DOTALL)
        self.assertIsNotNone(rail_rule)
        declarations = rail_rule.group("body")
        self.assertIn("min-height: 0", declarations)
        self.assertIn("height: 100%", declarations)
        self.assertIn("overflow-y: auto", declarations)
        self.assertIn("overflow-x: hidden", declarations)
        self.assertIn("scrollbar-gutter: stable", declarations)

    def test_industrial_hmi_theme_uses_neutral_surfaces_and_sparse_state_colour(self):
        self.assertIn("--bg: #202326", self.styles)
        self.assertIn("--surface: #292d30", self.styles)
        self.assertIn("--red: #d45b52", self.styles)
        header_rule = re.search(r"\.app-header\s*\{(?P<body>.*?)\}", self.styles, re.DOTALL)
        canvas_rule = re.search(r"\.canvas-wrap\s*\{(?P<body>.*?)\}", self.styles, re.DOTALL)
        self.assertIsNotNone(header_rule)
        self.assertIsNotNone(canvas_rule)
        self.assertNotIn("gradient", header_rule.group("body"))
        self.assertNotIn("gradient", canvas_rule.group("body"))
        self.assertIn("style.css?v=3.4", self.html)


if __name__ == "__main__":
    unittest.main()
