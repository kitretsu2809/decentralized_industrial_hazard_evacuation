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


if __name__ == "__main__":
    unittest.main()
