import time
import unittest
from unittest.mock import patch

from sim.services.perception.src.sensor_cnn import SensorAnomalyDetector
from simulator.engine.simulation import Simulation
from simulator.perception.video_pipeline import SimulatorPerceptionPipeline, build_hazard_vector


class PerceptionPipelineTests(unittest.TestCase):
    def test_colour_only_fire_fallback_requires_explicit_opt_in(self):
        class StubBackend:
            backend_status = {"name": "stub", "available": True}

        with patch.dict("os.environ", {"ENABLE_HSV_FIRE_FALLBACK": "false"}):
            pipeline = SimulatorPerceptionPipeline(
                on_update=lambda update: None,
                sensor_provider=lambda node_id: (8.0, 24.0, 52.0),
                detector=StubBackend(),
                sensor_detector=StubBackend(),
            )

        status = pipeline.capabilities()["fire_fallback"]
        self.assertFalse(status["available"])
        self.assertFalse(status["configured"])
        self.assertTrue(status["disclosed_fallback"])

    def test_hazard_vector_is_bounded_and_preserves_crowd_range(self):
        vector = build_hazard_vector(
            visual_hazard=1.4,
            crowd_density=1.6,
            visual_confidence=-0.2,
            sensor_anomaly=0.5,
        )
        self.assertGreaterEqual(vector["hazard"], 0.0)
        self.assertLessEqual(vector["hazard"], 1.0)
        self.assertEqual(vector["crowd_density"], 1.6)
        self.assertEqual(vector["visual_confidence"], 0.0)

    def test_sensor_cnn_never_uses_random_untrained_weights(self):
        detector = SensorAnomalyDetector(use_nn=True, checkpoint_path=None)
        self.assertFalse(detector.use_nn)
        self.assertTrue(
            "No trained" in detector.backend_status["load_error"]
            or "PyTorch is not installed" in detector.backend_status["load_error"]
        )

        detector.add_reading("node-a", 10.0, 24.0, 50.0)
        detector.add_reading("node-a", 180.0, 70.0, 45.0)
        is_anomaly, severity = detector.detect_anomaly("node-a")
        self.assertTrue(is_anomaly)
        self.assertGreater(severity, 0.0)

    def test_perception_update_enters_simulation_hazard_state(self):
        sim = Simulation()
        node_id = next(node for node in sim.node_positions if node not in sim.exits)
        update = {
            "node_id": node_id,
            "hazard_type": "FIRE",
            "person_count": 7,
            "vector": {
                "hazard": 0.65,
                "crowd_density": 1.2,
                "visual_confidence": 0.9,
                "sensor_anomaly": 0.4,
            },
            "updated_at": time.time(),
        }

        self.assertTrue(sim.apply_perception_update(update))
        packet = sim.perception_state()["nodes"][node_id]["communication"]
        self.assertEqual(packet["size_bits"], 32)
        self.assertEqual(len(bytes.fromhex(packet["hex"])), 4)
        self.assertEqual(len(packet["direction_mask"]), 8)
        for neighbor in sim.agent_neighbors[node_id]:
            received = sim.perception_state()["mesh"]["inboxes"][neighbor][node_id]
            self.assertEqual(received["hex"], packet["hex"])
        sim._drain_perception_updates()
        self.assertAlmostEqual(sim.hazard.levels[node_id], 0.65)
        self.assertTrue(sim.alarm_active)
        state = sim.state_dict()["perception"]["nodes"][node_id]
        self.assertEqual(state["person_count"], 7)
        self.assertEqual(state["vector"]["crowd_density"], 1.2)


if __name__ == "__main__":
    unittest.main()
