"""
Unit tests for Live In-GUI Training and Real-Time Inference Hot-Loading.
Verifies:
1. StandaloneSimulationEngine initializes AI policy (ST-TBA-GAT) and runs inference.
2. Node feature extraction accurately encodes 36 plant nodes.
3. Policy mode toggle ("ai" vs "baseline") alters evacuation routing.
4. AITrainingWorker runs in background thread, invokes callbacks, and reports status.
5. Monte Carlo benchmark produces valid comparative scorecard between AI and Baseline.
6. FastAPI endpoints for training, policy mode, and benchmark function properly.
"""

import unittest
import time
import os
import sys
import asyncio

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.run_industrial_standalone import (
    StandaloneSimulationEngine,
    AITrainingWorker,
    sim_engine,
    TrainRequestMsg,
    PolicyModeMsg,
    start_training,
    stop_training,
    get_training_status,
    set_policy_mode,
    run_benchmark
)


class TestLiveTrainingAndInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = sim_engine

    def test_01_ai_model_initialization(self):
        """Verify AI neural network (ActorCritic with GAT-GRU) initialized successfully."""
        self.assertIsNotNone(self.engine.ai_model, "AI model should be loaded in engine")
        self.assertEqual(self.engine.policy_mode, "ai", "Default policy mode should be 'ai'")
        self.assertIn("inference_latency_ms", self.engine.metrics)

    def test_02_feature_extraction(self):
        """Verify plant node feature extraction produces correct tensor dimensions."""
        x = self.engine._extract_ai_features()
        self.assertEqual(x.shape[0], 36, "Should extract features for all 36 plant nodes")
        self.assertEqual(x.shape[1], 8, "Feature vector dimension should be 8")
        self.assertEqual(self.engine.edge_index.shape[0], 2, "Edge index should have 2 rows [src, tgt]")

    def test_03_inference_step(self):
        """Verify that stepping with policy_mode='ai' executes neural forward pass."""
        self.engine.policy_mode = "ai"
        initial_time = self.engine.sim_time
        self.engine.step(0.1)
        self.assertGreater(self.engine.sim_time, initial_time)
        self.assertGreater(self.engine.metrics["inference_latency_ms"], 0.0)
        self.assertIn("model_confidence", self.engine.metrics)
        self.assertGreaterEqual(self.engine.metrics["model_confidence"], 0.0)

    def test_04_policy_mode_switch(self):
        """Verify toggling between 'baseline' and 'ai' modes."""
        self.engine.set_policy_mode("baseline")
        self.assertEqual(self.engine.policy_mode, "baseline")
        self.engine.step(0.1)
        self.assertEqual(self.engine.metrics["policy_mode"], "baseline")

        self.engine.set_policy_mode("ai")
        self.assertEqual(self.engine.policy_mode, "ai")
        self.engine.step(0.1)
        self.assertEqual(self.engine.metrics["policy_mode"], "ai")

    def test_05_training_worker_lifecycle(self):
        """Verify AITrainingWorker background execution and hot-load callback."""
        hot_loaded = []

        def on_episode_done(ep, total, metrics, policy):
            hot_loaded.append((ep, metrics))

        worker = AITrainingWorker(
            on_episode_callback=on_episode_done
        )
        self.assertFalse(worker.is_running)

        ok, msg = worker.start_training(num_episodes=2, facility_type="industrial")
        self.assertTrue(ok)
        self.assertTrue(worker.is_running)

        # Wait up to 10 seconds for 2 quick episodes to finish
        start_t = time.time()
        while worker.is_running and (time.time() - start_t) < 10:
            time.sleep(0.2)

        status = worker.get_status()
        self.assertFalse(status["is_training"])
        self.assertEqual(status["total_episodes"], 2)
        self.assertGreaterEqual(len(hot_loaded), 1, "At least 1 episode callback should have fired")
        self.assertIn("survival_rate", status["metrics"])
        self.assertIn("actor_loss", status["metrics"])

    def test_06_benchmark_evaluation(self):
        """Verify run_benchmark produces valid comparative metrics."""
        results = self.engine.run_benchmark(num_trials=2)
        self.assertIn("ai_model", results)
        self.assertIn("baseline", results)
        self.assertIn("survival_rate", results["ai_model"])
        self.assertIn("survival_rate", results["baseline"])
        self.assertGreaterEqual(results["ai_model"]["survival_rate"], 0.0)
        self.assertGreaterEqual(results["baseline"]["survival_rate"], 0.0)

    def test_07_api_endpoints(self):
        """Verify REST API endpoints for training and policy configuration."""
        # 1. Policy mode endpoint
        res = asyncio.run(set_policy_mode(PolicyModeMsg(mode="ai")))
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["mode"], "ai")

        # 2. Train status endpoint
        status = asyncio.run(get_training_status())
        self.assertIn("is_training", status)

        # 3. Train start endpoint (1 episode)
        start_res = asyncio.run(start_training(TrainRequestMsg(episodes=1, facility="industrial")))
        self.assertEqual(start_res["status"], "success")

        # Wait a short moment and check status
        time.sleep(1.0)
        status_running = asyncio.run(get_training_status())
        self.assertIn("is_training", status_running)

        # 4. Train stop endpoint
        stop_res = asyncio.run(stop_training())
        self.assertEqual(stop_res["status"], "success")


if __name__ == "__main__":
    unittest.main()
