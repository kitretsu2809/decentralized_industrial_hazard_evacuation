import unittest

import numpy as np

from scripts.generate_sensor_cnn_dataset import generate_dataset


class SensorCNNDatasetTests(unittest.TestCase):
    def test_generator_matches_runtime_cnn_layout_and_balances_labels(self):
        X, y, episode_ids = generate_dataset(episodes=20, window_size=50, seed=11)
        self.assertEqual(X.shape, (20, 3, 50))
        self.assertEqual(y.shape, (20,))
        self.assertEqual(episode_ids.shape, (20,))
        self.assertEqual(int(y.sum()), 10)
        self.assertTrue(np.isfinite(X).all())
        self.assertGreater(float(X[:, 0, :].max()), 50.0)

    def test_generator_is_reproducible(self):
        first = generate_dataset(episodes=8, window_size=20, seed=3)
        second = generate_dataset(episodes=8, window_size=20, seed=3)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)


if __name__ == "__main__":
    unittest.main()
