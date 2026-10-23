#!/usr/bin/env python3
"""Generate reproducible, domain-randomized telemetry for the sensor CNN.

The generated labels come from the simulator's injected hazard state, never
from the CNN or the threshold fallback.  This makes the file suitable for
training while keeping the simulator as the source of truth.

Example:
    python scripts/generate_sensor_cnn_dataset.py --episodes 2000 \
        --output data/sensor_cnn_synthetic.npz
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Allow the script to be launched as ``python scripts/...`` from the repo
# root, where Python otherwise places only ``scripts/`` on sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.hal_impl.sim_sensor import SimSensor


FEATURE_NAMES = ("gas_ppm", "temperature_c", "humidity_pct")


def _window(sensor: SimSensor, rng: np.random.Generator, episode: int,
            window_size: int, positive: bool) -> Tuple[np.ndarray, int]:
    """Create one temporal window, including realistic sensor imperfections."""
    node_id = f"dataset-node-{episode}"
    sensor.set_baseline(
        node_id,
        temperature=float(rng.uniform(16.0, 32.0)),
        humidity=float(rng.uniform(25.0, 75.0)),
        gas=float(rng.uniform(0.0, 12.0)),
    )

    onset = int(rng.integers(max(1, window_size // 3),
                             max(2, window_size * 2 // 3))) if positive else window_size
    peak = float(rng.uniform(0.25, 1.0)) if positive else 0.0
    duration = max(1, window_size - onset)
    readings = []
    previous = None

    for timestep in range(window_size):
        if positive and timestep >= onset:
            progress = (timestep - onset + 1) / duration
            hazard = min(peak, peak * progress)
            # Small fluctuations prevent the model from learning a perfectly
            # linear toy signal.
            hazard = max(0.0, min(1.0, hazard + float(rng.normal(0.0, 0.015))))
        else:
            hazard = 0.0

        sensor.inject_hazard(node_id, hazard)
        reading = np.asarray([
            sensor.read_gas(node_id),
            sensor.read_temperature(node_id),
            sensor.read_humidity(node_id),
        ], dtype=np.float32)

        # Domain randomization: occasional dropouts are held at the previous
        # value, matching what the runtime buffer can safely consume.
        if previous is not None and rng.random() < 0.025:
            reading = previous.copy()
        previous = reading
        readings.append(reading)

    sensor.clear_hazard(node_id)
    # CNN layout is channels x time, matching SensorCNN.forward().
    return np.stack(readings, axis=1), int(positive)


def generate_dataset(episodes: int = 2000, window_size: int = 50,
                     seed: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``X`` (N, 3, T), labels, and episode ids."""
    if episodes < 2:
        raise ValueError("episodes must be at least 2")
    if window_size < 8:
        raise ValueError("window_size must be at least 8")

    random.seed(seed)
    rng = np.random.default_rng(seed)
    sensor = SimSensor()
    windows, labels, episode_ids = [], [], []
    # Balanced classes make the first checkpoint useful; later training can
    # use the same generator with a production prevalence if desired.
    for episode in range(episodes):
        positive = episode % 2 == 1
        data, label = _window(sensor, rng, episode, window_size, positive)
        windows.append(data)
        labels.append(label)
        episode_ids.append(episode)

    return (np.asarray(windows, dtype=np.float32),
            np.asarray(labels, dtype=np.float32),
            np.asarray(episode_ids, dtype=np.int64))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=Path("data/sensor_cnn_synthetic.npz"))
    args = parser.parse_args()

    X, y, episode_ids = generate_dataset(args.episodes, args.window_size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, X=X, y=y, episode_ids=episode_ids,
                        feature_names=np.asarray(FEATURE_NAMES))
    metadata = {
        "source": "SimSensor with domain randomization",
        "seed": args.seed,
        "episodes": args.episodes,
        "window_size": args.window_size,
        "features": FEATURE_NAMES,
        "label_definition": "1 if a simulator hazard is injected in the window, else 0",
        "warning": "Synthetic data is pre-training; calibrate and validate with field telemetry.",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {args.output}: X={X.shape}, positives={int(y.sum())}, seed={args.seed}")


if __name__ == "__main__":
    main()
