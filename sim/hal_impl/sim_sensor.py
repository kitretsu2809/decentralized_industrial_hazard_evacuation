"""
Simulated sensor controller.

Generates synthetic sensor data (gas, temperature, humidity, presence)
correlated with the fire model state. In a real deployment, this would
read from MQ-2, DHT22, and IR ToF sensors via ADC/I2C.
"""
from __future__ import annotations

import logging
import random
from typing import Dict, Optional

from core.hal.base import SensorHAL

logger = logging.getLogger(__name__)


class SimSensor(SensorHAL):
    """Simulated environmental sensors producing synthetic data."""

    def __init__(self):
        # node_id -> {gas, temperature, humidity, presence}
        self._base_readings: Dict[str, Dict[str, float]] = {}
        # Injected hazard values affect sensor readings
        self._hazard_overrides: Dict[str, float] = {}

    def set_baseline(self, node_id: str, temperature: float = 22.0,
                     humidity: float = 45.0, gas: float = 0.0) -> None:
        """Set baseline sensor readings for a node (ambient conditions)."""
        self._base_readings[node_id] = {
            "temperature": temperature,
            "humidity": humidity,
            "gas": gas,
        }

    def inject_hazard(self, node_id: str, hazard_score: float) -> None:
        """Inject a hazard that affects sensor readings.

        Higher hazard scores simulate fire/smoke conditions:
        - Temperature rises proportionally
        - Gas levels increase
        - Humidity changes (fire dries air)

        Args:
            node_id: Node to affect.
            hazard_score: Hazard level (0.0 to 1.0).
        """
        self._hazard_overrides[node_id] = max(0.0, min(1.0, hazard_score))

    def clear_hazard(self, node_id: str) -> None:
        """Remove hazard override for a node."""
        self._hazard_overrides.pop(node_id, None)

    def read_gas(self, node_id: str) -> float:
        """Read simulated MQ-2 gas sensor (ppm).

        Normal: ~0-10 ppm. Fire: up to 500+ ppm.
        """
        base = self._base_readings.get(node_id, {}).get("gas", 0.0)
        hazard = self._hazard_overrides.get(node_id, 0.0)
        # Fire hazard dramatically increases gas readings
        gas_value = base + hazard * 500.0
        # Add small noise
        gas_value += random.gauss(0, 2.0)
        return max(0.0, gas_value)

    def read_temperature(self, node_id: str) -> float:
        """Read simulated temperature sensor (°C).

        Normal: ~20-25°C. Fire: up to 800°C.
        """
        base = self._base_readings.get(node_id, {}).get("temperature", 22.0)
        hazard = self._hazard_overrides.get(node_id, 0.0)
        # Exponential temperature rise with hazard
        temp = base + hazard * hazard * 780.0  # quadratic: peaks at ~800°C
        temp += random.gauss(0, 0.5)
        return max(-20.0, temp)

    def read_humidity(self, node_id: str) -> float:
        """Read simulated humidity sensor (%).

        Normal: ~40-60%. Fire reduces humidity.
        """
        base = self._base_readings.get(node_id, {}).get("humidity", 45.0)
        hazard = self._hazard_overrides.get(node_id, 0.0)
        # Fire dries the air
        humidity = base * (1.0 - hazard * 0.7)
        humidity += random.gauss(0, 1.0)
        return max(0.0, min(100.0, humidity))

    def read_presence(self, node_id: str) -> int:
        """Read simulated IR ToF presence count.

        Returns the number of people detected near the sensor.
        In simulation, this is set externally by the crowd sim.
        """
        # Default to 0 — crowd sim will override this
        return self._presence_counts.get(node_id, 0)

    def set_presence_count(self, node_id: str, count: int) -> None:
        """Set presence count (called by crowd simulation)."""
        if not hasattr(self, '_presence_counts'):
            self._presence_counts: Dict[str, int] = {}
        self._presence_counts[node_id] = max(0, count)
