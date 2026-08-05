from __future__ import annotations
import math
from typing import List

def normalize_hazard(raw_value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a hazard value to the specified range."""
    return max(min_val, min(raw_value, max_val))

def compute_fire_hazard(temperature: float, smoke_density: float, co_ppm: float) -> float:
    """Compute multi-factor fire hazard based on temp, smoke, and CO."""
    # Temperature contribution: sigmoid centered around 60°C
    temp_h = 1.0 / (1.0 + math.exp(-0.1 * (temperature - 60.0)))
    
    # Smoke contribution: linear
    smoke_h = normalize_hazard(smoke_density, 0.0, 1.0)
    
    # CO contribution: linear approximation, 0.5 at 50, 1.0 at 200 => cap at 200
    if co_ppm <= 50:
        co_h = (co_ppm / 50.0) * 0.5
    else:
        co_h = 0.5 + ((co_ppm - 50.0) / 150.0) * 0.5
    co_h = normalize_hazard(co_h, 0.0, 1.0)

    max_h = max(temp_h, smoke_h, co_h)
    mean_h = (temp_h + smoke_h + co_h) / 3.0
    return normalize_hazard(max_h * 0.6 + mean_h * 0.4)

def compute_detection_hazard(threat_type_value: int, confidence: float, base_hazard: float) -> float:
    """Compute confidence-weighted hazard from a YOLO detection."""
    return normalize_hazard(confidence * base_hazard)

def combine_hazards(hazards: List[float]) -> float:
    """Combine multiple hazard sources for a single node."""
    if not hazards:
        return 0.0
    max_h = max(hazards)
    mean_h = sum(hazards) / len(hazards)
    return normalize_hazard(max_h * 0.7 + mean_h * 0.3)

def decay_hazard(current_h: float, decay_rate: float = 0.05, dt: float = 1.0) -> float:
    """Compute exponential decay when threat clears."""
    if current_h <= 0.0:
        return 0.0
    decayed = current_h * math.exp(-decay_rate * dt)
    return max(0.0, decayed) if decayed > 1e-4 else 0.0

def compute_edge_cost(distance: float, source_hazard: float, target_hazard: float, congestion_ratio: float) -> float:
    """Compute hazard-aware edge cost for pathfinding."""
    max_hazard = max(source_hazard, target_hazard)
    return distance * (1.0 + 10.0 * max_hazard) * (1.0 + 5.0 * congestion_ratio)

def is_impassable(hazard: float, threshold: float = 0.95) -> bool:
    """Check if node is impassable based on a hazard threshold."""
    return hazard >= threshold
