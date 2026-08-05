from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

@dataclass
class PropagationConfig:
    vertical_decay_rate: float = 0.3
    horizontal_decay_rate: float = 0.1
    smoke_rise_factor: float = 3.0
    heat_rise_factor: float = 2.0
    structural_spread_factor: float = 1.5

def compute_vertical_propagation(source_hazard: float, source_floor: int, target_floor: int, threat_type_value: int, config: Optional[PropagationConfig] = None) -> float:
    """Compute hazard propagation vertically across floors."""
    if config is None:
        config = PropagationConfig()
        
    floor_diff = target_floor - source_floor
    if floor_diff == 0:
        return source_hazard
        
    distance = abs(floor_diff)
    
    # 1: FIRE, 2: SMOKE, 3: STRUCTURAL_COLLAPSE
    if threat_type_value in (1, 2):  # Fire / Smoke
        rise_factor = config.smoke_rise_factor if threat_type_value == 2 else config.heat_rise_factor
        if floor_diff > 0:
            # Propagating upwards: less decay
            decay = config.vertical_decay_rate / rise_factor
        else:
            # Propagating downwards: standard decay
            decay = config.vertical_decay_rate
    elif threat_type_value == 3:  # Structural Collapse
        if floor_diff > 0:
            decay = config.vertical_decay_rate / config.structural_spread_factor
        else:
            decay = config.vertical_decay_rate / (config.structural_spread_factor * 0.8)
    else:
        decay = config.vertical_decay_rate
        
    propagated = source_hazard * math.exp(-decay * distance)
    return max(0.0, min(1.0, propagated))

def compute_floor_hazard_map(hazard_sources: List[Tuple[str, int, float, int]], building_floors: List[int], config: Optional[PropagationConfig] = None) -> Dict[int, float]:
    """
    Given a list of (node_id, floor, hazard, threat_type_value),
    compute the max propagated hazard reaching each floor.
    """
    if config is None:
        config = PropagationConfig()
        
    floor_hazards = {floor: 0.0 for floor in building_floors}
    
    for _, source_floor, source_hazard, threat_type_val in hazard_sources:
        for target_floor in building_floors:
            prop_hazard = compute_vertical_propagation(source_hazard, source_floor, target_floor, threat_type_val, config)
            if prop_hazard > floor_hazards[target_floor]:
                floor_hazards[target_floor] = prop_hazard
                
    return floor_hazards

def should_disable_elevators(floor_hazards: Dict[int, float], threshold: float = 0.3) -> bool:
    """Determine if elevators should be disabled based on floor hazards."""
    for hazard in floor_hazards.values():
        if hazard >= threshold:
            return True
    return False

def get_safest_vertical_route(current_floor: int, floor_hazards: Dict[int, float], exit_floors: List[int]) -> Optional[int]:
    """Find the safest exit floor to evacuate to."""
    best_exit = None
    min_cost = float('inf')
    
    for exit_floor in exit_floors:
        step = 1 if exit_floor > current_floor else -1
        path_floors = range(current_floor, exit_floor + step, step)
        
        path_hazard_sum = sum(floor_hazards.get(f, 0.0) for f in path_floors)
        distance = abs(exit_floor - current_floor)
        
        cost = path_hazard_sum * 10.0 + distance
        
        if cost < min_cost:
            min_cost = cost
            best_exit = exit_floor
            
    return best_exit
