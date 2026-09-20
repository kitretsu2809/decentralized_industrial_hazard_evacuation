import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.services.environment.src.floorplan_generator import generate_default_building
from core.graph.building_graph import BuildingGraph
from core.hazard.cross_floor_propagation import PropagationConfig, compute_vertical_propagation, compute_floor_hazard_map
from core.hazard.threat_types import ThreatType

def test_hazard_propagation():
    print("Testing vertical hazard propagation...")
    
    config = PropagationConfig()
    
    # Fire on floor 1, intensity 0.8
    # Fire propagates UP strongly (smoke_rise_factor = 3.0), DOWN weakly.
    
    source_floor = 1
    hazard = 0.8
    
    f2_hazard = compute_vertical_propagation(hazard, source_floor, target_floor=2, threat_type_value=ThreatType.FIRE.value, config=config)
    f3_hazard = compute_vertical_propagation(hazard, source_floor, target_floor=3, threat_type_value=ThreatType.FIRE.value, config=config)
    
    print(f"Floor 1 Fire (0.8) -> F2 Hazard: {f2_hazard:.3f}, F3 Hazard: {f3_hazard:.3f}")
    assert f2_hazard > 0, "Hazard should propagate to floor 2"
    assert f2_hazard > f3_hazard, "Hazard should decay over distance"
    
    # Test floor hazard map
    building_floors = [1, 2, 3]
    hazard_sources = [
        ('lobby', 1, 0.9, ThreatType.FIRE.value)
    ]
    
    floor_hazards = compute_floor_hazard_map(hazard_sources, building_floors, config)
    print("Floor Hazards Map:", floor_hazards)
    
    assert floor_hazards[1] == 0.9
    assert floor_hazards[2] > 0
    assert floor_hazards[3] > 0
    assert floor_hazards[2] > floor_hazards[3]
    
    print("Hazard propagation test PASSED!")

if __name__ == '__main__':
    test_hazard_propagation()
