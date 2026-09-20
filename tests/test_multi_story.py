import sys
import os

# Add LBP to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.services.environment.src.floorplan_generator import generate_default_building
from core.graph.building_graph import BuildingGraph
from core.graph.multi_floor import MultiFloorManager

def test_multi_story_pathing():
    print("Testing multi-story building pathfinding...")
    b = generate_default_building()
    bg = BuildingGraph(b)
    mfm = MultiFloorManager(bg)

    # Validate cross-floor connections exist
    assert len(b.cross_floor_edges) > 0, "No cross-floor edges found!"
    print(f"Found {len(b.cross_floor_edges)} cross-floor edges.")
    
    # Floor 3 node (e.g. office_3_1) -> try to find path to exits (main_exit on F1)
    office_f3 = 'office_3_1'
    
    # Sever edges to F3 exits to force routing downstairs
    # Find edges connecting to 'fire_escape_3' and 'roof_access'
    for edge in b.all_edges:
        if edge.target in ['fire_escape_3', 'roof_access'] or edge.source in ['fire_escape_3', 'roof_access']:
            bg.sever_edge(edge.id)
    
    paths = bg.get_all_exit_paths(office_f3)
    
    assert len(paths) > 0, "No paths to exit found from Floor 3!"
    best_path, cost = paths[0]
    print(f"Best path from Floor 3 to Exit: {best_path} (Cost: {cost})")
    
    # Inject fire hazard in stairwell_a_f2 to block the A stairs
    print("Injecting hazard at stairwell_a_f2...")
    bg.update_hazard('stairwell_a_f2', 1.0)
    
    paths_after = bg.get_all_exit_paths(office_f3)
    best_path_after, cost_after = paths_after[0]
    print(f"Best path after hazard: {best_path_after} (Cost: {cost_after})")
    
    assert 'stairwell_a_f2' not in best_path_after, "Path should avoid hazardous stairwell!"
    assert 'stairwell_b_f2' in best_path_after or 'elevator_f2' in best_path_after, "Path must route via alternative vertical connection"
    print("Multi-story pathfinding test PASSED!")

if __name__ == '__main__':
    test_multi_story_pathing()
