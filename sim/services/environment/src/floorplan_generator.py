import yaml
from typing import Dict, List, Any
from core.graph.types import Building, Floor, Node, Edge, NodeType, EdgeState, DoorType

def generate_default_building(name: str = 'LBP Office Building') -> Building:
    b = Building(name=name)

    # Floor 1
    f1 = Floor(level=1)
    f1.nodes['lobby'] = Node(id='lobby', type=NodeType.INTERSECTION, floor=1, capacity=50, position=(100.0, 50.0))
    f1.nodes['reception'] = Node(id='reception', type=NodeType.ROOM, floor=1, capacity=10, position=(100.0, 40.0))
    for i in range(1, 5):
        f1.nodes[f'office_1_{i}'] = Node(id=f'office_1_{i}', type=NodeType.ROOM, floor=1, capacity=6, position=(40.0 * i, 20.0))
    f1.nodes['corridor_main_1'] = Node(id='corridor_main_1', type=NodeType.CORRIDOR, floor=1, capacity=30, position=(100.0, 30.0))
    f1.nodes['corridor_sec_1'] = Node(id='corridor_sec_1', type=NodeType.CORRIDOR, floor=1, capacity=20, position=(100.0, 20.0))
    f1.nodes['stairwell_a_f1'] = Node(id='stairwell_a_f1', type=NodeType.STAIRWELL, floor=1, capacity=8, position=(20.0, 80.0), is_cross_floor=True)
    f1.nodes['stairwell_b_f1'] = Node(id='stairwell_b_f1', type=NodeType.STAIRWELL, floor=1, capacity=8, position=(180.0, 80.0), is_cross_floor=True)
    f1.nodes['elevator_f1'] = Node(id='elevator_f1', type=NodeType.ELEVATOR, floor=1, capacity=6, position=(100.0, 80.0), is_cross_floor=True)
    f1.nodes['main_exit'] = Node(id='main_exit', type=NodeType.EXIT, floor=1, capacity=100, position=(100.0, 100.0))
    f1.nodes['fire_escape_south'] = Node(id='fire_escape_south', type=NodeType.EXIT, floor=1, capacity=30, position=(100.0, 0.0))
    f1.exits.extend(['main_exit', 'fire_escape_south'])
    
    # Internal edges for F1
    edges_f1 = [
        ('lobby', 'main_exit', 10.0, 3.0, 15),
        ('lobby', 'reception', 5.0, 2.0, 10),
        ('lobby', 'corridor_main_1', 10.0, 3.0, 15),
        ('corridor_main_1', 'corridor_sec_1', 10.0, 2.5, 12),
        ('corridor_main_1', 'elevator_f1', 15.0, 2.0, 8),
        ('corridor_sec_1', 'fire_escape_south', 15.0, 2.0, 10),
        ('corridor_main_1', 'stairwell_a_f1', 20.0, 1.5, 5),
        ('corridor_main_1', 'stairwell_b_f1', 20.0, 1.5, 5)
    ]
    for i in range(1, 5):
        edges_f1.append((f'office_1_{i}', 'corridor_sec_1', 5.0, 1.5, 5))
        
    for idx, (src, tgt, dist, width, tp) in enumerate(edges_f1):
        f1.edges.append(Edge(id=f'e_f1_{idx}', source=src, target=tgt, distance=dist, width=width, max_throughput=tp))

    b.floors[1] = f1

    # Floor 2
    f2 = Floor(level=2)
    for i in range(1, 9):
        f2.nodes[f'office_2_{i}'] = Node(id=f'office_2_{i}', type=NodeType.ROOM, floor=2, capacity=6, position=(20.0 * i, 20.0))
    f2.nodes['conf_room_1'] = Node(id='conf_room_1', type=NodeType.ROOM, floor=2, capacity=20, position=(50.0, 50.0))
    f2.nodes['conf_room_2'] = Node(id='conf_room_2', type=NodeType.ROOM, floor=2, capacity=20, position=(150.0, 50.0))
    f2.nodes['kitchen_2'] = Node(id='kitchen_2', type=NodeType.ROOM, floor=2, capacity=8, position=(100.0, 40.0))
    f2.nodes['corridor_main_2'] = Node(id='corridor_main_2', type=NodeType.CORRIDOR, floor=2, capacity=30, position=(100.0, 30.0))
    f2.nodes['corridor_east_2'] = Node(id='corridor_east_2', type=NodeType.CORRIDOR, floor=2, capacity=20, position=(150.0, 30.0))
    f2.nodes['corridor_west_2'] = Node(id='corridor_west_2', type=NodeType.CORRIDOR, floor=2, capacity=20, position=(50.0, 30.0))
    f2.nodes['intersection_2'] = Node(id='intersection_2', type=NodeType.INTERSECTION, floor=2, capacity=15, position=(100.0, 30.0))
    f2.nodes['stairwell_a_f2'] = Node(id='stairwell_a_f2', type=NodeType.STAIRWELL, floor=2, capacity=8, position=(20.0, 80.0), is_cross_floor=True)
    f2.nodes['stairwell_b_f2'] = Node(id='stairwell_b_f2', type=NodeType.STAIRWELL, floor=2, capacity=8, position=(180.0, 80.0), is_cross_floor=True)
    f2.nodes['elevator_f2'] = Node(id='elevator_f2', type=NodeType.ELEVATOR, floor=2, capacity=6, position=(100.0, 80.0), is_cross_floor=True)
    f2.nodes['fire_escape_2'] = Node(id='fire_escape_2', type=NodeType.EXIT, floor=2, capacity=30, position=(100.0, 0.0))
    f2.exits.append('fire_escape_2')

    edges_f2 = [
        ('intersection_2', 'corridor_main_2', 5.0, 3.0, 15),
        ('intersection_2', 'corridor_east_2', 10.0, 2.5, 12),
        ('intersection_2', 'corridor_west_2', 10.0, 2.5, 12),
        ('corridor_main_2', 'kitchen_2', 5.0, 1.5, 5),
        ('corridor_east_2', 'conf_room_2', 10.0, 2.0, 8),
        ('corridor_west_2', 'conf_room_1', 10.0, 2.0, 8),
        ('corridor_main_2', 'elevator_f2', 15.0, 2.0, 8),
        ('corridor_main_2', 'fire_escape_2', 15.0, 2.0, 10),
        ('corridor_west_2', 'stairwell_a_f2', 15.0, 1.5, 5),
        ('corridor_east_2', 'stairwell_b_f2', 15.0, 1.5, 5)
    ]
    for i in range(1, 9):
        corr = 'corridor_west_2' if i <= 4 else 'corridor_east_2'
        edges_f2.append((f'office_2_{i}', corr, 5.0, 1.5, 5))
        
    for idx, (src, tgt, dist, width, tp) in enumerate(edges_f2):
        f2.edges.append(Edge(id=f'e_f2_{idx}', source=src, target=tgt, distance=dist, width=width, max_throughput=tp))
        
    b.floors[2] = f2

    # Floor 3
    f3 = Floor(level=3)
    for i in range(1, 7):
        f3.nodes[f'office_3_{i}'] = Node(id=f'office_3_{i}', type=NodeType.ROOM, floor=3, capacity=6, position=(30.0 * i, 20.0))
    f3.nodes['server_room'] = Node(id='server_room', type=NodeType.ROOM, floor=3, capacity=4, position=(20.0, 50.0))
    f3.nodes['auditorium'] = Node(id='auditorium', type=NodeType.ROOM, floor=3, capacity=100, position=(150.0, 50.0))
    f3.nodes['corridor_main_3'] = Node(id='corridor_main_3', type=NodeType.CORRIDOR, floor=3, capacity=25, position=(100.0, 30.0))
    f3.nodes['corridor_east_3'] = Node(id='corridor_east_3', type=NodeType.CORRIDOR, floor=3, capacity=15, position=(150.0, 30.0))
    f3.nodes['intersection_3'] = Node(id='intersection_3', type=NodeType.INTERSECTION, floor=3, capacity=15, position=(100.0, 30.0))
    f3.nodes['stairwell_a_f3'] = Node(id='stairwell_a_f3', type=NodeType.STAIRWELL, floor=3, capacity=8, position=(20.0, 80.0), is_cross_floor=True)
    f3.nodes['stairwell_b_f3'] = Node(id='stairwell_b_f3', type=NodeType.STAIRWELL, floor=3, capacity=8, position=(180.0, 80.0), is_cross_floor=True)
    f3.nodes['elevator_f3'] = Node(id='elevator_f3', type=NodeType.ELEVATOR, floor=3, capacity=6, position=(100.0, 80.0), is_cross_floor=True)
    f3.nodes['fire_escape_3'] = Node(id='fire_escape_3', type=NodeType.EXIT, floor=3, capacity=30, position=(100.0, 0.0))
    f3.nodes['roof_access'] = Node(id='roof_access', type=NodeType.EXIT, floor=3, capacity=15, position=(100.0, 100.0))
    f3.exits.extend(['fire_escape_3', 'roof_access'])

    edges_f3 = [
        ('intersection_3', 'corridor_main_3', 5.0, 3.0, 15),
        ('intersection_3', 'corridor_east_3', 10.0, 2.5, 12),
        ('corridor_main_3', 'server_room', 15.0, 1.5, 5),
        ('corridor_east_3', 'auditorium', 5.0, 3.0, 15),
        ('corridor_main_3', 'elevator_f3', 15.0, 2.0, 8),
        ('corridor_main_3', 'roof_access', 20.0, 2.0, 10),
        ('corridor_main_3', 'fire_escape_3', 15.0, 2.0, 10),
        ('corridor_main_3', 'stairwell_a_f3', 15.0, 1.5, 5),
        ('corridor_east_3', 'stairwell_b_f3', 15.0, 1.5, 5)
    ]
    for i in range(1, 7):
        corr = 'corridor_main_3' if i <= 3 else 'corridor_east_3'
        edges_f3.append((f'office_3_{i}', corr, 5.0, 1.5, 5))
        
    for idx, (src, tgt, dist, width, tp) in enumerate(edges_f3):
        f3.edges.append(Edge(id=f'e_f3_{idx}', source=src, target=tgt, distance=dist, width=width, max_throughput=tp))
        
    b.floors[3] = f3

    # Cross-floor edges
    cross_edges = [
        ('stairwell_a_f1', 'stairwell_a_f2', 5.0, 1.5, 5, True, (1, 2)),
        ('stairwell_a_f2', 'stairwell_a_f3', 5.0, 1.5, 5, True, (2, 3)),
        ('stairwell_b_f1', 'stairwell_b_f2', 5.0, 1.5, 5, True, (1, 2)),
        ('stairwell_b_f2', 'stairwell_b_f3', 5.0, 1.5, 5, True, (2, 3)),
        ('elevator_f1', 'elevator_f2', 5.0, 2.0, 8, True, (1, 2)),
        ('elevator_f2', 'elevator_f3', 5.0, 2.0, 8, True, (2, 3)),
    ]
    
    for idx, (src, tgt, dist, width, tp, is_vert, span) in enumerate(cross_edges):
        b.cross_floor_edges.append(
            Edge(id=f'e_cross_{idx}', source=src, target=tgt, distance=dist, width=width, 
                 max_throughput=tp, is_vertical=is_vert, floor_span=span)
        )

    return b


def generate_industrial_plant(name: str = 'LBP Petrochemical Refinery') -> Building:
    """
    Generates a realistic multi-story chemical / petrochemical industrial facility graph.
    Models processing units, storage tank farms, pipe racks, hazardous containment,
    and ISO 7010 emergency muster assembly points with realistic physical distances (meters).
    """
    b = Building(name=name)

    # -------------------------------------------------------------
    # FLOOR 1: Ground Operations, Reactor Yard & Tank Farm
    # -------------------------------------------------------------
    f1 = Floor(level=1)
    
    # Process & Storage Units
    f1.nodes['tank_farm_a'] = Node(id='tank_farm_a', type=NodeType.ROOM, floor=1, capacity=8, position=(40.0, 30.0))
    f1.nodes['tank_farm_b'] = Node(id='tank_farm_b', type=NodeType.ROOM, floor=1, capacity=8, position=(40.0, 75.0))
    f1.nodes['hazmat_basin'] = Node(id='hazmat_basin', type=NodeType.ROOM, floor=1, capacity=12, position=(70.0, 52.0))
    f1.nodes['pump_house'] = Node(id='pump_house', type=NodeType.ROOM, floor=1, capacity=10, position=(70.0, 85.0))
    
    f1.nodes['reactor_1'] = Node(id='reactor_1', type=NodeType.ROOM, floor=1, capacity=15, position=(115.0, 35.0))
    f1.nodes['reactor_2'] = Node(id='reactor_2', type=NodeType.ROOM, floor=1, capacity=15, position=(145.0, 35.0))
    f1.nodes['compressor_shed'] = Node(id='compressor_shed', type=NodeType.ROOM, floor=1, capacity=12, position=(130.0, 85.0))
    
    f1.nodes['control_room'] = Node(id='control_room', type=NodeType.ROOM, floor=1, capacity=25, position=(195.0, 35.0))
    f1.nodes['loading_bay'] = Node(id='loading_bay', type=NodeType.ROOM, floor=1, capacity=20, position=(195.0, 75.0))
    f1.nodes['emergency_shower'] = Node(id='emergency_shower', type=NodeType.ROOM, floor=1, capacity=6, position=(160.0, 20.0))
    
    # Industrial Intersections & Pipe Racks (Routers)
    f1.nodes['pipe_rack_junc_1'] = Node(id='pipe_rack_junc_1', type=NodeType.INTERSECTION, floor=1, capacity=35, position=(100.0, 55.0))
    f1.nodes['pipe_rack_junc_2'] = Node(id='pipe_rack_junc_2', type=NodeType.INTERSECTION, floor=1, capacity=35, position=(160.0, 55.0))
    f1.nodes['corridor_perimeter_n'] = Node(id='corridor_perimeter_n', type=NodeType.CORRIDOR, floor=1, capacity=40, position=(130.0, 20.0))
    f1.nodes['corridor_perimeter_s'] = Node(id='corridor_perimeter_s', type=NodeType.CORRIDOR, floor=1, capacity=40, position=(130.0, 95.0))
    
    # Vertical Connectors
    f1.nodes['stair_north_f1'] = Node(id='stair_north_f1', type=NodeType.STAIRWELL, floor=1, capacity=12, position=(85.0, 20.0), is_cross_floor=True)
    f1.nodes['stair_south_f1'] = Node(id='stair_south_f1', type=NodeType.STAIRWELL, floor=1, capacity=12, position=(175.0, 95.0), is_cross_floor=True)
    f1.nodes['industrial_hoist_f1'] = Node(id='industrial_hoist_f1', type=NodeType.ELEVATOR, floor=1, capacity=8, position=(130.0, 55.0), is_cross_floor=True)
    
    # ISO 7010 Evacuation Muster Assembly Points (Exits)
    f1.nodes['muster_point_alpha'] = Node(id='muster_point_alpha', type=NodeType.EXIT, floor=1, capacity=150, position=(130.0, 5.0))
    f1.nodes['muster_point_bravo'] = Node(id='muster_point_bravo', type=NodeType.EXIT, floor=1, capacity=150, position=(130.0, 110.0))
    f1.exits.extend(['muster_point_alpha', 'muster_point_bravo'])
    
    # Edges F1 (source, target, distance_meters, width_meters, max_throughput)
    edges_f1 = [
        # Tank farm to containment & junctions
        ('tank_farm_a', 'hazmat_basin', 25.0, 2.5, 12),
        ('tank_farm_b', 'hazmat_basin', 25.0, 2.5, 12),
        ('tank_farm_b', 'pump_house', 20.0, 2.0, 10),
        ('pump_house', 'pipe_rack_junc_1', 30.0, 3.0, 15),
        ('hazmat_basin', 'pipe_rack_junc_1', 25.0, 3.0, 15),
        ('hazmat_basin', 'stair_north_f1', 30.0, 2.0, 8),
        
        # Central Pipe Rack Corridor
        ('pipe_rack_junc_1', 'industrial_hoist_f1', 20.0, 4.0, 25),
        ('industrial_hoist_f1', 'pipe_rack_junc_2', 20.0, 4.0, 25),
        ('pipe_rack_junc_1', 'reactor_1', 18.0, 2.5, 12),
        ('pipe_rack_junc_2', 'reactor_2', 18.0, 2.5, 12),
        ('reactor_1', 'reactor_2', 20.0, 2.5, 12),
        ('pipe_rack_junc_1', 'compressor_shed', 35.0, 3.0, 15),
        ('pipe_rack_junc_2', 'compressor_shed', 35.0, 3.0, 15),
        
        # East Control & Loading Section
        ('pipe_rack_junc_2', 'control_room', 30.0, 3.5, 20),
        ('pipe_rack_junc_2', 'loading_bay', 30.0, 3.5, 20),
        ('control_room', 'loading_bay', 35.0, 2.5, 12),
        ('control_room', 'emergency_shower', 22.0, 2.0, 10),
        ('emergency_shower', 'corridor_perimeter_n', 18.0, 2.5, 15),
        
        # North Perimeter Egress Path to Muster Alpha
        ('reactor_1', 'corridor_perimeter_n', 15.0, 3.0, 18),
        ('reactor_2', 'corridor_perimeter_n', 15.0, 3.0, 18),
        ('stair_north_f1', 'corridor_perimeter_n', 25.0, 2.5, 15),
        ('corridor_perimeter_n', 'muster_point_alpha', 12.0, 5.0, 50),
        
        # South Perimeter Egress Path to Muster Bravo
        ('compressor_shed', 'corridor_perimeter_s', 15.0, 3.0, 18),
        ('loading_bay', 'stair_south_f1', 28.0, 3.5, 20),
        ('stair_south_f1', 'corridor_perimeter_s', 45.0, 3.0, 20),
        ('pipe_rack_junc_2', 'stair_south_f1', 35.0, 2.0, 10),
        ('corridor_perimeter_s', 'muster_point_bravo', 12.0, 5.0, 50)
    ]
    for idx, (src, tgt, dist, width, tp) in enumerate(edges_f1):
        f1.edges.append(Edge(id=f'e_ind_f1_{idx}', source=src, target=tgt, distance=dist, width=width, max_throughput=tp))
    b.floors[1] = f1

    # -------------------------------------------------------------
    # FLOOR 2: Processing Mezzanine & Pipe Gallery
    # -------------------------------------------------------------
    f2 = Floor(level=2)
    f2.nodes['distillation_deck'] = Node(id='distillation_deck', type=NodeType.ROOM, floor=2, capacity=12, position=(115.0, 35.0))
    f2.nodes['catalyst_feed'] = Node(id='catalyst_feed', type=NodeType.ROOM, floor=2, capacity=10, position=(145.0, 35.0))
    f2.nodes['substation_f2'] = Node(id='substation_f2', type=NodeType.ROOM, floor=2, capacity=8, position=(195.0, 35.0))
    f2.nodes['catwalk_west_f2'] = Node(id='catwalk_west_f2', type=NodeType.INTERSECTION, floor=2, capacity=20, position=(100.0, 55.0))
    f2.nodes['catwalk_east_f2'] = Node(id='catwalk_east_f2', type=NodeType.INTERSECTION, floor=2, capacity=20, position=(160.0, 55.0))
    f2.nodes['mezzanine_walkway'] = Node(id='mezzanine_walkway', type=NodeType.CORRIDOR, floor=2, capacity=25, position=(130.0, 35.0))
    
    f2.nodes['stair_north_f2'] = Node(id='stair_north_f2', type=NodeType.STAIRWELL, floor=2, capacity=12, position=(85.0, 20.0), is_cross_floor=True)
    f2.nodes['stair_south_f2'] = Node(id='stair_south_f2', type=NodeType.STAIRWELL, floor=2, capacity=12, position=(175.0, 95.0), is_cross_floor=True)
    f2.nodes['industrial_hoist_f2'] = Node(id='industrial_hoist_f2', type=NodeType.ELEVATOR, floor=2, capacity=8, position=(130.0, 55.0), is_cross_floor=True)
    
    # Emergency Quick-Egress Chute from Mezzanine
    f2.nodes['slide_escape_f2'] = Node(id='slide_escape_f2', type=NodeType.EXIT, floor=2, capacity=40, position=(215.0, 55.0))
    f2.exits.append('slide_escape_f2')
    
    edges_f2 = [
        ('catwalk_west_f2', 'industrial_hoist_f2', 20.0, 2.0, 12),
        ('industrial_hoist_f2', 'catwalk_east_f2', 20.0, 2.0, 12),
        ('catwalk_west_f2', 'distillation_deck', 18.0, 1.8, 10),
        ('distillation_deck', 'mezzanine_walkway', 15.0, 2.0, 12),
        ('mezzanine_walkway', 'catalyst_feed', 15.0, 2.0, 12),
        ('catalyst_feed', 'catwalk_east_f2', 18.0, 1.8, 10),
        ('catwalk_east_f2', 'substation_f2', 25.0, 1.8, 10),
        ('catwalk_west_f2', 'stair_north_f2', 30.0, 1.8, 10),
        ('distillation_deck', 'stair_north_f2', 25.0, 1.8, 10),
        ('catwalk_east_f2', 'stair_south_f2', 35.0, 1.8, 10),
        ('catwalk_east_f2', 'slide_escape_f2', 30.0, 2.0, 15),
        ('substation_f2', 'slide_escape_f2', 25.0, 2.0, 15)
    ]
    for idx, (src, tgt, dist, width, tp) in enumerate(edges_f2):
        f2.edges.append(Edge(id=f'e_ind_f2_{idx}', source=src, target=tgt, distance=dist, width=width, max_throughput=tp))
    b.floors[2] = f2

    # -------------------------------------------------------------
    # FLOOR 3: Flare Header & Vent Scrubber Platform
    # -------------------------------------------------------------
    f3 = Floor(level=3)
    f3.nodes['scrubber_deck'] = Node(id='scrubber_deck', type=NodeType.ROOM, floor=3, capacity=10, position=(115.0, 35.0))
    f3.nodes['flare_platform'] = Node(id='flare_platform', type=NodeType.ROOM, floor=3, capacity=10, position=(150.0, 35.0))
    f3.nodes['catwalk_f3'] = Node(id='catwalk_f3', type=NodeType.CORRIDOR, floor=3, capacity=20, position=(130.0, 55.0))
    
    f3.nodes['stair_north_f3'] = Node(id='stair_north_f3', type=NodeType.STAIRWELL, floor=3, capacity=12, position=(85.0, 20.0), is_cross_floor=True)
    f3.nodes['stair_south_f3'] = Node(id='stair_south_f3', type=NodeType.STAIRWELL, floor=3, capacity=12, position=(175.0, 95.0), is_cross_floor=True)
    f3.nodes['industrial_hoist_f3'] = Node(id='industrial_hoist_f3', type=NodeType.ELEVATOR, floor=3, capacity=8, position=(130.0, 55.0), is_cross_floor=True)
    
    # Aerial Helicopter Rescue Muster Deck
    f3.nodes['helipad_muster'] = Node(id='helipad_muster', type=NodeType.EXIT, floor=3, capacity=30, position=(195.0, 55.0))
    f3.exits.append('helipad_muster')
    
    edges_f3 = [
        ('stair_north_f3', 'scrubber_deck', 22.0, 1.8, 10),
        ('scrubber_deck', 'flare_platform', 25.0, 2.0, 12),
        ('scrubber_deck', 'catwalk_f3', 18.0, 1.8, 10),
        ('flare_platform', 'catwalk_f3', 18.0, 1.8, 10),
        ('catwalk_f3', 'industrial_hoist_f3', 10.0, 2.0, 10),
        ('catwalk_f3', 'stair_south_f3', 35.0, 1.8, 10),
        ('flare_platform', 'helipad_muster', 30.0, 2.5, 15),
        ('catwalk_f3', 'helipad_muster', 35.0, 2.0, 12)
    ]
    for idx, (src, tgt, dist, width, tp) in enumerate(edges_f3):
        f3.edges.append(Edge(id=f'e_ind_f3_{idx}', source=src, target=tgt, distance=dist, width=width, max_throughput=tp))
    b.floors[3] = f3

    # -------------------------------------------------------------
    # Cross-Floor Inter-Level Connectors
    # -------------------------------------------------------------
    cross_edges = [
        ('stair_north_f1', 'stair_north_f2', 6.0, 1.8, 10, True, (1, 2)),
        ('stair_north_f2', 'stair_north_f3', 6.0, 1.8, 10, True, (2, 3)),
        ('stair_south_f1', 'stair_south_f2', 6.0, 1.8, 10, True, (1, 2)),
        ('stair_south_f2', 'stair_south_f3', 6.0, 1.8, 10, True, (2, 3)),
        ('industrial_hoist_f1', 'industrial_hoist_f2', 6.0, 2.2, 10, True, (1, 2)),
        ('industrial_hoist_f2', 'industrial_hoist_f3', 6.0, 2.2, 10, True, (2, 3))
    ]
    for idx, (src, tgt, dist, width, tp, is_vert, span) in enumerate(cross_edges):
        b.cross_floor_edges.append(
            Edge(id=f'e_ind_cross_{idx}', source=src, target=tgt, distance=dist, width=width,
                 max_throughput=tp, is_vertical=is_vert, floor_span=span)
        )

    return b


def _append_floor_edges(floor: Floor, prefix: str, edges: List[tuple]) -> None:
    """Add compact layout edge specifications to a floor."""
    for idx, (source, target, distance, width, throughput) in enumerate(edges):
        floor.edges.append(Edge(
            id=f"{prefix}_{idx}",
            source=source,
            target=target,
            distance=float(distance),
            width=float(width),
            max_throughput=int(throughput),
        ))


def generate_hexagonal_cell_factory(name: str = 'LBP Hexagonal Cell Manufacturing Plant') -> Building:
    """Hexagonal cellular-manufacturing hall with a central robotic assembly cell.

    The six-sided circulation loop provides alternate paths around a disabled
    production area, while three separated exit discharges avoid a single
    egress dependency.  The centre controller has six directional neighbours,
    within the eight-direction mesh protocol limit.
    """
    building = Building(name=name)
    floor = Floor(level=1)
    node_specs = [
        ('north_cross_aisle', NodeType.CORRIDOR, 45, (130.0, 18.0)),
        ('quality_metrology', NodeType.ROOM, 22, (180.0, 43.0)),
        ('packaging_cell', NodeType.ROOM, 34, (180.0, 97.0)),
        ('south_cross_aisle', NodeType.CORRIDOR, 45, (130.0, 122.0)),
        ('cnc_machining_cell', NodeType.ROOM, 32, (80.0, 97.0)),
        ('receiving_kitting', NodeType.ROOM, 30, (80.0, 43.0)),
        ('robotic_assembly_cell', NodeType.INTERSECTION, 52, (130.0, 70.0)),
        ('battery_charge_room', NodeType.ROOM, 10, (130.0, 43.0)),
        ('muster_north', NodeType.EXIT, 120, (130.0, -5.0)),
        ('muster_east', NodeType.EXIT, 120, (220.0, 70.0)),
        ('muster_west', NodeType.EXIT, 120, (40.0, 70.0)),
    ]
    for node_id, node_type, capacity, position in node_specs:
        floor.nodes[node_id] = Node(
            id=node_id,
            type=node_type,
            floor=1,
            capacity=capacity,
            position=position,
        )
    floor.exits.extend(['muster_north', 'muster_east', 'muster_west'])

    ring = [
        ('north_cross_aisle', 'quality_metrology', 32, 3.2, 18),
        ('quality_metrology', 'packaging_cell', 30, 2.8, 15),
        ('packaging_cell', 'south_cross_aisle', 32, 3.2, 18),
        ('south_cross_aisle', 'cnc_machining_cell', 32, 3.2, 18),
        ('cnc_machining_cell', 'receiving_kitting', 30, 3.0, 16),
        ('receiving_kitting', 'north_cross_aisle', 32, 3.2, 18),
    ]
    spokes = [
        ('robotic_assembly_cell', 'north_cross_aisle', 29, 3.8, 24),
        ('robotic_assembly_cell', 'quality_metrology', 29, 2.8, 15),
        ('robotic_assembly_cell', 'packaging_cell', 29, 3.2, 18),
        ('robotic_assembly_cell', 'south_cross_aisle', 29, 3.8, 24),
        ('robotic_assembly_cell', 'cnc_machining_cell', 29, 3.2, 18),
        ('robotic_assembly_cell', 'receiving_kitting', 29, 3.5, 20),
        ('battery_charge_room', 'north_cross_aisle', 25, 2.0, 8),
        ('battery_charge_room', 'quality_metrology', 32, 2.0, 8),
    ]
    egress = [
        ('north_cross_aisle', 'muster_north', 15, 4.5, 36),
        ('quality_metrology', 'muster_east', 28, 3.5, 24),
        ('packaging_cell', 'muster_east', 28, 3.5, 24),
        ('receiving_kitting', 'muster_west', 28, 3.5, 24),
        ('cnc_machining_cell', 'muster_west', 28, 3.5, 24),
    ]
    _append_floor_edges(floor, 'e_hex', ring + spokes + egress)
    building.floors[1] = floor
    return building


def generate_u_shaped_factory(name: str = 'LBP U-Shaped High-Bay Manufacturing Hall') -> Building:
    """U-shaped material-flow hall with cross aisles and separated muster exits."""
    building = Building(name=name)
    floor = Floor(level=1)
    node_specs = [
        ('receiving_dock', NodeType.ROOM, 28, (35.0, 25.0)),
        ('raw_material_store', NodeType.ROOM, 38, (35.0, 55.0)),
        ('cnc_machining', NodeType.ROOM, 44, (35.0, 90.0)),
        ('welding_cell', NodeType.ROOM, 34, (75.0, 118.0)),
        ('main_assembly_line', NodeType.ROOM, 70, (120.0, 118.0)),
        ('paint_finishing', NodeType.ROOM, 26, (165.0, 118.0)),
        ('quality_control', NodeType.ROOM, 24, (205.0, 90.0)),
        ('packaging_zone', NodeType.ROOM, 36, (205.0, 55.0)),
        ('shipping_dock', NodeType.ROOM, 28, (205.0, 25.0)),
        ('west_cross_aisle', NodeType.INTERSECTION, 45, (80.0, 73.0)),
        ('central_cross_aisle', NodeType.INTERSECTION, 55, (120.0, 73.0)),
        ('east_cross_aisle', NodeType.INTERSECTION, 45, (160.0, 73.0)),
        ('utilities_control', NodeType.ROOM, 14, (120.0, 35.0)),
        ('muster_west', NodeType.EXIT, 160, (8.0, 118.0)),
        ('muster_east', NodeType.EXIT, 160, (232.0, 118.0)),
        ('muster_north', NodeType.EXIT, 120, (120.0, 5.0)),
    ]
    for node_id, node_type, capacity, position in node_specs:
        floor.nodes[node_id] = Node(
            id=node_id,
            type=node_type,
            floor=1,
            capacity=capacity,
            position=position,
        )
    floor.exits.extend(['muster_west', 'muster_east', 'muster_north'])

    production_flow = [
        ('receiving_dock', 'raw_material_store', 24, 4.0, 26),
        ('raw_material_store', 'cnc_machining', 28, 3.6, 22),
        ('cnc_machining', 'welding_cell', 34, 3.2, 18),
        ('welding_cell', 'main_assembly_line', 34, 4.0, 26),
        ('main_assembly_line', 'paint_finishing', 34, 4.0, 26),
        ('paint_finishing', 'quality_control', 34, 3.2, 18),
        ('quality_control', 'packaging_zone', 28, 3.6, 22),
        ('packaging_zone', 'shipping_dock', 24, 4.0, 26),
    ]
    cross_aisles = [
        ('west_cross_aisle', 'central_cross_aisle', 30, 4.5, 30),
        ('central_cross_aisle', 'east_cross_aisle', 30, 4.5, 30),
        ('west_cross_aisle', 'raw_material_store', 31, 3.5, 22),
        ('west_cross_aisle', 'cnc_machining', 32, 3.5, 22),
        ('central_cross_aisle', 'main_assembly_line', 35, 4.5, 30),
        ('east_cross_aisle', 'quality_control', 32, 3.5, 22),
        ('east_cross_aisle', 'packaging_zone', 31, 3.5, 22),
        ('central_cross_aisle', 'utilities_control', 28, 3.0, 16),
        ('utilities_control', 'receiving_dock', 58, 3.0, 16),
        ('utilities_control', 'shipping_dock', 58, 3.0, 16),
    ]
    egress = [
        ('cnc_machining', 'muster_west', 24, 4.5, 34),
        ('welding_cell', 'muster_west', 42, 4.0, 28),
        ('paint_finishing', 'muster_east', 42, 4.0, 28),
        ('quality_control', 'muster_east', 24, 4.5, 34),
        ('utilities_control', 'muster_north', 24, 4.0, 28),
    ]
    _append_floor_edges(floor, 'e_ushape', production_flow + cross_aisles + egress)
    building.floors[1] = floor
    return building


BUILDING_LAYOUTS = {
    'petrochemical': {
        'name': 'Petrochemical Refinery',
        'shape': 'Multi-level process yard',
        'description': 'Tank farm, reactors, pipe racks, mezzanines and elevated process decks.',
        'generator': generate_industrial_plant,
    },
    'hexagonal_cell': {
        'name': 'Hexagonal Cell Factory',
        'shape': 'Six-sided cellular loop',
        'description': 'Central robotic assembly with a redundant hexagonal material-flow aisle.',
        'generator': generate_hexagonal_cell_factory,
    },
    'u_shaped_hall': {
        'name': 'U-Shaped High-Bay Hall',
        'shape': 'U-flow production hall',
        'description': 'Receiving-to-shipping production flow with three protected cross-aisle exits.',
        'generator': generate_u_shaped_factory,
    },
}


def building_layout_catalog() -> Dict[str, Dict[str, str]]:
    return {
        layout_id: {
            'id': layout_id,
            'name': spec['name'],
            'shape': spec['shape'],
            'description': spec['description'],
        }
        for layout_id, spec in BUILDING_LAYOUTS.items()
    }


def generate_industrial_layout(layout_id: str = 'petrochemical') -> Building:
    try:
        generator = BUILDING_LAYOUTS[layout_id]['generator']
    except KeyError as exc:
        raise ValueError(f"Unknown industrial layout: {layout_id}") from exc
    return generator()



def export_building_to_yaml(building: Building, path: str) -> None:
    data = {
        'name': building.name,
        'floors': {},
        'cross_floor_edges': []
    }
    
    for level, floor in building.floors.items():
        data['floors'][level] = {
            'nodes': [
                {
                    'id': n.id,
                    'type': n.type.value,
                    'floor': n.floor,
                    'capacity': n.capacity,
                    'position': list(n.position),
                    'is_cross_floor': n.is_cross_floor
                } for n in floor.nodes.values()
            ],
            'edges': [
                {
                    'id': e.id,
                    'source': e.source,
                    'target': e.target,
                    'distance': e.distance,
                    'width': e.width,
                    'max_throughput': e.max_throughput
                } for e in floor.edges
            ]
        }
        
    for e in building.cross_floor_edges:
        data['cross_floor_edges'].append({
            'id': e.id,
            'source': e.source,
            'target': e.target,
            'distance': e.distance,
            'width': e.width,
            'max_throughput': e.max_throughput,
            'is_vertical': e.is_vertical,
            'floor_span': list(e.floor_span) if e.floor_span else None
        })
        
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)


def generate_building_from_yaml(path: str) -> Building:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        
    b = Building(name=data['name'])
    
    for level, f_data in data.get('floors', {}).items():
        floor = Floor(level=level)
        for n_data in f_data.get('nodes', []):
            node = Node(
                id=n_data['id'],
                type=NodeType(n_data['type']),
                floor=n_data['floor'],
                capacity=n_data['capacity'],
                position=tuple(n_data['position']),
                is_cross_floor=n_data.get('is_cross_floor', False)
            )
            floor.nodes[node.id] = node
            if node.type == NodeType.EXIT:
                floor.exits.append(node.id)
                
        for e_data in f_data.get('edges', []):
            edge = Edge(
                id=e_data['id'],
                source=e_data['source'],
                target=e_data['target'],
                distance=e_data['distance'],
                width=e_data['width'],
                max_throughput=e_data['max_throughput']
            )
            floor.edges.append(edge)
            
        b.floors[level] = floor
        
    for e_data in data.get('cross_floor_edges', []):
        span = tuple(e_data['floor_span']) if e_data.get('floor_span') else None
        edge = Edge(
            id=e_data['id'],
            source=e_data['source'],
            target=e_data['target'],
            distance=e_data['distance'],
            width=e_data['width'],
            max_throughput=e_data['max_throughput'],
            is_vertical=e_data.get('is_vertical', False),
            floor_span=span
        )
        b.cross_floor_edges.append(edge)
        
    return b
