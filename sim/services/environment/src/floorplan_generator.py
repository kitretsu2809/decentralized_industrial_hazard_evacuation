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
