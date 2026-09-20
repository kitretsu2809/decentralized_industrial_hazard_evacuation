import heapq
from typing import List, Set, Tuple, Optional, Dict
from core.graph.types import Building, Node, Edge, EdgeState

class BuildingGraph:
    def __init__(self, building: Building):
        self.building = building
        # adjacency list: node_id -> {neighbor_id: edge}
        self.adj: Dict[str, Dict[str, Edge]] = {}
        self.edge_map: Dict[str, Edge] = {}
        self._build_adj()
        
    def _build_adj(self):
        self.adj.clear()
        self.edge_map.clear()
        for node_id in self.building.all_nodes.keys():
            self.adj[node_id] = {}
            
        for edge in self.building.all_edges:
            self.edge_map[edge.id] = edge
            if edge.source in self.adj:
                self.adj[edge.source][edge.target] = edge
            if edge.target in self.adj:
                self.adj[edge.target][edge.source] = edge
                
    def get_neighbors(self, node_id: str) -> List[str]:
        return list(self.adj.get(node_id, {}).keys())
        
    def get_k_hop_neighbors(self, node_id: str, k: int) -> Set[str]:
        if k < 0:
            return set()
        visited = set([node_id])
        current_level = set([node_id])
        
        for _ in range(k):
            next_level = set()
            for node in current_level:
                for neighbor in self.get_neighbors(node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
            current_level = next_level
            
        if node_id in visited:
            visited.remove(node_id)
        return visited

    def sever_edge(self, edge_id: str):
        if edge_id in self.edge_map:
            self.edge_map[edge_id].state = EdgeState.SEVERED

    def restore_edge(self, edge_id: str):
        if edge_id in self.edge_map:
            self.edge_map[edge_id].state = EdgeState.OPEN

    def lock_door(self, edge_id: str):
        if edge_id in self.edge_map:
            self.edge_map[edge_id].state = EdgeState.LOCKED

    def unlock_door(self, edge_id: str):
        if edge_id in self.edge_map:
            self.edge_map[edge_id].state = EdgeState.OPEN

    def emergency_seal(self, edge_id: str):
        if edge_id in self.edge_map:
            self.edge_map[edge_id].state = EdgeState.EMERGENCY_SEALED

    def get_active_edges(self, node_id: str) -> List[Edge]:
        active_edges = []
        for neighbor, edge in self.adj.get(node_id, {}).items():
            if edge.state == EdgeState.OPEN:
                active_edges.append(edge)
        return active_edges
        
    def compute_shortest_path(self, src: str, dst: str, avoid_nodes: Set[str] = None) -> Tuple[List[str], float]:
        if avoid_nodes is None:
            avoid_nodes = set()
            
        if src not in self.adj or dst not in self.adj:
            return [], float('inf')
            
        distances = {node_id: float('inf') for node_id in self.adj}
        distances[src] = 0
        pq = [(0.0, src)]
        previous = {node_id: None for node_id in self.adj}
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node == dst:
                break
                
            if current_dist > distances[current_node]:
                continue
                
            node_obj = self.building.get_node(current_node)
            hazard = node_obj.hazard_score if node_obj else 0.0
                
            for neighbor, edge in self.adj.get(current_node, {}).items():
                if neighbor in avoid_nodes:
                    continue
                if edge.state != EdgeState.OPEN:
                    continue
                    
                cost = edge.distance * (1 + hazard * 10)
                distance = current_dist + cost
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
                    
        if distances[dst] == float('inf'):
            return [], float('inf')
            
        path = []
        curr = dst
        while curr is not None:
            path.append(curr)
            curr = previous[curr]
        path.reverse()
        
        return path, distances[dst]

    def get_all_exit_paths(self, node_id: str) -> List[Tuple[List[str], float]]:
        paths = []
        for exit_id in self.building.all_exits:
            path, cost = self.compute_shortest_path(node_id, exit_id)
            if path:
                paths.append((path, cost))
        paths.sort(key=lambda x: x[1])
        return paths

    def get_floor_subgraph_nodes(self, floor: int) -> List[str]:
        if floor in self.building.floors:
            return list(self.building.floors[floor].nodes.keys())
        return []

    def is_path_exists(self, src: str, dst: str) -> bool:
        path, _ = self.compute_shortest_path(src, dst)
        return len(path) > 0

    def update_hazard(self, node_id: str, hazard: float):
        node = self.building.get_node(node_id)
        if node:
            node.hazard_score = hazard

    def get_edge_by_id(self, edge_id: str) -> Optional[Edge]:
        return self.edge_map.get(edge_id)

    def get_edge_between(self, src: str, dst: str) -> Optional[Edge]:
        return self.adj.get(src, {}).get(dst)

    def to_dict(self) -> dict:
        nodes = []
        for n in self.building.all_nodes.values():
            nodes.append({
                "id": n.id,
                "type": n.type.value,
                "floor": n.floor,
                "occupancy": n.current_occupancy,
                "hazard": n.hazard_score
            })
            
        edges = []
        for e in self.building.all_edges:
            edges.append({
                "id": e.id,
                "source": e.source,
                "target": e.target,
                "state": e.state.value
            })
            
        floors = {}
        for level, floor in self.building.floors.items():
            floors[level] = {
                "level": level,
                "nodes": [n.id for n in floor.nodes.values()],
                "edges": [{
                    "id": e.id,
                    "source": e.source,
                    "target": e.target,
                    "state": e.state.value,
                    "distance": e.distance,
                    "width": e.width,
                    "has_door": e.has_door
                } for e in floor.edges],
                "exits": floor.exits
            }

        return {
            "building_name": self.building.name,
            "total_capacity": self.building.total_capacity,
            "nodes": nodes,
            "edges": edges,
            "floors": floors
        }

