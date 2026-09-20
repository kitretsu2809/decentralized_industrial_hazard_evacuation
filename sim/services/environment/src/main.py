import os
import time
import json
import logging
from typing import Dict, Any

from core.messaging.schemas import (
    EnvironmentStateMsg, PolicyActionMsg, 
    HazardUpdateMsg, DisasterInjectionMsg, SimControlMsg, ActiveThreatMsg
)
from core.graph.types import NodeType
from core.graph.building_graph import BuildingGraph
from sim.services.environment.src.floorplan_generator import generate_industrial_plant, generate_default_building
from sim.services.environment.src.evac_env import EvacuationEnv
from sim.services.environment.src.event_bus import EnvironmentEventBus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentService:
    def __init__(self):
        self.redis_url = os.environ.get("REDIS_URL", "redis://redis:6379")
        self.fps = float(os.environ.get("SIM_STATE_FPS", "10"))
        self.sim_speed = float(os.environ.get("SIM_SPEED", "1.0"))
        
        # Initialize with Industrial Plant facility (Petrochemical / Chemical complex)
        self.building = generate_industrial_plant()
        self.building_graph = BuildingGraph(self.building)
        self.env = EvacuationEnv(self.building, max_steps=10000)
        self.env.reset()
        
        # Event bus
        self.bus = EnvironmentEventBus(self.redis_url)
        
        # Latest actions from policy
        self.latest_actions = {agent: [0]*(len(self.env.action_spaces[agent].nvec)) for agent in self.env.agents}
        
        self.visual_evacuees = {}
        self._init_visual_evacuees()
        
        self.running = True
        
        # Setup callbacks
        self.bus.subscribe_policy_actions(self._on_policy_action)
        self.bus.subscribe_disaster_injection(self._on_disaster_inject)
        self.bus.subscribe_control(self._on_control)
        
        # Publish static graph to redis for GUI
        self._publish_static_graph()

    def _init_visual_evacuees(self):
        import random
        self.visual_evacuees.clear()
        evac_id = 0
        for n_id, count in self.env.evacuees_at_node.items():
            if count > 0:
                node = self.building.get_node(n_id)
                for _ in range(int(count)):
                    self.visual_evacuees[f"p_{evac_id}"] = {
                        "id": f"p_{evac_id}",
                        "node": n_id,
                        "pos": [node.position[0] + random.uniform(-3, 3), node.position[1] + random.uniform(-3, 3)],
                        "target_node": n_id,
                        "progress": 1.0,
                        "floor": node.floor,
                        "status": "moving"
                    }
                    evac_id += 1

    def _publish_static_graph(self):
        bg = BuildingGraph(self.building)
        graph_dict = bg.to_dict()
        self.bus.redis_client.set("lbp:building:graph", json.dumps(graph_dict))
        logger.info("Published building graph to Redis")

    def _on_policy_action(self, msg: PolicyActionMsg):
        # Update latest actions. Ensure we handle missing agents gracefully.
        for agent_id, agent_action in msg.actions.items():
            if agent_id in self.env.agents:
                # Convert AgentAction dicts back to multidiscrete array
                action_len = len(self.env.action_spaces[agent_id].nvec)
                arr = [0] * action_len
                # Simple mapping: first N are edge actions, last is door action
                edges = list(agent_action.get("edge_actions", {}).values())
                for i in range(min(len(edges), action_len - 1)):
                    arr[i] = edges[i]
                door_cmds = agent_action.get("door_commands", {})
                if len(arr) > 0 and door_cmds:
                    arr[-1] = list(door_cmds.values())[0]
                self.latest_actions[agent_id] = arr

    def _on_disaster_inject(self, msg: DisasterInjectionMsg):
        threat_type = msg.threat_type.upper()
        if threat_type == 'CLEAR_ALL' or (msg.intensity <= 0.0 and msg.target_node == 'ALL'):
            logger.info("Clearing all industrial disasters from plant")
            self.env.fire_model.clear_all()
            return
            
        if threat_type == 'SCENARIO':
            scenario = msg.metadata.get('scenario', 'major_catastrophe')
            logger.info(f"Injecting industrial scenario: {scenario}")
            self.env.fire_model.inject_scenario(scenario)
            return

        if threat_type == 'SET_WIND':
            angle = float(msg.metadata.get('angle', 45.0))
            speed = float(msg.metadata.get('speed', 4.2))
            logger.info(f"Updating plant wind: {angle} deg, {speed} m/s")
            self.env.fire_model.set_wind(angle, speed)
            return

        logger.info(f"Injecting industrial disaster: {threat_type} at {msg.target_node} (Floor {msg.floor}, Intensity {msg.intensity})")
        if msg.intensity <= 0.0:
            self.env.fire_model.remove_hazard(msg.target_node)
        else:
            self.env.fire_model.inject_hazard(
                threat_type=threat_type,
                node_id=msg.target_node,
                intensity=msg.intensity,
                spread_rate=msg.spread_rate,
                metadata=msg.metadata
            )

    def _on_control(self, msg: SimControlMsg):
        logger.info(f"Sim control command: {msg.command}")
        if msg.command == 'reset':
            self.env.reset()
            self._init_visual_evacuees()
        elif msg.command == 'pause':
            self.running = False
        elif msg.command == 'play':
            self.running = True
        elif msg.command == 'fps' and msg.value:
            self.fps = float(msg.value)
        elif msg.command == 'speed' and msg.value:
            self.sim_speed = float(msg.value)

    def run(self):
        logger.info("Starting Environment Service loop...")
        self.bus.start_listening()
        
        try:
            while True:
                start_time = time.time()
                step_duration = 1.0 / self.fps
                
                if self.running:
                    # Step environment
                    obs, rewards, term, trunc, infos = self.env.step(self.latest_actions)
                    
                    # If done, reset
                    if not self.env.agents:
                        self.env.reset()
                        self._init_visual_evacuees()
                        self.latest_actions = {agent: [0]*(len(self.env.action_spaces[agent].nvec)) for agent in self.env.agents}
                    
                    # Build NodeStateMsgs
                    nodes_state = {}
                    bg = self.env.building_graph
                    for n_id, n in self.building.all_nodes.items():
                        edge_states = {}
                        sign_directions = {}
                        for edge in bg.adj.get(n_id, {}).values():
                            edge_states[edge.id] = edge.state.name
                            if hasattr(self.env, 'edge_signs') and self.env.edge_signs.get(edge.id, False):
                                sign_directions[edge.id] = "REDIRECT"
                            else:
                                sign_directions[edge.id] = "NONE"
                        
                        nodes_state[n_id] = {
                            "node_id": n_id,
                            "floor": n.floor,
                            "hazard_score": n.hazard_score,
                            "crowd_count": self.env.evacuees_at_node.get(n_id, 0),
                            "capacity": n.capacity,
                            "node_type": n.type.name,
                            "position": list(n.position),
                            "edge_states": edge_states,
                            "sign_directions": sign_directions
                        }

                    # Update visual evacuees based on flows
                    import random
                    for src, dst, amount in getattr(self.env, 'last_flows', []):
                        moved = 0
                        for ev in self.visual_evacuees.values():
                            if ev["node"] == src and ev["target_node"] == src:
                                ev["target_node"] = dst
                                ev["progress"] = 0.0
                                moved += 1
                                if moved >= amount:
                                    break
                                    
                    # Interpolate positions based on physical speed and edge distance
                    evacuees_list = []
                    for ev in self.visual_evacuees.values():
                        node_obj = self.building.get_node(ev["node"])
                        if ev["progress"] < 1.0:
                            edge = self.building_graph.get_edge_between(ev["node"], ev["target_node"])
                            edge_dist = edge.distance if edge else 20.0
                            # Physical progress step governed by walking velocity and simulation speed
                            step_increment = max(0.04, min(0.35, (1.34 * 2.2 * self.sim_speed) / max(1.0, edge_dist)))
                            ev["progress"] += step_increment
                            
                            if ev["progress"] >= 1.0:
                                ev["progress"] = 1.0
                                ev["node"] = ev["target_node"]
                                node_obj = self.building.get_node(ev["node"])
                                if node_obj:
                                    ev["pos"] = [node_obj.position[0] + random.uniform(-2, 2), node_obj.position[1] + random.uniform(-2, 2)]
                                    ev["floor"] = node_obj.floor
                            else:
                                target_obj = self.building.get_node(ev["target_node"])
                                if target_obj and node_obj:
                                    prog = ev["progress"]
                                    ev["pos"][0] = node_obj.position[0] * (1.0 - prog) + target_obj.position[0] * prog
                                    ev["pos"][1] = node_obj.position[1] * (1.0 - prog) + target_obj.position[1] * prog
                                    
                        if node_obj:
                            if node_obj.type.name == "EXIT":
                                ev["status"] = "evacuated"
                            elif node_obj.hazard_score > 0.75:
                                ev["status"] = "casualty"
                                
                        evacuees_list.append({
                            "id": ev["id"],
                            "position": ev["pos"],
                            "floor": ev["floor"],
                            "current_edge": None,
                            "status": ev["status"]
                        })

                    # Build active threats list with true industrial threat types
                    active_threats_list = []
                    for threat in self.env.fire_model.get_active_threats():
                        node_obj = self.env.building.get_node(threat['node_id'])
                        t_type = threat.get('type', 'FIRE').upper()
                        active_threats_list.append(ActiveThreatMsg(
                            threat_id=f"{t_type.lower()}_{threat['node_id']}",
                            threat_type=t_type,
                            node_id=threat['node_id'],
                            floor=threat.get('floor', node_obj.floor if node_obj else 1),
                            hazard_score=threat['intensity'],
                            graph_action="UPDATE"
                        ))

                    # Create EnvironmentStateMsg
                    state = EnvironmentStateMsg(
                        step=self.env.current_step,
                        timestamp=time.time(),
                        nodes=nodes_state,
                        evacuees=evacuees_list,
                        active_threats=active_threats_list,
                        metrics={
                            "total_people": self.env.total_evacuees,
                            "evacuated": self.env.evacuated,
                            "casualties": self.env.casualties,
                            "wind_x": float(self.env.fire_model.wind_vector[0]),
                            "wind_y": float(self.env.fire_model.wind_vector[1]),
                            "wind_speed": float(getattr(self.env.fire_model, 'wind_speed', 4.2)),
                            "active_disasters": float(len(self.env.fire_model.active_threats))
                        }
                    )
                    
                    self.bus.publish_state(state)
                    
                elapsed = time.time() - start_time
                sleep_time = max(0, step_duration - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Shutting down Environment Service...")
        finally:
            self.bus.close()

if __name__ == "__main__":
    service = EnvironmentService()
    service.run()
