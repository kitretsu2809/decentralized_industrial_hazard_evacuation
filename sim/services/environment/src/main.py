import os
import time
import json
import logging
from typing import Dict, Any

from core.messaging.schemas import (
    EnvironmentStateMsg, PolicyActionMsg, 
    HazardUpdateMsg, DisasterInjectionMsg, SimControlMsg
)
from core.graph.types import NodeType
from core.graph.building_graph import BuildingGraph
from sim.services.environment.src.floorplan_generator import generate_default_building
from sim.services.environment.src.evac_env import EvacuationEnv
from sim.services.environment.src.event_bus import EnvironmentEventBus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentService:
    def __init__(self):
        self.redis_url = os.environ.get("REDIS_URL", "redis://redis:6379")
        self.fps = float(os.environ.get("SIM_STATE_FPS", "10"))
        self.sim_speed = float(os.environ.get("SIM_SPEED", "1.0"))
        
        # Init building and env
        self.building = generate_default_building()
        self.env = EvacuationEnv(self.building, max_steps=10000)
        self.env.reset()
        
        # Event bus
        self.bus = EnvironmentEventBus(self.redis_url)
        
        # Latest actions from policy
        self.latest_actions = {agent: [0]*(len(self.env.action_spaces[agent].nvec)) for agent in self.env.agents}
        
        self.running = True
        
        # Setup callbacks
        self.bus.subscribe_policy_actions(self._on_policy_action)
        self.bus.subscribe_disaster_injection(self._on_disaster_inject)
        self.bus.subscribe_control(self._on_control)
        
        # Publish static graph to redis for GUI
        self._publish_static_graph()

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
                edges = list(agent_action.edge_actions.values())
                for i in range(min(len(edges), action_len - 1)):
                    arr[i] = edges[i]
                if len(arr) > 0 and agent_action.door_commands:
                    arr[-1] = list(agent_action.door_commands.values())[0]
                self.latest_actions[agent_id] = arr

    def _on_disaster_inject(self, msg: DisasterInjectionMsg):
        logger.info(f"Injecting disaster: {msg.threat_type} at {msg.target_node}")
        if msg.threat_type.name == 'FIRE':
            self.env.fire_model.inject_fire(msg.target_node, intensity=msg.intensity, spread_rate=msg.spread_rate)
        # Handle other types if needed

    def _on_control(self, msg: SimControlMsg):
        logger.info(f"Sim control command: {msg.command}")
        if msg.command == 'reset':
            self.env.reset()
        elif msg.command == 'pause':
            self.running = False
        elif msg.command == 'play':
            self.running = True

    def run(self):
        logger.info("Starting Environment Service loop...")
        self.bus.start_listening()
        
        step_duration = 1.0 / self.fps
        
        try:
            while True:
                start_time = time.time()
                
                if self.running:
                    # Step environment
                    obs, rewards, term, trunc, infos = self.env.step(self.latest_actions)
                    
                    # If done, reset
                    if not self.env.agents:
                        self.env.reset()
                        self.latest_actions = {agent: [0]*(len(self.env.action_spaces[agent].nvec)) for agent in self.env.agents}
                    
                    # Build NodeStateMsgs
                    nodes_state = {}
                    bg = self.env.building_graph
                    for n_id, n in self.building.all_nodes.items():
                        edge_states = {}
                        sign_directions = {}
                        for edge in bg.get_active_edges(n_id):
                            edge_states[edge.id] = edge.state.name
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

                    # Create EnvironmentStateMsg
                    state = EnvironmentStateMsg(
                        step=self.env.current_step,
                        timestamp=time.time(),
                        nodes=nodes_state,
                        evacuees=[],
                        active_threats=[],
                        metrics={
                            "total_people": self.env.total_evacuees,
                            "evacuated": self.env.evacuated,
                            "casualties": self.env.casualties
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
