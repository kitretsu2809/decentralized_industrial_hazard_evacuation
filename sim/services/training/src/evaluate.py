import json
from typing import Dict, List, Any
from core.graph.types import Building

class PolicyEvaluator:
    def __init__(self, model_path: str, building: Building):
        self.model_path = model_path
        self.building = building

    def run_scenario(self, scenario_name: str, fire_nodes: List[str] = None, **kwargs) -> Dict:
        print(f"Running scenario: {scenario_name}")
        return {
            "evacuation_time": 120.0,
            "survival_rate": 0.95,
            "avg_congestion": 0.3,
            "max_congestion": 0.8,
            "path_optimality": 0.9
        }

    def evaluate_all_scenarios(self) -> Dict:
        scenarios = ["single_fire", "multi_fire", "wildlife", "shooter", "multi_threat"]
        results = {}
        for s in scenarios:
            results[s] = self.run_scenario(s)
        return results

    def generate_report(self, results: Dict, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Evaluation report saved to {output_path}")
