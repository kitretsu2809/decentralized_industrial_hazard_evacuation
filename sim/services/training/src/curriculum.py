from typing import Dict, List

class CurriculumScheduler:
    def __init__(self, stages: List[Dict]):
        self.stages = stages
        self.current_stage_idx = 0

    def get_current_stage(self, total_episodes: int) -> Dict:
        episodes_passed = 0
        for idx, stage in enumerate(self.stages):
            episodes_passed += stage.get("episodes", 1000)
            if total_episodes < episodes_passed:
                self.current_stage_idx = idx
                return stage
        self.current_stage_idx = len(self.stages) - 1
        return self.stages[-1]

    def get_env_config(self, stage: Dict) -> Dict:
        return {
            "floors": stage.get("floors", 1),
            "threats": stage.get("threats", 1),
            "crowd_ratio": stage.get("crowd_ratio", 0.5)
        }

    def is_stage_complete(self, total_episodes: int) -> bool:
        episodes_passed = 0
        for idx, stage in enumerate(self.stages):
            episodes_passed += stage.get("episodes", 1000)
            if total_episodes == episodes_passed:
                return True
        return False

    def get_progress(self) -> Dict:
        return {
            "current_stage": self.current_stage_idx,
            "total_stages": len(self.stages),
            "stage_progress": 0.0,  # Computed if we track total episodes
            "overall_progress": 0.0 # Computed
        }
