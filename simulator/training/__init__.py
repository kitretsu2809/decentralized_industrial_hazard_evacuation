"""MAPPO training pipeline and benchmark evaluation for ST-TBA-GAT."""
from .train_mappo import MAPPOTrainer, run_benchmark

__all__ = ["MAPPOTrainer", "run_benchmark"]
