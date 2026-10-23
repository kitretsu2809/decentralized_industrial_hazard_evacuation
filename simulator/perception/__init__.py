"""Simulator-native perception adapters for recorded node camera feeds."""

from .video_pipeline import SimulatorPerceptionPipeline, build_hazard_vector

__all__ = ["SimulatorPerceptionPipeline", "build_hazard_vector"]
