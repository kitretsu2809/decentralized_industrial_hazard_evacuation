import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False
    logger.warning("onnxruntime is not installed. Inference will not be available.")

class PolicyInference:
    """ONNX Runtime inference wrapper for running trained policy on any device."""

    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.session = None
        self._available = False
        self._last_latency = 0.0

        if HAS_ONNXRUNTIME:
            try:
                providers = ['CPUExecutionProvider']
                if device == 'gpu':
                    providers = ['CUDAExecutionProvider'] + providers
                
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                self._available = True
            except Exception as e:
                logger.error(f"Failed to load ONNX model from {model_path}: {e}")
                self._available = False
        else:
            self._available = False

    def is_available(self) -> bool:
        """Whether the model is successfully loaded and ready for inference."""
        return self._available

    def infer(self, node_features: np.ndarray, neighbor_features: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
        """Run a single inference, returning action probabilities."""
        if not self.is_available():
            raise RuntimeError("Inference session is not available.")
        
        start_time = time.time()
        
        input_name_1 = self.session.get_inputs()[0].name
        input_name_2 = self.session.get_inputs()[1].name
        input_name_3 = self.session.get_inputs()[2].name
        
        inputs = {
            input_name_1: node_features,
            input_name_2: neighbor_features,
            input_name_3: edge_mask
        }
        
        outputs = self.session.run(None, inputs)
        
        end_time = time.time()
        self._last_latency = (end_time - start_time) * 1000.0  # ms
        
        return outputs[0]

    def infer_batch(self, batch_features: List[np.ndarray], batch_neighbors: List[np.ndarray], batch_masks: List[np.ndarray]) -> List[np.ndarray]:
        """Run batched inference."""
        if not self.is_available():
            raise RuntimeError("Inference session is not available.")
            
        start_time = time.time()
        
        input_name_1 = self.session.get_inputs()[0].name
        input_name_2 = self.session.get_inputs()[1].name
        input_name_3 = self.session.get_inputs()[2].name
        
        stacked_features = np.stack(batch_features)
        stacked_neighbors = np.stack(batch_neighbors)
        stacked_masks = np.stack(batch_masks)
        
        inputs = {
            input_name_1: stacked_features,
            input_name_2: stacked_neighbors,
            input_name_3: stacked_masks
        }
        
        outputs = self.session.run(None, inputs)
        
        end_time = time.time()
        self._last_latency = (end_time - start_time) * 1000.0  # ms
        
        # Return list of results for each item in the batch
        return list(outputs[0])

    def get_latency_ms(self) -> float:
        """Get the latency of the last inference call in ms."""
        return self._last_latency

    def benchmark(self, n_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference performance over N runs."""
        if not self.is_available():
            return {}
            
        latencies = []
        
        # Dummy inputs based on expected shape (assuming some standard shape here)
        input_shapes = [inp.shape for inp in self.session.get_inputs()]
        
        try:
            dummy_f = np.zeros([shape if isinstance(shape, int) else 1 for shape in input_shapes[0]], dtype=np.float32)
            dummy_n = np.zeros([shape if isinstance(shape, int) else 1 for shape in input_shapes[1]], dtype=np.float32)
            dummy_m = np.zeros([shape if isinstance(shape, int) else 1 for shape in input_shapes[2]], dtype=np.float32)
            
            for _ in range(n_runs):
                self.infer(dummy_f, dummy_n, dummy_m)
                latencies.append(self.get_latency_ms())
                
            return {
                'avg_ms': float(np.mean(latencies)),
                'min_ms': float(np.min(latencies)),
                'max_ms': float(np.max(latencies))
            }
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {}
