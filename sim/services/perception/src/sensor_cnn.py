import numpy as np
from typing import Tuple, Dict, List

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class SensorCNN(nn.Module):
        def __init__(self, window_size: int = 50):
            super().__init__()
            # Input: (batch, 3, window_size)
            self.conv1 = nn.Conv1d(3, 16, kernel_size=5, padding=2)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.conv3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
            self.relu3 = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(16, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.relu3(self.conv3(x))
            x = self.pool(x).squeeze(-1)
            x = self.sigmoid(self.fc(x))
            return x

class SensorAnomalyDetector:
    def __init__(self, window_size: int = 50, use_nn: bool = True):
        self.window_size = window_size
        self.use_nn = use_nn and HAS_TORCH
        self.buffers: Dict[str, List[Tuple[float, float, float]]] = {} # node_id -> list of (gas, temp, hum)
        
        if self.use_nn:
            self.model = SensorCNN(window_size)
            self.model.eval()
            
    def add_reading(self, node_id: str, gas: float, temperature: float, humidity: float):
        if node_id not in self.buffers:
            self.buffers[node_id] = []
        self.buffers[node_id].append((gas, temperature, humidity))
        if len(self.buffers[node_id]) > self.window_size:
            self.buffers[node_id].pop(0)
            
    def detect_anomaly(self, node_id: str) -> Tuple[bool, float]:
        if node_id not in self.buffers or len(self.buffers[node_id]) < 2:
            return False, 0.0
            
        data = self.buffers[node_id]
        
        if self.use_nn and len(data) == self.window_size:
            # Prepare tensor (batch, channels, length)
            tensor_data = torch.tensor(data, dtype=torch.float32).T.unsqueeze(0)
            with torch.no_grad():
                pred = self.model(tensor_data).item()
            return pred > 0.5, pred
        else:
            # Threshold fallback
            latest = data[-1]
            gas, temp, _ = latest
            
            # rate of change for temp
            prev_temp = data[-2][1]
            temp_rate = temp - prev_temp
            
            is_anomaly = gas > 50.0 or temp > 50.0 or temp_rate > 5.0
            
            severity = 0.0
            if is_anomaly:
                gas_sev = min(1.0, max(0.0, (gas - 50.0) / 100.0))
                temp_sev = min(1.0, max(0.0, (temp - 50.0) / 50.0))
                rate_sev = min(1.0, max(0.0, (temp_rate - 5.0) / 10.0))
                severity = max(gas_sev, temp_sev, rate_sev)
                
            return is_anomaly, severity
            
    def get_hazard_score(self, node_id: str) -> float:
        is_anomaly, severity = self.detect_anomaly(node_id)
        if is_anomaly:
            return severity
        return 0.0
        
    def clear(self, node_id: str):
        if node_id in self.buffers:
            del self.buffers[node_id]
