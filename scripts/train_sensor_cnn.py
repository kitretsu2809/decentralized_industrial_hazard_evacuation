#!/usr/bin/env python3
"""Train and evaluate SensorCNN on an episode-safe telemetry dataset.

Example:
    python scripts/train_sensor_cnn.py \
      --dataset data/sensor_cnn_synthetic.npz \
      --checkpoint data/models/sensor_cnn.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.services.perception.src.sensor_cnn import SensorCNN  # noqa: E402


def normalize(X: np.ndarray) -> np.ndarray:
    """Use the same feature scaling as SensorAnomalyDetector."""
    result = X.astype(np.float32, copy=True)
    result[:, 0, :] = np.clip(result[:, 0, :] / 200.0, 0.0, 2.0)
    result[:, 1, :] = np.clip((result[:, 1, :] - 20.0) / 80.0, -0.25, 2.0)
    result[:, 2, :] = np.clip(result[:, 2, :] / 100.0, 0.0, 1.0)
    return result


def split_by_episode(episode_ids: np.ndarray, seed: int) -> Tuple[np.ndarray, ...]:
    """Split whole episodes so adjacent windows cannot leak across partitions."""
    unique = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = max(1, int(len(unique) * 0.70))
    n_val = max(1, int(len(unique) * 0.15))
    train_ids = set(unique[:n_train].tolist())
    val_ids = set(unique[n_train:n_train + n_val].tolist())
    test_ids = set(unique[n_train + n_val:].tolist())
    return tuple(np.asarray([eid in group for eid in episode_ids])
                 for group in (train_ids, val_ids, test_ids))


def metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    truth, scores = [], []
    with torch.no_grad():
        for xb, yb in loader:
            scores.extend(model(xb.to(device)).squeeze(1).cpu().numpy().tolist())
            truth.extend(yb.numpy().tolist())
    y_true = np.asarray(truth, dtype=np.int64)
    y_score = np.asarray(scores)
    y_pred = (y_score >= 0.5).astype(np.int64)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {"accuracy": (tp + tn) / max(1, len(y_true)),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-8, precision + recall),
            "false_positive_rate": fp / max(1, fp + tn),
            "false_negative_rate": fn / max(1, fn + tp),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def train(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    payload = np.load(args.dataset)
    X = normalize(payload["X"])
    y = payload["y"].astype(np.float32)
    episode_ids = payload["episode_ids"]
    train_mask, val_mask, test_mask = split_by_episode(episode_ids, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SensorCNN(window_size=X.shape[-1]).to(device)
    train_y = torch.from_numpy(y[train_mask])
    # SensorCNN currently applies sigmoid, so use BCELoss-compatible targets.
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loaders = {
        "train": DataLoader(TensorDataset(torch.from_numpy(X[train_mask]), train_y),
                             batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(TensorDataset(torch.from_numpy(X[val_mask]), torch.from_numpy(y[val_mask])),
                           batch_size=args.batch_size),
        "test": DataLoader(TensorDataset(torch.from_numpy(X[test_mask]), torch.from_numpy(y[test_mask])),
                            batch_size=args.batch_size),
    }

    best_val = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb.to(device)).squeeze(1), yb.to(device))
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_losses = []
            for xb, yb in loaders["val"]:
                val_losses.append(float(loss_fn(model(xb.to(device)).squeeze(1), yb.to(device))))
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % max(1, args.log_every) == 0:
            print(f"epoch={epoch} val_loss={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    result = {name: metrics(model, loader, device) for name, loader in loaders.items() if name != "train"}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "window_size": X.shape[-1],
                "feature_names": ["gas_ppm", "temperature_c", "humidity_pct"],
                "normalization": "sensor_cnn_v1", "metrics": result}, args.checkpoint)
    args.checkpoint.with_suffix(".metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved checkpoint: {args.checkpoint} ({device})")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("data/models/sensor_cnn.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
