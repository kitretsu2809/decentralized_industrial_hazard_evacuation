"""Kaggle GPU runner for the eight-direction ST-TBA-GAT policy.

Run from the repository root::

    python scripts/kaggle_train.py --episodes 100

The runner deliberately validates the saved model interface before creating the
download archive.  A successful archive therefore cannot silently contain an
old six-corridor checkpoint.
"""
import os
import sys
import argparse
import json
import subprocess
import zipfile
from pathlib import Path


EXPECTED_NODE_DIM = 51
EXPECTED_MAX_CORRIDORS = 8
EXPECTED_NUM_ACTIONS = 9


def validate_checkpoint(path: Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    training_meta = payload.get("extra_meta", {})
    actual = {
        "node_dim": int(payload.get("node_dim", -1)),
        "max_corridors": int(payload.get("max_corridors", -1)),
        "num_actions": int(payload.get("num_actions", -1)),
        "episode": int(training_meta.get("episode", -1)),
        "curriculum_stage": int(training_meta.get("curriculum_stage", -1)),
        "survival_rate": float(training_meta.get("survival_rate", -1.0)),
    }
    expected = {
        "node_dim": EXPECTED_NODE_DIM,
        "max_corridors": EXPECTED_MAX_CORRIDORS,
        "num_actions": EXPECTED_NUM_ACTIONS,
    }
    mismatches = {
        key: (actual[key], value)
        for key, value in expected.items()
        if actual[key] != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}={got} (expected {wanted})"
            for key, (got, wanted) in mismatches.items()
        )
        raise RuntimeError(f"Incompatible checkpoint {path}: {details}")
    return actual

def main():
    parser = argparse.ArgumentParser(description="Kaggle ST-TBA-GAT Training Runner")
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Final episode number (also when resuming)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda or cpu)")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/kaggle/working/checkpoints/directional",
        help="Output directory",
    )
    parser.add_argument("--min-evacuees", type=int, default=30, help="Minimum crowd headcount")
    parser.add_argument("--max-evacuees", type=int, default=400, help="Maximum crowd headcount")
    parser.add_argument("--ppo-batch-size", type=int, default=16, help="PPO rollout minibatch size")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume")
    parser.add_argument("--skip-eval", action="store_true", help="Skip benchmark after training")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 LAUNCHING KAGGLE ST-TBA-GAT MAPPO GPU TRAINING")
    print(f"   Final episode: {args.episodes} | Device: {args.device} | Output: {save_dir}")
    print("=" * 80)

    cmd = [
        sys.executable,
        "simulator/training/train_mappo.py",
        "--device", args.device,
        "--episodes", str(args.episodes),
        "--save-dir", str(save_dir),
        "--min-evacuees", str(args.min_evacuees),
        "--max-evacuees", str(args.max_evacuees),
        "--ppo-batch-size", str(args.ppo_batch_size),
        "--curriculum",
    ]
    if not args.skip_eval:
        cmd.append("--eval")
    if args.resume:
        cmd.extend(["--resume", str(Path(args.resume).expanduser().resolve())])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    ret = subprocess.run(cmd, env=env, cwd=repo_root)
    if ret.returncode != 0:
        print(f"❌ Training failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)

    best_path = save_dir / "best_policy.pt"
    latest_path = save_dir / "st_tba_gat_latest.pt"
    if not best_path.exists() or not latest_path.exists():
        raise FileNotFoundError("Training finished without both expected checkpoints")

    metadata = validate_checkpoint(best_path)
    validate_checkpoint(latest_path)
    summary_path = save_dir / "checkpoint_metadata.json"
    summary_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    archive_path = save_dir.parent.parent / "lbp-eight-direction-model.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for artifact in (best_path, latest_path, summary_path):
            archive.write(artifact, arcname=artifact.name)

    print("\n✅ Training complete and checkpoint interface validated:", metadata)
    print(f"   Download this file from Kaggle Output: {archive_path}")
    for artifact in (best_path, latest_path, summary_path, archive_path):
        print(f"   📁 {artifact} ({artifact.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
