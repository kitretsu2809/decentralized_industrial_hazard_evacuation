"""
Kaggle GPU Training Script for ST-TBA-GAT MARL Evacuation Policy.

Run this script directly in a Kaggle notebook cell or bash session:
    python scripts/kaggle_train.py --episodes 100 --save-dir /kaggle/working/checkpoints
"""
import os
import sys
import shutil
import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Kaggle ST-TBA-GAT Training Runner")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda or cpu)")
    parser.add_argument("--save-dir", type=str, default="/kaggle/working/checkpoints", help="Output directory")
    parser.add_argument("--min-evacuees", type=int, default=30, help="Minimum crowd headcount")
    parser.add_argument("--max-evacuees", type=int, default=400, help="Maximum crowd headcount")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 80)
    print("🚀 LAUNCHING KAGGLE ST-TBA-GAT MAPPO GPU TRAINING")
    print(f"   Episodes: {args.episodes} | Device: {args.device} | Output: {args.save_dir}")
    print("=" * 80)

    cmd = [
        sys.executable,
        "simulator/training/train_mappo.py",
        "--device", args.device,
        "--episodes", str(args.episodes),
        "--save-dir", args.save_dir,
        "--min-evacuees", str(args.min_evacuees),
        "--max-evacuees", str(args.max_evacuees),
        "--curriculum",
        "--eval",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        print(f"❌ Training failed with exit code {ret.returncode}")
        sys.exit(ret.returncode)

    # Copy ONNX if generated
    if os.path.exists("data/models/policy.onnx"):
        shutil.copy("data/models/policy.onnx", os.path.join(args.save_dir, "policy.onnx"))

    print("\n✅ Training complete! Generated artifacts in", args.save_dir)
    for fname in os.listdir(args.save_dir):
        fpath = os.path.join(args.save_dir, fname)
        print(f"   📁 {fname} ({os.path.getsize(fpath)/1024:.1f} KB)")

if __name__ == "__main__":
    main()
