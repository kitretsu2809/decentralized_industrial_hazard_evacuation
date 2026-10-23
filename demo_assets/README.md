# Demonstration video assets

## `pscs_i_crowd_demo_12s.mp4`

This 12-second, 1280x720 clip is an excerpt from the official PSCS-I dataset
demonstration video published by the Multimodal Data Fusion and Analytics
Group. It is included only to demonstrate predictive analysis in the simulator:
YOLO person detection, crowd-density estimation, node-level hazard-vector
fusion, and routing input generation.

- Source: https://m4d.iti.gr/wp-content/uploads/2025/12/crowd_demo_v2.mp4
- Dataset page: https://m4d.iti.gr/pscs-i-dataset/
- Dataset: *Photorealistic Synthetic Crowds Simulation in Indoor environments
  (PSCS-I)*, Pasios et al., IEEE Access, 2025,
  https://doi.org/10.1109/ACCESS.2025.3636727
- Content: synthetic Unreal Engine 5 footage; no real people are depicted.
- Permitted project use: predictive processing of the original content, such
  as object detection and generation of classification/density tags. Review the
  dataset page and FAB EULA before redistribution or any different use.

The source demonstration is not used to train a generative model. The local
excerpt was resized and encoded as H.264 with its audio removed so it stays
well below the simulator's 250 MB upload limit.

The video is kept local and ignored by Git to avoid redistributing a third-party
dataset asset. Re-create it from the authors' public demonstration with:

```bash
curl -L --fail \
  --output /tmp/pscs-i-crowd-demo-source.mp4 \
  https://m4d.iti.gr/wp-content/uploads/2025/12/crowd_demo_v2.mp4
ffmpeg -y -i /tmp/pscs-i-crowd-demo-source.mp4 \
  -t 12 -vf "scale=1280:-2:flags=lanczos" -an \
  -c:v libx264 -preset medium -crf 23 -movflags +faststart \
  demo_assets/pscs_i_crowd_demo_12s.mp4
```

## Local YOLO prerequisite

Model weights are intentionally ignored by Git. Before demonstrating on a new
clone, download the official COCO-pretrained YOLO11n checkpoint:

```bash
mkdir -p data/models
curl -L --fail \
  --output data/models/yolo11n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

Then copy `.env.example` to `.env`, or add this setting to the existing `.env`:

```dotenv
YOLO_MODEL_PATH=data/models/yolo11n.pt
YOLO_DEVICE=cpu
ENABLE_HSV_FIRE_FALLBACK=false
```

The standard COCO checkpoint detects people. Fire and smoke require custom
weights. The colour-only HSV fallback is disabled by default because red
equipment and video overlays can cause false alarms; enable it only for a
clearly labelled heuristic demonstration.
