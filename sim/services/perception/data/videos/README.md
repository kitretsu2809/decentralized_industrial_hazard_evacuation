# Perception Service Sample Videos

To test the Perception Service, you can download sample footage representing various threats. Place the downloaded `.mp4` or `.avi` files in this directory.

## Fire & Smoke
- Search YouTube or Kaggle for "fire and smoke detection dataset"
- Ensure videos show distinct flames or thick smoke.
- *Creative Commons Search: "building fire footage CC"*

## Structural Collapse
- Look for disaster simulation videos or earthquake footage.
- The YOLO model will look for rubble or collapse-like classes if available in the model.
- *Keywords: "building collapse CCTV"*

## Wildlife
- Download videos containing wildlife (cats, dogs, bears, horses, etc.).
- The system correctly detects wildlife intrusions.
- *Example Dataset: COCO dataset snippets with animals.*

> **Note:** Intruder or generic person detections are ignored by design in the Perception Service, so there is no need for person-only videos unless testing false positive rejections.
