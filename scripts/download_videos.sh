#!/bin/bash
# Download sample threat detection videos
# These are placeholder instructions - user should source their own videos
set -e

VIDEO_DIR="sim/services/perception/data/videos"
mkdir -p "$VIDEO_DIR"

echo "========================================"
echo "  Sample Video Download Guide"
echo "========================================"
echo ""
echo "Place test videos in: $VIDEO_DIR/"
echo ""
echo "Recommended sources:"
echo "  Fire/Smoke: Search 'fire detection dataset' on Kaggle"
echo "  Debris:     Search 'structural damage video dataset'"
echo "  Wildlife:   Any animal footage from YouTube (Creative Commons)"
echo ""
echo "Expected filenames:"
echo "  $VIDEO_DIR/fire_test.mp4"
echo "  $VIDEO_DIR/debris_test.mp4"
echo "  $VIDEO_DIR/wildlife_test.mp4"
echo ""
echo "Note: You can also use your webcam feed for live testing."
