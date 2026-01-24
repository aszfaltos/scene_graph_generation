#!/bin/bash
# Train Faster R-CNN detector for VRD dataset
# Run this FIRST before training the relation model
#
# Usage:
#   ./run_train_vrd_detector.sh
#
# Output:
#   checkpoints/detection/vrd_detector/model_final.pth
#   Copy this to: checkpoints/detection/pretrained_faster_rcnn/vrd_faster_det.pth

set -e

NUM_GPUS=${NUM_GPUS:-2}
CONFIG="configs/detector_vrd.yaml"
OUTPUT_DIR="checkpoints/detection/vrd_detector"

echo "======================================"
echo "Training VRD Object Detector"
echo "======================================"
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "======================================"

mkdir -p "$OUTPUT_DIR"
mkdir -p "checkpoints/detection/pretrained_faster_rcnn"

# Train the detector
uv run torchrun --nproc_per_node=$NUM_GPUS tools/detector_pretrain_net.py \
    --config-file "$CONFIG" \
    OUTPUT_DIR "$OUTPUT_DIR"

# Copy final model to expected location
if [ -f "$OUTPUT_DIR/model_final.pth" ]; then
    cp "$OUTPUT_DIR/model_final.pth" "checkpoints/detection/pretrained_faster_rcnn/vrd_faster_det.pth"
    echo "======================================"
    echo "Detector training complete!"
    echo "Checkpoint saved to: checkpoints/detection/pretrained_faster_rcnn/vrd_faster_det.pth"
    echo "======================================"
else
    echo "ERROR: model_final.pth not found"
    exit 1
fi
