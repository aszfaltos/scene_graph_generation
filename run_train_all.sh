#!/bin/bash
# Train and evaluate all VRD IMP Phase 1 configs
#
# Usage:
#   # Train all configs with defaults (30k iterations, 2 GPUs)
#   ./run_train_all.sh
#
#   # Train with custom iterations
#   ./run_train_all.sh --max-iter 50000
#
#   # Train single config
#   ./run_train_all.sh --config configs/e2e_relIMP_vrd_bert_pce.yaml
#
#   # Quick test run (1000 iterations)
#   ./run_train_all.sh --quick

set -e

# Defaults
NUM_GPUS=2
MAX_ITER=30000
BATCH_SIZE=8
OUTPUT_DIR="checkpoints/vrd_experiments"
RESULTS_DIR="results/vrd_imp_phase1"

# Parse arguments
EXTRA_ARGS=""
TRAIN_ALL=true
CONFIG_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MAX_ITER=1000
            CHECKPOINT_PERIOD=500
            VAL_PERIOD=500
            EXTRA_ARGS="$EXTRA_ARGS --checkpoint-period $CHECKPOINT_PERIOD --val-period $VAL_PERIOD"
            shift
            ;;
        --max-iter)
            MAX_ITER=$2
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE=$2
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS=$2
            shift 2
            ;;
        --config)
            CONFIG_FILE=$2
            TRAIN_ALL=false
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR=$2
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "======================================"
echo "VRD IMP Training Pipeline"
echo "======================================"
echo "GPUs: $NUM_GPUS"
echo "Max iterations: $MAX_ITER"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo "Results dir: $RESULTS_DIR"
echo "======================================"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$RESULTS_DIR"

# Build the command
CMD="torchrun --nproc_per_node=$NUM_GPUS train_eval_vrd.py"
CMD="$CMD --max-iter $MAX_ITER"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --results-dir $RESULTS_DIR"

if [ "$TRAIN_ALL" = true ]; then
    CMD="$CMD --all"
else
    CMD="$CMD --config-file $CONFIG_FILE"
fi

CMD="$CMD $EXTRA_ARGS"

echo "Running: $CMD"
echo "======================================"

# Run training
$CMD

echo "======================================"
echo "Training complete!"
echo "Results saved to: $RESULTS_DIR/summary.json"
echo "======================================"
