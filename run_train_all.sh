#!/bin/bash
# Train and evaluate all VRD IMP Phase 1 configs
#
# Usage:
#   # Train all configs with defaults (2 GPUs per job)
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
MAX_ITER=30000
BATCH_SIZE=12  # Per GPU, so 24 total with 2 GPUs
NUM_GPUS=2
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
echo "Max iterations: $MAX_ITER"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "GPUs: $NUM_GPUS"
echo "Output dir: $OUTPUT_DIR"
echo "Results dir: $RESULTS_DIR"
echo "======================================"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "logs"

# Get all config files
if [ "$TRAIN_ALL" = true ]; then
    CONFIGS=(
        "configs/e2e_relIMP_vrd_glove.yaml"
        "configs/e2e_relIMP_vrd_glove_pce.yaml"
        "configs/e2e_relIMP_vrd_bert.yaml"
        "configs/e2e_relIMP_vrd_bert_pce.yaml"
    )
else
    CONFIGS=("$CONFIG_FILE")
fi

# Function to run a single training job with multi-GPU
run_training() {
    local config=$1
    local config_name=$(basename "$config" .yaml)
    local log_file="logs/${config_name}.log"

    echo ""
    echo "======================================"
    echo "Training: $config_name"
    echo "Log: $log_file"
    echo "======================================"

    uv run torchrun --nproc_per_node=$NUM_GPUS tools/relation_train_net.py \
        --config-file "$config" \
        SOLVER.IMS_PER_BATCH $BATCH_SIZE \
        SOLVER.MAX_ITER $MAX_ITER \
        OUTPUT_DIR "${OUTPUT_DIR}/${config_name}" \
        DATALOADER.NUM_WORKERS 0 \
        $EXTRA_ARGS 2>&1 | tee "$log_file"

    return $?
}

# Train all configs sequentially (each uses 2 GPUs)
FAILED=()
PASSED=()

for config in "${CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        if run_training "$config"; then
            PASSED+=("$config")
            echo "PASSED: $config"
        else
            FAILED+=("$config")
            echo "FAILED: $config"
        fi
    else
        echo "SKIPPING (not found): $config"
    fi
done

echo ""
echo "======================================"
echo "TRAINING SUMMARY"
echo "======================================"
echo "Passed: ${#PASSED[@]}"
for cfg in "${PASSED[@]}"; do
    echo "  ✓ $cfg"
done
echo "Failed: ${#FAILED[@]}"
for cfg in "${FAILED[@]}"; do
    echo "  ✗ $cfg"
done
echo "======================================"
echo "Results saved to: $OUTPUT_DIR"

if [ ${#FAILED[@]} -gt 0 ]; then
    exit 1
fi
