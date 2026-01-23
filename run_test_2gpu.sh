#!/bin/bash
# Run forward/backward test on 2 A100 GPUs with 128 samples (64 per GPU)
#
# Usage:
#   ./run_test_2gpu.sh configs/e2e_relIMP_vrd_word2vec_pce.yaml
#
# Or test all new configs:
#   ./run_test_2gpu.sh --all

set -e

CONFIG_FILE=${1:-"configs/e2e_relIMP_vrd_word2vec_pce.yaml"}
TOTAL_SAMPLES=${2:-128}
NUM_GPUS=2

if [ "$1" = "--all" ]; then
    echo "Testing all VRD IMP configs..."
    echo "=============================="

    CONFIGS=(
        # Baselines (no PCE)
        "configs/e2e_relIMP_vrd_no_semantics.yaml"
        "configs/e2e_relIMP_vrd_glove.yaml"
        "configs/e2e_relIMP_vrd_word2vec.yaml"
        "configs/e2e_relIMP_vrd_bert.yaml"
        "configs/e2e_relIMP_vrd_minilm.yaml"
        # With PCE (learnable gating)
        "configs/e2e_relIMP_vrd_no_semantics_pce.yaml"
        "configs/e2e_relIMP_vrd_glove_pce.yaml"
        "configs/e2e_relIMP_vrd_word2vec_pce.yaml"
        "configs/e2e_relIMP_vrd_bert_pce.yaml"
        "configs/e2e_relIMP_vrd_minilm_pce.yaml"
        # Alternative gating
        "configs/e2e_relIMP_vrd_bert_pce_sigmoid.yaml"
        "configs/e2e_relIMP_vrd_bert_pce_poly.yaml"
        "configs/e2e_relIMP_vrd_word2vec_pce_sigmoid.yaml"
        "configs/e2e_relIMP_vrd_word2vec_pce_poly.yaml"
    )

    FAILED=()
    PASSED=()

    for config in "${CONFIGS[@]}"; do
        if [ -f "$config" ]; then
            echo ""
            echo "Testing: $config"
            echo "----------------------------------------"
            if torchrun --nproc_per_node=$NUM_GPUS test_forward_backward.py \
                --config-file "$config" \
                --total-samples $TOTAL_SAMPLES; then
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
    echo "=============================="
    echo "SUMMARY"
    echo "=============================="
    echo "Passed: ${#PASSED[@]}"
    for cfg in "${PASSED[@]}"; do
        echo "  ✓ $cfg"
    done
    echo ""
    echo "Failed: ${#FAILED[@]}"
    for cfg in "${FAILED[@]}"; do
        echo "  ✗ $cfg"
    done

    if [ ${#FAILED[@]} -gt 0 ]; then
        exit 1
    fi
else
    echo "Running test with $NUM_GPUS GPUs on $TOTAL_SAMPLES samples"
    echo "Config: $CONFIG_FILE"
    echo "=============================="

    torchrun --nproc_per_node=$NUM_GPUS test_forward_backward.py \
        --config-file "$CONFIG_FILE" \
        --total-samples $TOTAL_SAMPLES
fi
