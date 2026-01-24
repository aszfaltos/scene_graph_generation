#!/bin/bash
# Run forward/backward test on 1 GPU (single GPU mode for debugging)
#
# Usage:
#   ./run_test_2gpu.sh configs/e2e_relIMP_vrd_word2vec_pce.yaml
#
# Test all new configs (training):
#   ./run_test_2gpu.sh --all
#
# Test all configs (training + evaluation):
#   ./run_test_2gpu.sh --all --eval

set -e

CONFIG_FILE=${1:-"configs/e2e_relIMP_vrd_word2vec_pce.yaml"}
TOTAL_SAMPLES=${2:-256}  # 128 per GPU with 2 GPUs
NUM_GPUS=2

# Check for --eval flag
RUN_EVAL=false
for arg in "$@"; do
    if [ "$arg" = "--eval" ]; then
        RUN_EVAL=true
    fi
done

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
    EVAL_FAILED=()
    EVAL_PASSED=()

    # Arrays for memory tracking
    declare -A TRAIN_MEMORY
    declare -A EVAL_MEMORY

    for config in "${CONFIGS[@]}"; do
        if [ -f "$config" ]; then
            echo ""
            echo "Testing (train): $config"
            echo "----------------------------------------"

            # Run training test and capture output
            output=$(uv run torchrun --nproc_per_node=$NUM_GPUS test_forward_backward.py \
                --config-file "$config" \
                --total-samples $TOTAL_SAMPLES 2>&1) && train_status=0 || train_status=$?

            echo "$output"

            if [ $train_status -eq 0 ]; then
                PASSED+=("$config")
                echo "PASSED (train): $config"

                # Extract GPU memory from output
                mem=$(echo "$output" | grep "GPU_MEMORY_GB:" | tail -1 | cut -d':' -f2)
                if [ -n "$mem" ]; then
                    TRAIN_MEMORY["$config"]="$mem"
                else
                    TRAIN_MEMORY["$config"]="N/A"
                fi

                # Run evaluation test if --eval flag is set
                if [ "$RUN_EVAL" = true ]; then
                    echo ""
                    echo "Testing (eval): $config"
                    echo "----------------------------------------"

                    eval_output=$(uv run torchrun --nproc_per_node=$NUM_GPUS test_forward_backward.py \
                        --config-file "$config" \
                        --total-samples 16 \
                        --mode eval 2>&1) && eval_status=0 || eval_status=$?

                    echo "$eval_output"

                    if [ $eval_status -eq 0 ]; then
                        EVAL_PASSED+=("$config")
                        echo "PASSED (eval): $config"

                        # Extract GPU memory from output
                        eval_mem=$(echo "$eval_output" | grep "GPU_MEMORY_GB:" | tail -1 | cut -d':' -f2)
                        if [ -n "$eval_mem" ]; then
                            EVAL_MEMORY["$config"]="$eval_mem"
                        else
                            EVAL_MEMORY["$config"]="N/A"
                        fi
                    else
                        EVAL_FAILED+=("$config")
                        echo "FAILED (eval): $config"
                        EVAL_MEMORY["$config"]="FAIL"
                    fi
                fi
            else
                FAILED+=("$config")
                echo "FAILED (train): $config"
                TRAIN_MEMORY["$config"]="FAIL"
            fi
        else
            echo "SKIPPING (not found): $config"
        fi
    done

    echo ""
    echo "=============================="
    echo "SUMMARY"
    echo "=============================="
    echo "Training Tests:"
    echo "  Passed: ${#PASSED[@]}"
    for cfg in "${PASSED[@]}"; do
        echo "    ✓ $cfg"
    done
    echo "  Failed: ${#FAILED[@]}"
    for cfg in "${FAILED[@]}"; do
        echo "    ✗ $cfg"
    done

    if [ "$RUN_EVAL" = true ]; then
        echo ""
        echo "Evaluation Tests:"
        echo "  Passed: ${#EVAL_PASSED[@]}"
        for cfg in "${EVAL_PASSED[@]}"; do
            echo "    ✓ $cfg"
        done
        echo "  Failed: ${#EVAL_FAILED[@]}"
        for cfg in "${EVAL_FAILED[@]}"; do
            echo "    ✗ $cfg"
        done
    fi

    # Print GPU memory table
    echo ""
    echo "=============================="
    echo "GPU MEMORY USAGE (GB)"
    echo "=============================="
    printf "%-55s | %8s" "Config" "Train"
    if [ "$RUN_EVAL" = true ]; then
        printf " | %8s" "Eval"
    fi
    printf "\n"
    printf "%s\n" "$(printf '%.0s-' {1..78})"

    for config in "${CONFIGS[@]}"; do
        if [ -f "$config" ]; then
            config_short=$(basename "$config")
            train_mem="${TRAIN_MEMORY[$config]:-N/A}"
            printf "%-55s | %8s" "$config_short" "$train_mem"
            if [ "$RUN_EVAL" = true ]; then
                eval_mem="${EVAL_MEMORY[$config]:-N/A}"
                printf " | %8s" "$eval_mem"
            fi
            printf "\n"
        fi
    done
    echo "=============================="

    if [ ${#FAILED[@]} -gt 0 ] || [ ${#EVAL_FAILED[@]} -gt 0 ]; then
        exit 1
    fi
else
    echo "Running test with $NUM_GPUS GPUs on $TOTAL_SAMPLES samples"
    echo "Config: $CONFIG_FILE"
    echo "=============================="

    uv run torchrun --nproc_per_node=$NUM_GPUS test_forward_backward.py \
        --config-file "$CONFIG_FILE" \
        --total-samples $TOTAL_SAMPLES
fi
