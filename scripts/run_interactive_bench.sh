#!/bin/bash
##############################################################################
# Interactive Generation Benchmark
# Tests real-world prompt/response quality + tok/sec with llama-cli
# Runs a set of prompts through each model and captures timing
##############################################################################

set -e

cd /home/pendakwahteknologi/benchmark-rocm

LLAMA_CLI="/home/pendakwahteknologi/finetune-rocm-v0/llama.cpp/build/bin/llama-cli"
export LD_LIBRARY_PATH="/home/pendakwahteknologi/finetune-rocm-v0/llama.cpp/build/bin:$LD_LIBRARY_PATH"
MODEL_DIR="$(pwd)/models"
USB_MODEL_DIR="/mnt/usb/models"
RESULTS_DIR="$(pwd)/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$RESULTS_DIR/interactive_bench_${TIMESTAMP}.jsonl"
LOG_FILE="$RESULTS_DIR/interactive_bench_${TIMESTAMP}.log"

# Test prompts - mix of simple and complex
PROMPTS=(
    "Explain what a GPU is in one paragraph."
    "Write a Python function to check if a number is prime."
    "What are the main differences between TCP and UDP?"
    "Summarise the theory of relativity in simple terms."
    "Write a bash script that finds the largest file in a directory."
)

# Qwen2.5 GGUF Q4_K_M model files
declare -A QUANT_FILES
QUANT_FILES["3B_Q4"]="qwen2.5-3b-instruct-q4_k_m.gguf"
QUANT_FILES["7B_Q4"]="qwen2.5-7b-instruct-q4_k_m.gguf"
QUANT_FILES["14B_Q4"]="qwen2.5-14b-instruct-q4_k_m.gguf"
QUANT_FILES["32B_Q4"]="qwen2.5-32b-instruct-q4_k_m.gguf"

log() {
    echo "$1" | tee -a "$LOG_FILE"
}

echo "================================================================================"
echo " Interactive Generation Benchmark"
echo " AMD Radeon AI PRO R9700 • 32GB VRAM • ROCm 7.2"
echo "================================================================================"
echo ""
echo " Tests real prompts with Q4_K_M quant across all model sizes"
echo " Captures: response quality, tok/sec, time-to-first-token"
echo ""
echo " Output: $OUTPUT_FILE"
echo ""
echo "================================================================================"
echo ""

MODEL_SIZES=("3B" "7B" "14B" "32B")

for size in "${MODEL_SIZES[@]}"; do
    log "================================================================================"
    log " Testing Qwen2.5-${size}-Instruct (Q4_K_M)"
    log "================================================================================"

    model_file="${QUANT_FILES[${size}_Q4]}"
    model_path="$MODEL_DIR/$model_file"

    # Copy from USB drive if needed
    if [ ! -f "$model_path" ]; then
        if [ ! -f "$USB_MODEL_DIR/$model_file" ]; then
            log "[ERROR] Model not found on USB drive: $USB_MODEL_DIR/$model_file"
            continue
        fi
        file_size_gb=$(awk "BEGIN {printf \"%.1f\", $(stat -c%s "$USB_MODEL_DIR/$model_file") / 1073741824}")
        log "[COPY] $model_file (${file_size_gb}GB) from USB drive ..."
        cp "$USB_MODEL_DIR/$model_file" "$model_path"
    fi

    for i in "${!PROMPTS[@]}"; do
        prompt="${PROMPTS[$i]}"
        log ""
        log "   [PROMPT $((i+1))/${#PROMPTS[@]}] $prompt"
        log ""

        # Run llama-cli and capture stderr for timing stats
        start_time=$(date +%s%N)

        response=$("$LLAMA_CLI" \
            -m "$model_path" \
            -ngl 99 \
            -fa \
            -c 2048 \
            -n 256 \
            --temp 0.7 \
            -p "<|im_start|>user
${prompt}<|im_end|>
<|im_start|>assistant
" \
            --no-display-prompt \
            2>"$RESULTS_DIR/.stderr_tmp" || true)

        end_time=$(date +%s%N)
        elapsed_ms=$(( (end_time - start_time) / 1000000 ))

        # Parse timing from stderr
        eval_speed=$(grep "eval time" "$RESULTS_DIR/.stderr_tmp" | tail -1 | grep -oP '[\d.]+\s+tokens per second' | head -1 || echo "N/A")
        prompt_speed=$(grep "prompt eval time" "$RESULTS_DIR/.stderr_tmp" | grep -oP '[\d.]+\s+tokens per second' | head -1 || echo "N/A")
        total_tokens=$(grep "eval count" "$RESULTS_DIR/.stderr_tmp" | grep -oP '\d+' | head -1 || echo "0")

        # Truncate response for display
        display_resp=$(echo "$response" | head -5)
        log "   [RESPONSE] $display_resp"
        log "   [STATS] Eval: $eval_speed | Prompt: $prompt_speed | Time: ${elapsed_ms}ms"

        # Write JSONL
        python3 -c "
import json, sys
entry = {
    'model_size': '${size}',
    'quant': 'Q4_K_M',
    'prompt_num': $((i+1)),
    'prompt': '''${prompt}''',
    'response_preview': '''${display_resp}'''[:200],
    'eval_speed': '${eval_speed}',
    'prompt_speed': '${prompt_speed}',
    'total_ms': ${elapsed_ms},
}
print(json.dumps(entry))
" >> "$OUTPUT_FILE" 2>/dev/null || true

    done

    # Cleanup
    log ""
    log "[CLEANUP] Removing ${size} model files..."
    rm -f "$model_path"
    log "[CLEANUP] Done."
    log ""
done

rm -f "$RESULTS_DIR/.stderr_tmp"

log "================================================================================"
log " Interactive Benchmark Complete!"
log "================================================================================"
log ""
log " Results: $OUTPUT_FILE"
log " Log:     $LOG_FILE"
log ""
