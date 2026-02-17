#!/bin/bash
##############################################################################
# Interactive Generation Benchmark
# Tests real-world prompt/response quality + tok/sec with llama-cli
# Runs a fixed prompt set across model sizes and quantisations
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
BENCH_START_TIME=$(date +%s)

mkdir -p "$MODEL_DIR" "$RESULTS_DIR"

##############################################################################
# Terminal colours
##############################################################################
BOLD="\033[1m"
DIM="\033[2m"
RESET="\033[0m"
RED="\033[1;31m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
BLUE="\033[1;34m"
MAGENTA="\033[1;35m"
CYAN="\033[1;36m"
WHITE="\033[1;37m"

##############################################################################
# Prompt set
##############################################################################
PROMPTS=(
    "Explain what a GPU is in one paragraph."
    "Write a Python function to check if a number is prime."
    "What are the main differences between TCP and UDP?"
    "Summarise the theory of relativity in simple terms."
    "Write a bash script that finds the largest file in a directory."
)

# Quant files per model size (same mapping style as run_benchmark.sh)
declare -A QUANT_FILES_Q4
QUANT_FILES_Q4["3B"]="qwen2.5-3b-instruct-q4_k_m.gguf"
QUANT_FILES_Q4["7B"]="qwen2.5-7b-instruct-q4_k_m.gguf"
QUANT_FILES_Q4["14B"]="qwen2.5-14b-instruct-q4_k_m.gguf"
QUANT_FILES_Q4["32B"]="qwen2.5-32b-instruct-q4_k_m.gguf"

declare -A QUANT_FILES_Q5
QUANT_FILES_Q5["3B"]="qwen2.5-3b-instruct-q5_k_m.gguf"
QUANT_FILES_Q5["7B"]="qwen2.5-7b-instruct-q5_k_m.gguf"
QUANT_FILES_Q5["14B"]="qwen2.5-14b-instruct-q5_k_m.gguf"
QUANT_FILES_Q5["32B"]="qwen2.5-32b-instruct-q5_k_m.gguf"

declare -A QUANT_FILES_Q8
QUANT_FILES_Q8["3B"]="qwen2.5-3b-instruct-q8_0.gguf"
QUANT_FILES_Q8["7B"]="qwen2.5-7b-instruct-q8_0.gguf"
QUANT_FILES_Q8["14B"]="qwen2.5-14b-instruct-q8_0.gguf"
QUANT_FILES_Q8["32B"]="qwen2.5-32b-instruct-q8_0.gguf"

# Same Q8 VRAM fit logic as run_benchmark.sh
declare -A VRAM_ESTIMATE_Q8
VRAM_ESTIMATE_Q8["3B"]=4
VRAM_ESTIMATE_Q8["7B"]=8
VRAM_ESTIMATE_Q8["14B"]=16
VRAM_ESTIMATE_Q8["32B"]=34
TOTAL_VRAM_GB=32

MODEL_SIZES=("3B" "7B" "14B" "32B")
TOTAL_MODELS=${#MODEL_SIZES[@]}
PROMPT_COUNT=${#PROMPTS[@]}
RUN_NUM=0
TOTAL_RUNS=0

# Pre-calculate total runs based on quant availability per model size.
for size in "${MODEL_SIZES[@]}"; do
    quant_count=2  # Q4 + Q5
    if [ "${VRAM_ESTIMATE_Q8[$size]}" -le "$TOTAL_VRAM_GB" ]; then
        quant_count=$((quant_count + 1))
    fi
    TOTAL_RUNS=$((TOTAL_RUNS + quant_count * PROMPT_COUNT))
done

##############################################################################
# Helper functions
##############################################################################

logfile() {
    echo "$1" >> "$LOG_FILE"
}

out() {
    echo -e "$1"
}

outlog() {
    echo -e "$1"
    echo "$2" >> "$LOG_FILE"
}

elapsed_time() {
    local now=$(date +%s)
    local diff=$((now - BENCH_START_TIME))
    local hours=$((diff / 3600))
    local mins=$(((diff % 3600) / 60))
    local secs=$((diff % 60))
    printf "%02d:%02d:%02d" $hours $mins $secs
}

draw_line() {
    local char="${1:-=}"
    printf '%0.s'"$char" {1..80}
    echo ""
}

draw_progress_bar() {
    local current=$1
    local total=$2
    local width=40
    local filled=$((current * width / total))
    local empty=$((width - filled))
    local pct=$((current * 100 / total))

    printf "${CYAN}["
    printf '%0.s#' $(seq 1 $filled 2>/dev/null) || true
    printf '%0.s-' $(seq 1 $empty 2>/dev/null) || true
    printf "] %3d%% ${DIM}(%d/%d)${RESET}" $pct $current $total
}

model_file_for_quant() {
    local size="$1"
    local quant="$2"

    case "$quant" in
        Q4_K_M) echo "${QUANT_FILES_Q4[$size]}" ;;
        Q5_K_M) echo "${QUANT_FILES_Q5[$size]}" ;;
        Q8_0)   echo "${QUANT_FILES_Q8[$size]}" ;;
        *)      echo "" ;;
    esac
}

copy_model() {
    local model_file="$1"
    local src="$USB_MODEL_DIR/$model_file"
    local dest="$MODEL_DIR/$model_file"

    if [ -f "$dest" ]; then
        outlog "  ${GREEN}OK${RESET}  Already present: ${WHITE}$model_file${RESET}" \
               "[OK] Already present: $model_file"
        return 0
    fi

    if [ ! -f "$src" ]; then
        outlog "  ${RED}ERROR${RESET}  Model not found on USB: ${WHITE}$src${RESET}" \
               "[ERROR] Model not found on USB: $src"
        return 1
    fi

    local file_size_gb
    local file_size_bytes
    file_size_bytes=$(stat -c%s "$src" 2>/dev/null || echo 0)
    file_size_gb=$(awk -v b="$file_size_bytes" 'BEGIN {printf "%.1f", b / 1073741824}')
    echo -ne "  ${YELLOW}COPYING${RESET}  $model_file ${DIM}(${file_size_gb} GB)${RESET} ... "
    logfile "[COPY] $model_file (${file_size_gb}GB) from USB drive"
    cp "$src" "$dest"
    out "${GREEN}done${RESET}"
    return 0
}

cleanup_models() {
    local size="$1"
    out "  ${DIM}Cleaning up ${size} model files...${RESET}"
    logfile "[CLEANUP] Removing ${size} model files"

    for f in "$MODEL_DIR"/qwen2.5-${size,,}b-instruct-*.gguf; do
        if [ -f "$f" ]; then
            rm -f "$f"
            logfile "[CLEANUP] Deleted: $(basename "$f")"
        fi
    done

    out "  ${GREEN}Cleanup done${RESET}"
    logfile "[CLEANUP] Done"
}

##############################################################################
# Opening banner
##############################################################################
clear
echo ""
out "${CYAN}$(draw_line)${RESET}"
out "${WHITE}${BOLD}"
echo "     ___                      ___  _____   ____                  _     "
echo "    / _ \\__      _____ _ __  |__ \\| ____| | __ )  ___ _ __   ___| |__  "
echo "   | | | \\ \\ /\\ / / _ \\ '_ \\   ) | |__   |  _ \\ / _ \\ '_ \\ / __| '_ \\ "
echo "   | |_| |\\ V  V /  __/ | | | / /|___ \\  | |_) |  __/ | | | (__| | | |"
echo "    \\__\\_\\ \\_/\\_/ \\___|_| |_||____|___/  |____/ \\___|_| |_|\\___|_| |_|"
out "${RESET}"
out "${CYAN}$(draw_line)${RESET}"
echo ""
out "  ${WHITE}${BOLD}GPU${RESET}          AMD Radeon AI PRO R9700"
out "  ${WHITE}${BOLD}Architecture${RESET} RDNA4 (gfx1201)"
out "  ${WHITE}${BOLD}VRAM${RESET}         32 GB"
out "  ${WHITE}${BOLD}ROCm${RESET}         7.2"
out "  ${WHITE}${BOLD}Backend${RESET}      llama.cpp (HIP/ROCm)"
out "  ${WHITE}${BOLD}Mode${RESET}         Interactive Prompt Benchmark"
out ""
out "${CYAN}$(draw_line '-')${RESET}"
out ""
out "  ${WHITE}${BOLD}Models${RESET}       Qwen2.5-3B / 7B / 14B / 32B Instruct GGUF"
out "  ${WHITE}${BOLD}Quants${RESET}       Q4_K_M, Q5_K_M, Q8_0 (where it fits in VRAM)"
out "  ${WHITE}${BOLD}Prompts${RESET}      ${PROMPT_COUNT} fixed prompts"
out "  ${WHITE}${BOLD}Runs${RESET}         ${TOTAL_RUNS} total generations"
out ""
out "  ${DIM}Results: $OUTPUT_FILE${RESET}"
out "  ${DIM}Log:     $LOG_FILE${RESET}"
out ""
out "${CYAN}$(draw_line)${RESET}"
echo ""
sleep 1

logfile "[INFO] Interactive benchmark started at $(date)"
logfile ""

##############################################################################
# Check dependencies
##############################################################################
out "  ${YELLOW}Checking dependencies...${RESET}"
echo ""

if [ ! -f "$LLAMA_CLI" ]; then
    out "  ${RED}ERROR: llama-cli not found at $LLAMA_CLI${RESET}"
    out "  ${RED}       Build llama.cpp with ROCm first.${RESET}"
    exit 1
fi
out "  ${GREEN}OK${RESET}  llama-cli found"

if [ ! -d "$USB_MODEL_DIR" ]; then
    out "  ${RED}ERROR: USB model directory not found at $USB_MODEL_DIR${RESET}"
    out "  ${RED}       Mount the USB drive first: sudo mount /dev/sda1 /mnt/usb${RESET}"
    exit 1
fi

USB_MODEL_COUNT=$(ls "$USB_MODEL_DIR"/*.gguf 2>/dev/null | wc -l)
out "  ${GREEN}OK${RESET}  USB drive mounted - ${USB_MODEL_COUNT} model files available"
echo ""
sleep 1

for model_num in "${!MODEL_SIZES[@]}"; do
    size="${MODEL_SIZES[$model_num]}"

    echo ""
    out "${CYAN}$(draw_line)${RESET}"
    out "  ${WHITE}${BOLD}MODEL $((model_num + 1))/${TOTAL_MODELS}: Qwen2.5-${size}-Instruct-GGUF${RESET}"
    out "  ${DIM}Elapsed: $(elapsed_time)${RESET}"
    out "${CYAN}$(draw_line '-')${RESET}"
    echo ""

    logfile "================================================================================"
    logfile " MODEL $((model_num + 1))/${TOTAL_MODELS}: Qwen2.5-${size}-Instruct-GGUF"
    logfile "================================================================================"

    quants_to_test=("Q4_K_M" "Q5_K_M")
    q8_vram="${VRAM_ESTIMATE_Q8[$size]}"
    if [ "$q8_vram" -le "$TOTAL_VRAM_GB" ]; then
        quants_to_test+=("Q8_0")
    else
        outlog "  ${YELLOW}NOTE${RESET}  Skipping Q8_0 - estimated ${q8_vram}GB exceeds ${TOTAL_VRAM_GB}GB VRAM" \
               "[SKIP] Q8_0 for ${size}: estimated ${q8_vram}GB exceeds ${TOTAL_VRAM_GB}GB VRAM"
    fi

    out "  Quantisations: ${MAGENTA}${quants_to_test[*]}${RESET}"
    echo ""

    out "  ${CYAN}--- Copying models from USB drive ---${RESET}"
    for quant in "${quants_to_test[@]}"; do
        model_file="$(model_file_for_quant "$size" "$quant")"
        if [ -z "$model_file" ]; then
            outlog "  ${RED}ERROR${RESET}  Unknown quant mapping: ${WHITE}$quant${RESET}" \
                   "[ERROR] Unknown quant mapping: $quant"
            continue
        fi
        copy_model "$model_file" || true
    done
    echo ""

    out "  ${CYAN}--- Running interactive prompts ---${RESET}"
    for quant in "${quants_to_test[@]}"; do
        model_file="$(model_file_for_quant "$size" "$quant")"
        model_path="$MODEL_DIR/$model_file"

        if [ ! -f "$model_path" ]; then
            outlog "  ${RED}SKIP${RESET}  Missing local model for ${quant}: ${WHITE}$model_file${RESET}" \
                   "[SKIP] Missing local model for ${quant}: $model_file"
            continue
        fi

        out "  ${WHITE}${BOLD}Quant: ${quant}${RESET}"
        logfile ""
        logfile "[QUANT] ${size} ${quant}"

        for i in "${!PROMPTS[@]}"; do
            prompt_num=$((i + 1))
            prompt="${PROMPTS[$i]}"
            RUN_NUM=$((RUN_NUM + 1))

            out "    ${MAGENTA}[${RUN_NUM}/${TOTAL_RUNS}]${RESET} ${WHITE}${BOLD}PROMPT ${prompt_num}/${PROMPT_COUNT}${RESET}"
            out "    ${DIM}${prompt}${RESET}"
            out "    ${DIM}Elapsed: $(elapsed_time)${RESET}"

            logfile "[PROMPT ${prompt_num}/${PROMPT_COUNT}] (${quant}) $prompt"

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
            elapsed_ms=$(((end_time - start_time) / 1000000))

            eval_speed=$(grep "eval time" "$RESULTS_DIR/.stderr_tmp" | tail -1 | grep -oP '[\d.]+\s+tokens per second' | head -1 || echo "N/A")
            prompt_speed=$(grep "prompt eval time" "$RESULTS_DIR/.stderr_tmp" | grep -oP '[\d.]+\s+tokens per second' | head -1 || echo "N/A")

            display_resp=$(echo "$response" | head -5)

            out "       ${BLUE}Response${RESET}  $(echo "$display_resp" | head -1)"
            out "       ${GREEN}Eval${RESET}      ${eval_speed}"
            out "       ${CYAN}Prompt${RESET}    ${prompt_speed}"
            out "       ${YELLOW}Time${RESET}      ${elapsed_ms}ms"
            echo -ne "       Progress: "
            draw_progress_bar $RUN_NUM $TOTAL_RUNS
            echo ""
            echo ""

            logfile "[RESPONSE] $display_resp"
            logfile "[STATS] Quant: ${quant} | Eval: $eval_speed | Prompt: $prompt_speed | Time: ${elapsed_ms}ms"

            python3 -c "
import json
entry = {
    'model_size': '${size}',
    'quant': '${quant}',
    'prompt_num': ${prompt_num},
    'prompt': '''${prompt}''',
    'response_preview': '''${display_resp}'''[:200],
    'eval_speed': '${eval_speed}',
    'prompt_speed': '${prompt_speed}',
    'total_ms': ${elapsed_ms}
}
print(json.dumps(entry))
" >> "$OUTPUT_FILE" 2>/dev/null || true
        done
    done

    cleanup_models "$size"
done

rm -f "$RESULTS_DIR/.stderr_tmp"

echo ""
out "${CYAN}$(draw_line)${RESET}"
out "  ${WHITE}${BOLD}INTERACTIVE BENCHMARK COMPLETE${RESET}"
out "  ${DIM}Total time: $(elapsed_time)${RESET}"
out ""
out "  ${DIM}Results saved to:${RESET}"
out "    JSONL  $OUTPUT_FILE"
out "    Log    $LOG_FILE"
out ""
out "${CYAN}$(draw_line)${RESET}"
echo ""

logfile ""
logfile "================================================================================"
logfile " Interactive benchmark complete - Total time: $(elapsed_time)"
logfile " Results: $OUTPUT_FILE"
logfile " Log:     $LOG_FILE"
logfile "================================================================================"
