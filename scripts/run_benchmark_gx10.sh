#!/bin/bash
##############################################################################
# Qwen2.5 Benchmark Suite for NVIDIA GB10 (Project DIGITS / GX10)
# Tests tok/sec, prompt processing, and generation across model sizes & quants
# Models downloaded from HuggingFace into local models/ directory
##############################################################################

set -e

cd /home/gx10/benchmark-gx10

LLAMA_BENCH="/home/gx10/llama.cpp/build/bin/llama-bench"
LLAMA_CLI="/home/gx10/llama.cpp/build/bin/llama-cli"
export LD_LIBRARY_PATH="/home/gx10/llama.cpp/build/bin:$LD_LIBRARY_PATH"
MODEL_DIR="$(pwd)/models"
RESULTS_DIR="$(pwd)/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_CSV="$RESULTS_DIR/benchmark_${TIMESTAMP}.csv"
RESULTS_JSON="$RESULTS_DIR/benchmark_${TIMESTAMP}.json"
LOG_FILE="$RESULTS_DIR/benchmark_${TIMESTAMP}.log"
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
# Quant files per model size
##############################################################################
declare -A QUANT_FILES_Q4
QUANT_FILES_Q4["3B"]="qwen2.5-3b-instruct-q4_k_m.gguf"
QUANT_FILES_Q4["7B"]="qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
QUANT_FILES_Q4["14B"]="qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf"
QUANT_FILES_Q4["32B"]="qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf"

declare -A QUANT_FILES_Q5
QUANT_FILES_Q5["3B"]="qwen2.5-3b-instruct-q5_k_m.gguf"
QUANT_FILES_Q5["7B"]="qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf"
QUANT_FILES_Q5["14B"]="qwen2.5-14b-instruct-q5_k_m-00001-of-00003.gguf"
QUANT_FILES_Q5["32B"]="qwen2.5-32b-instruct-q5_k_m-00001-of-00006.gguf"

declare -A QUANT_FILES_Q8
QUANT_FILES_Q8["3B"]="qwen2.5-3b-instruct-q8_0.gguf"
QUANT_FILES_Q8["7B"]="qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf"
QUANT_FILES_Q8["14B"]="qwen2.5-14b-instruct-q8_0-00001-of-00004.gguf"
QUANT_FILES_Q8["32B"]="qwen2.5-32b-instruct-q8_0-00001-of-00009.gguf"

# GB10 has ~128GB unified memory - all models fit including Q8_0 for 32B
TOTAL_VRAM_GB=128

# Prompt lengths for benchmarking
PP_LENGTHS="128,256,512"
TG_LENGTH=128
N_REPS=3

MODEL_SIZES=("3B" "7B" "14B" "32B")
TOTAL_MODELS=${#MODEL_SIZES[@]}

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
    local mins=$(( (diff % 3600) / 60 ))
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

##############################################################################
# System info gathering
##############################################################################
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GB10")
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//' || echo "Unknown")
CPU_MODEL=$(lscpu | grep "Model name" | head -1 | sed 's/.*: *//')
RAM_TOTAL=$(free -h | awk '/Mem:/ {print $2}')

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
out "  ${WHITE}${BOLD}GPU${RESET}          ${GPU_NAME}"
out "  ${WHITE}${BOLD}Architecture${RESET} Blackwell (GB10)"
out "  ${WHITE}${BOLD}Memory${RESET}       ${RAM_TOTAL} Unified (CPU+GPU shared)"
out "  ${WHITE}${BOLD}CUDA${RESET}         ${CUDA_VER}"
out "  ${WHITE}${BOLD}Driver${RESET}       ${DRIVER_VER}"
out "  ${WHITE}${BOLD}CPU${RESET}          ${CPU_MODEL}"
out "  ${WHITE}${BOLD}Backend${RESET}      llama.cpp (CUDA)"
out "  ${WHITE}${BOLD}Flash Attn${RESET}   Enabled"
echo ""
out "${CYAN}$(draw_line '-')${RESET}"
echo ""
out "  ${WHITE}${BOLD}Models${RESET}       Qwen2.5-3B / 7B / 14B / 32B Instruct GGUF"
out "  ${WHITE}${BOLD}Quants${RESET}       Q4_K_M, Q5_K_M, Q8_0 (all fit in unified memory)"
out "  ${WHITE}${BOLD}Tests${RESET}        Prompt Processing (PP) at 128, 256, 512 tokens"
out "                Text Generation  (TG) at 128 tokens"
out "  ${WHITE}${BOLD}Repetitions${RESET}  ${N_REPS} per configuration (averaged)"
out "  ${WHITE}${BOLD}GPU Layers${RESET}   All offloaded (ngl=99)"
echo ""
out "  ${DIM}Results: $RESULTS_CSV${RESET}"
out "  ${DIM}Log:     $LOG_FILE${RESET}"
echo ""
out "${CYAN}$(draw_line)${RESET}"
echo ""
sleep 2

# Initialize JSON results
cat > "$RESULTS_JSON" << JSONEOF
{
  "metadata": {
    "gpu": "${GPU_NAME}",
    "arch": "Blackwell (GB10)",
    "memory_gb": "$(echo $RAM_TOTAL | sed 's/Gi//')",
    "memory_type": "Unified (CPU+GPU shared)",
    "cuda_version": "${CUDA_VER}",
    "driver_version": "${DRIVER_VER}",
    "cpu": "${CPU_MODEL}",
    "backend": "llama.cpp (CUDA)",
    "timestamp": "$(date -Iseconds)",
    "pp_lengths": "128,256,512",
    "tg_length": 128,
    "repetitions": 3
  },
  "results": []
}
JSONEOF

# Initialize CSV
echo "model_size,quant,pp_tokens,pp_tok_sec,tg_tokens,tg_tok_sec,model_file,vram_used_mb" > "$RESULTS_CSV"

logfile "[INFO] Benchmark started at $(date)"
logfile "[INFO] GPU: ${GPU_NAME}"
logfile "[INFO] CUDA: ${CUDA_VER} | Driver: ${DRIVER_VER}"
logfile "[INFO] CPU: ${CPU_MODEL}"
logfile "[INFO] Memory: ${RAM_TOTAL} Unified"
logfile ""

##############################################################################
# Check dependencies
##############################################################################
out "  ${YELLOW}Checking dependencies...${RESET}"
echo ""

if [ ! -f "$LLAMA_BENCH" ]; then
    out "  ${RED}ERROR: llama-bench not found at $LLAMA_BENCH${RESET}"
    out "  ${RED}       Build llama.cpp with CUDA first.${RESET}"
    exit 1
fi
out "  ${GREEN}OK${RESET}  llama-bench found"

# Check nvidia-smi
if ! command -v nvidia-smi &>/dev/null; then
    out "  ${YELLOW}WARN${RESET}  nvidia-smi not found — power monitoring disabled"
else
    out "  ${GREEN}OK${RESET}  nvidia-smi available"
fi

# Check models exist
MODEL_COUNT=$(ls "$MODEL_DIR"/*.gguf 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    out "  ${RED}ERROR: No .gguf model files found in $MODEL_DIR${RESET}"
    out "  ${RED}       Download models first.${RESET}"
    exit 1
fi
out "  ${GREEN}OK${RESET}  ${MODEL_COUNT} model files available"

FREE_DISK=$(df -h /home/gx10/ | tail -1 | awk '{print $4}')
out "  ${GREEN}OK${RESET}  Free disk space: ${FREE_DISK}"
echo ""
sleep 1

##############################################################################
# Run benchmark for a single model file
##############################################################################
BENCH_COUNT=0
# GB10 can run all quants for all sizes — total = 4 sizes x 3 quants = 12
TOTAL_BENCH_COUNT=$((TOTAL_MODELS * 3))

run_bench() {
    local model_size="$1"
    local quant="$2"
    local model_file="$3"
    local model_path="$MODEL_DIR/$model_file"

    BENCH_COUNT=$((BENCH_COUNT + 1))

    if [ ! -f "$model_path" ]; then
        outlog "         ${RED}SKIP${RESET}  File not found: $model_file" \
               "   [SKIP] File not found: $model_file"
        return 1
    fi

    local file_size_gb=$(awk "BEGIN {printf \"%.1f\", $(stat -c%s "$model_path") / 1073741824}")
    local elapsed=$(elapsed_time)

    echo ""
    out "     ${CYAN}[${BENCH_COUNT}/${TOTAL_BENCH_COUNT}]${RESET} ${WHITE}${BOLD}Qwen2.5-${model_size}-Instruct-GGUF${RESET} ${MAGENTA}${quant}${RESET} ${DIM}(${file_size_gb} GB)${RESET}"
    out "     ${DIM}PP: 128, 256, 512 tokens | TG: 128 tokens | Reps: ${N_REPS} | Elapsed: ${elapsed}${RESET}"
    out "     ${YELLOW}Running llama-bench...${RESET}"
    echo ""

    logfile ""
    logfile "   [BENCH] Qwen2.5-${model_size}-Instruct-GGUF ${quant} (${file_size_gb}GB)"
    logfile "           PP lengths: ${PP_LENGTHS} | TG: ${TG_LENGTH} | Reps: ${N_REPS}"
    logfile ""

    local bench_start=$(date +%s)

    # Run llama-bench with CSV output
    local bench_output
    bench_output=$("$LLAMA_BENCH" \
        -m "$model_path" \
        -p "$PP_LENGTHS" \
        -n "$TG_LENGTH" \
        -r "$N_REPS" \
        -ngl 99 \
        -fa 1 \
        -o csv \
        2>&1) || true

    local bench_end=$(date +%s)
    local bench_dur=$((bench_end - bench_start))

    # Log raw output to file only
    echo "$bench_output" >> "$LOG_FILE"
    logfile ""

    # Parse CSV output - strip quotes and extract correct fields
    # llama-bench CSV columns (0-indexed):
    #   31=n_prompt, 32=n_gen, 37=avg_ts (tokens/sec)
    echo "$bench_output" | grep -v "^build_commit\|^ggml\|^  Device\|^$" | \
        sed 's/"//g' | while IFS=',' read -r \
        f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 \
        f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 \
        f20 f21 f22 f23 f24 f25 f26 f27 f28 f29 \
        f30 n_prompt n_gen n_depth test_time avg_ns stddev_ns avg_ts stddev_ts; do

        if [[ "$n_prompt" =~ ^[0-9]+$ ]] && [ "$n_prompt" -gt 0 ] && [ "$n_gen" -eq 0 ]; then
            echo "${model_size},${quant},${n_prompt},${avg_ts},0,0,${model_file},0" >> "$RESULTS_CSV"
            printf "         ${BLUE}PP %4s tokens${RESET}  ->  ${WHITE}${BOLD}%10s tok/s${RESET}\n" "$n_prompt" "$avg_ts"
            logfile "         PP ${n_prompt} tokens -> ${avg_ts} tok/s"
        elif [[ "$n_gen" =~ ^[0-9]+$ ]] && [ "$n_gen" -gt 0 ] && [ "$n_prompt" -eq 0 ]; then
            echo "${model_size},${quant},0,0,${n_gen},${avg_ts},${model_file},0" >> "$RESULTS_CSV"
            printf "         ${GREEN}TG %4s tokens${RESET}  ->  ${WHITE}${BOLD}%10s tok/s${RESET}\n" "$n_gen" "$avg_ts"
            logfile "         TG ${n_gen} tokens -> ${avg_ts} tok/s"
        fi
    done 2>/dev/null || true

    echo ""
    outlog "     ${GREEN}DONE${RESET}  Qwen2.5-${model_size}-Instruct-GGUF ${quant} ${DIM}(${bench_dur}s)${RESET}" \
           "   [DONE] Qwen2.5-${model_size}-Instruct-GGUF ${quant} (${bench_dur}s)"
}

##############################################################################
# Main benchmark loop
##############################################################################
MODEL_NUM=0

for size in "${MODEL_SIZES[@]}"; do
    MODEL_NUM=$((MODEL_NUM + 1))

    echo ""
    out "${CYAN}$(draw_line)${RESET}"
    out "  ${WHITE}${BOLD}MODEL ${MODEL_NUM}/${TOTAL_MODELS}: Qwen2.5-${size}-Instruct-GGUF${RESET}"
    out "  ${DIM}Elapsed: $(elapsed_time)${RESET}"
    out "${CYAN}$(draw_line)${RESET}"
    echo ""

    logfile "================================================================================"
    logfile " MODEL ${MODEL_NUM}/${TOTAL_MODELS}: Qwen2.5-${size}-Instruct-GGUF"
    logfile "================================================================================"
    logfile ""

    # GB10 has enough unified memory for all quants
    quants_to_test=("Q4_K_M" "Q5_K_M" "Q8_0")

    out "     Quantizations: ${MAGENTA}${quants_to_test[*]}${RESET}"
    echo ""

    # Benchmark phase
    out "     ${CYAN}--- Running benchmarks ---${RESET}"
    for quant in "${quants_to_test[@]}"; do
        case "$quant" in
            Q4_K_M) run_bench "$size" "$quant" "${QUANT_FILES_Q4[$size]}" ;;
            Q5_K_M) run_bench "$size" "$quant" "${QUANT_FILES_Q5[$size]}" ;;
            Q8_0)   run_bench "$size" "$quant" "${QUANT_FILES_Q8[$size]}" ;;
        esac
    done

    # Progress bar
    echo ""
    echo -ne "  Overall progress: "
    draw_progress_bar $MODEL_NUM $TOTAL_MODELS
    echo ""
done

##############################################################################
# Print summary table
##############################################################################
echo ""
echo ""
out "${CYAN}$(draw_line)${RESET}"
out "${CYAN}$(draw_line)${RESET}"
echo ""
out "  ${WHITE}${BOLD}BENCHMARK COMPLETE${RESET}"
out "  ${DIM}Total time: $(elapsed_time)${RESET}"
echo ""
out "${CYAN}$(draw_line)${RESET}"
echo ""
out "  ${WHITE}${BOLD}${GPU_NAME}${RESET}  ${DIM}|${RESET}  Blackwell (GB10)  ${DIM}|${RESET}  ${RAM_TOTAL} Unified  ${DIM}|${RESET}  CUDA ${CUDA_VER}"
out "  llama.cpp (CUDA)   ${DIM}|${RESET}  Flash Attention ON   ${DIM}|${RESET}  All layers on GPU"
echo ""
out "${CYAN}$(draw_line '-')${RESET}"

logfile ""
logfile "================================================================================"
logfile " BENCHMARK COMPLETE - Total time: $(elapsed_time)"
logfile "================================================================================"
logfile ""
logfile " ${GPU_NAME} (Blackwell GB10) | ${RAM_TOTAL} Unified | CUDA ${CUDA_VER}"
logfile " llama.cpp (CUDA) | Flash Attention ON | All layers on GPU"
logfile ""

# -- Prompt Processing Table --
echo ""
out "  ${WHITE}${BOLD}PROMPT PROCESSING (tok/s)${RESET}"
out "  ${DIM}How fast the model reads your input prompt${RESET}"
echo ""

logfile "  PROMPT PROCESSING (tok/s)"
logfile "  -----------------------------------------------------------------------------------------------------------"

printf "  ${BOLD}%-35s %-10s %12s %12s %12s${RESET}\n" "Model" "Quant" "PP 128" "PP 256" "PP 512"
out "  ${DIM}$(printf '%0.s-' {1..93})${RESET}"

printf "  %-35s %-10s %12s %12s %12s\n" "Model" "Quant" "PP 128" "PP 256" "PP 512" >> "$LOG_FILE"
logfile "  -----------------------------------------------------------------------------------------------------------"

for size in "${MODEL_SIZES[@]}"; do
    for quant in Q4_K_M Q5_K_M Q8_0; do
        pp128=$(awk -F',' -v s="$size" -v q="$quant" '$1==s && $2==q && $3=="128" {printf "%.1f", $4}' "$RESULTS_CSV")
        pp256=$(awk -F',' -v s="$size" -v q="$quant" '$1==s && $2==q && $3=="256" {printf "%.1f", $4}' "$RESULTS_CSV")
        pp512=$(awk -F',' -v s="$size" -v q="$quant" '$1==s && $2==q && $3=="512" {printf "%.1f", $4}' "$RESULTS_CSV")
        if [ -n "$pp128" ] || [ -n "$pp256" ] || [ -n "$pp512" ]; then
            printf "  ${WHITE}%-35s${RESET} ${MAGENTA}%-10s${RESET} %12s %12s %12s\n" \
                "Qwen2.5-${size}-Instruct-GGUF" "$quant" \
                "${pp128:---}" "${pp256:---}" "${pp512:---}"
            printf "  %-35s %-10s %12s %12s %12s\n" \
                "Qwen2.5-${size}-Instruct-GGUF" "$quant" \
                "${pp128:---}" "${pp256:---}" "${pp512:---}" >> "$LOG_FILE"
        fi
    done
done

out "  ${DIM}$(printf '%0.s-' {1..93})${RESET}"
logfile "  -----------------------------------------------------------------------------------------------------------"

# -- Text Generation Table --
echo ""
out "  ${WHITE}${BOLD}TEXT GENERATION (tok/s)${RESET}"
out "  ${DIM}How fast the model writes output tokens - the main speed you feel${RESET}"
echo ""

logfile ""
logfile "  TEXT GENERATION (tok/s)"
logfile "  -----------------------------------------------------------------------------------------------------------"

printf "  ${BOLD}%-35s %-10s %12s${RESET}\n" "Model" "Quant" "TG 128"
out "  ${DIM}$(printf '%0.s-' {1..60})${RESET}"

printf "  %-35s %-10s %12s\n" "Model" "Quant" "TG 128" >> "$LOG_FILE"
logfile "  -----------------------------------------------------------------------------------------------------------"

for size in "${MODEL_SIZES[@]}"; do
    for quant in Q4_K_M Q5_K_M Q8_0; do
        tg=$(awk -F',' -v s="$size" -v q="$quant" '$1==s && $2==q && $5=="128" {printf "%.1f", $6}' "$RESULTS_CSV")
        if [ -n "$tg" ]; then
            printf "  ${WHITE}%-35s${RESET} ${MAGENTA}%-10s${RESET} ${GREEN}${BOLD}%12s${RESET}\n" \
                "Qwen2.5-${size}-Instruct-GGUF" "$quant" "$tg"
            printf "  %-35s %-10s %12s\n" \
                "Qwen2.5-${size}-Instruct-GGUF" "$quant" "$tg" >> "$LOG_FILE"
        fi
    done
done

out "  ${DIM}$(printf '%0.s-' {1..60})${RESET}"
logfile "  -----------------------------------------------------------------------------------------------------------"

# -- Peak values --
PEAK_PP=$(awk -F',' 'NR>1 && $4+0>0 {if($4+0>max) max=$4+0} END {printf "%.1f", max}' "$RESULTS_CSV")
PEAK_TG=$(awk -F',' 'NR>1 && $6+0>0 {if($6+0>max) max=$6+0} END {printf "%.1f", max}' "$RESULTS_CSV")
PEAK_PP_MODEL=$(awk -F',' 'NR>1 && $4+0>0 {if($4+0>max) {max=$4+0; m=$1; q=$2}} END {print "Qwen2.5-"m"-Instruct-GGUF "q}' "$RESULTS_CSV")
PEAK_TG_MODEL=$(awk -F',' 'NR>1 && $6+0>0 {if($6+0>max) {max=$6+0; m=$1; q=$2}} END {print "Qwen2.5-"m"-Instruct-GGUF "q}' "$RESULTS_CSV")

echo ""
out "${CYAN}$(draw_line '-')${RESET}"
echo ""
out "  ${WHITE}${BOLD}PEAK PERFORMANCE${RESET}"
echo ""
out "  ${BLUE}Prompt Processing${RESET}   ${WHITE}${BOLD}${PEAK_PP} tok/s${RESET}   ${DIM}(${PEAK_PP_MODEL})${RESET}"
out "  ${GREEN}Text Generation${RESET}     ${WHITE}${BOLD}${PEAK_TG} tok/s${RESET}   ${DIM}(${PEAK_TG_MODEL})${RESET}"
echo ""

logfile ""
logfile "  PEAK PROMPT PROCESSING:  ${PEAK_PP} tok/s  (${PEAK_PP_MODEL})"
logfile "  PEAK TEXT GENERATION:    ${PEAK_TG} tok/s  (${PEAK_TG_MODEL})"
logfile ""

out "${CYAN}$(draw_line)${RESET}"
echo ""
out "  ${DIM}Results saved to:${RESET}"
out "    CSV   $RESULTS_CSV"
out "    JSON  $RESULTS_JSON"
out "    Log   $LOG_FILE"
echo ""
out "  ${DIM}Generate HTML report:${RESET}"
out "    python3 scripts/generate_report_gx10.py"
echo ""
out "${CYAN}$(draw_line)${RESET}"
echo ""

logfile "================================================================================"
logfile " Results saved to:"
logfile "   CSV:  $RESULTS_CSV"
logfile "   JSON: $RESULTS_JSON"
logfile "   Log:  $LOG_FILE"
logfile "================================================================================"
