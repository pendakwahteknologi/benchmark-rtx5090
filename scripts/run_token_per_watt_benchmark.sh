#!/bin/bash
##############################################################################
# Token-Per-Watt Benchmark Suite for AMD Radeon AI PRO R9700
# Measures generation throughput efficiency: tok/s, avg watts, tok/W, J/token
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

##############################################################################
# Paths and benchmark configuration
##############################################################################
DEFAULT_LLAMA_BENCH_CANDIDATES=(
    "/home/pendakwahteknologi/finetune-rocm-v0/llama.cpp/build/bin/llama-bench"
    "/home/pendakwahteknologi/llama.cpp/build/bin/llama-bench"
)

if [ -n "${LLAMA_BENCH:-}" ]; then
    LLAMA_BENCH="$LLAMA_BENCH"
else
    LLAMA_BENCH=""
    for candidate in "${DEFAULT_LLAMA_BENCH_CANDIDATES[@]}"; do
        if [ -x "$candidate" ]; then
            LLAMA_BENCH="$candidate"
            break
        fi
    done
    if [ -z "$LLAMA_BENCH" ] && command -v llama-bench >/dev/null 2>&1; then
        LLAMA_BENCH="$(command -v llama-bench)"
    fi
    if [ -z "$LLAMA_BENCH" ]; then
        LLAMA_BENCH="${DEFAULT_LLAMA_BENCH_CANDIDATES[0]}"
    fi
fi

LLAMA_BIN_DIR="$(dirname "$LLAMA_BENCH")"
export LD_LIBRARY_PATH="${LLAMA_BIN_DIR}:${LD_LIBRARY_PATH:-}"

MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/models}"
USB_MODEL_DIR="${USB_MODEL_DIR:-/mnt/usb/models}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/results}"
mkdir -p "$MODEL_DIR" "$RESULTS_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_CSV="$RESULTS_DIR/token_per_watt_${TIMESTAMP}.csv"
RESULTS_JSON="$RESULTS_DIR/token_per_watt_${TIMESTAMP}.json"
LOG_FILE="$RESULTS_DIR/token_per_watt_${TIMESTAMP}.log"
BENCH_START_TIME="$(date +%s)"

TG_LENGTH="${TG_LENGTH:-512}"
N_REPS="${N_REPS:-5}"
POWER_SAMPLE_INTERVAL_SEC="${POWER_SAMPLE_INTERVAL_SEC:-0.5}"
GPU_INDEX="${GPU_INDEX:-0}"
USD_TO_MYR="${USD_TO_MYR:-4.70}"
if [ -n "${ELECTRICITY_COST_MYR_PER_KWH:-}" ]; then
    ELECTRICITY_COST_MYR_PER_KWH="$ELECTRICITY_COST_MYR_PER_KWH"
    COST_RATE_SOURCE="myr_direct"
elif [ -n "${ELECTRICITY_COST_USD_PER_KWH:-}" ]; then
    ELECTRICITY_COST_MYR_PER_KWH="$(awk -v usd="$ELECTRICITY_COST_USD_PER_KWH" -v fx="$USD_TO_MYR" 'BEGIN { printf "%.6f", usd * fx }')"
    COST_RATE_SOURCE="usd_converted"
else
    ELECTRICITY_COST_MYR_PER_KWH="0.55"
    COST_RATE_SOURCE="myr_default"
fi
TOTAL_VRAM_GB="${TOTAL_VRAM_GB:-32}"

DEFAULT_MODEL_SIZES=("3B" "7B" "14B" "32B")
if [ -n "${MODEL_SIZES_CSV:-}" ]; then
    IFS=',' read -r -a MODEL_SIZES <<< "$MODEL_SIZES_CSV"
else
    MODEL_SIZES=("${DEFAULT_MODEL_SIZES[@]}")
fi

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

# Estimated model sizes in GB for VRAM fit check (approximate)
declare -A VRAM_ESTIMATE_Q8
VRAM_ESTIMATE_Q8["3B"]=4
VRAM_ESTIMATE_Q8["7B"]=8
VRAM_ESTIMATE_Q8["14B"]=16
VRAM_ESTIMATE_Q8["32B"]=34

##############################################################################
# Helpers
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
    local now
    now="$(date +%s)"
    local diff=$((now - BENCH_START_TIME))
    local hours=$((diff / 3600))
    local mins=$(((diff % 3600) / 60))
    local secs=$((diff % 60))
    printf "%02d:%02d:%02d" "$hours" "$mins" "$secs"
}

draw_line() {
    local char="${1:-=}"
    printf '%0.s'"$char" {1..80}
    echo ""
}

is_number() {
    [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

is_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
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

get_gpu_power_w() {
    rocm-smi -d "$GPU_INDEX" --showpower 2>/dev/null | \
        awk '/Power \(W\)/ {gsub(/[^0-9.]/, "", $NF); if ($NF != "") {print $NF; exit}}'
}

sample_power_during_pid() {
    local target_pid="$1"
    local sample_file="$2"
    : > "$sample_file"

    while kill -0 "$target_pid" 2>/dev/null; do
        local p
        p="$(get_gpu_power_w || true)"
        if is_number "${p:-}"; then
            echo "$p" >> "$sample_file"
        fi
        sleep "$POWER_SAMPLE_INTERVAL_SEC"
    done
}

summarise_power_samples() {
    local sample_file="$1"
    awk '
        BEGIN { sum = 0; n = 0; min = ""; max = "" }
        /^[0-9]+([.][0-9]+)?$/ {
            p = $1 + 0;
            sum += p;
            n++;
            if (min == "" || p < min) min = p;
            if (max == "" || p > max) max = p;
        }
        END {
            if (n == 0) {
                printf "0,0,0,0";
            } else {
                printf "%.6f,%.6f,%.6f,%d", sum / n, min, max, n;
            }
        }
    ' "$sample_file"
}

extract_bench_tok_sec() {
    local bench_output="$1"
    local expected_tokens="$2"

    echo "$bench_output" | sed 's/"//g' | awk -F',' -v expected="$expected_tokens" '
        NF < 38 { next }
        $32 !~ /^[0-9]+$/ || $33 !~ /^[0-9]+$/ { next }
        {
            np = $32 + 0
            ng = $33 + 0
            ts = $38 + 0
            if (ts <= 0) next

            # Preferred: explicit TG row
            if (np == 0 && ng == expected) {
                printf "%.6f,tg_exact", ts
                found = 1
                exit
            }

            # Compatibility: some builds emit TG in n_prompt with n_gen=0
            if (np == expected && ng == 0 && compat_exact == "") {
                compat_exact = sprintf("%.6f,tg_compat_prompt_column", ts)
            }

            # Fallback candidates
            if (np == 0 && ng > 0 && compat_tg_any == "") {
                compat_tg_any = sprintf("%.6f,tg_nonzero_n_gen", ts)
            }
            if (np > 0 && ng == 0 && compat_prompt_any == "") {
                compat_prompt_any = sprintf("%.6f,prompt_column_only", ts)
            }
            if (first_any == "") {
                first_any = sprintf("%.6f,first_numeric_row", ts)
            }
        }
        END {
            if (found == 1) {
                exit
            } else if (compat_exact != "") {
                print compat_exact
            } else if (compat_tg_any != "") {
                print compat_tg_any
            } else if (compat_prompt_any != "") {
                print compat_prompt_any
            } else if (first_any != "") {
                print first_any
            }
        }
    '
}

copy_model() {
    local filename="$1"
    local src="$USB_MODEL_DIR/$filename"
    local dest="$MODEL_DIR/$filename"

    if [ -f "$dest" ]; then
        outlog "         ${GREEN}OK${RESET}  Already present: ${WHITE}$filename${RESET}" \
               "   [OK] Already present: $filename"
        return 0
    fi

    if [ ! -f "$src" ]; then
        outlog "         ${RED}ERROR${RESET}  Model not found on USB: $src" \
               "   [ERROR] Model not found on USB drive: $src"
        return 1
    fi

    local size_gb
    size_gb="$(awk "BEGIN {printf \"%.1f\", $(stat -c%s "$src") / 1073741824}")"
    echo -ne "         ${YELLOW}COPYING${RESET}  $filename ${DIM}(${size_gb} GB)${RESET} ... "
    logfile "   [COPY] $filename (${size_gb}GB) from USB drive ..."
    cp "$src" "$dest"

    if [ -f "$dest" ]; then
        out "${GREEN}done${RESET}"
        logfile "   [OK] Copied: $filename (${size_gb}GB)"
        return 0
    fi

    out "${RED}failed${RESET}"
    logfile "   [ERROR] Failed to copy: $filename"
    return 1
}

cleanup_models() {
    local size="$1"
    echo ""
    out "     ${DIM}Cleaning up ${size} model files to free disk space...${RESET}"
    logfile ""
    logfile "[CLEANUP] Removing $size model files to free disk space..."

    for f in "$MODEL_DIR"/qwen2.5-${size,,}b-instruct-*.gguf; do
        if [ -f "$f" ]; then
            local fname
            local fsize
            fname="$(basename "$f")"
            fsize="$(du -h "$f" | cut -f1)"
            rm -f "$f"
            logfile "   [DELETED] $fname ($fsize)"
        fi
    done

    local free
    free="$(df -h "$ROOT_DIR" | tail -1 | awk '{print $4}')"
    out "     ${DIM}Done. Free disk: ${free}${RESET}"
    logfile "[CLEANUP] Done. Free disk: $free"
    logfile ""
}

##############################################################################
# Opening banner
##############################################################################
clear
echo ""
out "${CYAN}$(draw_line)${RESET}"
out "${WHITE}${BOLD}TOKEN-PER-WATT BENCHMARK${RESET}"
out "${CYAN}$(draw_line)${RESET}"
echo ""
out "  ${WHITE}${BOLD}GPU${RESET}             AMD Radeon AI PRO R9700"
out "  ${WHITE}${BOLD}Benchmark${RESET}       TG-only throughput vs power draw"
out "  ${WHITE}${BOLD}TG Tokens${RESET}       ${TG_LENGTH}"
out "  ${WHITE}${BOLD}Repetitions${RESET}     ${N_REPS}"
out "  ${WHITE}${BOLD}Power Sampling${RESET}  every ${POWER_SAMPLE_INTERVAL_SEC}s (GPU index ${GPU_INDEX})"
out "  ${WHITE}${BOLD}Model Sizes${RESET}     ${MODEL_SIZES[*]}"
out "  ${WHITE}${BOLD}Electricity${RESET}     RM ${ELECTRICITY_COST_MYR_PER_KWH}/kWh"
echo ""
out "  ${DIM}Results: $RESULTS_CSV${RESET}"
out "  ${DIM}Log:     $LOG_FILE${RESET}"
echo ""
out "  ${WHITE}${BOLD}How to read this benchmark${RESET}"
out "  ${DIM}- tok/s: higher = faster generation${RESET}"
out "  ${DIM}- tok/W: higher = better efficiency${RESET}"
out "  ${DIM}- sec/1k tok and RM/1k tok: lower = better${RESET}"
if [ "$COST_RATE_SOURCE" = "usd_converted" ]; then
    out "  ${DIM}Using converted rate from USD: ${ELECTRICITY_COST_USD_PER_KWH} USD/kWh x ${USD_TO_MYR} = RM ${ELECTRICITY_COST_MYR_PER_KWH}/kWh${RESET}"
fi
echo ""
out "${CYAN}$(draw_line)${RESET}"
echo ""

logfile "[INFO] Token-per-watt benchmark started at $(date)"
logfile "[INFO] Config: TG_LENGTH=${TG_LENGTH}, N_REPS=${N_REPS}, POWER_SAMPLE_INTERVAL_SEC=${POWER_SAMPLE_INTERVAL_SEC}, GPU_INDEX=${GPU_INDEX}, MODEL_SIZES=${MODEL_SIZES[*]}, ELECTRICITY_COST_MYR_PER_KWH=${ELECTRICITY_COST_MYR_PER_KWH}, COST_RATE_SOURCE=${COST_RATE_SOURCE}, USD_TO_MYR=${USD_TO_MYR}"
logfile ""

##############################################################################
# Dependency checks
##############################################################################
out "  ${YELLOW}Checking dependencies...${RESET}"
echo ""

if [ ! -x "$LLAMA_BENCH" ]; then
    out "  ${RED}ERROR: llama-bench not found/executable at $LLAMA_BENCH${RESET}"
    out "  ${RED}       Set LLAMA_BENCH or edit this script path.${RESET}"
    exit 1
fi
out "  ${GREEN}OK${RESET}  llama-bench found"

if ! command -v rocm-smi >/dev/null 2>&1; then
    out "  ${RED}ERROR: rocm-smi not found in PATH.${RESET}"
    out "  ${RED}       Install ROCm SMI tools first.${RESET}"
    exit 1
fi
out "  ${GREEN}OK${RESET}  rocm-smi found"

if [ ! -d "$USB_MODEL_DIR" ]; then
    out "  ${RED}ERROR: USB model directory not found at $USB_MODEL_DIR${RESET}"
    out "  ${RED}       Mount the USB drive first (or set USB_MODEL_DIR).${RESET}"
    exit 1
fi
USB_MODEL_COUNT=$(find "$USB_MODEL_DIR" -maxdepth 1 -type f -name '*.gguf' | wc -l)
out "  ${GREEN}OK${RESET}  USB model directory available - ${USB_MODEL_COUNT} model files"

if ! is_number "$ELECTRICITY_COST_MYR_PER_KWH"; then
    out "  ${RED}ERROR: ELECTRICITY_COST_MYR_PER_KWH must be numeric.${RESET}"
    exit 1
fi
if ! awk -v v="$ELECTRICITY_COST_MYR_PER_KWH" 'BEGIN { exit !(v > 0) }'; then
    out "  ${RED}ERROR: ELECTRICITY_COST_MYR_PER_KWH must be > 0.${RESET}"
    exit 1
fi
if [ "$COST_RATE_SOURCE" = "usd_converted" ] && ! is_number "$USD_TO_MYR"; then
    out "  ${RED}ERROR: USD_TO_MYR must be numeric when using USD conversion.${RESET}"
    exit 1
fi

if ! is_integer "$TG_LENGTH" || [ "$TG_LENGTH" -le 0 ]; then
    out "  ${RED}ERROR: TG_LENGTH must be a positive integer.${RESET}"
    exit 1
fi

if ! is_integer "$N_REPS" || [ "$N_REPS" -le 0 ]; then
    out "  ${RED}ERROR: N_REPS must be a positive integer.${RESET}"
    exit 1
fi

if ! is_number "$POWER_SAMPLE_INTERVAL_SEC"; then
    out "  ${RED}ERROR: POWER_SAMPLE_INTERVAL_SEC must be numeric.${RESET}"
    exit 1
fi
if ! awk -v v="$POWER_SAMPLE_INTERVAL_SEC" 'BEGIN { exit !(v > 0) }'; then
    out "  ${RED}ERROR: POWER_SAMPLE_INTERVAL_SEC must be > 0.${RESET}"
    exit 1
fi

if ! is_integer "$GPU_INDEX"; then
    out "  ${RED}ERROR: GPU_INDEX must be a non-negative integer.${RESET}"
    exit 1
fi

if [ "${#MODEL_SIZES[@]}" -eq 0 ]; then
    out "  ${RED}ERROR: MODEL_SIZES_CSV resolved to empty list.${RESET}"
    exit 1
fi
for size in "${MODEL_SIZES[@]}"; do
    case "$size" in
        3B|7B|14B|32B) ;;
        *)
            out "  ${RED}ERROR: Invalid model size '${size}'. Use: 3B,7B,14B,32B${RESET}"
            exit 1
            ;;
    esac
done
echo ""

##############################################################################
# Initialize CSV
##############################################################################
echo "model_size,quant,tg_tokens,tg_tok_sec,avg_power_w,min_power_w,max_power_w,power_samples,tokens_per_watt,joules_per_token,sec_per_1k_tokens,wh_per_1k_tokens,kwh_per_1m_tokens,rm_per_1k_tokens,rm_per_1m_tokens,electricity_cost_myr_per_kwh,model_file,bench_seconds" > "$RESULTS_CSV"

##############################################################################
# Benchmark counters
##############################################################################
BENCH_COUNT=0
TOTAL_BENCH_COUNT=0
FAILED_BENCHES=0
for size in "${MODEL_SIZES[@]}"; do
    TOTAL_BENCH_COUNT=$((TOTAL_BENCH_COUNT + 2))
    q8_vram="${VRAM_ESTIMATE_Q8[$size]}"
    if [ "$q8_vram" -le "$TOTAL_VRAM_GB" ]; then
        TOTAL_BENCH_COUNT=$((TOTAL_BENCH_COUNT + 1))
    fi
done

run_power_bench() {
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

    local file_size_gb
    file_size_gb="$(awk "BEGIN {printf \"%.1f\", $(stat -c%s "$model_path") / 1073741824}")"

    echo ""
    out "     ${CYAN}[${BENCH_COUNT}/${TOTAL_BENCH_COUNT}]${RESET} ${WHITE}${BOLD}Qwen2.5-${model_size}-Instruct-GGUF${RESET} ${MAGENTA}${quant}${RESET} ${DIM}(${file_size_gb} GB)${RESET}"
    out "     ${DIM}TG: ${TG_LENGTH} tokens | Reps: ${N_REPS} | Sampling: ${POWER_SAMPLE_INTERVAL_SEC}s | Elapsed: $(elapsed_time)${RESET}"
    out "     ${YELLOW}Running benchmark + power sampler...${RESET}"
    echo ""

    logfile ""
    logfile "   [BENCH] Qwen2.5-${model_size}-Instruct-GGUF ${quant} (${file_size_gb}GB)"
    logfile "           TG: ${TG_LENGTH} tokens | Reps: ${N_REPS} | Sampling: ${POWER_SAMPLE_INTERVAL_SEC}s"
    logfile ""

    local bench_tmp power_tmp
    bench_tmp="$(mktemp)"
    power_tmp="$(mktemp)"

    local bench_start bench_end bench_dur
    bench_start="$(date +%s)"

    "$LLAMA_BENCH" \
        -m "$model_path" \
        -p 0 \
        -n "$TG_LENGTH" \
        -r "$N_REPS" \
        -ngl 99 \
        -fa 1 \
        -o csv > "$bench_tmp" 2>&1 &
    local bench_pid=$!

    sample_power_during_pid "$bench_pid" "$power_tmp" &
    local sampler_pid=$!

    local bench_rc=0
    if wait "$bench_pid"; then
        bench_rc=0
    else
        bench_rc=$?
    fi
    wait "$sampler_pid" || true

    bench_end="$(date +%s)"
    bench_dur=$((bench_end - bench_start))

    local bench_output
    bench_output="$(cat "$bench_tmp")"
    rm -f "$bench_tmp"

    echo "$bench_output" >> "$LOG_FILE"
    logfile ""

    if [ "$bench_rc" -ne 0 ]; then
        outlog "     ${RED}ERROR${RESET}  llama-bench failed for Qwen2.5-${model_size} ${quant}" \
               "   [ERROR] llama-bench failed for Qwen2.5-${model_size} ${quant}"
        rm -f "$power_tmp"
        return 1
    fi

    local parse_info tg_tok_sec parse_mode
    parse_info="$(extract_bench_tok_sec "$bench_output" "$TG_LENGTH" || true)"
    IFS=',' read -r tg_tok_sec parse_mode <<< "${parse_info:-,}"
    if ! is_number "${tg_tok_sec:-}"; then
        outlog "     ${RED}ERROR${RESET}  Could not parse TG tok/s from llama-bench output" \
               "   [ERROR] Could not parse TG tok/s from llama-bench output"
        rm -f "$power_tmp"
        return 1
    fi
    parse_mode="${parse_mode:-unknown}"

    local avg_power_w min_power_w max_power_w power_samples
    IFS=',' read -r avg_power_w min_power_w max_power_w power_samples < <(summarise_power_samples "$power_tmp")
    rm -f "$power_tmp"

    if [ "$power_samples" -eq 0 ]; then
        local fallback_power
        fallback_power="$(get_gpu_power_w || true)"
        if is_number "${fallback_power:-}"; then
            avg_power_w="$fallback_power"
            min_power_w="$fallback_power"
            max_power_w="$fallback_power"
            power_samples=1
        fi
    fi

    local tokens_per_watt joules_per_token sec_per_1k_tokens wh_per_1k_tokens kwh_per_1m_tokens rm_per_1k_tokens rm_per_1m_tokens
    tokens_per_watt="$(awk -v t="$tg_tok_sec" -v p="$avg_power_w" 'BEGIN { if (p > 0) printf "%.6f", t / p; else printf "0" }')"
    joules_per_token="$(awk -v t="$tg_tok_sec" -v p="$avg_power_w" 'BEGIN { if (t > 0) printf "%.6f", p / t; else printf "0" }')"
    sec_per_1k_tokens="$(awk -v t="$tg_tok_sec" 'BEGIN { if (t > 0) printf "%.6f", 1000 / t; else printf "0" }')"
    wh_per_1k_tokens="$(awk -v p="$avg_power_w" -v s="$sec_per_1k_tokens" 'BEGIN { printf "%.6f", (p * s) / 3600 }')"
    kwh_per_1m_tokens="$(awk -v j="$joules_per_token" 'BEGIN { printf "%.6f", (j * 1000000) / 3600000 }')"
    rm_per_1m_tokens="$(awk -v k="$kwh_per_1m_tokens" -v c="$ELECTRICITY_COST_MYR_PER_KWH" 'BEGIN { printf "%.6f", k * c }')"
    rm_per_1k_tokens="$(awk -v r="$rm_per_1m_tokens" 'BEGIN { printf "%.6f", r / 1000 }')"

    echo "${model_size},${quant},${TG_LENGTH},${tg_tok_sec},${avg_power_w},${min_power_w},${max_power_w},${power_samples},${tokens_per_watt},${joules_per_token},${sec_per_1k_tokens},${wh_per_1k_tokens},${kwh_per_1m_tokens},${rm_per_1k_tokens},${rm_per_1m_tokens},${ELECTRICITY_COST_MYR_PER_KWH},${model_file},${bench_dur}" >> "$RESULTS_CSV"

    printf "         ${GREEN}TG %4s tokens${RESET}  ->  ${WHITE}${BOLD}%10.2f tok/s${RESET}\n" "$TG_LENGTH" "$tg_tok_sec"
    printf "         ${DIM}Parser mode: %s${RESET}\n" "$parse_mode"
    printf "         ${BLUE}Avg Power${RESET}       ->  ${WHITE}${BOLD}%10.2f W${RESET} ${DIM}(samples: %s, min %.2fW, max %.2fW)${RESET}\n" "$avg_power_w" "$power_samples" "$min_power_w" "$max_power_w"
    printf "         ${MAGENTA}Efficiency${RESET}      ->  ${WHITE}${BOLD}%10.3f tok/W${RESET}\n" "$tokens_per_watt"
    printf "         ${CYAN}Latency${RESET}         ->  ${WHITE}${BOLD}%10.2f sec / 1k tokens${RESET}\n" "$sec_per_1k_tokens"
    printf "         ${YELLOW}Energy${RESET}          ->  ${WHITE}${BOLD}%10.4f Wh / 1k tokens${RESET}\n" "$wh_per_1k_tokens"
    printf "         ${YELLOW}Cost${RESET}            ->  ${WHITE}${BOLD}%10.6f RM / 1k${RESET} ${DIM}|${RESET} ${WHITE}${BOLD}%10.4f RM / 1M${RESET}\n" "$rm_per_1k_tokens" "$rm_per_1m_tokens"

    logfile "         TG ${TG_LENGTH} tokens -> ${tg_tok_sec} tok/s"
    logfile "         Parser mode -> ${parse_mode}"
    logfile "         Avg Power -> ${avg_power_w}W (samples=${power_samples}, min=${min_power_w}, max=${max_power_w})"
    logfile "         Efficiency -> ${tokens_per_watt} tok/W | ${joules_per_token} J/token"
    logfile "         Time -> ${sec_per_1k_tokens} sec per 1k tokens"
    logfile "         Energy -> ${wh_per_1k_tokens} Wh per 1k tokens | ${kwh_per_1m_tokens} kWh per 1M tokens"
    logfile "         Estimated cost -> ${rm_per_1k_tokens} RM per 1k tokens | ${rm_per_1m_tokens} RM per 1M tokens @ ${ELECTRICITY_COST_MYR_PER_KWH} RM/kWh"
    logfile ""
}

##############################################################################
# Main benchmark loop
##############################################################################
MODEL_NUM=0
TOTAL_MODELS="${#MODEL_SIZES[@]}"

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

    quants_to_test=("Q4_K_M" "Q5_K_M")
    q8_vram="${VRAM_ESTIMATE_Q8[$size]}"
    if [ "$q8_vram" -le "$TOTAL_VRAM_GB" ]; then
        quants_to_test+=("Q8_0")
    else
        outlog "     ${YELLOW}NOTE${RESET}  Skipping Q8_0 — estimated ${q8_vram}GB exceeds ${TOTAL_VRAM_GB}GB VRAM" \
               "   [SKIP] Q8_0 for ${size}: estimated ${q8_vram}GB exceeds ${TOTAL_VRAM_GB}GB VRAM"
    fi

    out "     Quantizations: ${MAGENTA}${quants_to_test[*]}${RESET}"
    echo ""

    out "     ${CYAN}--- Copying models from USB drive ---${RESET}"
    logfile "[COPY] Copying Qwen2.5-${size} model files from USB drive..."
    for quant in "${quants_to_test[@]}"; do
        copy_model "$(model_file_for_quant "$size" "$quant")"
    done
    echo ""

    out "     ${CYAN}--- Running token-per-watt benchmark ---${RESET}"
    for quant in "${quants_to_test[@]}"; do
        if ! run_power_bench "$size" "$quant" "$(model_file_for_quant "$size" "$quant")"; then
            FAILED_BENCHES=$((FAILED_BENCHES + 1))
            outlog "     ${YELLOW}WARN${RESET}  Benchmark failed for ${size} ${quant}; continuing..." \
                   "   [WARN] Benchmark failed for ${size} ${quant}; continuing..."
        fi
    done

    cleanup_models "$size"
done

##############################################################################
# Summary table
##############################################################################
echo ""
out "${CYAN}$(draw_line)${RESET}"
out "  ${WHITE}${BOLD}TOKEN-PER-WATT BENCHMARK COMPLETE${RESET}"
out "  ${DIM}Total time: $(elapsed_time)${RESET}"
out "${CYAN}$(draw_line '-')${RESET}"
echo ""
out "  ${WHITE}${BOLD}SUMMARY (sorted by tok/W)${RESET}"
out ""

DATA_ROWS=$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "$RESULTS_CSV")
SUCCESS_BENCHES="$DATA_ROWS"
ATTEMPTED_BENCHES=$((SUCCESS_BENCHES + FAILED_BENCHES))

out "  Attempted: ${ATTEMPTED_BENCHES}   Succeeded: ${SUCCESS_BENCHES}   Failed: ${FAILED_BENCHES}"
echo ""

if [ "$DATA_ROWS" -gt 0 ]; then
    printf "  ${BOLD}%-4s %-23s %-8s %9s %9s %9s %10s %11s${RESET}\n" "#" "Model" "Quant" "tok/s" "Avg W" "tok/W" "sec/1k" "RM/1k"
    out "  ${DIM}$(printf '%0.s-' {1..102})${RESET}"

    RANK=0
    tail -n +2 "$RESULTS_CSV" | sort -t',' -k9,9gr | while IFS=',' read -r \
        model_size quant tg_tokens tg_tok_sec avg_power_w min_power_w max_power_w power_samples \
        tokens_per_watt joules_per_token sec_per_1k_tokens wh_per_1k_tokens kwh_per_1m_tokens \
        rm_per_1k_tokens rm_per_1m_tokens electricity_cost_myr_per_kwh model_file bench_seconds; do
        RANK=$((RANK + 1))
        printf "  %-4s %-23s %-8s %9.2f %9.2f %9.3f %10.2f %11.6f\n" \
            "${RANK}" "Qwen2.5-${model_size}" "$quant" "$tg_tok_sec" "$avg_power_w" "$tokens_per_watt" "$sec_per_1k_tokens" "$rm_per_1k_tokens"
    done

    out "  ${DIM}$(printf '%0.s-' {1..102})${RESET}"
    echo ""

    FASTEST_ROW="$(awk -F',' 'NR>1 && $4+0>max {max=$4+0; r=$0} END {print r}' "$RESULTS_CSV")"
    EFFICIENT_ROW="$(awk -F',' 'NR>1 && $9+0>max {max=$9+0; r=$0} END {print r}' "$RESULTS_CSV")"
    CHEAPEST_ROW="$(awk -F',' 'NR>1 && (min=="" || $14+0<min) {min=$14+0; r=$0} END {print r}' "$RESULTS_CSV")"

    if [ -n "$FASTEST_ROW" ]; then
        IFS=',' read -r m q _ tg _ _ _ _ _ _ sec1k _ _ rm1k rm1m _ _ _ <<< "$FASTEST_ROW"
        out "  ${GREEN}${BOLD}Fastest:${RESET} Qwen2.5-${m} ${q} -> ${tg} tok/s (${sec1k} sec/1k)"
    fi
    if [ -n "$EFFICIENT_ROW" ]; then
        IFS=',' read -r m q _ _ _ _ _ _ tokw _ _ _ _ rm1k rm1m _ _ _ <<< "$EFFICIENT_ROW"
        out "  ${MAGENTA}${BOLD}Most Efficient:${RESET} Qwen2.5-${m} ${q} -> ${tokw} tok/W"
    fi
    if [ -n "$CHEAPEST_ROW" ]; then
        IFS=',' read -r m q _ _ _ _ _ _ _ _ _ _ _ rm1k rm1m _ _ _ <<< "$CHEAPEST_ROW"
        out "  ${YELLOW}${BOLD}Cheapest Energy:${RESET} Qwen2.5-${m} ${q} -> RM ${rm1k}/1k (${rm1m}/1M tokens)"
    fi
    echo ""
else
    out "  ${RED}No successful benchmark rows were captured.${RESET}"
    out "  ${DIM}Check log for parser/llama-bench output mismatch: $LOG_FILE${RESET}"
    echo ""
fi

##############################################################################
# Write JSON output from CSV rows
##############################################################################
python3 - "$RESULTS_CSV" "$RESULTS_JSON" "$TG_LENGTH" "$N_REPS" "$POWER_SAMPLE_INTERVAL_SEC" "$GPU_INDEX" "$ELECTRICITY_COST_MYR_PER_KWH" << 'PYEOF'
import csv
import json
import sys
from datetime import datetime

csv_path = sys.argv[1]
json_path = sys.argv[2]
tg_length = int(sys.argv[3])
n_reps = int(sys.argv[4])
sample_interval = float(sys.argv[5])
gpu_index = int(sys.argv[6])
elec_cost_myr = float(sys.argv[7])

rows = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(
            {
                "model_size": row["model_size"],
                "quant": row["quant"],
                "tg_tokens": int(row["tg_tokens"]),
                "tg_tok_sec": float(row["tg_tok_sec"]),
                "avg_power_w": float(row["avg_power_w"]),
                "min_power_w": float(row["min_power_w"]),
                "max_power_w": float(row["max_power_w"]),
                "power_samples": int(row["power_samples"]),
                "tokens_per_watt": float(row["tokens_per_watt"]),
                "joules_per_token": float(row["joules_per_token"]),
                "sec_per_1k_tokens": float(row["sec_per_1k_tokens"]),
                "wh_per_1k_tokens": float(row["wh_per_1k_tokens"]),
                "kwh_per_1m_tokens": float(row["kwh_per_1m_tokens"]),
                "rm_per_1k_tokens": float(row["rm_per_1k_tokens"]),
                "rm_per_1m_tokens": float(row["rm_per_1m_tokens"]),
                "electricity_cost_myr_per_kwh": float(row["electricity_cost_myr_per_kwh"]),
                "model_file": row["model_file"],
                "bench_seconds": int(row["bench_seconds"]),
            }
        )

payload = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "gpu": "AMD Radeon AI PRO R9700",
        "arch": "gfx1201 (RDNA4)",
        "vram_gb": 32,
        "backend": "llama.cpp (HIP/ROCm)",
        "benchmark_type": "token_per_watt_tg_only",
        "tg_tokens": tg_length,
        "repetitions": n_reps,
        "power_sample_interval_sec": sample_interval,
        "gpu_index": gpu_index,
        "electricity_cost_myr_per_kwh": elec_cost_myr,
        "currency": "MYR",
    },
    "results": rows,
}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PYEOF

out "${CYAN}$(draw_line '-')${RESET}"
out "  ${DIM}Results saved to:${RESET}"
out "    CSV   $RESULTS_CSV"
out "    JSON  $RESULTS_JSON"
out "    Log   $LOG_FILE"
echo ""
out "  ${DIM}Replay this result in CLI later:${RESET}"
out "    cat \"$RESULTS_CSV\""
out "    tail -n +2 \"$RESULTS_CSV\" | sort -t',' -k9,9gr"
echo ""
out "  ${DIM}Tip:${RESET} set electricity rate at runtime, e.g."
out "    ELECTRICITY_COST_MYR_PER_KWH=0.55 bash scripts/run_token_per_watt_benchmark.sh"
echo ""
out "${CYAN}$(draw_line)${RESET}"
echo ""

logfile "================================================================================"
logfile " TOKEN-PER-WATT BENCHMARK COMPLETE — Total time: $(elapsed_time)"
logfile " Results saved to:"
logfile "   CSV:  $RESULTS_CSV"
logfile "   JSON: $RESULTS_JSON"
logfile "   Log:  $LOG_FILE"
logfile "================================================================================"
