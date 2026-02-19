#!/bin/bash
##############################################################################
# Show Token-Per-Watt Results in CLI
# Usage:
#   bash scripts/show_token_per_watt_results.sh
#   bash scripts/show_token_per_watt_results.sh results/token_per_watt_YYYYMMDD_HHMMSS.csv
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

if [ $# -ge 1 ]; then
    CSV_PATH="$1"
else
    CSV_PATH="$(ls -t results/token_per_watt_*.csv 2>/dev/null | head -1 || true)"
fi

if [ -z "${CSV_PATH:-}" ] || [ ! -f "$CSV_PATH" ]; then
    echo "[ERROR] No token-per-watt CSV found."
    echo "        Run: bash scripts/run_token_per_watt_benchmark.sh"
    exit 1
fi

DATA_ROWS=$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "$CSV_PATH")

echo "================================================================================"
echo " Token-Per-Watt Results (CLI)"
echo "================================================================================"
echo " File: $CSV_PATH"
echo " Rows: $DATA_ROWS"
echo ""

if [ "$DATA_ROWS" -le 0 ]; then
    echo "[WARN] CSV has header only (no completed benchmark rows)."
    exit 0
fi

printf " %-4s %-23s %-8s %9s %9s %9s %10s %11s\n" "#" "Model" "Quant" "tok/s" "Avg W" "tok/W" "sec/1k" "RM/1k"
printf '%0.s-' {1..102}
echo ""

RANK=0
tail -n +2 "$CSV_PATH" | sort -t',' -k9,9gr | while IFS=',' read -r \
    model_size quant tg_tokens tg_tok_sec avg_power_w min_power_w max_power_w power_samples \
    tokens_per_watt joules_per_token sec_per_1k_tokens wh_per_1k_tokens kwh_per_1m_tokens \
    rm_per_1k_tokens rm_per_1m_tokens electricity_cost_myr_per_kwh model_file bench_seconds; do
    RANK=$((RANK + 1))
    printf " %-4s %-23s %-8s %9.2f %9.2f %9.3f %10.2f %11.6f\n" \
        "$RANK" "Qwen2.5-${model_size}" "$quant" "$tg_tok_sec" "$avg_power_w" "$tokens_per_watt" "$sec_per_1k_tokens" "$rm_per_1k_tokens"
done

printf '%0.s-' {1..102}
echo ""
echo ""

FASTEST_ROW="$(awk -F',' 'NR>1 && $4+0>max {max=$4+0; r=$0} END {print r}' "$CSV_PATH")"
EFFICIENT_ROW="$(awk -F',' 'NR>1 && $9+0>max {max=$9+0; r=$0} END {print r}' "$CSV_PATH")"
CHEAPEST_ROW="$(awk -F',' 'NR>1 && (min=="" || $14+0<min) {min=$14+0; r=$0} END {print r}' "$CSV_PATH")"

if [ -n "$FASTEST_ROW" ]; then
    IFS=',' read -r m q _ tg _ _ _ _ _ _ sec1k _ _ rm1k rm1m _ _ _ <<< "$FASTEST_ROW"
    echo " Fastest:        Qwen2.5-${m} ${q} -> ${tg} tok/s (${sec1k} sec/1k)"
fi
if [ -n "$EFFICIENT_ROW" ]; then
    IFS=',' read -r m q _ _ _ _ _ _ tokw _ _ _ _ rm1k rm1m _ _ _ <<< "$EFFICIENT_ROW"
    echo " Most Efficient: Qwen2.5-${m} ${q} -> ${tokw} tok/W"
fi
if [ -n "$CHEAPEST_ROW" ]; then
    IFS=',' read -r m q _ _ _ _ _ _ _ _ _ _ _ rm1k rm1m _ _ _ <<< "$CHEAPEST_ROW"
    echo " Cheapest:       Qwen2.5-${m} ${q} -> RM ${rm1k}/1k (${rm1m}/1M)"
fi
echo "================================================================================"
