#!/bin/bash
##############################################################################
# Background Benchmark Script - Run Full Qwen2.5 Benchmark Suite
# Safe to disconnect SSH - logs to results/benchmark.log
##############################################################################

set -e

cd /home/pendakwahteknologi/benchmark-rocm

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="results/benchmark_${TIMESTAMP}.log"

echo "================================================================================"
echo " Starting Qwen2.5 Benchmark Suite in Background"
echo "================================================================================"
echo ""
echo " Benchmark will run in background. You can safely disconnect SSH."
echo ""
echo " Monitor progress:"
echo "   tail -f $LOG_FILE"
echo ""
echo " Check GPU usage:"
echo "   watch -n 1 rocm-smi"
echo ""
echo " Check if running:"
echo "   ps aux | grep run_benchmark"
echo ""
echo "================================================================================"
echo ""

nohup bash scripts/run_benchmark.sh > "$LOG_FILE" 2>&1 &

BG_PID=$!

echo " Started! PID: $BG_PID"
echo " Log: $LOG_FILE"
echo ""
echo " Watch progress:"
echo "   tail -f $LOG_FILE"
echo ""
