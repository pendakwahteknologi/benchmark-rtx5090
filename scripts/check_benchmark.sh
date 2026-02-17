#!/bin/bash
##############################################################################
# Check Benchmark Status
# Shows GPU usage, running processes, and latest log output
##############################################################################

echo "================================================================================"
echo " Benchmark Status Check"
echo "================================================================================"
echo ""

# Check if benchmark is running
BENCH_PID=$(pgrep -f "run_benchmark.sh" 2>/dev/null || true)
if [ -n "$BENCH_PID" ]; then
    echo "[RUNNING] Benchmark is active (PID: $BENCH_PID)"
else
    echo "[STOPPED] No benchmark process found"
fi
echo ""

# GPU status
echo "--- GPU Status ---"
rocm-smi --showuse --showmeminfo vram 2>/dev/null | grep -E "GPU\[|=====" || echo "rocm-smi not available"
echo ""

# Disk space
echo "--- Disk Space ---"
df -h /home/pendakwahteknologi/ | tail -1 | awk '{print "  Used: "$3" / "$2"  Free: "$4"  ("$5" used)"}'
echo ""

# Model files on disk
echo "--- Model Files ---"
if ls /home/pendakwahteknologi/benchmark-rocm/models/*.gguf 1>/dev/null 2>&1; then
    du -sh /home/pendakwahteknologi/benchmark-rocm/models/*.gguf 2>/dev/null
else
    echo "  No model files currently on disk (good - space is being managed)"
fi
echo ""

# Latest log
echo "--- Latest Log Output (last 30 lines) ---"
LATEST_LOG=$(ls -t /home/pendakwahteknologi/benchmark-rocm/results/benchmark_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "  Log file: $LATEST_LOG"
    echo ""
    tail -30 "$LATEST_LOG"
else
    echo "  No log files found. Run the benchmark first:"
    echo "    ./scripts/run_benchmark.sh"
fi
echo ""
echo "================================================================================"
echo ""
echo " Monitor in real-time:"
echo "   tail -f $LATEST_LOG"
echo ""
echo " Watch GPU:"
echo "   watch -n 1 rocm-smi"
echo ""
echo "================================================================================"
