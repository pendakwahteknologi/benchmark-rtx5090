# Qwen2.5 GGUF Benchmark Suite for ASUS TURBO Radeon AI PRO R9700

Comprehensive benchmarking suite for testing Qwen2.5 language models (3B, 7B, 14B, 32B) across multiple quantization levels on AMD's RDNA4 GPU using ROCm and llama.cpp.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Hardware & Software](#hardware--software)
- [What We're Benchmarking](#what-were-benchmarking)
- [Setup Instructions](#setup-instructions)
- [Path Portability (Important)](#path-portability-important)
- [Running Benchmarks](#running-benchmarks)
- [Understanding Results](#understanding-results)
- [Script Reference](#script-reference)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Measured Results (2026-02-17)](#measured-results-2026-02-17)

---

## Overview

This benchmark suite measures **inference performance** of Qwen2.5 language models in GGUF format, testing how fast the ASUS TURBO Radeon AI PRO R9700 can:

1. **Process input prompts** (Prompt Processing / PP) - How quickly the model reads and understands your input
2. **Generate output tokens** (Text Generation / TG) - How quickly the model writes responses

The suite automatically:
- Copies models from USB drive to local storage
- Runs benchmarks across 4 model sizes × 3 quantization levels
- Manages disk space by deleting models after testing each size
- Generates detailed results with pretty terminal output + HTML reports

---

## Hardware & Software

### Hardware Specifications

| Component | Specification |
|-----------|---------------|
| **GPU** | ASUS TURBO Radeon AI PRO R9700 |
| **Architecture** | RDNA4 (gfx1201) |
| **VRAM** | 32 GB |
| **CPU** | Intel Core Ultra 7 265K |
| **Storage** | 476.9 GB SSD + USB drive for models |

### Software Stack

| Software | Version | Purpose |
|----------|---------|---------|
| **ROCm** | 7.2 | AMD's GPU compute platform |
| **llama.cpp** | Latest (build 8030) | LLM inference engine with HIP/ROCm backend |
| **Python** | 3.x | Report generation |
| **Bash** | 4.x+ | Benchmark orchestration |

### Model Source

- **Location**: USB drive mounted at `/mnt/usb/models/`
- **Format**: GGUF (GPT-Generated Unified Format)
- **Repository**: Qwen/Qwen2.5-{SIZE}-Instruct-GGUF on Hugging Face

---

## What We're Benchmarking

### Model Sizes

We test 4 model sizes from the Qwen2.5-Instruct family:

| Model | Parameters | Use Case |
|-------|-----------|----------|
| **3B** | 3 billion | Fast, lightweight responses |
| **7B** | 7 billion | Balanced performance/quality |
| **14B** | 14 billion | High quality responses |
| **32B** | 32 billion | Maximum quality (where it fits) |

### Quantization Levels

**Quantization** reduces model size/memory by using lower precision numbers. We test 3 levels:

| Quant | Bits | Size Impact | Quality | Speed |
|-------|------|-------------|---------|-------|
| **Q4_K_M** | 4-bit | Smallest (~2GB for 3B) | Good | Fastest |
| **Q5_K_M** | 5-bit | Medium (~2.3GB for 3B) | Better | Fast |
| **Q8_0** | 8-bit | Largest (~3.4GB for 3B) | Best | Slower (tested where VRAM allows) |

**Note**: Q8_0 is only tested when the model fits in 32GB VRAM. For 32B models, Q8 would require ~34GB, so it's automatically skipped.

### Test Types

#### 1. Prompt Processing (PP)
- **What it measures**: How fast the model processes input text
- **Test cases**: 128, 256, 512 token prompts
- **Why it matters**: Faster PP = less time waiting before the model starts responding
- **Metric**: tokens per second (tok/s)

#### 2. Text Generation (TG)
- **What it measures**: How fast the model generates output
- **Test case**: 128 tokens of generation
- **Why it matters**: This is the speed you *feel* during conversations
- **Metric**: tokens per second (tok/s)

### Benchmark Configuration

```bash
Prompt Processing: 128, 256, 512 tokens (3 tests per quantization)
Text Generation:   128 tokens output (1 test per quantization)
Repetitions:       3 per test (results are averaged)
GPU Offload:       All layers on GPU (ngl=99)
Flash Attention:   Enabled (fa=1)
Batch Size:        2048 (n_batch)
Context:           8192 tokens max
```

---

## Setup Instructions

### Prerequisites

1. **Mount USB Drive** (contains model files)
   ```bash
   sudo mount /dev/sda1 /mnt/usb
   ls /mnt/usb/models/  # Verify 11 GGUF files present
   ```

2. **Verify llama.cpp** is built with ROCm
   ```bash
   /path/to/llama.cpp/build/bin/llama-bench --help
   # Should show "found 1 ROCm devices: ASUS TURBO Radeon AI PRO R9700"
   ```

3. **Check Disk Space**
   ```bash
   df -h /path/to/benchmark-storage
   # Need at least 50GB free (largest models: 32B Q5 = 21.7GB)
   ```

4. **Set script execute permissions (recommended after clone)**
   ```bash
   chmod +x /path/to/benchmark-rocm/scripts/*.sh
   chmod +x /path/to/benchmark-rocm/scripts/*.py
   ```
   If you skip this, you can still run scripts via interpreter:
   ```bash
   bash scripts/run_benchmark.sh
   python3 scripts/generate_report.py
   ```

## Path Portability (Important)

The scripts are currently tied to one Ubuntu server layout. If you run them as-is on another machine, they will fail unless your paths match.

Hardcoded paths currently used:

- Benchmark root: `/home/pendakwahteknologi/benchmark-rocm`
- llama.cpp binaries: `/home/pendakwahteknologi/finetune-rocm-v0/llama.cpp/build/bin`
- USB model source: `/mnt/usb/models`
- Disk checks in `check_benchmark.sh`: `/home/pendakwahteknologi/`

Where these appear:

- `scripts/run_benchmark.sh`
- `scripts/run_interactive_bench.sh`
- `scripts/run_benchmark_background.sh`
- `scripts/check_benchmark.sh`

### Make It Work On Your Machine

1. Copy `benchmark-rocm` anywhere you want.
2. Edit the top path variables in `scripts/run_benchmark.sh`:
   - `cd ...`
   - `LLAMA_BENCH=...`
   - `LLAMA_CLI=...`
   - `LD_LIBRARY_PATH=...`
   - `USB_MODEL_DIR=...`
3. Edit the same path variables in `scripts/run_interactive_bench.sh` (`cd`, `LLAMA_CLI`, `LD_LIBRARY_PATH`, `USB_MODEL_DIR`).
4. Edit `scripts/run_benchmark_background.sh` so its `cd ...` points to your local `benchmark-rocm`.
5. Edit absolute paths in `scripts/check_benchmark.sh`:
   - `df -h /home/pendakwahteknologi/`
   - model/log paths under `/home/pendakwahteknologi/benchmark-rocm/...`

If you prefer fewer absolute paths, convert scripts to use `SCRIPT_DIR`/`ROOT_DIR` derived from the script location and keep only llama.cpp + USB paths configurable.

### Directory Structure

```
benchmark-rocm/
├── models/                      # Temporary model storage (auto-managed)
├── results/                     # Benchmark outputs
│   ├── benchmark_YYYYMMDD_HHMMSS.csv   # Raw data
│   ├── benchmark_YYYYMMDD_HHMMSS.json  # Metadata + results
│   ├── benchmark_YYYYMMDD_HHMMSS.log   # Detailed log
│   └── benchmark_report.html            # Visual report (generated separately)
└── scripts/
    ├── run_benchmark.sh                 # Main benchmark (foreground)
    ├── run_benchmark_background.sh      # Background runner
    ├── run_interactive_bench.sh         # Real prompt testing
    ├── check_benchmark.sh               # Status checker
    └── generate_report.py               # HTML report generator
```

---

## Running Benchmarks

### Quick Start

```bash
cd /path/to/benchmark-rocm
bash scripts/run_benchmark.sh
```

**Duration**: ~15-20 minutes for all 11 model configurations

### Background Mode (Recommended for SSH)

```bash
bash scripts/run_benchmark_background.sh
```

This runs the benchmark in the background so you can:
- Disconnect from SSH safely
- Monitor progress with `tail -f results/benchmark_*.log`
- Check GPU usage with `watch -n 1 rocm-smi`

### Interactive Prompt Benchmark

Use this to run fixed real-world prompts through each model size and supported quantisations with `llama-cli`.

```bash
bash scripts/run_interactive_bench.sh
```

This script:
- Runs unattended (no manual input during execution)
- Tests 5 built-in prompts across `3B`, `7B`, `14B`, and `32B`
- Tests quants `Q4_K_M`, `Q5_K_M`, and `Q8_0` where VRAM estimate allows
- Automatically skips `Q8_0` when estimated VRAM use exceeds `TOTAL_VRAM_GB` (for example 32B)
- Writes output to:
  - `results/interactive_bench_YYYYMMDD_HHMMSS.jsonl`
  - `results/interactive_bench_YYYYMMDD_HHMMSS.log`
- Cleans up model files after each size

### Check Status

```bash
bash scripts/check_benchmark.sh
```

Shows:
- Running processes
- GPU utilisation
- Disk space
- Latest log output (last 30 lines)

---

## Understanding Results

### Terminal Output

During the benchmark, you'll see:

#### 1. **Opening Banner**
```
     ___                      ___  _____   ____                  _
    / _ \__      _____ _ __  |__ \| ____| | __ )  ___ _ __   ___| |__
   | | | \ \ /\ / / _ \ '_ \   ) | |__   |  _ \ / _ \ '_ \ / __| '_ \
   | |_| |\ V  V /  __/ | | | / /|___ \  | |_) |  __/ | | | (__| | | |
    \__\_\ \_/\_/ \___|_| |_||____|___/  |____/ \___|_| |_|\_____|_| |_|

===============================================================================
  GPU          ASUS TURBO Radeon AI PRO R9700
  Architecture RDNA4 (gfx1201)
  VRAM         32 GB
  ROCm         7.2
  Backend      llama.cpp (HIP/ROCm)
  Flash Attn   Enabled
```

#### 2. **Per-Model Progress**
```
================================================================================
  MODEL 1/4: Qwen2.5-3B-Instruct-GGUF
  Elapsed: 00:00:00
================================================================================

     Quantizations: Q4_K_M Q5_K_M Q8_0

     --- Copying models from USB drive ---
         COPYING  qwen2.5-3b-instruct-q4_k_m.gguf (2.0 GB) ... done

     --- Running benchmarks ---

     [1/11] Qwen2.5-3B-Instruct-GGUF Q4_K_M (2.0 GB)
     PP: 128, 256, 512 tokens | TG: 128 tokens | Reps: 3 | Elapsed: 00:00:15
     Running llama-bench...

         PP  128 tokens  ->  3977.7 tok/s
         PP  256 tokens  ->  6347.8 tok/s
         PP  512 tokens  ->  8117.7 tok/s
         TG  128 tokens  ->   142.9 tok/s

     DONE  Qwen2.5-3B-Instruct-GGUF Q4_K_M (5s)
```

#### 3. **Final Summary Tables**

**Prompt Processing:**
```
  Model                               Quant            PP 128       PP 256       PP 512
  ---------------------------------------------------------------------------------------------
  Qwen2.5-3B-Instruct-GGUF           Q4_K_M         3977.7       6347.8       8117.7
  Qwen2.5-3B-Instruct-GGUF           Q5_K_M         3810.4       6082.4       7729.0
  Qwen2.5-3B-Instruct-GGUF           Q8_0           1349.7       2274.3       3083.9
  Qwen2.5-7B-Instruct-GGUF           Q4_K_M         2156.3       3845.6       5432.1
  ...
```

**Text Generation:**
```
  Model                               Quant            TG 128
  ------------------------------------------------------------
  Qwen2.5-3B-Instruct-GGUF           Q4_K_M          142.9
  Qwen2.5-3B-Instruct-GGUF           Q5_K_M          137.7
  Qwen2.5-3B-Instruct-GGUF           Q8_0            113.1
  ...
```

**Peak Performance:**
```
  Prompt Processing   8117.7 tok/s  (Qwen2.5-3B-Instruct-GGUF Q4_K_M)
  Text Generation      142.9 tok/s  (Qwen2.5-3B-Instruct-GGUF Q4_K_M)
```

### Output Files

#### 1. **CSV File** (`benchmark_YYYYMMDD_HHMMSS.csv`)
Raw data suitable for spreadsheets/analysis:

```csv
model_size,quant,pp_tokens,pp_tok_sec,tg_tokens,tg_tok_sec,model_file,vram_used_mb
3B,Q4_K_M,128,3977.732217,0,0,qwen2.5-3b-instruct-q4_k_m.gguf,0
3B,Q4_K_M,256,6347.781998,0,0,qwen2.5-3b-instruct-q4_k_m.gguf,0
3B,Q4_K_M,512,8117.655426,0,0,qwen2.5-3b-instruct-q4_k_m.gguf,0
3B,Q4_K_M,0,0,128,142.894636,qwen2.5-3b-instruct-q4_k_m.gguf,0
```

**Columns:**
- `model_size`: 3B, 7B, 14B, 32B
- `quant`: Q4_K_M, Q5_K_M, Q8_0
- `pp_tokens`: Prompt processing test size (128, 256, 512) or 0 if TG test
- `pp_tok_sec`: Prompt processing speed in tokens/second
- `tg_tokens`: Text generation test size (128) or 0 if PP test
- `tg_tok_sec`: Text generation speed in tokens/second
- `model_file`: GGUF filename
- `vram_used_mb`: Reserved for future use (currently 0)

#### 2. **JSON File** (`benchmark_YYYYMMDD_HHMMSS.json`)
Structured format with metadata:

```json
{
  "metadata": {
    "gpu": "ASUS TURBO Radeon AI PRO R9700",
    "arch": "gfx1201 (RDNA4)",
    "vram_gb": 32,
    "rocm_version": "7.2",
    "backend": "llama.cpp (HIP/ROCm)",
    "timestamp": "2026-02-16T21:40:53+08:00",
    "pp_lengths": "128,256,512",
    "tg_length": 128,
    "repetitions": 3
  },
  "results": []
}
```

#### 3. **Log File** (`benchmark_YYYYMMDD_HHMMSS.log`)
Complete execution log including:
- llama-bench raw CSV output
- Copy/cleanup operations
- Timing information
- Any errors or warnings

#### 4. **HTML Report** (`benchmark_report.html`)
Interactive visual report with:
- Bar charts comparing models
- Summary statistics
- System specifications
- Colour-coded performance tiers

Generate with:
```bash
python3 scripts/generate_report.py
```

Then open in browser:
```bash
firefox results/benchmark_report.html
```

---

## Script Reference

### Main Scripts

#### `run_benchmark.sh`
**Purpose**: Main synthetic benchmark runner (foreground).

**What it actually does**:
- Uses `llama-bench` (not `llama-cli`) for PP/TG throughput testing.
- Tests model sizes `3B, 7B, 14B, 32B`.
- Tests quants `Q4_K_M`, `Q5_K_M`, and `Q8_0` where VRAM estimate allows.
- Copies models from `/mnt/usb/models`, benchmarks, then deletes local model files per size.
- Writes three outputs per run:
  - `results/benchmark_YYYYMMDD_HHMMSS.csv`
  - `results/benchmark_YYYYMMDD_HHMMSS.json`
  - `results/benchmark_YYYYMMDD_HHMMSS.log`

**Hardcoded machine-specific paths**:
- Yes (`cd`, llama.cpp bin paths, USB path, disk path assumptions).

**Usage**:
```bash
bash scripts/run_benchmark.sh
```

---

#### `run_benchmark_background.sh`
**Purpose**: Convenience wrapper to run `run_benchmark.sh` with `nohup` in background.

**What it actually does**:
- Changes directory to hardcoded benchmark root.
- Starts `bash scripts/run_benchmark.sh > results/benchmark_<timestamp>.log 2>&1 &`.
- Prints PID and tail command.

**Hardcoded machine-specific paths**:
- Yes (`cd /home/pendakwahteknologi/benchmark-rocm`).

**Usage**:
```bash
bash scripts/run_benchmark_background.sh
```

**Monitoring**:
```bash
tail -f results/benchmark_*.log          # Follow log output
watch -n 1 rocm-smi                       # GPU utilisation
ps aux | grep run_benchmark               # Check if running
```

---

#### `check_benchmark.sh`
**Purpose**: Read-only status helper for active/recent benchmark runs.

**What it actually does**:
- Checks whether `run_benchmark.sh` is running (`pgrep -f`).
- Prints GPU usage/memory via `rocm-smi`.
- Prints disk usage.
- Lists model files currently in the local `models/` folder.
- Prints last 30 lines of latest benchmark log.

**Hardcoded machine-specific paths**:
- Yes, strongly. Uses absolute `/home/pendakwahteknologi/...` paths for disk/model/log checks.

**Usage**:
```bash
bash scripts/check_benchmark.sh
```

---

#### `run_interactive_bench.sh`
**Purpose**: Automated prompt-response benchmark with `llama-cli` across model sizes and quantisations.

**What it actually does**:
- Runs 5 fixed prompts on each size (`3B/7B/14B/32B`).
- Tests quants `Q4_K_M`, `Q5_K_M`, and `Q8_0` where VRAM fit check allows.
- Uses the same `Q8_0` skip logic as `run_benchmark.sh` when estimated VRAM exceeds `TOTAL_VRAM_GB`.
- Copies each model from `/mnt/usb/models` if missing.
- Captures response preview and timing lines from stderr.
- Writes:
  - `results/interactive_bench_YYYYMMDD_HHMMSS.jsonl`
  - `results/interactive_bench_YYYYMMDD_HHMMSS.log`
- Removes the model file after each size.

**Human interaction required**:
- No. It is fully unattended once launched.

**Hardcoded machine-specific paths**:
- Yes (`cd`, `LLAMA_CLI`, `LD_LIBRARY_PATH`, `USB_MODEL_DIR`).

**Prompt set used**:
1. "Explain what a GPU is in one paragraph."
2. "Write a Python function to check if a number is prime."
3. "What are the main differences between TCP and UDP?"
4. "Summarise the theory of relativity in simple terms."
5. "Write a bash script that finds the largest file in a directory."

**Usage**:
```bash
bash scripts/run_interactive_bench.sh
```

**Output**: JSONL file with model size, quant, prompts, responses, and timing

---

#### `generate_report.py`
**Purpose**: Convert latest `benchmark_*.csv` into a single HTML report.

**What it actually does**:
- Looks for newest `results/benchmark_*.csv`.
- Parses rows and embeds them into a static HTML template.
- Writes `results/benchmark_report.html`.

**Important runtime note**:
- Run this from the `benchmark-rocm` root so `results/` resolves correctly.
- This script has no hardcoded `/home/...` path; it uses relative `results/`.

**Usage**:
```bash
python3 scripts/generate_report.py
```

**Output**: `results/benchmark_report.html`

---

### Configuration Variables

Edit at top of `run_benchmark.sh` and `run_interactive_bench.sh`:

```bash
# Benchmark configuration
PP_LENGTHS="128,256,512"      # Prompt processing test sizes
TG_LENGTH=128                  # Text generation test size
N_REPS=3                       # Repetitions per test (averaged)

# Machine paths (must match your system)
LLAMA_BENCH="/path/to/llama.cpp/build/bin/llama-bench"
LLAMA_CLI="/path/to/llama.cpp/build/bin/llama-cli"
export LD_LIBRARY_PATH="/path/to/llama.cpp/build/bin:$LD_LIBRARY_PATH"
USB_MODEL_DIR="/mnt/usb/models"

# Model sizes to test
MODEL_SIZES=("3B" "7B" "14B" "32B")

# VRAM available
TOTAL_VRAM_GB=32

# VRAM estimates for Q8_0 (auto-skip if exceeds TOTAL_VRAM_GB)
VRAM_ESTIMATE_Q8["3B"]=4
VRAM_ESTIMATE_Q8["7B"]=8
VRAM_ESTIMATE_Q8["14B"]=16
VRAM_ESTIMATE_Q8["32B"]=34
```

---

## Troubleshooting

### USB Drive Issues

**Problem**: `ERROR: USB model directory not found at /mnt/usb/models`

**Solution**:
```bash
# Mount the USB drive
sudo mount /dev/sda1 /mnt/usb

# Verify models are present
ls /mnt/usb/models/*.gguf
# Should show 11 GGUF files
```

---

### Permission Denied On Script Run

**Problem**: Running `./scripts/run_benchmark.sh` (or other scripts) returns `Permission denied`.

**Cause**: Execute bits are missing (for example files show `-rw-rw-r--`).

**Solution**:
```bash
chmod +x /path/to/benchmark-rocm/scripts/*.sh
chmod +x /path/to/benchmark-rocm/scripts/*.py
```

**Alternative** (without execute bit):
```bash
bash scripts/run_benchmark.sh
bash scripts/run_interactive_bench.sh
python3 scripts/generate_report.py
```

---

### Library Not Found

**Problem**: `error while loading shared libraries: libllama.so.0`

**Solution**: The script already sets `LD_LIBRARY_PATH`. If you still see this:
```bash
export LD_LIBRARY_PATH="/home/pendakwahteknologi/finetune-rocm-v0/llama.cpp/build/bin:$LD_LIBRARY_PATH"
```

---

### Disk Space

**Problem**: "No space left on device" during copy

**Solution**:
```bash
# Check free space
df -h /home/pendakwahteknologi/

# Manual cleanup if needed
rm -rf ~/benchmark-rocm/models/*.gguf
```

The script automatically cleans up after each model size, but if interrupted, models may remain.

---

### Empty Results Tables

**Problem**: Summary tables show all zeros or are empty

**Causes**:
1. llama-bench failed to run (check log for errors)
2. CSV parsing failed (llama-bench output format changed)

**Debug**:
```bash
# Check if CSV has data
cat results/benchmark_*.csv

# Test llama-bench manually
LD_LIBRARY_PATH="/home/pendakwahteknologi/finetune-rocm-v0/llama.cpp/build/bin:$LD_LIBRARY_PATH" \
    /home/pendakwahteknologi/finetune-rocm-v0/llama.cpp/build/bin/llama-bench \
    -m /mnt/usb/models/qwen2.5-3b-instruct-q4_k_m.gguf \
    -p 128 -n 128 -r 1 -ngl 99 -fa 1 -o csv
```

---

### Model Loading Failures

**Problem**: `main: error: failed to load model`

**Causes**:
1. Corrupted model file (incomplete copy)
2. Incompatible GGUF version

**Solution**:
```bash
# Remove potentially corrupted file
rm ~/benchmark-rocm/models/qwen2.5-*.gguf

# Verify source file integrity
ls -lh /mnt/usb/models/qwen2.5-3b-instruct-q4_k_m.gguf
# Compare size with successful copies

# Re-run benchmark (will re-copy)
```

---

### ROCm Device Not Found

**Problem**: `ggml_cuda_init: no ROCm devices found`

**Solution**:
```bash
# Check ROCm installation
rocminfo | grep "Name:"

# Check GPU is visible
rocm-smi

# Verify llama.cpp was built with ROCm
ldd /home/pendakwahteknologi/finetune-rocm-v0/llama.cpp/build/bin/llama-bench | grep hip
# Should show libamdhip64.so
```

---

## Advanced Usage

### Custom Model Sizes

To test only specific models, edit `run_benchmark.sh`:

```bash
# Test only 3B and 7B
MODEL_SIZES=("3B" "7B")
```

### Custom Prompt Lengths

```bash
# Test different prompt sizes
PP_LENGTHS="64,128,256,512,1024"
```

### More Repetitions

For more stable results (slower):

```bash
N_REPS=5  # Instead of 3
```

### Skip Quantization Levels

Comment out unwanted quants in the loop:

```bash
# In run_benchmark.sh, around line 389
quants_to_test=("Q4_K_M")  # Only test Q4
# quants_to_test=("Q4_K_M" "Q5_K_M")  # Skip Q8
```

### Parallel Benchmarking

**DO NOT** run multiple benchmarks simultaneously - GPU contention will skew results. Run sequentially.

---

## Measured Results (2026-02-17)

This section records real benchmark output from this project on the test machine.

- **Benchmark report timestamp**: 2026-02-17 12:31:50
- **Source files**:
  - `benchmark_report.html`
  - `SYSTEM_INFO.md`
- **Benchmark mode**: `run_benchmark.sh` (llama-bench, 3 repetitions)

### System Overview (Test Bench)

| Component | Value |
|----------|-------|
| **OS** | Ubuntu 24.04.4 LTS |
| **Kernel** | 6.17.0-14-generic |
| **ROCm** | 7.2.0 |
| **Docker** | 29.2.1 |
| **Python** | 3.12.3 |
| **CPU** | Intel Core Ultra 7 265K |
| **CPU Cores** | 20 |
| **CPU Boost** | Up to 5.6 GHz |
| **CPU Cache** | 30MB L3 |
| **GPU (Primary AI Accelerator)** | ASUS TURBO Radeon AI PRO R9700 |
| **GPU Architecture** | RDNA4 (`gfx1201`) |
| **GPU VRAM** | 32GB GDDR |
| **GPU Driver** | amdgpu 6.18.4 |
| **PCIe Link** | 32.0GT/s x16 |
| **GPU Max Power** | 300W |
| **ROCm GPU Support** | Native `gfx1201` support |
| **Memory** | 46GB DDR5 |
| **DIMM Configuration** | 2 x 24GB DDR5-8200 |
| **Memory Speed (running)** | 6400 MT/s |
| **Storage** | WDC PC SN810 NVMe 512GB |
| **Inference Backend** | llama.cpp HIP backend |
| **PyTorch Stack** | PyTorch 2.9.1 ROCm |
| **HSA Override** | Not required |
| **Benchmark repetitions** | 3 per configuration |

### Social Badge Text

Use this block for image/footer overlays:

```text
Powered By
ASUS TURBO Radeon AI PRO R9700 32GB
Intel Core Ultra 7 265K
46GB DDR5
ROCm 7.2 + llama.cpp HIP
```

### Headline Performance

- **Peak Prompt Processing (PP 512)**: **8124.8 tok/s**  
  `Qwen2.5-3B-Instruct` + `Q4_K_M`
- **Peak Text Generation (TG 128)**: **143.4 tok/s**  
  `Qwen2.5-3B-Instruct` + `Q4_K_M`

### TG 128 Comparison (tok/s)

| Model | Q4_K_M | Q5_K_M | Q8_0 |
|------|--------|--------|------|
| **Qwen2.5-3B** | 143.4 | 137.0 | 113.0 |
| **Qwen2.5-7B** | 103.7 | 93.3 | 69.5 |
| **Qwen2.5-14B** | 55.0 | 49.3 | 36.2 |
| **Qwen2.5-32B** | 26.8 | 23.7 | N/A (skipped; VRAM limit) |

### PP 512 Comparison (tok/s)

| Model | Q4_K_M | Q5_K_M | Q8_0 |
|------|--------|--------|------|
| **Qwen2.5-3B** | 8124.8 | 7737.2 | 3042.0 |
| **Qwen2.5-7B** | 4006.1 | 3040.4 | 1392.6 |
| **Qwen2.5-14B** | 2007.7 | 1887.9 | 685.6 |
| **Qwen2.5-32B** | 901.6 | 854.2 | N/A (skipped; VRAM limit) |

### Notes

- For this machine and configuration, `Q4_K_M` was fastest across all tested sizes.
- `Q8_0` gave lower throughput and, for 32B, did not fit the 32 GB VRAM limit used by the benchmark logic.
- Results are specific to this system, ROCm version, llama.cpp build, and benchmark settings.

---

## Performance Interpretation

### What's "Good" Performance?

**Prompt Processing (PP)**:
- **>5000 tok/s**: Excellent - instant prompt understanding
- **3000-5000 tok/s**: Good - barely noticeable delay
- **1000-3000 tok/s**: Acceptable - slight delay
- **<1000 tok/s**: Slow - noticeable wait time

**Text Generation (TG)**:
- **>100 tok/s**: Excellent - feels instant (typical reading speed ~200 tok/s)
- **50-100 tok/s**: Good - smooth experience
- **20-50 tok/s**: Acceptable - slight lag
- **<20 tok/s**: Slow - noticeable typing delay

### Trade-offs

| Metric | Q4_K_M | Q5_K_M | Q8_0 |
|--------|---------|---------|------|
| **Speed** | Fastest | Fast | Slower |
| **Quality** | Good | Better | Best |
| **VRAM** | Lowest | Medium | Highest |
| **Size** | Smallest | Medium | Largest |

**Recommendation**: Q4_K_M for most use cases - offers best speed/quality balance.

### Model Size Trade-offs

| Size | Speed | Quality | Use When |
|------|-------|---------|----------|
| **3B** | Fastest | Good | Need maximum speed, simple tasks |
| **7B** | Fast | Better | General-purpose, balanced needs |
| **14B** | Slower | Great | Complex reasoning, quality matters |
| **32B** | Slowest | Best | Maximum quality, can tolerate latency |

---

## Citation & Credits

**GPU**: ASUS TURBO Radeon AI PRO R9700 (RDNA4 / gfx1201)
**Models**: Qwen2.5-Instruct by Alibaba Cloud (Hugging Face: Qwen/Qwen2.5-{SIZE}-Instruct-GGUF)
**Inference Engine**: llama.cpp by ggerganov (HIP/ROCm backend)
**Compute Platform**: ROCm 7.2 by AMD

---

## License

This benchmark suite is provided as-is for educational and evaluation purposes.

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review log files in `results/`
3. Test individual components manually
4. Ensure all prerequisites are met

**Happy Benchmarking! 🚀**


---

# Qwen2.5 Instruct GGUF Models for Benchmarking

## Hardware Target

- **GPU:** ASUS TURBO Radeon AI PRO R9700 (RDNA4, gfx1201)
- **VRAM:** 32 GB
- **Backend:** llama.cpp with HIP/ROCm 7.2

## Models & Quantizations

| Model | Quant | Est. Size | Fits VRAM | Source |
|-------|-------|-----------|-----------|--------|
| Qwen2.5-3B-Instruct | Q4_K_M | ~2.1 GB | Yes | Official |
| Qwen2.5-3B-Instruct | Q5_K_M | ~2.4 GB | Yes | Official |
| Qwen2.5-3B-Instruct | Q8_0 | ~3.6 GB | Yes | Official |
| Qwen2.5-7B-Instruct | Q4_K_M | ~4.7 GB | Yes | Official |
| Qwen2.5-7B-Instruct | Q5_K_M | ~5.4 GB | Yes | Official |
| Qwen2.5-7B-Instruct | Q8_0 | ~8.1 GB | Yes | Official |
| Qwen2.5-14B-Instruct | Q4_K_M | ~9.0 GB | Yes | Official |
| Qwen2.5-14B-Instruct | Q5_K_M | ~10.5 GB | Yes | Official |
| Qwen2.5-14B-Instruct | Q8_0 | ~15.7 GB | Yes | Official |
| Qwen2.5-32B-Instruct | Q4_K_M | ~19.9 GB | Yes | Official |
| Qwen2.5-32B-Instruct | Q5_K_M | ~23.8 GB | Yes | Official |
| Qwen2.5-32B-Instruct | Q8_0 | ~34.8 GB | **No** (exceeds 32GB) | Community |

## Download Links

### Qwen2.5-3B-Instruct-GGUF (Official)

- **Repository:** https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
- **Files (single file per quant):**
  - `qwen2.5-3b-instruct-q4_k_m.gguf` (~2.1 GB)
  - `qwen2.5-3b-instruct-q5_k_m.gguf` (~2.4 GB)
  - `qwen2.5-3b-instruct-q8_0.gguf` (~3.6 GB)

### Qwen2.5-7B-Instruct-GGUF (Official)

- **Repository:** https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF
- **Files (split — requires merge after download):**
  - Q4_K_M: `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf` + `...00002-of-00002.gguf` (~4.7 GB total)
  - Q5_K_M: `qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf` + `...00002-of-00002.gguf` (~5.4 GB total)
  - Q8_0: `qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf` + 2 more parts (~8.1 GB total)

### Qwen2.5-14B-Instruct-GGUF (Official)

- **Repository:** https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF
- **Files (split — requires merge after download):**
  - Q4_K_M: 3 parts `qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf` ... (~9.0 GB total)
  - Q5_K_M: 3 parts `qwen2.5-14b-instruct-q5_k_m-00001-of-00003.gguf` ... (~10.5 GB total)
  - Q8_0: 4 parts `qwen2.5-14b-instruct-q8_0-00001-of-00004.gguf` ... (~15.7 GB total)

### Qwen2.5-32B-Instruct-GGUF (Official)

- **Repository:** https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF
- **Files (split — requires merge after download):**
  - Q4_K_M: 5 parts `qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf` ... (~19.9 GB total)
  - Q5_K_M: 6 parts `qwen2.5-32b-instruct-q5_k_m-00001-of-00006.gguf` ... (~23.8 GB total)
  - Q8_0: **Skipped** — ~34.8 GB exceeds 32 GB VRAM

## Important Notes

### Split GGUF Files (7B, 14B, 32B)

Models 7B and above have their GGUF files **split into multiple parts** due to Hugging Face file size limits. After downloading, they must be merged:

```bash
# llama-gguf-split is included in the llama.cpp build
llama-gguf-split --merge qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf qwen2.5-7b-instruct-q4_k_m.gguf
```

The benchmark script (`run_benchmark.sh`) handles this automatically.

### Disk Space Strategy

Since storage is limited, the benchmark script:

1. Downloads all quant files for **one model size** at a time
2. Runs all benchmarks for that size
3. **Deletes** all files for that size
4. Moves to the next model size

Peak disk usage per model size:

| Model Size | Peak Disk (all 3 quants) |
|------------|--------------------------|
| 3B | ~8.1 GB |
| 7B | ~18.2 GB |
| 14B | ~35.2 GB |
| 32B | ~43.7 GB (Q4+Q5 only) |

### 32B Q8_0 — Skipped

The Q8_0 quantization of Qwen2.5-32B requires ~34.8 GB, which exceeds the 32 GB VRAM on the R9700. This configuration is automatically skipped by the benchmark script.

A community-quantized version exists at https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF but is not used since it cannot run fully on GPU.

## Download Commands (Manual)

```bash
# 3B (single files)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir ./models

# 7B (split files — download all parts)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF --include "qwen2.5-7b-instruct-q4_k_m-*" --local-dir ./models

# 14B (split files)
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-GGUF --include "qwen2.5-14b-instruct-q4_k_m-*" --local-dir ./models

# 32B (split files)
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-GGUF --include "qwen2.5-32b-instruct-q4_k_m-*" --local-dir ./models
```
