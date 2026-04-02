# Qwen2.5 GGUF Benchmark Suite for ASUS Ascent GX10

Comprehensive benchmarking suite for testing Qwen2.5 language models (3B, 7B, 14B, 32B) across multiple quantization levels on the [ASUS Ascent GX10](https://www.asus.com/my/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/) desktop AI supercomputer, powered by the NVIDIA GB10 Grace Blackwell Superchip, using CUDA and llama.cpp.

---

## Table of Contents

- [Overview](#overview)
- [Cost Per Token (RM)](#cost-per-token-rm)
- [Hardware & Software](#hardware--software)
- [What We're Benchmarking](#what-were-benchmarking)
- [Setup Instructions](#setup-instructions)
- [Running Benchmarks](#running-benchmarks)
- [Understanding Results](#understanding-results)
- [Script Reference](#script-reference)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Measured Results (2026-04-03)](#measured-results-2026-04-03)
---

## Overview

This benchmark suite measures **inference performance** of Qwen2.5 language models in GGUF format, testing how fast the ASUS Ascent GX10 can:

1. **Process input prompts** (Prompt Processing / PP) - How quickly the model reads and understands your input
2. **Generate output tokens** (Text Generation / TG) - How quickly the model writes responses

The ASUS Ascent GX10 has **128GB LPDDR5X unified memory** shared between CPU and GPU via NVLink-C2C, meaning **all model sizes and quantizations fit** — including 32B Q8_0 (~34.8GB).

The suite automatically:
- Downloads models from Hugging Face (or uses pre-downloaded models)
- Runs benchmarks across 4 model sizes x 3 quantization levels = **12 configurations**
- Generates detailed results with pretty terminal output + HTML reports
- Measures power efficiency (tok/W) via `nvidia-smi`

---

## Cost Per Token (RM)

This repo includes a dedicated **cost-per-token benchmark**:

- Script: `scripts/run_token_per_watt_gx10.sh`
- Power monitoring: `nvidia-smi` (GPU power.draw query)
- Currency: **MYR (RM)** via `ELECTRICITY_COST_MYR_PER_KWH`

It measures generation speed and energy efficiency, then estimates electricity cost:

- `tok/s` (speed)
- `tok/W` (efficiency)
- `sec/1k tok` (latency style metric)
- `RM/1k tok` and `RM/1M tok` (energy cost estimate)

Quick start:

```bash
bash scripts/run_token_per_watt_gx10.sh
```

The main throughput benchmark:

```bash
bash scripts/run_benchmark_gx10.sh
```

---

## Hardware & Software

### Hardware Specifications

| Component | Specification |
|-----------|---------------|
| **Device** | [ASUS Ascent GX10](https://www.asus.com/my/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/) |
| **SoC** | NVIDIA GB10 Grace Blackwell Superchip |
| **GPU** | NVIDIA Blackwell GPU with 5th-gen Tensor Cores |
| **GPU Architecture** | Blackwell (Compute Capability 12.1) |
| **AI Performance** | Up to 1 PFLOP (FP4) |
| **CPU** | 20-core ARM (Cortex-X925 @ 3.9 GHz + Cortex-A725 @ 2.8 GHz) |
| **Memory** | 128 GB LPDDR5X Coherent Unified System Memory |
| **CPU-GPU Interconnect** | NVLink-C2C (5x PCIe 5.0 bandwidth) |
| **Storage** | 916 GB NVMe SSD |
| **Networking** | NVIDIA ConnectX-7, 10 GbE LAN |
| **Connectivity** | HDMI 2.1b, DisplayPort 2.1, USB 3.2 Gen 2x2 Type-C |
| **Dimensions** | 282.4 x 187.7 x 56.5 mm |
| **Cooling** | Vapor chamber with dual-fan, 7-level thermal control |
| **Power (GPU under load)** | ~33-47W |

### Software Stack

| Software | Version | Purpose |
|----------|---------|---------|
| **CUDA** | 13.0 | NVIDIA GPU compute platform |
| **Driver** | 580.142 | NVIDIA kernel driver |
| **OS** | Ubuntu 24.04.4 LTS | Operating system |
| **Kernel** | 6.17.0-1014-nvidia | Linux kernel (NVIDIA variant) |
| **llama.cpp** | Latest (CUDA backend) | LLM inference engine |
| **Python** | 3.12 | Report generation |
| **Bash** | 5.x | Benchmark orchestration |

### Model Source

- **Source**: Hugging Face (`Qwen/Qwen2.5-{SIZE}-Instruct-GGUF`)
- **Format**: GGUF (GPT-Generated Unified Format)
- **Local storage**: `models/` directory (downloaded via `huggingface-cli` / `hf`)
- **Note**: 7B+ models are split into multiple shard files; llama.cpp loads them natively from the first shard

---

## What We're Benchmarking

### Model Sizes

We test 4 model sizes from the Qwen2.5-Instruct family:

| Model | Parameters | Use Case |
|-------|-----------|----------|
| **3B** | 3 billion | Fast, lightweight responses |
| **7B** | 7 billion | Balanced performance/quality |
| **14B** | 14 billion | High quality responses |
| **32B** | 32 billion | Maximum quality |

### Quantization Levels

**Quantization** reduces model size/memory by using lower precision numbers. We test 3 levels:

| Quant | Bits | Size Impact | Quality | Speed |
|-------|------|-------------|---------|-------|
| **Q4_K_M** | 4-bit | Smallest (~2GB for 3B) | Good | Fastest |
| **Q5_K_M** | 5-bit | Medium (~2.3GB for 3B) | Better | Fast |
| **Q8_0** | 8-bit | Largest (~3.4GB for 3B) | Best | Slower |

**Note**: The ASUS Ascent GX10's 128GB unified memory fits **all** quantizations for **all** model sizes, including 32B Q8_0 (~34.8GB).

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
```

---

## Setup Instructions

### Prerequisites

1. **Build llama.cpp** with CUDA support
   ```bash
   git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
   cd llama.cpp
   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
   cmake --build build --target llama-bench llama-cli -j$(nproc)
   ```

2. **Download models** from Hugging Face
   ```bash
   pip install --user --break-system-packages huggingface-hub
   export PATH="$HOME/.local/bin:$PATH"

   cd benchmark-gx10/models

   # 3B (single files)
   hf download Qwen/Qwen2.5-3B-Instruct-GGUF \
     qwen2.5-3b-instruct-q4_k_m.gguf \
     qwen2.5-3b-instruct-q5_k_m.gguf \
     qwen2.5-3b-instruct-q8_0.gguf --local-dir .

   # 7B (split files)
   hf download Qwen/Qwen2.5-7B-Instruct-GGUF \
     qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
     qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf \
     qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf \
     qwen2.5-7b-instruct-q5_k_m-00002-of-00002.gguf \
     qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf \
     qwen2.5-7b-instruct-q8_0-00002-of-00003.gguf \
     qwen2.5-7b-instruct-q8_0-00003-of-00003.gguf --local-dir .

   # 14B (split files)
   hf download Qwen/Qwen2.5-14B-Instruct-GGUF \
     qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf \
     qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf \
     qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf \
     qwen2.5-14b-instruct-q5_k_m-00001-of-00003.gguf \
     qwen2.5-14b-instruct-q5_k_m-00002-of-00003.gguf \
     qwen2.5-14b-instruct-q5_k_m-00003-of-00003.gguf \
     qwen2.5-14b-instruct-q8_0-00001-of-00004.gguf \
     qwen2.5-14b-instruct-q8_0-00002-of-00004.gguf \
     qwen2.5-14b-instruct-q8_0-00003-of-00004.gguf \
     qwen2.5-14b-instruct-q8_0-00004-of-00004.gguf --local-dir .

   # 32B (split files — including Q8_0 which fits in 128GB unified memory)
   hf download Qwen/Qwen2.5-32B-Instruct-GGUF \
     qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf \
     qwen2.5-32b-instruct-q4_k_m-00002-of-00005.gguf \
     qwen2.5-32b-instruct-q4_k_m-00003-of-00005.gguf \
     qwen2.5-32b-instruct-q4_k_m-00004-of-00005.gguf \
     qwen2.5-32b-instruct-q4_k_m-00005-of-00005.gguf \
     qwen2.5-32b-instruct-q5_k_m-00001-of-00006.gguf \
     qwen2.5-32b-instruct-q5_k_m-00002-of-00006.gguf \
     qwen2.5-32b-instruct-q5_k_m-00003-of-00006.gguf \
     qwen2.5-32b-instruct-q5_k_m-00004-of-00006.gguf \
     qwen2.5-32b-instruct-q5_k_m-00005-of-00006.gguf \
     qwen2.5-32b-instruct-q5_k_m-00006-of-00006.gguf \
     qwen2.5-32b-instruct-q8_0-00001-of-00009.gguf \
     qwen2.5-32b-instruct-q8_0-00002-of-00009.gguf \
     qwen2.5-32b-instruct-q8_0-00003-of-00009.gguf \
     qwen2.5-32b-instruct-q8_0-00004-of-00009.gguf \
     qwen2.5-32b-instruct-q8_0-00005-of-00009.gguf \
     qwen2.5-32b-instruct-q8_0-00006-of-00009.gguf \
     qwen2.5-32b-instruct-q8_0-00007-of-00009.gguf \
     qwen2.5-32b-instruct-q8_0-00008-of-00009.gguf \
     qwen2.5-32b-instruct-q8_0-00009-of-00009.gguf --local-dir .
   ```

3. **Verify CUDA**
   ```bash
   nvidia-smi
   # Should show: NVIDIA GB10 (ASUS Ascent GX10), Driver 580.142, CUDA 13.0
   ```

4. **Set script execute permissions**
   ```bash
   chmod +x scripts/*.sh scripts/*.py
   ```

### Directory Structure

```
benchmark-gx10/
├── models/                      # GGUF model files (downloaded from HF)
├── results/                     # Benchmark outputs
│   ├── benchmark_YYYYMMDD_HHMMSS.csv     # Raw throughput data
│   ├── benchmark_YYYYMMDD_HHMMSS.json    # Metadata + results
│   ├── benchmark_YYYYMMDD_HHMMSS.log     # Detailed log
│   ├── benchmark_report_gx10.html        # Visual report
│   ├── token_per_watt_YYYYMMDD_HHMMSS.csv   # Power efficiency data
│   ├── token_per_watt_YYYYMMDD_HHMMSS.json  # Power efficiency metadata
│   └── token_per_watt_YYYYMMDD_HHMMSS.log   # Power efficiency log
├── scripts/
│   ├── run_benchmark_gx10.sh             # Main benchmark (CUDA)
│   ├── run_token_per_watt_gx10.sh        # Power efficiency benchmark (nvidia-smi)
│   └── generate_report_gx10.py           # HTML report generator
├── docs/                        # Documentation and reports
└── README.md
```

---

## Running Benchmarks

### Quick Start

```bash
cd ~/benchmark-gx10
bash scripts/run_benchmark_gx10.sh
```

**Duration**: ~15 minutes for all 12 model configurations

### Token-Per-Watt Benchmark

```bash
bash scripts/run_token_per_watt_gx10.sh
```

**Duration**: ~55 minutes (5 reps x 512 tokens x 12 configs, with power sampling)

This script:
- Runs TG-only `llama-bench` tests across all model sizes and quants
- Samples GPU power every 0.5s with `nvidia-smi --query-gpu=power.draw`
- Computes: `tok/s`, `tok/W`, `J/token`, `sec/1k tokens`, `RM/1k tokens`

Example with custom electricity rate:
```bash
ELECTRICITY_COST_MYR_PER_KWH=0.55 bash scripts/run_token_per_watt_gx10.sh
```

### Generate HTML Report

```bash
python3 scripts/generate_report_gx10.py
firefox results/benchmark_report_gx10.html
```

### Monitor GPU During Benchmark

```bash
watch -n 1 nvidia-smi
```

---

## Understanding Results

### Terminal Output

During the benchmark, you'll see:

#### 1. Opening Banner
```
  Device       ASUS Ascent GX10
  SoC          NVIDIA GB10 Grace Blackwell Superchip
  Memory       128 GB LPDDR5X Unified (CPU+GPU shared)
  CUDA         13.0
  Driver       580.142
  CPU          ARM Cortex-X925 + Cortex-A725 (20 cores)
  Backend      llama.cpp (CUDA)
  Flash Attn   Enabled
```

#### 2. Per-Model Progress
```
  MODEL 1/4: Qwen2.5-3B-Instruct-GGUF

     [1/12] Qwen2.5-3B-Instruct-GGUF Q4_K_M (2.0 GB)
     PP: 128, 256, 512 tokens | TG: 128 tokens | Reps: 3

         PP  128 tokens  ->  2466.1 tok/s
         PP  256 tokens  ->  3291.7 tok/s
         PP  512 tokens  ->  3674.1 tok/s
         TG  128 tokens  ->    44.1 tok/s

     DONE  Qwen2.5-3B-Instruct-GGUF Q4_K_M (12s)
```

### Output Files

| File | Format | Contents |
|------|--------|----------|
| `benchmark_*.csv` | CSV | Raw PP/TG throughput data |
| `benchmark_*.json` | JSON | Metadata + results |
| `benchmark_*.log` | Text | Full execution log |
| `benchmark_report_gx10.html` | HTML | Interactive visual report with charts |
| `token_per_watt_*.csv` | CSV | Power efficiency data |
| `token_per_watt_*.json` | JSON | Power efficiency metadata |

---

## Script Reference

### GX10-Specific Scripts

| Script | Purpose |
|--------|---------|
| `run_benchmark_gx10.sh` | Main throughput benchmark (CUDA, nvidia-smi) |
| `run_token_per_watt_gx10.sh` | Power efficiency benchmark (nvidia-smi power sampling) |
| `generate_report_gx10.py` | HTML report generator for GX10 results |

### Configuration Variables

Edit at top of `run_benchmark_gx10.sh`:

```bash
LLAMA_BENCH="/home/gx10/llama.cpp/build/bin/llama-bench"
LLAMA_CLI="/home/gx10/llama.cpp/build/bin/llama-cli"
MODEL_DIR="$(pwd)/models"

PP_LENGTHS="128,256,512"      # Prompt processing test sizes
TG_LENGTH=128                  # Text generation test size
N_REPS=3                       # Repetitions per test (averaged)
TOTAL_VRAM_GB=128              # GB10 unified memory
```

For token-per-watt runs:
```bash
TG_LENGTH=512
N_REPS=5
POWER_SAMPLE_INTERVAL_SEC=0.5
GPU_INDEX=0
ELECTRICITY_COST_MYR_PER_KWH=0.55
```

---

## Troubleshooting

### CUDA Error on Launch

**Problem**: `CUDA error` in `ggml_cuda_mul_mat_q`

**Solution**: Rebuild llama.cpp with native architecture:
```bash
cd ~/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target llama-bench llama-cli -j$(nproc)
```

The ASUS Ascent GX10's GB10 SoC is compute capability 12.1 — using `-DCMAKE_CUDA_ARCHITECTURES=native` ensures correct code generation.

### Library Not Found

**Problem**: `error while loading shared libraries: libllama.so.0`

**Solution**: The script sets `LD_LIBRARY_PATH` automatically. If you still see this:
```bash
export LD_LIBRARY_PATH="/home/gx10/llama.cpp/build/bin:$LD_LIBRARY_PATH"
```

### No Model Files

**Problem**: `ERROR: No .gguf model files found`

**Solution**: Download models from Hugging Face (see [Setup Instructions](#setup-instructions)).

### Empty Results Tables

**Problem**: Summary tables show all zeros

**Debug**:
```bash
# Test llama-bench manually
/home/gx10/llama.cpp/build/bin/llama-bench \
    -m models/qwen2.5-3b-instruct-q4_k_m.gguf \
    -p 128 -n 128 -r 1 -ngl 99 -fa 1 -o csv
```

---

## Advanced Usage

### Custom Model Sizes

```bash
# Test only 3B and 7B
MODEL_SIZES_CSV="3B,7B" bash scripts/run_token_per_watt_gx10.sh
```

### Quick Learning Run

```bash
MODEL_SIZES_CSV=3B N_REPS=2 TG_LENGTH=256 bash scripts/run_token_per_watt_gx10.sh
```

### Parallel Benchmarking

**DO NOT** run multiple benchmarks simultaneously — GPU contention will skew results.

---

## Measured Results (2026-04-03)

### System Overview (ASUS Ascent GX10)

| Component | Value |
|----------|-------|
| **Device** | ASUS Ascent GX10 |
| **SoC** | NVIDIA GB10 Grace Blackwell Superchip |
| **OS** | Ubuntu 24.04.4 LTS |
| **Kernel** | 6.17.0-1014-nvidia |
| **CUDA** | 13.0 |
| **Driver** | 580.142 |
| **GPU** | NVIDIA Blackwell with 5th-gen Tensor Cores |
| **GPU Architecture** | Blackwell (Compute Capability 12.1) |
| **AI Performance** | Up to 1 PFLOP (FP4) |
| **Memory** | 128 GB LPDDR5X Coherent Unified (CPU+GPU shared) |
| **CPU-GPU Interconnect** | NVLink-C2C |
| **GPU Power (under load)** | ~33-47W |
| **CPU** | 20-core ARM (Cortex-X925 + Cortex-A725, big.LITTLE) |
| **Storage** | 916 GB NVMe SSD |
| **Dimensions** | 282.4 x 187.7 x 56.5 mm |
| **Inference Backend** | llama.cpp CUDA backend |
| **Flash Attention** | Enabled |
| **Benchmark Repetitions** | 3 per configuration |

### Headline Performance

- **Peak Prompt Processing (PP 512)**: **6761.8 tok/s**
  `Qwen2.5-3B-Instruct` + `Q4_K_M`
- **Peak Text Generation (TG 128)**: **94.4 tok/s**
  `Qwen2.5-3B-Instruct` + `Q4_K_M`
- **Best Energy Efficiency**: **2.62 tok/W**
  `Qwen2.5-3B-Instruct` + `Q4_K_M`
- **All 12 configurations tested** (including 32B Q8_0)

### TG 128 — Text Generation Speed (tok/s)

| Model | Q4_K_M | Q5_K_M | Q8_0 |
|------|--------|--------|------|
| **Qwen2.5-3B** | **94.4** | 81.3 | 63.7 |
| **Qwen2.5-7B** | 44.0 | 37.2 | 28.7 |
| **Qwen2.5-14B** | 22.8 | 18.8 | 14.2 |
| **Qwen2.5-32B** | 9.7 | 8.5 | 6.3 |

### PP 512 — Prompt Processing Speed (tok/s)

| Model | Q4_K_M | Q5_K_M | Q8_0 |
|------|--------|--------|------|
| **Qwen2.5-3B** | **6761.8** | 6583.2 | 5790.4 |
| **Qwen2.5-7B** | 3424.1 | 3222.5 | 2581.6 |
| **Qwen2.5-14B** | 1672.2 | 1474.4 | 1167.0 |
| **Qwen2.5-32B** | 705.3 | 619.9 | 492.2 |

### Token-Per-Watt Efficiency (TG 512, sorted by tok/W)

| Model | Quant | tok/s | Avg W | tok/W | RM/1M tokens |
|-------|-------|------:|------:|------:|-------------:|
| 3B | Q4_K_M | 95.9 | 36.6 | **2.620** | RM 0.06 |
| 3B | Q5_K_M | 82.5 | 37.5 | 2.201 | RM 0.07 |
| 3B | Q8_0 | 64.0 | 32.7 | 1.960 | RM 0.08 |
| 7B | Q4_K_M | 44.3 | 40.0 | 1.108 | RM 0.14 |
| 7B | Q5_K_M | 37.4 | 39.7 | 0.942 | RM 0.16 |
| 7B | Q8_0 | 28.7 | 33.4 | 0.860 | RM 0.18 |
| 14B | Q4_K_M | 22.8 | 41.5 | 0.550 | RM 0.28 |
| 14B | Q8_0 | 14.2 | 32.6 | 0.436 | RM 0.35 |
| 14B | Q5_K_M | 18.7 | 47.1 | 0.397 | RM 0.38 |
| 32B | Q4_K_M | 10.1 | 44.3 | 0.229 | RM 0.67 |
| 32B | Q5_K_M | 8.5 | 42.4 | 0.200 | RM 0.76 |
| 32B | Q8_0 | 6.3 | 42.8 | 0.148 | RM 1.03 |

### Key Observations

- The ASUS Ascent GX10 draws only **33-47W under load** — remarkably power-efficient for a 128GB unified memory AI supercomputer
- All 12 configurations completed successfully, including **32B Q8_0** (34.8GB)
- `Q4_K_M` was fastest across all tested sizes
- 3B Q4_K_M achieves **94.4 tok/s** TG — well above human reading speed, ideal for real-time chat
- 7B models run at **28-44 tok/s** — smooth interactive experience
- 14B models at **14-23 tok/s** — usable for chat with higher quality
- Power draw is relatively flat across model sizes thanks to the unified memory architecture
- Electricity cost ranges from **RM 0.06/M tokens** (3B Q4) to **RM 1.03/M tokens** (32B Q8) — extremely affordable local inference

---

## Performance Interpretation

### What's "Good" Performance?

**Prompt Processing (PP)**:
- **>3000 tok/s**: Excellent - instant prompt understanding
- **1000-3000 tok/s**: Good - barely noticeable delay
- **500-1000 tok/s**: Acceptable - slight delay
- **<500 tok/s**: Slow - noticeable wait time

**Text Generation (TG)**:
- **>40 tok/s**: Excellent - faster than reading speed
- **20-40 tok/s**: Good - smooth experience
- **10-20 tok/s**: Acceptable - usable for chat
- **<10 tok/s**: Slow - noticeable lag

### Trade-offs

| Metric | Q4_K_M | Q5_K_M | Q8_0 |
|--------|---------|---------|------|
| **Speed** | Fastest | Fast | Slower |
| **Quality** | Good | Better | Best |
| **Memory** | Lowest | Medium | Highest |

**Recommendation**: Q4_K_M for most use cases — offers best speed/quality balance on the ASUS Ascent GX10.

---

## Models & Quantizations

| Model | Quant | Est. Size | Fits GX10 | Shard Files |
|-------|-------|-----------|-----------|-------------|
| Qwen2.5-3B-Instruct | Q4_K_M | ~2.0 GB | Yes | 1 |
| Qwen2.5-3B-Instruct | Q5_K_M | ~2.3 GB | Yes | 1 |
| Qwen2.5-3B-Instruct | Q8_0 | ~3.4 GB | Yes | 1 |
| Qwen2.5-7B-Instruct | Q4_K_M | ~4.4 GB | Yes | 2 |
| Qwen2.5-7B-Instruct | Q5_K_M | ~5.1 GB | Yes | 2 |
| Qwen2.5-7B-Instruct | Q8_0 | ~7.8 GB | Yes | 3 |
| Qwen2.5-14B-Instruct | Q4_K_M | ~8.7 GB | Yes | 3 |
| Qwen2.5-14B-Instruct | Q5_K_M | ~10.1 GB | Yes | 3 |
| Qwen2.5-14B-Instruct | Q8_0 | ~15.3 GB | Yes | 4 |
| Qwen2.5-32B-Instruct | Q4_K_M | ~19.5 GB | Yes | 5 |
| Qwen2.5-32B-Instruct | Q5_K_M | ~22.7 GB | Yes | 6 |
| Qwen2.5-32B-Instruct | Q8_0 | ~34.8 GB | **Yes** | 9 |

All 12 configurations fit in the ASUS Ascent GX10's 128GB unified memory.

---

## Citation & Credits

**Device**: [ASUS Ascent GX10](https://www.asus.com/my/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/) (NVIDIA GB10 Grace Blackwell Superchip)
**Models**: Qwen2.5-Instruct by Alibaba Cloud (Hugging Face: Qwen/Qwen2.5-{SIZE}-Instruct-GGUF)
**Inference Engine**: llama.cpp by ggerganov (CUDA backend)
**Compute Platform**: CUDA 13.0 by NVIDIA

---

## License

This benchmark suite is provided as-is for educational and evaluation purposes.
