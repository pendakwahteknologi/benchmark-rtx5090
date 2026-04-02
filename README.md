# Qwen2.5 GGUF Benchmark Suite for NVIDIA GeForce RTX 5090

Comprehensive benchmarking suite for testing Qwen2.5 language models (3B, 7B, 14B, 32B) across multiple quantization levels on a server equipped with the NVIDIA GeForce RTX 5090 (32GB VRAM), dual AMD EPYC 7543 CPUs (120 vCPUs), and 944GB system RAM, using CUDA and llama.cpp.

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

This benchmark suite measures **inference performance** of Qwen2.5 language models in GGUF format, testing how fast an NVIDIA GeForce RTX 5090 server can:

1. **Process input prompts** (Prompt Processing / PP) - How quickly the model reads and understands your input
2. **Generate output tokens** (Text Generation / TG) - How quickly the model writes responses

The RTX 5090 has **32GB GDDR7 VRAM** with **944GB system RAM** available for CPU offload, meaning all model sizes and quantizations can be tested — though 32B Q8_0 (~34.8GB) requires partial CPU offload as it exceeds the 32GB VRAM.

The suite automatically:
- Downloads models from Hugging Face (or uses pre-downloaded models)
- Runs benchmarks across 4 model sizes x 3 quantization levels = **12 configurations**
- Generates detailed results with pretty terminal output + HTML reports
- Measures power efficiency (tok/W) via `nvidia-smi`

---

## Cost Per Token (RM)

This repo includes a dedicated **cost-per-token benchmark**:

- Script: `scripts/run_token_per_watt_rtx5090.sh`
- Power monitoring: `nvidia-smi` (GPU power.draw query)
- Currency: **MYR (RM)** via `ELECTRICITY_COST_MYR_PER_KWH`

It measures generation speed and energy efficiency, then estimates electricity cost:

- `tok/s` (speed)
- `tok/W` (efficiency)
- `sec/1k tok` (latency style metric)
- `RM/1k tok` and `RM/1M tok` (energy cost estimate)

Quick start:

```bash
bash scripts/run_token_per_watt_rtx5090.sh
```

The main throughput benchmark:

```bash
bash scripts/run_benchmark_rtx5090.sh
```

---

## Hardware & Software

### Hardware Specifications

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA GeForce RTX 5090 |
| **GPU Architecture** | Blackwell (Compute Capability 12.0) |
| **VRAM** | 32 GB GDDR7 |
| **CPU** | 2x AMD EPYC 7543 32-Core Processor (120 vCPUs) |
| **System RAM** | 944 GB |
| **Storage** | 100 GB boot + 6.4 TB data |
| **Virtualization** | KVM (AMD-V) |
| **NUMA Nodes** | 2 (60 CPUs each) |

### Software Stack

| Software | Version | Purpose |
|----------|---------|---------|
| **CUDA** | 12.8 | NVIDIA GPU compute platform |
| **Driver** | 570.153.02 | NVIDIA kernel driver |
| **OS** | Ubuntu 24.04.3 LTS | Operating system |
| **Kernel** | 6.11.0-26-generic | Linux kernel |
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

**Note**: The RTX 5090's 32GB VRAM fits most configurations fully on GPU. The 32B Q8_0 (~34.8GB) exceeds VRAM and uses partial CPU offload via the server's 944GB system RAM.

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

   cd benchmark-rtx5090/models

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

   # 32B (split files — Q8_0 ~34.8GB exceeds 32GB VRAM, uses partial CPU offload)
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
   # Should show: NVIDIA GeForce RTX 5090, Driver 570.153.02, CUDA 12.8
   ```

4. **Set script execute permissions**
   ```bash
   chmod +x scripts/*.sh scripts/*.py
   ```

### Directory Structure

```
benchmark-rtx5090/
├── models/                      # GGUF model files (downloaded from HF)
├── results/                     # Benchmark outputs
│   ├── benchmark_YYYYMMDD_HHMMSS.csv     # Raw throughput data
│   ├── benchmark_YYYYMMDD_HHMMSS.json    # Metadata + results
│   ├── benchmark_YYYYMMDD_HHMMSS.log     # Detailed log
│   ├── benchmark_report_rtx5090.html        # Visual report
│   ├── token_per_watt_YYYYMMDD_HHMMSS.csv   # Power efficiency data
│   ├── token_per_watt_YYYYMMDD_HHMMSS.json  # Power efficiency metadata
│   └── token_per_watt_YYYYMMDD_HHMMSS.log   # Power efficiency log
├── scripts/
│   ├── run_benchmark_rtx5090.sh             # Main benchmark (CUDA)
│   ├── run_token_per_watt_rtx5090.sh        # Power efficiency benchmark (nvidia-smi)
│   └── generate_report_rtx5090.py           # HTML report generator
├── docs/                        # Documentation and reports
└── README.md
```

---

## Running Benchmarks

### Quick Start

```bash
cd ~/benchmark-rtx5090
bash scripts/run_benchmark_rtx5090.sh
```

**Duration**: ~15 minutes for all 12 model configurations

### Token-Per-Watt Benchmark

```bash
bash scripts/run_token_per_watt_rtx5090.sh
```

**Duration**: ~55 minutes (5 reps x 512 tokens x 12 configs, with power sampling)

This script:
- Runs TG-only `llama-bench` tests across all model sizes and quants
- Samples GPU power every 0.5s with `nvidia-smi --query-gpu=power.draw`
- Computes: `tok/s`, `tok/W`, `J/token`, `sec/1k tokens`, `RM/1k tokens`

Example with custom electricity rate:
```bash
ELECTRICITY_COST_MYR_PER_KWH=0.55 bash scripts/run_token_per_watt_rtx5090.sh
```

### Generate HTML Report

```bash
python3 scripts/generate_report_rtx5090.py
firefox results/benchmark_report_rtx5090.html
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
  GPU          NVIDIA GeForce RTX 5090
  Architecture Blackwell (Compute Capability 12.0)
  VRAM         32 GB GDDR7
  System RAM   944 GB
  CUDA         12.8
  Driver       570.153.02
  CPU          2x AMD EPYC 7543 (120 vCPUs)
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
| `benchmark_report_rtx5090.html` | HTML | Interactive visual report with charts |
| `token_per_watt_*.csv` | CSV | Power efficiency data |
| `token_per_watt_*.json` | JSON | Power efficiency metadata |

---

## Script Reference

### RTX 5090 Benchmark Scripts

| Script | Purpose |
|--------|---------|
| `run_benchmark_rtx5090.sh` | Main throughput benchmark (CUDA, nvidia-smi) |
| `run_token_per_watt_rtx5090.sh` | Power efficiency benchmark (nvidia-smi power sampling) |
| `generate_report_rtx5090.py` | HTML report generator for benchmark results |

### Configuration Variables

Edit at top of `run_benchmark_rtx5090.sh`:

```bash
LLAMA_BENCH="/workspace/llama.cpp/build/bin/llama-bench"
LLAMA_CLI="/workspace/llama.cpp/build/bin/llama-cli"
MODEL_DIR="$(pwd)/models"

PP_LENGTHS="128,256,512"      # Prompt processing test sizes
TG_LENGTH=128                  # Text generation test size
N_REPS=3                       # Repetitions per test (averaged)
TOTAL_VRAM_GB=32               # RTX 5090 VRAM
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

The RTX 5090 is compute capability 12.0 — using `-DCMAKE_CUDA_ARCHITECTURES=native` ensures correct code generation.

### Library Not Found

**Problem**: `error while loading shared libraries: libllama.so.0`

**Solution**: The script sets `LD_LIBRARY_PATH` automatically. If you still see this:
```bash
export LD_LIBRARY_PATH="/workspace/llama.cpp/build/bin:$LD_LIBRARY_PATH"
```

### No Model Files

**Problem**: `ERROR: No .gguf model files found`

**Solution**: Download models from Hugging Face (see [Setup Instructions](#setup-instructions)).

### Empty Results Tables

**Problem**: Summary tables show all zeros

**Debug**:
```bash
# Test llama-bench manually
/workspace/llama.cpp/build/bin/llama-bench \
    -m models/qwen2.5-3b-instruct-q4_k_m.gguf \
    -p 128 -n 128 -r 1 -ngl 99 -fa 1 -o csv
```

---

## Advanced Usage

### Custom Model Sizes

```bash
# Test only 3B and 7B
MODEL_SIZES_CSV="3B,7B" bash scripts/run_token_per_watt_rtx5090.sh
```

### Quick Learning Run

```bash
MODEL_SIZES_CSV=3B N_REPS=2 TG_LENGTH=256 bash scripts/run_token_per_watt_rtx5090.sh
```

### Parallel Benchmarking

**DO NOT** run multiple benchmarks simultaneously — GPU contention will skew results.

---

## Measured Results (2026-04-03)

### System Overview (NVIDIA RTX 5090 Server)

| Component | Value |
|----------|-------|
| **GPU** | NVIDIA GeForce RTX 5090 |
| **GPU Architecture** | Blackwell (Compute Capability 12.0) |
| **VRAM** | 32 GB GDDR7 |
| **CPU** | 2x AMD EPYC 7543 32-Core (120 vCPUs) |
| **System RAM** | 944 GB |
| **OS** | Ubuntu 24.04.3 LTS |
| **Kernel** | 6.11.0-26-generic |
| **CUDA** | 12.8 |
| **Driver** | 570.153.02 |
| **Storage** | 100 GB boot + 6.4 TB data |
| **Virtualization** | KVM (AMD-V) |
| **NUMA Nodes** | 2 (60 CPUs each) |
| **Inference Backend** | llama.cpp CUDA backend |
| **Flash Attention** | Enabled |
| **Benchmark Repetitions** | 3 per configuration |

### Headline Performance

*Results will be populated after running the benchmark on this server.*

*Detailed results tables will be populated after running the benchmark.*

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

**Recommendation**: Q4_K_M for most use cases — offers best speed/quality balance on the RTX 5090.

---

## Models & Quantizations

| Model | Quant | Est. Size | Fits in 32GB VRAM | Shard Files |
|-------|-------|-----------|-------------------|-------------|
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
| Qwen2.5-32B-Instruct | Q8_0 | ~34.8 GB | Partial (CPU offload) | 9 |

11 of 12 configurations fit fully in the RTX 5090's 32GB VRAM. The 32B Q8_0 (~34.8GB) uses partial CPU offload via the 944GB system RAM.

---

## Citation & Credits

**GPU**: NVIDIA GeForce RTX 5090 (32GB GDDR7)
**CPU**: 2x AMD EPYC 7543 32-Core (120 vCPUs)
**Models**: Qwen2.5-Instruct by Alibaba Cloud (Hugging Face: Qwen/Qwen2.5-{SIZE}-Instruct-GGUF)
**Inference Engine**: llama.cpp by ggerganov (CUDA backend)
**Compute Platform**: CUDA 12.8 by NVIDIA

---

## License

This benchmark suite is provided as-is for educational and evaluation purposes.
