#!/usr/bin/env python3
"""
Generate Comprehensive HTML Benchmark Report for NVIDIA GB10 (GX10)
Reads ALL CSV results from llama-bench and token-per-watt benchmarks,
aggregates across runs, and produces a professional interactive HTML report.

Note: This report uses locally-generated data only (from our own benchmark CSV).
No external/untrusted content is rendered. All innerHTML usage operates on
data produced by our own benchmark scripts.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import sys


def get_system_info():
    """Gather system information for the report."""
    info = {}
    try:
        info['gpu'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            text=True).strip()
    except Exception:
        info['gpu'] = 'NVIDIA GB10'

    try:
        info['driver'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            text=True).strip()
    except Exception:
        info['driver'] = 'Unknown'

    try:
        cuda_out = subprocess.check_output(['nvcc', '--version'], text=True)
        for line in cuda_out.splitlines():
            if 'release' in line:
                info['cuda'] = line.split('release ')[-1].split(',')[0]
                break
    except Exception:
        info['cuda'] = 'Unknown'

    try:
        lscpu = subprocess.check_output(['lscpu'], text=True)
        for line in lscpu.splitlines():
            if 'Model name' in line:
                info['cpu'] = line.split(':')[-1].strip()
                break
    except Exception:
        info['cpu'] = 'Unknown'

    try:
        mem = subprocess.check_output(['free', '-h'], text=True)
        for line in mem.splitlines():
            if 'Mem:' in line:
                info['ram'] = line.split()[1]
                break
    except Exception:
        info['ram'] = 'Unknown'

    try:
        info['vram'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader'],
            text=True).strip()
    except Exception:
        info['vram'] = 'Unknown'

    return info


def load_benchmark_csvs(results_dir):
    """Load all benchmark CSVs and return per-run data + aggregated best-run data."""
    csvs = sorted(results_dir.glob("benchmark_*.csv"), key=lambda p: p.name)
    all_runs = []
    for csv_path in csvs:
        rows = []
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if rows:
            ts = csv_path.stem.replace("benchmark_", "")
            all_runs.append({"timestamp": ts, "file": csv_path.name, "rows": rows})
    return all_runs


def load_tpw_csvs(results_dir):
    """Load all token-per-watt CSVs."""
    csvs = sorted(results_dir.glob("token_per_watt_*.csv"), key=lambda p: p.name)
    all_runs = []
    for csv_path in csvs:
        rows = []
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if rows:
            ts = csv_path.stem.replace("token_per_watt_", "")
            all_runs.append({"timestamp": ts, "file": csv_path.name, "rows": rows})
    return all_runs


def load_metadata(results_dir):
    """Load metadata from the latest benchmark JSON."""
    jsons = sorted(results_dir.glob("benchmark_*.json"), key=lambda p: p.name)
    if jsons:
        with jsons[-1].open("r") as f:
            return json.load(f).get("metadata", {})
    return {}


def aggregate_best_run(all_runs):
    """For each model/quant/test combo, pick the best (highest tok/s) across runs."""
    best = {}
    for run in all_runs:
        for row in run["rows"]:
            key = (row["model_size"], row["quant"],
                   int(row["pp_tokens"]), int(row["tg_tokens"]))
            pp_val = float(row["pp_tok_sec"])
            tg_val = float(row["tg_tok_sec"])
            val = pp_val if pp_val > 0 else tg_val
            if key not in best or val > best[key]["value"]:
                best[key] = {"row": row, "value": val, "run": run["timestamp"]}
    return [b["row"] for b in best.values()]


def aggregate_average(all_runs):
    """Average across all runs for each model/quant/test combo."""
    sums = defaultdict(lambda: {"pp_tok_sec": 0, "tg_tok_sec": 0, "count": 0})
    for run in all_runs:
        for row in run["rows"]:
            key = (row["model_size"], row["quant"],
                   int(row["pp_tokens"]), int(row["tg_tokens"]))
            sums[key]["pp_tok_sec"] += float(row["pp_tok_sec"])
            sums[key]["tg_tok_sec"] += float(row["tg_tok_sec"])
            sums[key]["count"] += 1
            sums[key]["model_size"] = row["model_size"]
            sums[key]["quant"] = row["quant"]
            sums[key]["pp_tokens"] = row["pp_tokens"]
            sums[key]["tg_tokens"] = row["tg_tokens"]
            sums[key]["model_file"] = row["model_file"]
            sums[key]["vram_used_mb"] = row.get("vram_used_mb", "0")
    result = []
    for key, d in sums.items():
        result.append({
            "model_size": d["model_size"],
            "quant": d["quant"],
            "pp_tokens": d["pp_tokens"],
            "tg_tokens": d["tg_tokens"],
            "pp_tok_sec": str(d["pp_tok_sec"] / d["count"]),
            "tg_tok_sec": str(d["tg_tok_sec"] / d["count"]),
            "model_file": d["model_file"],
            "vram_used_mb": d["vram_used_mb"],
        })
    return result


def aggregate_tpw_latest(tpw_runs):
    """Return the latest token-per-watt run data."""
    if not tpw_runs:
        return []
    return tpw_runs[-1]["rows"]


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASUS Ascent GX10 — Qwen2.5 GGUF Benchmark Report</title>
    <style>
        :root {
            --primary: #0f3460;
            --primary-light: #1a4a80;
            --accent: #76b900;
            --accent-dark: #5a9000;
            --bg-dark: #0a0f1a;
            --bg-gradient: linear-gradient(135deg, #0a0f1a 0%, #0f1f3a 50%, #0f3460 100%);
            --card-bg: #ffffff;
            --card-border: #e8ecf1;
            --text-primary: #1a202c;
            --text-secondary: #4a5568;
            --text-muted: #718096;
            --q4-color: #3b82f6;
            --q5-color: #10b981;
            --q8-color: #f59e0b;
            --q4-bg: #dbeafe;
            --q5-bg: #d1fae5;
            --q8-bg: #fef3c7;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.08);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.1);
            --shadow-lg: 0 10px 40px rgba(0,0,0,0.15);
            --shadow-xl: 0 20px 60px rgba(0,0,0,0.25);
            --radius: 12px;
            --radius-lg: 16px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', Roboto, sans-serif;
            background: var(--bg-gradient);
            min-height: 100vh;
            color: var(--text-primary);
            line-height: 1.6;
            -webkit-font-smoothing: antialiased;
        }

        .page-wrapper {
            max-width: 1440px;
            margin: 0 auto;
            padding: 24px;
        }

        /* ── Header ── */
        .header {
            background: linear-gradient(135deg, #0f1f3a 0%, #0f3460 40%, #1a4a80 100%);
            border-radius: var(--radius-lg);
            padding: 48px 48px 40px;
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow-xl);
        }

        .header::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 600px;
            height: 600px;
            background: radial-gradient(circle, rgba(118,185,0,0.08) 0%, transparent 70%);
            pointer-events: none;
        }

        .header-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }

        .header-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .header-logo {
            width: 48px;
            height: 48px;
            background: var(--accent);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            font-weight: 800;
            color: white;
            letter-spacing: -1px;
        }

        .header-device {
            color: rgba(255,255,255,0.7);
            font-size: 0.85em;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .header-date {
            color: rgba(255,255,255,0.5);
            font-size: 0.85em;
        }

        .header h1 {
            font-size: 2.4em;
            font-weight: 700;
            color: white;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }

        .header-subtitle {
            font-size: 1.15em;
            color: rgba(255,255,255,0.75);
            margin-bottom: 20px;
        }

        .header-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            letter-spacing: 0.3px;
        }

        .badge-gpu {
            background: rgba(118,185,0,0.2);
            border: 1px solid rgba(118,185,0,0.4);
            color: #a8e060;
        }

        .badge-info {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            color: rgba(255,255,255,0.8);
        }

        .badge-runs {
            background: rgba(59,130,246,0.2);
            border: 1px solid rgba(59,130,246,0.4);
            color: #93c5fd;
        }

        /* ── Navigation ── */
        .nav {
            display: flex;
            gap: 4px;
            background: rgba(255,255,255,0.95);
            backdrop-filter: blur(10px);
            border-radius: var(--radius);
            padding: 5px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-md);
            position: sticky;
            top: 12px;
            z-index: 100;
            overflow-x: auto;
        }

        .nav-btn {
            flex: 1;
            min-width: 120px;
            padding: 12px 16px;
            text-align: center;
            cursor: pointer;
            background: transparent;
            border: none;
            font-size: 0.88em;
            font-weight: 600;
            color: var(--text-secondary);
            border-radius: 8px;
            transition: all 0.2s ease;
            white-space: nowrap;
        }

        .nav-btn:hover { background: #f0f4f8; color: var(--primary); }

        .nav-btn.active {
            background: var(--primary);
            color: white;
            box-shadow: 0 2px 8px rgba(15,52,96,0.3);
        }

        /* ── Panels ── */
        .panel { display: none; }
        .panel.active { display: block; }

        /* ── Cards ── */
        .card {
            background: var(--card-bg);
            border-radius: var(--radius);
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--card-border);
            overflow: hidden;
            margin-bottom: 20px;
        }

        .card-header {
            padding: 20px 24px 16px;
            border-bottom: 1px solid var(--card-border);
        }

        .card-title {
            font-size: 1.15em;
            font-weight: 700;
            color: var(--primary);
        }

        .card-subtitle {
            font-size: 0.85em;
            color: var(--text-muted);
            margin-top: 2px;
        }

        .card-body { padding: 24px; }

        /* ── Stat Grid ── */
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .stat-card {
            background: var(--card-bg);
            border-radius: var(--radius);
            padding: 20px 24px;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--card-border);
            position: relative;
            overflow: hidden;
        }

        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
        }

        .stat-card.accent-green::before { background: var(--accent); }
        .stat-card.accent-blue::before { background: var(--q4-color); }
        .stat-card.accent-amber::before { background: var(--q8-color); }
        .stat-card.accent-teal::before { background: #14b8a6; }
        .stat-card.accent-purple::before { background: #8b5cf6; }
        .stat-card.accent-rose::before { background: #f43f5e; }

        .stat-label {
            font-size: 0.78em;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        .stat-value {
            font-size: 2em;
            font-weight: 800;
            color: var(--primary);
            line-height: 1.1;
        }

        .stat-unit {
            font-size: 0.4em;
            font-weight: 600;
            color: var(--text-muted);
            margin-left: 2px;
        }

        .stat-detail {
            font-size: 0.8em;
            color: var(--text-muted);
            margin-top: 4px;
        }

        /* ── Info Box ── */
        .info-box {
            background: #f0f7ff;
            border-left: 4px solid var(--q4-color);
            padding: 16px 20px;
            border-radius: 0 8px 8px 0;
            margin-bottom: 24px;
            font-size: 0.9em;
            color: var(--text-secondary);
        }

        .info-box strong { color: var(--primary); }

        /* ── Tables ── */
        .table-wrapper {
            overflow-x: auto;
            border-radius: var(--radius);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }

        thead th {
            background: var(--primary);
            color: white;
            padding: 12px 16px;
            font-weight: 600;
            text-align: left;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            white-space: nowrap;
        }

        tbody td {
            padding: 11px 16px;
            border-bottom: 1px solid #f0f3f6;
            white-space: nowrap;
        }

        tbody tr:hover { background: #f8fafc; }
        tbody tr:last-child td { border-bottom: none; }

        .cell-value {
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }

        .cell-best {
            color: var(--accent-dark);
            position: relative;
        }

        .quant-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 10px;
            font-size: 0.82em;
            font-weight: 700;
            letter-spacing: 0.3px;
        }

        .qb-q4 { background: var(--q4-bg); color: #1d4ed8; }
        .qb-q5 { background: var(--q5-bg); color: #047857; }
        .qb-q8 { background: var(--q8-bg); color: #b45309; }

        /* ── Charts ── */
        .chart-wrap {
            background: #f8fafc;
            border-radius: var(--radius);
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid #eef2f6;
        }

        .chart-title {
            font-size: 1em;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 16px;
        }

        .bar-chart {
            display: flex;
            align-items: flex-end;
            gap: 12px;
            height: 260px;
            padding: 10px 0;
            border-bottom: 2px solid #e2e8f0;
        }

        .bar-group {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 4px;
            min-width: 80px;
        }

        .bar-group-bars {
            display: flex;
            align-items: flex-end;
            gap: 4px;
            width: 100%;
            justify-content: center;
        }

        .chart-bar {
            width: 32px;
            border-radius: 6px 6px 0 0;
            transition: all 0.2s ease;
            position: relative;
            cursor: pointer;
            min-height: 2px;
        }

        .chart-bar:hover {
            filter: brightness(1.15);
            transform: scaleY(1.02);
            transform-origin: bottom;
        }

        .chart-bar .tip {
            display: none;
            position: absolute;
            bottom: calc(100% + 8px);
            left: 50%;
            transform: translateX(-50%);
            background: var(--bg-dark);
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.78em;
            font-weight: 600;
            white-space: nowrap;
            z-index: 10;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }

        .chart-bar .tip::after {
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 5px solid transparent;
            border-top-color: var(--bg-dark);
        }

        .chart-bar:hover .tip { display: block; }

        .bar-label {
            font-size: 0.82em;
            font-weight: 600;
            color: var(--text-secondary);
            margin-top: 10px;
            text-align: center;
        }

        .legend {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 16px;
            flex-wrap: wrap;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.82em;
            font-weight: 600;
            color: var(--text-secondary);
        }

        .legend-dot {
            width: 14px;
            height: 14px;
            border-radius: 4px;
        }

        /* ── Horizontal bar chart ── */
        .hbar-chart { margin: 16px 0; }

        .hbar-row {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            gap: 12px;
        }

        .hbar-label {
            width: 160px;
            font-size: 0.82em;
            font-weight: 600;
            color: var(--text-secondary);
            text-align: right;
            flex-shrink: 0;
        }

        .hbar-track {
            flex: 1;
            height: 28px;
            background: #eef2f6;
            border-radius: 6px;
            overflow: hidden;
            position: relative;
        }

        .hbar-fill {
            height: 100%;
            border-radius: 6px;
            display: flex;
            align-items: center;
            padding-left: 10px;
            font-size: 0.78em;
            font-weight: 700;
            color: white;
            min-width: 60px;
            transition: width 0.5s ease;
        }

        .hbar-value {
            width: 90px;
            font-size: 0.85em;
            font-weight: 700;
            color: var(--text-primary);
            text-align: right;
            flex-shrink: 0;
            font-variant-numeric: tabular-nums;
        }

        /* ── Two-column layout ── */
        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        /* ── Specs grid ── */
        .specs-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
        }

        .spec-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 16px;
            border-bottom: 1px solid #f0f3f6;
        }

        .spec-row:last-child { border-bottom: none; }
        .spec-key { color: var(--text-muted); font-size: 0.88em; }
        .spec-val { font-weight: 700; color: var(--text-primary); font-size: 0.88em; }

        /* ── Run comparison ── */
        .run-tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.75em;
            font-weight: 700;
        }

        .run-latest { background: #dcfce7; color: #166534; }
        .run-older { background: #f1f5f9; color: #475569; }

        /* ── Footer ── */
        .footer {
            text-align: center;
            padding: 32px 24px;
            color: rgba(255,255,255,0.4);
            font-size: 0.82em;
        }

        .footer a { color: rgba(255,255,255,0.6); text-decoration: none; }

        /* ── Responsive ── */
        @media (max-width: 900px) {
            .page-wrapper { padding: 12px; }
            .header { padding: 28px 24px; }
            .header h1 { font-size: 1.6em; }
            .stat-grid { grid-template-columns: repeat(2, 1fr); }
            .grid-2 { grid-template-columns: 1fr; }
            .specs-grid { grid-template-columns: 1fr; }
            .bar-chart { gap: 6px; height: 200px; }
            .chart-bar { width: 22px; }
            .nav-btn { min-width: 90px; font-size: 0.8em; padding: 10px 8px; }
        }

        @media (max-width: 600px) {
            .stat-grid { grid-template-columns: 1fr; }
            .header-top { flex-direction: column; align-items: flex-start; gap: 8px; }
        }

        /* ── Print ── */
        @media print {
            body { background: white; }
            .nav { display: none; }
            .panel { display: block !important; page-break-inside: avoid; }
            .header { box-shadow: none; }
        }
    </style>
</head>
<body>
<div class="page-wrapper">

    <!-- Header -->
    <div class="header">
        <div class="header-top">
            <div class="header-brand">
                <div class="header-logo">GX</div>
                <div>
                    <div class="header-device">ASUS Ascent GX10</div>
                </div>
            </div>
            <div class="header-date">{TIMESTAMP}</div>
        </div>
        <h1>Qwen2.5 GGUF Benchmark Report</h1>
        <div class="header-subtitle">llama.cpp inference performance &mdash; NVIDIA Blackwell GB10 CUDA</div>
        <div class="header-badges">
            <span class="badge badge-gpu">{GPU_NAME}</span>
            <span class="badge badge-info">Blackwell (GB10)</span>
            <span class="badge badge-info">{RAM_TOTAL} Unified Memory</span>
            <span class="badge badge-info">CUDA {CUDA_VER}</span>
            <span class="badge badge-runs">{N_BENCH_RUNS} benchmark runs &bull; {N_TPW_RUNS} power runs</span>
        </div>
    </div>

    <!-- Navigation -->
    <div class="nav" id="nav">
        <button class="nav-btn active" onclick="switchTab('overview',this)">Overview</button>
        <button class="nav-btn" onclick="switchTab('pp',this)">Prompt Processing</button>
        <button class="nav-btn" onclick="switchTab('tg',this)">Text Generation</button>
        <button class="nav-btn" onclick="switchTab('power',this)">Power &amp; Efficiency</button>
        <button class="nav-btn" onclick="switchTab('runs',this)">Run History</button>
        <button class="nav-btn" onclick="switchTab('raw',this)">Raw Data</button>
    </div>

    <!-- ════════════ OVERVIEW ════════════ -->
    <div id="overview" class="panel active">
        <div class="stat-grid" id="ov-stats"></div>

        <div class="grid-2">
            <div class="chart-wrap" id="ov-tg-chart"></div>
            <div class="chart-wrap" id="ov-pp-chart"></div>
        </div>

        <div class="grid-2" id="ov-efficiency-row" style="display:none;">
            <div class="chart-wrap" id="ov-tpw-chart"></div>
            <div class="chart-wrap" id="ov-cost-chart"></div>
        </div>

        <div class="card">
            <div class="card-header">
                <div class="card-title">System Specifications</div>
                <div class="card-subtitle">Hardware and software configuration</div>
            </div>
            <div class="card-body" style="padding:0;">
                <div class="specs-grid" id="sys-specs"></div>
            </div>
        </div>
    </div>

    <!-- ════════════ PROMPT PROCESSING ════════════ -->
    <div id="pp" class="panel">
        <div class="info-box">
            <strong>Prompt Processing (PP)</strong> measures how fast the model ingests the input prompt.
            Higher tok/s = faster time-to-first-token. Tested at 128, 256, and 512 token prompt lengths.
            Values shown are <strong>best across {N_BENCH_RUNS} runs</strong>.
        </div>
        <div class="stat-grid" id="pp-stats"></div>
        <div class="chart-wrap" id="pp-chart-128"></div>
        <div class="chart-wrap" id="pp-chart-256"></div>
        <div class="chart-wrap" id="pp-chart-512"></div>
        <div class="card">
            <div class="card-header">
                <div class="card-title">Prompt Processing &mdash; Detailed Results</div>
                <div class="card-subtitle">Best observed values across all benchmark runs (tok/s)</div>
            </div>
            <div class="card-body" style="padding:0;">
                <div class="table-wrapper">
                    <table id="pp-table"><thead><tr>
                        <th>Model</th><th>Quant</th><th>PP 128</th><th>PP 256</th><th>PP 512</th>
                    </tr></thead><tbody></tbody></table>
                </div>
            </div>
        </div>
    </div>

    <!-- ════════════ TEXT GENERATION ════════════ -->
    <div id="tg" class="panel">
        <div class="info-box">
            <strong>Text Generation (TG)</strong> measures how fast the model produces output tokens.
            Higher tok/s = faster responses. This is the primary speed metric for interactive use.
            Tested at 128 output tokens. Values shown are <strong>best across {N_BENCH_RUNS} runs</strong>.
        </div>
        <div class="stat-grid" id="tg-stats"></div>
        <div class="chart-wrap" id="tg-chart-detail"></div>
        <div class="card">
            <div class="card-header">
                <div class="card-title">Text Generation &mdash; Detailed Results</div>
                <div class="card-subtitle">Best observed values across all benchmark runs (tok/s)</div>
            </div>
            <div class="card-body" style="padding:0;">
                <div class="table-wrapper">
                    <table id="tg-table"><thead><tr>
                        <th>Model</th><th>Quant</th><th>TG 128 (tok/s)</th><th>Usability</th>
                    </tr></thead><tbody></tbody></table>
                </div>
            </div>
        </div>
    </div>

    <!-- ════════════ POWER & EFFICIENCY ════════════ -->
    <div id="power" class="panel">
        <div class="info-box">
            <strong>Power Efficiency</strong> measures energy consumption during text generation (512 tokens).
            Metrics include tokens-per-watt, energy cost, and average GPU power draw.
            Electricity cost calculated at <strong>{ELEC_RATE} MYR/kWh</strong>.
        </div>
        <div class="stat-grid" id="pw-stats"></div>
        <div class="grid-2">
            <div class="chart-wrap" id="pw-tpw-chart"></div>
            <div class="chart-wrap" id="pw-power-chart"></div>
        </div>
        <div class="chart-wrap" id="pw-cost-hbar"></div>
        <div class="card">
            <div class="card-header">
                <div class="card-title">Power Efficiency &mdash; Full Results</div>
                <div class="card-subtitle">Latest token-per-watt benchmark data</div>
            </div>
            <div class="card-body" style="padding:0;">
                <div class="table-wrapper">
                    <table id="pw-table"><thead><tr>
                        <th>Model</th><th>Quant</th><th>TG tok/s</th><th>Avg Power (W)</th><th>Tokens/Watt</th><th>J/Token</th><th>kWh/1M tok</th><th>RM/1M tok</th>
                    </tr></thead><tbody></tbody></table>
                </div>
            </div>
        </div>
    </div>

    <!-- ════════════ RUN HISTORY ════════════ -->
    <div id="runs" class="panel">
        <div class="info-box">
            <strong>Run History</strong> compares performance across all benchmark sessions.
            This helps identify variance, warm-up effects, and consistency.
        </div>
        <div class="card">
            <div class="card-header">
                <div class="card-title">Benchmark Runs Summary</div>
                <div class="card-subtitle">Performance overview per run</div>
            </div>
            <div class="card-body" style="padding:0;">
                <div class="table-wrapper">
                    <table id="runs-table"><thead><tr>
                        <th>Run</th><th>Timestamp</th><th>Data Points</th><th>Peak PP (tok/s)</th><th>Peak TG (tok/s)</th><th>Status</th>
                    </tr></thead><tbody></tbody></table>
                </div>
            </div>
        </div>
        <div class="card" style="margin-top:20px;">
            <div class="card-header">
                <div class="card-title">Run-by-Run TG Comparison &mdash; 3B Q4_K_M</div>
                <div class="card-subtitle">Fastest config tracked across runs to show consistency</div>
            </div>
            <div class="card-body">
                <div class="chart-wrap" id="run-compare-chart"></div>
            </div>
        </div>
    </div>

    <!-- ════════════ RAW DATA ════════════ -->
    <div id="raw" class="panel">
        <div class="info-box">
            Complete benchmark data from llama-bench. Best values across <strong>{N_BENCH_RUNS} runs</strong>
            ({N_REPS} repetitions each). All values are tok/s.
        </div>
        <div class="card">
            <div class="card-header">
                <div class="card-title">Complete Benchmark Data</div>
            </div>
            <div class="card-body" style="padding:0;">
                <div class="table-wrapper">
                    <table id="raw-table"><thead><tr>
                        <th>Model</th><th>Quant</th><th>Type</th><th>Tokens</th><th>Speed (tok/s)</th><th>Source Run</th>
                    </tr></thead><tbody></tbody></table>
                </div>
            </div>
        </div>
    </div>

</div>

<div class="footer">
    <p>Generated {TIMESTAMP} &bull; {GPU_NAME} &bull; Blackwell (GB10) &bull; {RAM_TOTAL} Unified Memory &bull; CUDA {CUDA_VER} &bull; llama.cpp (CUDA)</p>
    <p style="margin-top:4px;">ASUS Ascent GX10 &mdash; AI Workstation Benchmark Suite</p>
</div>

<script>
// ── Data ──
var bestData = {BEST_DATA};
var avgData = {AVG_DATA};
var tpwData = {TPW_DATA};
var runsMeta = {RUNS_META};
var allRunsData = {ALL_RUNS_DATA};

// ── Helpers ──
var models = ['3B', '7B', '14B', '32B'];
var quants = ['Q4_K_M', 'Q5_K_M', 'Q8_0'];
var ppLens = [128, 256, 512];
var qColors = { 'Q4_K_M': '#3b82f6', 'Q5_K_M': '#10b981', 'Q8_0': '#f59e0b' };

function switchTab(id, btn) {
    document.querySelectorAll('.panel').forEach(function(p) { p.classList.remove('active'); });
    document.querySelectorAll('.nav-btn').forEach(function(b) { b.classList.remove('active'); });
    document.getElementById(id).classList.add('active');
    btn.classList.add('active');
}

function qClass(q) {
    if (q.indexOf('Q4') >= 0) return 'qb-q4';
    if (q.indexOf('Q5') >= 0) return 'qb-q5';
    return 'qb-q8';
}

function qBadge(q) {
    return '<span class="quant-badge ' + qClass(q) + '">' + q + '</span>';
}

function fmt(v, d) { return Number(v).toFixed(d === undefined ? 1 : d); }

function usabilityLabel(tg) {
    if (tg >= 30) return '<span style="color:#059669;font-weight:700">Excellent</span>';
    if (tg >= 15) return '<span style="color:#2563eb;font-weight:700">Good</span>';
    if (tg >= 8) return '<span style="color:#d97706;font-weight:700">Usable</span>';
    return '<span style="color:#dc2626;font-weight:700">Slow</span>';
}

// ── Parse benchmark data ──
var ppData = {};
var tgData = {};
bestData.forEach(function(r) {
    var key = r.model_size + '_' + r.quant;
    if (parseInt(r.pp_tokens) > 0) {
        if (!ppData[key]) ppData[key] = {};
        ppData[key][parseInt(r.pp_tokens)] = parseFloat(r.pp_tok_sec);
    }
    if (parseInt(r.tg_tokens) > 0) {
        tgData[key] = parseFloat(r.tg_tok_sec);
    }
});

// ── Peak values ──
var allTG = []; var allPP = [];
Object.keys(tgData).forEach(function(k) { if (tgData[k] > 0) allTG.push(tgData[k]); });
Object.keys(ppData).forEach(function(k) {
    Object.keys(ppData[k]).forEach(function(p) { if (ppData[k][p] > 0) allPP.push(ppData[k][p]); });
});
var peakTG = allTG.length ? Math.max.apply(null, allTG) : 0;
var peakPP = allPP.length ? Math.max.apply(null, allPP) : 0;

// ── Chart builder ──
function makeBarChart(containerId, title, chartData, maxVal, unit) {
    var c = document.getElementById(containerId);
    if (!c || !chartData.length) return;
    if (!maxVal) maxVal = Math.max.apply(null, chartData.map(function(d) {
        return d.values.length ? Math.max.apply(null, d.values) : 0;
    })) * 1.15;
    if (!unit) unit = 'tok/s';

    var html = '<div class="chart-title">' + title + '</div>';
    html += '<div class="bar-chart">';
    chartData.forEach(function(g) {
        html += '<div class="bar-group"><div class="bar-group-bars">';
        g.values.forEach(function(v, i) {
            var h = Math.max(2, (v / maxVal) * 230);
            var col = qColors[g.quants[i]] || '#999';
            html += '<div class="chart-bar" style="height:' + h + 'px;background:' + col + '">';
            html += '<div class="tip">' + g.quants[i] + ': ' + fmt(v) + ' ' + unit + '</div></div>';
        });
        html += '</div><div class="bar-label">' + g.label + '</div></div>';
    });
    html += '</div>';
    html += '<div class="legend">';
    quants.forEach(function(q) {
        html += '<div class="legend-item"><div class="legend-dot" style="background:' + qColors[q] + '"></div>' + q + '</div>';
    });
    html += '</div>';
    c.innerHTML = html;
}

function buildChartData(mode, ppLen) {
    return models.map(function(m) {
        var vals = []; var qs = [];
        quants.forEach(function(q) {
            if (mode === 'tg') {
                var v = tgData[m + '_' + q];
                if (v && v > 0) { vals.push(v); qs.push(q); }
            } else {
                var d = ppData[m + '_' + q];
                if (d && d[ppLen]) { vals.push(d[ppLen]); qs.push(q); }
            }
        });
        return { label: 'Qwen2.5-' + m, values: vals, quants: qs };
    }).filter(function(d) { return d.values.length > 0; });
}

// ── Stat card builder ──
function addStat(container, label, value, unit, detail, accent) {
    var html = '<div class="stat-card ' + (accent || 'accent-green') + '">';
    html += '<div class="stat-label">' + label + '</div>';
    html += '<div class="stat-value">' + value + '<span class="stat-unit">' + (unit || '') + '</span></div>';
    if (detail) html += '<div class="stat-detail">' + detail + '</div>';
    html += '</div>';
    container.innerHTML += html;
}

// ── Horizontal bar chart ──
function makeHBar(containerId, title, items, maxVal, unit, colorFn) {
    var c = document.getElementById(containerId);
    if (!c || !items.length) return;
    if (!maxVal) maxVal = Math.max.apply(null, items.map(function(i) { return i.value; })) * 1.05;

    var html = '<div class="chart-title">' + title + '</div><div class="hbar-chart">';
    items.forEach(function(item) {
        var pct = Math.max(1, (item.value / maxVal) * 100);
        var col = colorFn ? colorFn(item) : (qColors[item.quant] || '#999');
        html += '<div class="hbar-row">';
        html += '<div class="hbar-label">' + item.label + '</div>';
        html += '<div class="hbar-track"><div class="hbar-fill" style="width:' + pct + '%;background:' + col + '">' + fmt(item.value, item.decimals !== undefined ? item.decimals : 1) + '</div></div>';
        html += '<div class="hbar-value">' + fmt(item.value, item.decimals !== undefined ? item.decimals : 1) + ' ' + (unit || '') + '</div>';
        html += '</div>';
    });
    html += '</div>';
    c.innerHTML = html;
}

// ══════════════════════════════════════
// OVERVIEW
// ══════════════════════════════════════
var ovStats = document.getElementById('ov-stats');
addStat(ovStats, 'Peak Text Generation', fmt(peakTG), ' tok/s', '3B Q4_K_M', 'accent-green');
addStat(ovStats, 'Peak Prompt Processing', fmt(peakPP), ' tok/s', '3B Q4_K_M @ 512 tokens', 'accent-blue');
addStat(ovStats, 'Models Tested', models.length, '', 'Qwen2.5 3B / 7B / 14B / 32B', 'accent-amber');
addStat(ovStats, 'Configurations', bestData.length, '', models.length + ' models x ' + quants.length + ' quants x ' + (ppLens.length + 1) + ' tests', 'accent-teal');
addStat(ovStats, 'Benchmark Runs', runsMeta.length, '', 'Best values selected', 'accent-purple');

if (tpwData.length > 0) {
    var bestTPW = Math.max.apply(null, tpwData.map(function(r) { return parseFloat(r.tokens_per_watt); }));
    addStat(ovStats, 'Best Efficiency', fmt(bestTPW, 2), ' tok/W', '3B Q4_K_M', 'accent-rose');
}

makeBarChart('ov-tg-chart', 'Text Generation — 128 tokens output', buildChartData('tg'));
makeBarChart('ov-pp-chart', 'Prompt Processing — 512 tokens input', buildChartData('pp', 512));

// TPW overview charts
if (tpwData.length > 0) {
    document.getElementById('ov-efficiency-row').style.display = 'grid';

    var tpwChartData = models.map(function(m) {
        var vals = []; var qs = [];
        quants.forEach(function(q) {
            tpwData.forEach(function(r) {
                if (r.model_size === m && r.quant === q) {
                    vals.push(parseFloat(r.tokens_per_watt));
                    qs.push(q);
                }
            });
        });
        return { label: 'Qwen2.5-' + m, values: vals, quants: qs };
    }).filter(function(d) { return d.values.length > 0; });

    makeBarChart('ov-tpw-chart', 'Energy Efficiency — Tokens per Watt', tpwChartData, null, 'tok/W');

    var costItems = [];
    tpwData.forEach(function(r) {
        costItems.push({
            label: r.model_size + ' ' + r.quant,
            value: parseFloat(r.rm_per_1m_tokens),
            quant: r.quant,
            decimals: 2
        });
    });
    costItems.sort(function(a, b) { return a.value - b.value; });
    makeHBar('ov-cost-chart', 'Electricity Cost per 1M Tokens (MYR)', costItems, null, 'RM', function(i) { return qColors[i.quant]; });
}

// System specs
var specsContainer = document.getElementById('sys-specs');
var specs = [
    ['GPU', '{GPU_NAME}'], ['Architecture', 'Blackwell (GB10)'],
    ['Memory', '{RAM_TOTAL} Unified (CPU+GPU shared)'], ['CUDA Version', '{CUDA_VER}'],
    ['Driver', '{DRIVER_VER}'], ['CPU', '{CPU_NAME}'],
    ['Backend', 'llama.cpp (CUDA)'], ['Flash Attention', 'Enabled'],
    ['GPU Layers', 'All offloaded (ngl=99)'], ['Repetitions', '{N_REPS} per config per run']
];
specs.forEach(function(s) {
    var row = document.createElement('div');
    row.className = 'spec-row';
    row.innerHTML = '<span class="spec-key">' + s[0] + '</span><span class="spec-val">' + s[1] + '</span>';
    specsContainer.appendChild(row);
});

// ══════════════════════════════════════
// PROMPT PROCESSING
// ══════════════════════════════════════
var ppStats = document.getElementById('pp-stats');
ppLens.forEach(function(len) {
    var best = 0;
    Object.keys(ppData).forEach(function(k) { if (ppData[k][len] > best) best = ppData[k][len]; });
    addStat(ppStats, 'Peak PP ' + len, fmt(best), ' tok/s', '', 'accent-blue');
});

ppLens.forEach(function(len) {
    makeBarChart('pp-chart-' + len, 'Prompt Processing — ' + len + ' tokens input', buildChartData('pp', len));
});

var ppTbody = document.querySelector('#pp-table tbody');
models.forEach(function(m) {
    quants.forEach(function(q) {
        var d = ppData[m + '_' + q];
        if (!d) return;
        var row = '<tr>';
        row += '<td>Qwen2.5-' + m + '</td>';
        row += '<td>' + qBadge(q) + '</td>';
        ppLens.forEach(function(len) {
            var v = d[len];
            row += '<td class="cell-value">' + (v ? fmt(v) : '—') + '</td>';
        });
        row += '</tr>';
        ppTbody.innerHTML += row;
    });
});

// ══════════════════════════════════════
// TEXT GENERATION
// ══════════════════════════════════════
var tgStats = document.getElementById('tg-stats');
var fastestTG = ''; var slowestTG = ''; var fastVal = 0; var slowVal = Infinity;
Object.keys(tgData).forEach(function(k) {
    if (tgData[k] > fastVal) { fastVal = tgData[k]; fastestTG = k; }
    if (tgData[k] < slowVal && tgData[k] > 0) { slowVal = tgData[k]; slowestTG = k; }
});
addStat(tgStats, 'Fastest', fmt(fastVal), ' tok/s', fastestTG.replace('_', ' '), 'accent-green');
addStat(tgStats, 'Slowest', fmt(slowVal), ' tok/s', slowestTG.replace('_', ' '), 'accent-amber');
var interactive = 0;
Object.keys(tgData).forEach(function(k) { if (tgData[k] >= 15) interactive++; });
addStat(tgStats, 'Interactive (>15 tok/s)', interactive + '/' + Object.keys(tgData).length, '', 'Configs suitable for chat', 'accent-blue');

makeBarChart('tg-chart-detail', 'Text Generation — 128 tokens output', buildChartData('tg'));

var tgTbody = document.querySelector('#tg-table tbody');
models.forEach(function(m) {
    quants.forEach(function(q) {
        var v = tgData[m + '_' + q];
        if (v === undefined) return;
        var row = '<tr>';
        row += '<td>Qwen2.5-' + m + '</td>';
        row += '<td>' + qBadge(q) + '</td>';
        row += '<td class="cell-value">' + fmt(v) + '</td>';
        row += '<td>' + usabilityLabel(v) + '</td>';
        row += '</tr>';
        tgTbody.innerHTML += row;
    });
});

// ══════════════════════════════════════
// POWER & EFFICIENCY
// ══════════════════════════════════════
if (tpwData.length > 0) {
    var pwStats = document.getElementById('pw-stats');
    var bestEfficiency = 0; var avgPower = 0; var lowestCost = Infinity; var bestEffModel = '';
    tpwData.forEach(function(r) {
        var tpw = parseFloat(r.tokens_per_watt);
        avgPower += parseFloat(r.avg_power_w);
        if (tpw > bestEfficiency) { bestEfficiency = tpw; bestEffModel = r.model_size + ' ' + r.quant; }
        var cost = parseFloat(r.rm_per_1m_tokens);
        if (cost < lowestCost) lowestCost = cost;
    });
    avgPower /= tpwData.length;

    addStat(pwStats, 'Best Efficiency', fmt(bestEfficiency, 2), ' tok/W', bestEffModel, 'accent-green');
    addStat(pwStats, 'Avg GPU Power Draw', fmt(avgPower, 1), ' W', 'Across all configs', 'accent-amber');
    addStat(pwStats, 'Cheapest per 1M Tokens', 'RM ' + fmt(lowestCost, 2), '', 'At 0.55 MYR/kWh', 'accent-blue');
    addStat(pwStats, 'Configs Tested', tpwData.length, '', '512 token generation', 'accent-teal');

    // TPW bar chart
    var tpwItems = models.map(function(m) {
        var vals = []; var qs = [];
        quants.forEach(function(q) {
            tpwData.forEach(function(r) {
                if (r.model_size === m && r.quant === q) {
                    vals.push(parseFloat(r.tokens_per_watt));
                    qs.push(q);
                }
            });
        });
        return { label: 'Qwen2.5-' + m, values: vals, quants: qs };
    }).filter(function(d) { return d.values.length > 0; });
    makeBarChart('pw-tpw-chart', 'Tokens per Watt', tpwItems, null, 'tok/W');

    // Power draw chart
    var powerItems = models.map(function(m) {
        var vals = []; var qs = [];
        quants.forEach(function(q) {
            tpwData.forEach(function(r) {
                if (r.model_size === m && r.quant === q) {
                    vals.push(parseFloat(r.avg_power_w));
                    qs.push(q);
                }
            });
        });
        return { label: 'Qwen2.5-' + m, values: vals, quants: qs };
    }).filter(function(d) { return d.values.length > 0; });
    makeBarChart('pw-power-chart', 'Average Power Draw (Watts)', powerItems, null, 'W');

    // Cost horizontal bar
    var costBarItems = [];
    tpwData.forEach(function(r) {
        costBarItems.push({
            label: r.model_size + ' ' + r.quant,
            value: parseFloat(r.rm_per_1m_tokens),
            quant: r.quant,
            decimals: 2
        });
    });
    costBarItems.sort(function(a, b) { return a.value - b.value; });
    makeHBar('pw-cost-hbar', 'Electricity Cost — RM per 1M Tokens (lower is better)', costBarItems, null, 'RM', function(i) { return qColors[i.quant]; });

    // Power table
    var pwTbody = document.querySelector('#pw-table tbody');
    tpwData.forEach(function(r) {
        var row = '<tr>';
        row += '<td>Qwen2.5-' + r.model_size + '</td>';
        row += '<td>' + qBadge(r.quant) + '</td>';
        row += '<td class="cell-value">' + fmt(r.tg_tok_sec) + '</td>';
        row += '<td>' + fmt(r.avg_power_w) + '</td>';
        row += '<td class="cell-value">' + fmt(r.tokens_per_watt, 2) + '</td>';
        row += '<td>' + fmt(r.joules_per_token, 2) + '</td>';
        row += '<td>' + fmt(r.kwh_per_1m_tokens, 3) + '</td>';
        row += '<td class="cell-value">' + fmt(r.rm_per_1m_tokens, 2) + '</td>';
        row += '</tr>';
        pwTbody.innerHTML += row;
    });
}

// ══════════════════════════════════════
// RUN HISTORY
// ══════════════════════════════════════
var runsTbody = document.querySelector('#runs-table tbody');
runsMeta.forEach(function(run, i) {
    var isLatest = (i === runsMeta.length - 1);
    var peakPPr = 0; var peakTGr = 0;
    allRunsData[i].forEach(function(r) {
        var pp = parseFloat(r.pp_tok_sec); if (pp > peakPPr) peakPPr = pp;
        var tg = parseFloat(r.tg_tok_sec); if (tg > peakTGr) peakTGr = tg;
    });
    var ts = run.timestamp;
    var display = ts.substring(0,4) + '-' + ts.substring(4,6) + '-' + ts.substring(6,8) + ' ' + ts.substring(9,11) + ':' + ts.substring(11,13);
    var tag = isLatest ? '<span class="run-tag run-latest">Latest</span>' : '<span class="run-tag run-older">Run ' + (i+1) + '</span>';
    var row = '<tr>';
    row += '<td>' + tag + '</td>';
    row += '<td>' + display + '</td>';
    row += '<td>' + allRunsData[i].length + '</td>';
    row += '<td class="cell-value">' + fmt(peakPPr) + '</td>';
    row += '<td class="cell-value">' + fmt(peakTGr) + '</td>';
    row += '<td>' + (allRunsData[i].length >= 48 ? 'Complete' : 'Partial (' + allRunsData[i].length + '/48)') + '</td>';
    row += '</tr>';
    runsTbody.innerHTML += row;
});

// Run comparison chart: track 3B Q4_K_M TG across runs
var runCompareData = [];
allRunsData.forEach(function(runRows, i) {
    var tgVal = 0;
    runRows.forEach(function(r) {
        if (r.model_size === '3B' && r.quant === 'Q4_K_M' && parseInt(r.tg_tokens) > 0) {
            tgVal = parseFloat(r.tg_tok_sec);
        }
    });
    if (tgVal > 0) {
        var ts = runsMeta[i].timestamp;
        runCompareData.push({
            label: 'Run ' + (i+1),
            values: [tgVal],
            quants: ['Q4_K_M']
        });
    }
});
if (runCompareData.length > 0) {
    makeBarChart('run-compare-chart', '3B Q4_K_M — Text Generation across runs (tok/s)', runCompareData);
}

// ══════════════════════════════════════
// RAW DATA
// ══════════════════════════════════════
var rawTbody = document.querySelector('#raw-table tbody');
bestData.forEach(function(r) {
    var isPP = parseInt(r.pp_tokens) > 0;
    var row = '<tr>';
    row += '<td>Qwen2.5-' + r.model_size + '</td>';
    row += '<td>' + qBadge(r.quant) + '</td>';
    row += '<td>' + (isPP ? 'Prompt Processing' : 'Text Generation') + '</td>';
    row += '<td>' + (isPP ? r.pp_tokens : r.tg_tokens) + '</td>';
    row += '<td class="cell-value">' + fmt(isPP ? r.pp_tok_sec : r.tg_tok_sec) + '</td>';
    row += '<td>Best of ' + runsMeta.length + ' runs</td>';
    row += '</tr>';
    rawTbody.innerHTML += row;
});
</script>
</body>
</html>"""


def report_is_stale(results_dir):
    """Check if the HTML report is missing or older than any result CSV."""
    report = results_dir / "benchmark_report_gx10.html"
    if not report.exists():
        return True

    report_mtime = report.stat().st_mtime
    for pattern in ("benchmark_*.csv", "token_per_watt_*.csv"):
        for csv_file in results_dir.glob(pattern):
            if csv_file.stat().st_mtime > report_mtime:
                return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML benchmark report for NVIDIA GB10 (GX10)")
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate report even if it is up to date")
    parser.add_argument(
        "--check", action="store_true",
        help="Only check if report needs regeneration, then exit")
    args = parser.parse_args()

    results_dir = Path("results")
    output_path = results_dir / "benchmark_report_gx10.html"

    stale = report_is_stale(results_dir)

    if args.check:
        if stale:
            print("[CHECK] Report is missing or out of date — regeneration needed.")
            sys.exit(1)
        else:
            print(f"[CHECK] Report is up to date: {output_path}")
            sys.exit(0)

    if not args.force and not stale:
        print(f"[SKIP] Report already up to date: {output_path}")
        print("       Use --force to regenerate anyway.")
        sys.exit(0)

    if stale:
        if not output_path.exists():
            print("[INFO] Report HTML not found — generating new report.")
        else:
            print("[INFO] New result data detected — regenerating report.")
    else:
        print("[INFO] Force regeneration requested.")

    # Load all benchmark data
    bench_runs = load_benchmark_csvs(results_dir)
    if not bench_runs:
        print("[ERROR] No benchmark CSV found in results/")
        print("        Run ./scripts/run_benchmark_gx10.sh first.")
        sys.exit(1)

    print(f"[INFO] Found {len(bench_runs)} benchmark runs:")
    for run in bench_runs:
        print(f"       - {run['file']} ({len(run['rows'])} data points)")

    # Load token-per-watt data
    tpw_runs = load_tpw_csvs(results_dir)
    print(f"[INFO] Found {len(tpw_runs)} token-per-watt runs")

    # Load metadata
    metadata = load_metadata(results_dir)

    # Aggregate data
    best_rows = aggregate_best_run(bench_runs)
    avg_rows = aggregate_average(bench_runs)
    tpw_latest = aggregate_tpw_latest(tpw_runs)

    print(f"[INFO] Best-of-runs: {len(best_rows)} data points")
    print(f"[INFO] Token-per-watt: {len(tpw_latest)} data points")

    # Get system info
    sysinfo = get_system_info()

    # Prepare template data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_reps = str(metadata.get("repetitions", 3))
    elec_rate = "0.55"
    if tpw_latest:
        elec_rate = str(tpw_latest[0].get("electricity_cost_myr_per_kwh", "0.55"))

    runs_meta = [{"timestamp": r["timestamp"], "file": r["file"]} for r in bench_runs]
    all_runs_rows = [r["rows"] for r in bench_runs]

    # Build HTML
    html = HTML_TEMPLATE
    html = html.replace("{BEST_DATA}", json.dumps(best_rows))
    html = html.replace("{AVG_DATA}", json.dumps(avg_rows))
    html = html.replace("{TPW_DATA}", json.dumps(tpw_latest))
    html = html.replace("{RUNS_META}", json.dumps(runs_meta))
    html = html.replace("{ALL_RUNS_DATA}", json.dumps(all_runs_rows))
    html = html.replace("{TIMESTAMP}", timestamp)
    html = html.replace("{N_REPS}", n_reps)
    html = html.replace("{N_BENCH_RUNS}", str(len(bench_runs)))
    html = html.replace("{N_TPW_RUNS}", str(len(tpw_runs)))
    html = html.replace("{ELEC_RATE}", elec_rate)
    html = html.replace("{GPU_NAME}", sysinfo.get('gpu', 'NVIDIA GB10'))
    html = html.replace("{RAM_TOTAL}", sysinfo.get('ram', '128Gi'))
    html = html.replace("{CUDA_VER}", sysinfo.get('cuda', '13.0'))
    html = html.replace("{DRIVER_VER}", sysinfo.get('driver', 'Unknown'))
    html = html.replace("{CPU_NAME}", sysinfo.get('cpu', 'ARM Cortex-X925 + A725'))

    output_path = results_dir / "benchmark_report_gx10.html"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[SUCCESS] HTML report generated: {output_path}")
    print(f"[INFO] Open in browser: file://{output_path.absolute()}")
    print(f"[INFO] Report includes:")
    print(f"         - {len(bench_runs)} benchmark runs aggregated (best-of selected)")
    print(f"         - {len(tpw_runs)} power efficiency runs")
    print(f"         - 6 tabs: Overview, PP, TG, Power, Run History, Raw Data")


if __name__ == "__main__":
    main()
