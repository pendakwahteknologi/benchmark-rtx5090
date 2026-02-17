#!/usr/bin/env python3
"""
Generate Comprehensive HTML Benchmark Report
Reads CSV results from llama-bench and produces an interactive HTML report

Note: This report uses locally-generated data only (from our own benchmark CSV).
No external/untrusted content is rendered. All innerHTML usage operates on
data produced by our own benchmark scripts.
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import glob
import sys

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen2.5 GGUF Benchmark Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            padding: 20px;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .header .gpu-badge {
            display: inline-block;
            background: rgba(255,255,255,0.15);
            padding: 8px 20px;
            border-radius: 20px;
            margin-top: 15px;
            font-size: 0.95em;
        }

        .nav-tabs {
            display: flex;
            background: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .nav-tab {
            flex: 1;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            background: #f8f9fa;
            border: none;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s;
        }

        .nav-tab:hover {
            background: #e9ecef;
        }

        .nav-tab.active {
            background: white;
            border-bottom: 3px solid #2c5364;
            color: #2c5364;
        }

        .tab-content {
            display: none;
            padding: 30px;
        }

        .tab-content.active {
            display: block;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }

        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }

        .stat-value {
            font-size: 2.2em;
            font-weight: bold;
            color: #2c5364;
            margin: 10px 0;
        }

        .stat-label {
            color: #666;
            font-size: 0.9em;
        }

        .stat-unit {
            font-size: 0.5em;
            color: #999;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }

        th, td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }

        th {
            background: #2c5364;
            color: white;
            font-weight: 600;
        }

        tr:hover {
            background: #f0f7ff;
        }

        .quant-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 0.8em;
            font-weight: 600;
        }

        .quant-q4 { background: #d0ebff; color: #1971c2; }
        .quant-q5 { background: #d3f9d8; color: #2f9e44; }
        .quant-q8 { background: #fff3bf; color: #e67700; }

        .chart-container {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
        }

        .chart-title {
            font-size: 1.3em;
            font-weight: 600;
            color: #2c5364;
            margin-bottom: 15px;
        }

        .bar-chart {
            display: flex;
            align-items: flex-end;
            gap: 8px;
            height: 250px;
            padding: 10px 0;
            border-bottom: 2px solid #dee2e6;
        }

        .bar-group {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 4px;
        }

        .bar-group-bars {
            display: flex;
            align-items: flex-end;
            gap: 3px;
            width: 100%;
            justify-content: center;
        }

        .chart-bar {
            width: 28px;
            border-radius: 4px 4px 0 0;
            transition: all 0.3s;
            position: relative;
            cursor: pointer;
        }

        .chart-bar:hover {
            opacity: 0.8;
        }

        .chart-bar .tooltip {
            display: none;
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75em;
            white-space: nowrap;
            margin-bottom: 4px;
        }

        .chart-bar:hover .tooltip {
            display: block;
        }

        .bar-label {
            font-size: 0.85em;
            color: #666;
            margin-top: 8px;
            text-align: center;
        }

        .legend {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 15px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85em;
        }

        .legend-colour {
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }

        .info-box {
            background: #e7f5ff;
            border-left: 4px solid #339af0;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }

        .section-title {
            color: #2c5364;
            margin: 30px 0 15px 0;
            font-size: 1.5em;
        }

        .gpu-specs {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }

        .spec-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px dotted #dee2e6;
        }

        .spec-label { color: #666; }
        .spec-value { font-weight: 600; color: #333; }

        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
        }

        @media (max-width: 768px) {
            .stats { grid-template-columns: 1fr; }
            .gpu-specs { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Qwen2.5 GGUF Benchmark Report</h1>
            <p>llama.cpp Inference Performance on AMD ROCm</p>
            <div class="gpu-badge">AMD Radeon AI PRO R9700 &bull; RDNA4 (gfx1201) &bull; 32GB VRAM &bull; ROCm 7.2</div>
        </div>

        <div class="nav-tabs">
            <button class="nav-tab active" onclick="switchTab('overview')">Overview</button>
            <button class="nav-tab" onclick="switchTab('pp')">Prompt Processing</button>
            <button class="nav-tab" onclick="switchTab('tg')">Text Generation</button>
            <button class="nav-tab" onclick="switchTab('raw')">Raw Data</button>
        </div>

        <div id="overview" class="tab-content active">
            <div class="stats" id="overview-stats"></div>
            <div style="padding: 0 30px 30px 30px;">
                <h2 class="section-title">Text Generation Speed (tok/s)</h2>
                <div class="chart-container" id="tg-chart-overview"></div>
                <h2 class="section-title">Prompt Processing Speed (tok/s) — 512 tokens</h2>
                <div class="chart-container" id="pp-chart-overview"></div>
                <h2 class="section-title">System Specifications</h2>
                <div class="gpu-specs" id="system-specs"></div>
            </div>
        </div>

        <div id="pp" class="tab-content">
            <div style="padding: 30px;">
                <h2 class="section-title">Prompt Processing Performance</h2>
                <div class="info-box">
                    <p><strong>Prompt Processing (PP)</strong> measures how fast the model processes the input prompt.</p>
                    <p>Higher tok/s = faster time to first token. Tested at 128, 256, and 512 token prompt lengths.</p>
                </div>
                <div class="chart-container" id="pp-chart-128"></div>
                <div class="chart-container" id="pp-chart-256"></div>
                <div class="chart-container" id="pp-chart-512"></div>
                <h2 class="section-title">Detailed Results</h2>
                <table id="pp-table"><thead><tr>
                    <th>Model</th><th>Quant</th><th>PP 128 (tok/s)</th><th>PP 256 (tok/s)</th><th>PP 512 (tok/s)</th>
                </tr></thead><tbody></tbody></table>
            </div>
        </div>

        <div id="tg" class="tab-content">
            <div style="padding: 30px;">
                <h2 class="section-title">Text Generation Performance</h2>
                <div class="info-box">
                    <p><strong>Text Generation (TG)</strong> measures how fast the model generates output tokens.</p>
                    <p>Higher tok/s = faster response. This is the primary speed metric for interactive use. Tested at 128 output tokens.</p>
                </div>
                <div class="chart-container" id="tg-chart-detail"></div>
                <h2 class="section-title">Detailed Results</h2>
                <table id="tg-table"><thead><tr>
                    <th>Model</th><th>Quant</th><th>TG 128 (tok/s)</th>
                </tr></thead><tbody></tbody></table>
            </div>
        </div>

        <div id="raw" class="tab-content">
            <div style="padding: 30px;">
                <h2 class="section-title">Raw Benchmark Data</h2>
                <div class="info-box">
                    <p>Complete benchmark data from llama-bench. All values are averages over {N_REPS} repetitions.</p>
                </div>
                <table id="raw-table"><thead><tr>
                    <th>Model</th><th>Quant</th><th>Test Type</th><th>Tokens</th><th>Speed (tok/s)</th>
                </tr></thead><tbody></tbody></table>
            </div>
        </div>

        <div class="footer">
            <p>Generated on {TIMESTAMP}</p>
            <p>AMD Radeon AI PRO R9700 &bull; RDNA4 (gfx1201) &bull; 32GB VRAM &bull; ROCm 7.2 &bull; llama.cpp (HIP)</p>
        </div>
    </div>

    <script>
        // Data is generated locally by our own benchmark scripts - safe to render
        const data = {DATA};

        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(function(t) { t.classList.remove('active'); });
            document.querySelectorAll('.nav-tab').forEach(function(t) { t.classList.remove('active'); });
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }

        var quantColours = {
            'Q4_K_M': '#4dabf7',
            'Q5_K_M': '#69db7c',
            'Q8_0':   '#ffa94d'
        };

        function escapeText(str) {
            var div = document.createElement('div');
            div.textContent = str;
            return div.textContent;
        }

        function makeBarChart(containerId, title, chartData, maxVal) {
            var container = document.getElementById(containerId);
            if (!container) return;
            if (!maxVal) maxVal = Math.max.apply(null, chartData.map(function(d) { return Math.max.apply(null, d.values); })) * 1.15;

            // Build chart using DOM methods
            while (container.firstChild) container.removeChild(container.firstChild);

            var titleEl = document.createElement('div');
            titleEl.className = 'chart-title';
            titleEl.textContent = title;
            container.appendChild(titleEl);

            var chart = document.createElement('div');
            chart.className = 'bar-chart';

            chartData.forEach(function(group) {
                var groupDiv = document.createElement('div');
                groupDiv.className = 'bar-group';

                var barsDiv = document.createElement('div');
                barsDiv.className = 'bar-group-bars';

                group.values.forEach(function(val, i) {
                    var height = Math.max(2, (val / maxVal) * 220);
                    var colour = quantColours[group.quants[i]] || '#999';
                    var bar = document.createElement('div');
                    bar.className = 'chart-bar';
                    bar.style.height = height + 'px';
                    bar.style.background = colour;

                    var tip = document.createElement('div');
                    tip.className = 'tooltip';
                    tip.textContent = group.quants[i] + ': ' + val.toFixed(1) + ' tok/s';
                    bar.appendChild(tip);
                    barsDiv.appendChild(bar);
                });

                groupDiv.appendChild(barsDiv);
                var label = document.createElement('div');
                label.className = 'bar-label';
                label.textContent = group.label;
                groupDiv.appendChild(label);
                chart.appendChild(groupDiv);
            });

            container.appendChild(chart);

            var legend = document.createElement('div');
            legend.className = 'legend';
            Object.keys(quantColours).forEach(function(q) {
                var item = document.createElement('div');
                item.className = 'legend-item';
                var swatch = document.createElement('div');
                swatch.className = 'legend-colour';
                swatch.style.background = quantColours[q];
                item.appendChild(swatch);
                item.appendChild(document.createTextNode(q));
                legend.appendChild(item);
            });
            container.appendChild(legend);
        }

        // Organise data
        var models = ['3B', '7B', '14B', '32B'];
        var quants = ['Q4_K_M', 'Q5_K_M', 'Q8_0'];
        var ppLengths = [128, 256, 512];

        var ppData = {};
        var tgData = {};
        data.forEach(function(row) {
            var key = row.model_size + '_' + row.quant;
            if (parseInt(row.pp_tokens) > 0) {
                if (!ppData[key]) ppData[key] = {};
                ppData[key][parseInt(row.pp_tokens)] = parseFloat(row.pp_tok_sec);
            }
            if (parseInt(row.tg_tokens) > 0) {
                tgData[key] = parseFloat(row.tg_tok_sec);
            }
        });

        // Overview stats
        var allTG = Object.keys(tgData).map(function(k) { return tgData[k]; }).filter(function(v) { return v > 0; });
        var allPP = Object.keys(ppData).reduce(function(acc, k) {
            return acc.concat(Object.keys(ppData[k]).map(function(pk) { return ppData[k][pk]; }));
        }, []).filter(function(v) { return v > 0; });
        var maxTG = allTG.length > 0 ? Math.max.apply(null, allTG) : 0;
        var maxPP = allPP.length > 0 ? Math.max.apply(null, allPP) : 0;

        // Build stats with DOM methods
        var statsContainer = document.getElementById('overview-stats');
        var statsData = [
            { label: 'Peak Generation', value: maxTG.toFixed(1), unit: ' tok/s' },
            { label: 'Peak Prompt Processing', value: maxPP.toFixed(1), unit: ' tok/s' },
            { label: 'Models Tested', value: models.length.toString(), unit: '' },
            { label: 'Configurations', value: data.length.toString(), unit: '' }
        ];
        statsData.forEach(function(s) {
            var card = document.createElement('div');
            card.className = 'stat-card';
            var lbl = document.createElement('div');
            lbl.className = 'stat-label';
            lbl.textContent = s.label;
            var val = document.createElement('div');
            val.className = 'stat-value';
            val.textContent = s.value;
            if (s.unit) {
                var unitSpan = document.createElement('span');
                unitSpan.className = 'stat-unit';
                unitSpan.textContent = s.unit;
                val.appendChild(unitSpan);
            }
            card.appendChild(lbl);
            card.appendChild(val);
            statsContainer.appendChild(card);
        });

        // Build charts
        function buildChartData(ppLen) {
            return models.map(function(m) {
                var vals = [];
                var qs = [];
                quants.forEach(function(q) {
                    if (ppLen === 'tg') {
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

        makeBarChart('tg-chart-overview', 'Text Generation — 128 tokens output', buildChartData('tg'));
        makeBarChart('tg-chart-detail', 'Text Generation — 128 tokens output', buildChartData('tg'));
        makeBarChart('pp-chart-overview', 'Prompt Processing — 512 tokens input', buildChartData(512));
        ppLengths.forEach(function(ppLen) {
            makeBarChart('pp-chart-' + ppLen, 'Prompt Processing — ' + ppLen + ' tokens input', buildChartData(ppLen));
        });

        // Build tables with DOM methods
        function getQuantClass(q) {
            if (q.indexOf('Q4') >= 0) return 'quant-q4';
            if (q.indexOf('Q5') >= 0) return 'quant-q5';
            if (q.indexOf('Q8') >= 0 || q.indexOf('q8') >= 0) return 'quant-q8';
            return '';
        }

        // PP table
        var ppTbody = document.querySelector('#pp-table tbody');
        models.forEach(function(m) {
            quants.forEach(function(q) {
                var d = ppData[m + '_' + q];
                if (!d) return;
                var row = ppTbody.insertRow();
                var c1 = row.insertCell(); c1.textContent = 'Qwen2.5-' + m;
                var c2 = row.insertCell();
                var badge = document.createElement('span');
                badge.className = 'quant-badge ' + getQuantClass(q);
                badge.textContent = q;
                c2.appendChild(badge);
                var c3 = row.insertCell(); c3.textContent = d[128] ? d[128].toFixed(1) : '-';
                var c4 = row.insertCell(); c4.textContent = d[256] ? d[256].toFixed(1) : '-';
                var c5 = row.insertCell(); c5.textContent = d[512] ? d[512].toFixed(1) : '-';
            });
        });

        // TG table
        var tgTbody = document.querySelector('#tg-table tbody');
        models.forEach(function(m) {
            quants.forEach(function(q) {
                var v = tgData[m + '_' + q];
                if (!v) return;
                var row = tgTbody.insertRow();
                var c1 = row.insertCell(); c1.textContent = 'Qwen2.5-' + m;
                var c2 = row.insertCell();
                var badge = document.createElement('span');
                badge.className = 'quant-badge ' + getQuantClass(q);
                badge.textContent = q;
                c2.appendChild(badge);
                var c3 = row.insertCell();
                var bold = document.createElement('strong');
                bold.textContent = v.toFixed(1);
                c3.appendChild(bold);
            });
        });

        // Raw table
        var rawTbody = document.querySelector('#raw-table tbody');
        data.forEach(function(r) {
            var isPP = parseInt(r.pp_tokens) > 0;
            var row = rawTbody.insertRow();
            var c1 = row.insertCell(); c1.textContent = 'Qwen2.5-' + r.model_size;
            var c2 = row.insertCell();
            var badge = document.createElement('span');
            badge.className = 'quant-badge ' + getQuantClass(r.quant);
            badge.textContent = r.quant;
            c2.appendChild(badge);
            var c3 = row.insertCell(); c3.textContent = isPP ? 'Prompt Processing' : 'Text Generation';
            var c4 = row.insertCell(); c4.textContent = isPP ? r.pp_tokens : r.tg_tokens;
            var c5 = row.insertCell();
            var bold = document.createElement('strong');
            bold.textContent = isPP ? parseFloat(r.pp_tok_sec).toFixed(1) : parseFloat(r.tg_tok_sec).toFixed(1);
            c5.appendChild(bold);
        });

        // System specs
        var specsContainer = document.getElementById('system-specs');
        var specsCols = [
            [
                ['GPU', 'AMD Radeon AI PRO R9700'],
                ['Architecture', 'RDNA4 (gfx1201)'],
                ['VRAM', '32 GB'],
                ['ROCm Version', '7.2']
            ],
            [
                ['Backend', 'llama.cpp (HIP/ROCm)'],
                ['Flash Attention', 'Enabled'],
                ['GPU Layers', 'All offloaded (ngl=99)'],
                ['Repetitions', '{N_REPS} per config']
            ]
        ];
        specsCols.forEach(function(col) {
            var colDiv = document.createElement('div');
            col.forEach(function(spec) {
                var item = document.createElement('div');
                item.className = 'spec-item';
                var lbl = document.createElement('span');
                lbl.className = 'spec-label';
                lbl.textContent = spec[0];
                var val = document.createElement('span');
                val.className = 'spec-value';
                val.textContent = spec[1];
                item.appendChild(lbl);
                item.appendChild(val);
                colDiv.appendChild(item);
            });
            specsContainer.appendChild(colDiv);
        });
    </script>
</body>
</html>"""


def main():
    results_dir = Path("results")

    # Find latest CSV
    csvs = sorted(results_dir.glob("benchmark_*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        print("[ERROR] No benchmark CSV found in results/")
        print("        Run ./scripts/run_benchmark.sh first.")
        sys.exit(1)

    csv_path = csvs[-1]
    print(f"[INFO] Reading results from: {csv_path}")

    # Parse CSV
    rows = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("[ERROR] No data rows found in CSV")
        sys.exit(1)

    print(f"[INFO] Found {len(rows)} data points")

    # Generate HTML
    data_json = json.dumps(rows)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_reps = "3"

    html = HTML_TEMPLATE
    html = html.replace("{DATA}", data_json)
    html = html.replace("{TIMESTAMP}", timestamp)
    html = html.replace("{N_REPS}", n_reps)

    output_path = results_dir / "benchmark_report.html"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html)

    print(f"[SUCCESS] HTML report generated: {output_path}")
    print(f"[INFO] Open in browser: file://{output_path.absolute()}")


if __name__ == "__main__":
    main()
