# Phase 5: Report Generation {{SKIP_LABEL}}

## Objective
Generate a comprehensive benchmark report summarizing all results, analysis, and recommendations.

## Steps

### 1. Load All Data
Read from:
- `{{OUTPUT_DIR}}/results/benchmark_summary.json`
- `{{OUTPUT_DIR}}/results/profile_analysis.json` (if exists)
- `{{OUTPUT_DIR}}/results/bottlenecks.json` (if exists)
- `{{OUTPUT_DIR}}/progress.json`

### 2. Generate Markdown Report
Create `{{REPORT_DIR}}/benchmark_report.md` using the **exact template below**. If an old report already exists, **overwrite it completely** — do NOT copy or replicate the old report's structure.

```markdown
# InferenceX Benchmark Report

## Configuration
- **Config Key**: {{CONFIG_KEY}}
- **Date**: <current date>
- **GPU**: <detected GPU>
- **Framework**: <framework from config>
- **Model**: <model name>
- **Precision**: <precision>
- **Docker Image**: <image from sweep config>

## Benchmark Results

### Throughput Summary
| Metric | Value |
|--------|-------|
| Request Throughput | <req/s> |
| Input Token Throughput | <tok/s> |
| Output Token Throughput | <tok/s> |
| Total Token Throughput | <tok/s> |

Note: If the raw benchmark data does not include `input_throughput`, compute it as `total_token_throughput - output_token_throughput`.

### Latency Summary
| Concurrency | ISL×OSL | TTFT Mean (ms) | ITL Mean (ms) | End-to-end Latency (s) |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

### Throughput per GPU, Interactivity & Latency

**IMPORTANT**: This MUST be exactly ONE table. Do NOT split into separate "Throughput vs Interactivity" and "Throughput vs Latency" tables.

Shows throughput efficiency per GPU, per-user interactivity, and end-to-end latency across concurrency levels.
- `Token Throughput per GPU = total_token_throughput / tp` (total throughput per GPU)
- `Input Token Throughput per GPU = input_token_throughput / tp`
- `Output Token Throughput per GPU = output_token_throughput / tp`
- `Interactivity (tok/s/user) = 1 / TPOT` where `TPOT = mean_itl_ms / 1000` (Time Per Output Token in seconds)
- `End-to-end Latency (s) = mean_e2el_ms / 1000`

| Concurrency | ISL×OSL | TP | Token Throughput per GPU (tok/s/gpu) | Input Token Throughput per GPU | Output Token Throughput per GPU | Interactivity (tok/s/user) | End-to-end Latency (s) |
|---|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... | ... |

### Scaling Analysis
- How throughput scales with concurrency
- Optimal concurrency point
- Memory utilization patterns

## Profiling Results (if available)
- Top 10 operations by time
- GPU utilization breakdown
- Memory allocation patterns

## Bottlenecks & Recommendations
- Identified bottlenecks
- Optimization suggestions
- Comparison with expected performance

## Raw Data
- Location of result files
- Profile trace file locations (viewable at https://ui.perfetto.dev/)
```

### 3. Generate JSON Summary
Create `{{REPORT_DIR}}/report_summary.json` with machine-readable results.

## Completion
Update progress.json:
```json
{
  "phase": "report",
  "phases_completed": ["env", "config", "benchmark", "profile", "analyze", "report"],
  "current_step": "report generated",
  "final_report": "{{REPORT_DIR}}/benchmark_report.md"
}
```
