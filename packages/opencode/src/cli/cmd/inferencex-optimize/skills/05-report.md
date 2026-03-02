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
Create `{{REPORT_DIR}}/benchmark_report.md` with:

```markdown
# InferenceX Benchmark Report

## Configuration
- **Config Key**: {{CONFIG_KEY}}
- **Date**: <current date>
- **GPU**: <detected GPU>
- **Framework**: <framework from config>
- **Model**: <model name>
- **Precision**: <precision>

## Benchmark Results

### Throughput Summary
| Concurrency | ISL×OSL | Throughput (tok/s) | TTFT (ms) | ITL (ms) |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

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
