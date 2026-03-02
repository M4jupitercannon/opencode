# Phase 4: Results Analysis {{SKIP_LABEL}}

## Objective
Analyze benchmark results and profiling data to identify performance characteristics and bottlenecks.

## Steps

### 1. Collect All Benchmark Results
Gather all result files from `{{OUTPUT_DIR}}/results/` and `{{REPO_DIR}}/`.
Look for JSON result files matching the experiment naming pattern.

### 2. Parse Benchmark Metrics
For each result file, extract key metrics:
- **Throughput** (tokens/second)
- **Time to First Token (TTFT)**
- **Inter-Token Latency (ITL)**
- **Total latency**
- **Request success rate**

### 3. Build Comparison Table
Create a table comparing performance across:
- Different concurrency levels
- Different sequence lengths (ISL×OSL)
- Framework/precision combinations

Save as `{{OUTPUT_DIR}}/results/benchmark_summary.json`.

### 4. Analyze Profiling Data (if available)
If profile traces exist in `{{PROFILE_DIR}}/`:
- Identify top time-consuming operations
- Look for GPU utilization patterns
- Detect memory bottlenecks
- Note kernel execution times

Save analysis to `{{OUTPUT_DIR}}/results/profile_analysis.json`.

### 5. Identify Bottlenecks
Based on benchmark and profiling data:
- Rank operations by time consumption
- Identify scaling bottlenecks (how throughput changes with concurrency)
- Note memory pressure points
- Flag any anomalous results

Save to `{{OUTPUT_DIR}}/results/bottlenecks.json`.

## Completion
Update progress.json:
```json
{
  "phase": "analyze",
  "phases_completed": ["env", "config", "benchmark", "profile", "analyze"],
  "current_step": "analysis complete",
  "details": {
    "results_analyzed": <N>,
    "bottlenecks_found": <M>
  }
}
```
