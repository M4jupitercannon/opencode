# Phase 4: Results Analysis {{SKIP_LABEL}}

## Objective
Analyze benchmark results and profiling data to identify performance characteristics and bottlenecks.

## Steps

### 1. Collect All Benchmark Results
Gather all result files from `{{OUTPUT_DIR}}/results/` and `{{REPO_DIR}}/`.
Look for JSON result files matching the experiment naming pattern.

### 2. Parse Benchmark Metrics
For each result file, extract key metrics:
- **Request Throughput** (requests/second)
- **Input Token Throughput** (tokens/second) — if not in raw data, compute as `total_token_throughput - output_token_throughput` or `total_input_tokens / duration_s`
- **Output Token Throughput** (tokens/second)
- **Total Token Throughput** (tokens/second)
- **Time to First Token (TTFT)**
- **Inter-Token Latency (ITL)**
- **Total latency**
- **Request success rate**

### 3. Compute Derived Metrics
For each benchmark result, compute the following derived values and present them in **exactly ONE table** (do **NOT** split into two separate tables):
- `Token Thpt/GPU = total_token_throughput / tp` (total throughput per GPU)
- `In Thpt/GPU = input_token_throughput / tp`
- `Out Thpt/GPU = output_token_throughput / tp`
- `Interactivity (tok/s/user) = 1 / TPOT` where `TPOT = mean_itl_ms / 1000` (Time Per Output Token in seconds)
- `End-to-end Latency (s) = mean_e2el_ms / 1000`

**IMPORTANT**: Print ALL derived metrics in a single table with this exact format. Do NOT create separate "Throughput vs Interactivity" and "Throughput vs Latency" tables.

```
Throughput per GPU, Interactivity & Latency:
--------------------------------------------------------------------------------
Conc   ISLxOSL      TP   Token Thpt/GPU   In Thpt/GPU   Out Thpt/GPU   Interactivity(tok/s/user)   End-to-end Latency (s)
...
```

Include `tok_per_s_per_gpu`, `in_tok_per_s_per_gpu`, `out_tok_per_s_per_gpu`, `interactivity_tok_per_s`, and `e2el_s` as derived fields in the benchmark summary JSON alongside the raw metrics.

### 4. Build Comparison Table
Create a table comparing performance across:
- Different concurrency levels
- Different sequence lengths (ISL×OSL)
- Framework/precision combinations

Save as `{{OUTPUT_DIR}}/results/benchmark_summary.json`.

### 5. Analyze Profiling Data (if available)
If profile traces exist in `{{PROFILE_DIR}}/`:
- Identify top time-consuming operations
- Look for GPU utilization patterns
- Detect memory bottlenecks
- Note kernel execution times

Save analysis to `{{OUTPUT_DIR}}/results/profile_analysis.json`.

### 6. Identify Bottlenecks
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
