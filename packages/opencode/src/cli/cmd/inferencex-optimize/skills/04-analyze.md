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

### 5. Analyze Profiling Data with TraceLens (if available)
If profile traces exist in `{{PROFILE_DIR}}/`:

#### 5a. Start TraceLens Analysis Container
Start a persistent container for TraceLens analysis using the same image as profiling.
Mount the profile traces directory and the output directory so results are accessible on the host.
Detect GPU vendor and set appropriate flags:
```bash
# For AMD GPUs (runner starts with "mi")
GPU_FLAGS="--device=/dev/kfd --device=/dev/dri --group-add video --security-opt seccomp=unconfined"

# For NVIDIA GPUs
GPU_FLAGS="--gpus all"
```

```bash
TRACELENS_CONTAINER="inferencex-tracelens-{{CONFIG_KEY}}"

docker run -d \
    --name "$TRACELENS_CONTAINER" \
    --label inferencex-pipeline=true \
    --entrypoint /bin/bash \
    $GPU_FLAGS \
    --shm-size 16g \
    --ipc=host \
    -v {{PROFILE_DIR}}:/traces \
    -v {{OUTPUT_DIR}}/results:/results \
    -w /traces \
    $IMAGE \
    -c "sleep infinity"
```

#### 5b. Install TraceLens Inside Container

**Bash call 1 — Print DOCKER_LOG (separate bash call):**
```bash
DOCKER_LOG="{{OUTPUT_DIR}}/results/tracelens_install_docker.log"
echo "DOCKER_LOG: $DOCKER_LOG"
```

**Bash call 2 — Execute docker exec (separate bash call):**
```bash
docker exec "$TRACELENS_CONTAINER" bash -c '
    if ! command -v TraceLens_generate_perf_report_pytorch &>/dev/null; then
        echo "Installing TraceLens..."
        pip install git+https://github.com/AMD-AIG-AIMA/TraceLens.git
    fi
    TraceLens_generate_perf_report_pytorch --help > /dev/null 2>&1 && echo "TraceLens CLI available" || echo "ERROR: TraceLens CLI not found"
' > "$DOCKER_LOG" 2>&1
echo "Install exit code: $?"
```

Do NOT print or display the contents of the docker log file. The log is saved for debugging purposes only.

#### 5c. Find and Validate Trace Files
Locate torch profiler trace files inside the container, filtering out async_llm traces and benchmark result JSONs. **Critically**, validate that files actually contain `traceEvents` (Chrome Trace Event format) before passing them to TraceLens — benchmark result JSONs (with keys like `request_throughput`, `model_id`) will cause `KeyError: 'traceEvents'`.
```bash
docker exec "$TRACELENS_CONTAINER" bash -c '
    CANDIDATE_FILES=$(find /traces -name "*.json.gz" -o -name "*.json" | grep -v -i "async_llm" | grep -v "_docker.log" | sort)
    
    TRACE_FILES=""
    for f in $CANDIDATE_FILES; do
        if python3 -c "
import json, gzip, sys
try:
    opener = gzip.open if \"$f\".endswith(\".gz\") else open
    with opener(\"$f\", \"rt\") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and \"traceEvents\" in data:
        sys.exit(0)
    else:
        sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null; then
            TRACE_FILES="${TRACE_FILES}${f}"$'"'"'\n'"'"'
            echo "VALID torch trace: $f"
        else
            echo "SKIPPED (not a torch trace): $f"
        fi
    done
    
    TRACE_FILES=$(echo -e "$TRACE_FILES" | sed "/^$/d")
    TRACE_COUNT=$(echo "$TRACE_FILES" | grep -c . || true)
    echo "TRACE_COUNT=$TRACE_COUNT"
    echo "$TRACE_FILES"
'
```
If TRACE_COUNT is 0 (no valid torch profiler traces found), print a warning that profiling did not produce usable traces and skip to step 5g to clean up the container. Do NOT run TraceLens on files that lack `traceEvents`.

#### 5d. Run Single-Rank Performance Report
Run `TraceLens_generate_perf_report_pytorch` on the **first (rank-0) validated trace file** (one that contains `traceEvents`) to generate per-rank performance CSVs.

**Bash call 1 — Print DOCKER_LOG (separate bash call):**
```bash
DOCKER_LOG="{{OUTPUT_DIR}}/results/tracelens_rank0_docker.log"
echo "DOCKER_LOG: $DOCKER_LOG"
```

**Bash call 2 — Execute docker exec (separate bash call):**
```bash
docker exec "$TRACELENS_CONTAINER" bash -c '
    # Find first valid torch trace (must contain traceEvents)
    RANK0_TRACE=""
    for f in $(find /traces -name "*.json.gz" -o -name "*.json" | grep -v -i "async_llm" | grep -v "_docker.log" | sort); do
        if python3 -c "
import json, gzip, sys
try:
    opener = gzip.open if \"$f\".endswith(\".gz\") else open
    with opener(\"$f\", \"rt\") as fh:
        data = json.load(fh)
    sys.exit(0 if isinstance(data, dict) and \"traceEvents\" in data else 1)
except:
    sys.exit(1)
" 2>/dev/null; then
            RANK0_TRACE="$f"
            break
        fi
    done
    
    if [ -z "$RANK0_TRACE" ]; then
        echo "ERROR: No valid torch profiler trace found (no files contain traceEvents)"
        exit 1
    fi
    
    echo "Using trace: $RANK0_TRACE"
    mkdir -p /results/tracelens_rank0_csvs

    TraceLens_generate_perf_report_pytorch \
        --profile_json_path "$RANK0_TRACE" \
        --output_csvs_dir /results/tracelens_rank0_csvs \
        --enable_kernel_summary

    echo "Rank-0 TraceLens output:"
    ls -lh /results/tracelens_rank0_csvs/
' > "$DOCKER_LOG" 2>&1
echo "Rank-0 report exit code: $?"
```

Do NOT print or display the contents of the docker log file. The log is saved for debugging purposes only.

This produces CSV files including:
- `gpu_timeline.csv` — GPU activity timeline
- `ops_summary.csv` — Operator-level time breakdown
- `ops_summary_by_category.csv` — Time grouped by op category
- `kernel_summary.csv` — GPU kernel execution statistics
- `coll_analysis.csv` — Collective communication analysis

#### 5e. Run Multi-Rank Collective Report (if multiple ranks)
If more than one rank trace exists (TP > 1), run the multi-rank collective analysis.

**Bash call 1 — Print DOCKER_LOG (separate bash call):**
```bash
DOCKER_LOG="{{OUTPUT_DIR}}/results/tracelens_collective_docker.log"
echo "DOCKER_LOG: $DOCKER_LOG"
```

**Bash call 2 — Execute docker exec (separate bash call):**
```bash
docker exec "$TRACELENS_CONTAINER" bash -c '
    TRACE_COUNT=$(find /traces -name "*.json.gz" -o -name "*.json" | grep -v -i "async_llm" | wc -l)
    if [ "$TRACE_COUNT" -gt 1 ]; then
        mkdir -p /results/tracelens_collective_csvs

        TraceLens_generate_multi_rank_collective_report_pytorch \
            --trace_dir /traces \
            --world_size "$TRACE_COUNT" \
            --output_csvs_dir /results/tracelens_collective_csvs

        echo "Multi-rank collective output:"
        ls -lh /results/tracelens_collective_csvs/
    else
        echo "Single rank trace — skipping multi-rank collective report"
    fi
' > "$DOCKER_LOG" 2>&1
echo "Collective report exit code: $?"
```

Do NOT print or display the contents of the docker log file. The log is saved for debugging purposes only.

#### 5f. Parse TraceLens Results
Read the generated CSV files from `{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/` (and `tracelens_collective_csvs/` if present) on the host and extract key insights:
- From `ops_summary.csv`: top time-consuming operations, their cumulative GPU time
- From `kernel_summary.csv`: most expensive GPU kernels, call counts, average duration
- From `ops_summary_by_category.csv`: time distribution across categories (GEMM, attention, communication, etc.)
- From `coll_analysis.csv`: collective communication overhead and patterns
- From `gpu_timeline.csv`: GPU utilization and idle gaps

Summarize findings into a structured JSON:
```json
{
  "tracelens_version": "<version>",
  "trace_file": "<rank0 trace path>",
  "num_ranks_analyzed": <N>,
  "top_ops": [{"name": "...", "total_time_us": ..., "pct": ...}, ...],
  "top_kernels": [{"name": "...", "calls": ..., "avg_time_us": ..., "pct": ...}, ...],
  "category_breakdown": {"gemm": ..., "attention": ..., "communication": ..., ...},
  "collective_overhead_pct": <if multi-rank>,
  "gpu_utilization_pct": <estimated from timeline>,
  "output_csv_dirs": ["tracelens_rank0_csvs/", "tracelens_collective_csvs/"]
}
```

Save analysis to `{{OUTPUT_DIR}}/results/profile_analysis.json`.

#### 5g. Clean Up TraceLens Container
```bash
docker stop "$TRACELENS_CONTAINER"
docker rm "$TRACELENS_CONTAINER"
```

### 6. Identify Bottlenecks
Based on benchmark metrics and TraceLens profiling data:
- Rank GPU kernels by cumulative time from `kernel_summary.csv`
- Identify scaling bottlenecks (how throughput changes with concurrency)
- Quantify collective communication overhead from `coll_analysis.csv` (time spent in AllReduce, AllGather, etc.)
- Detect GPU idle gaps from `gpu_timeline.csv` indicating pipeline bubbles or CPU-bound phases
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
