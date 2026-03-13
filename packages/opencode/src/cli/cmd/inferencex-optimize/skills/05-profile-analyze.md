# Phase 5: Profile Analysis {{SKIP_LABEL}}

## Objective
Analyze profiling traces to identify GPU kernel-level performance bottlenecks and optimization opportunities.

{{PROFILE_ANALYSIS_NOTE}}

If profile traces exist in `{{PROFILE_DIR}}/`:

## Steps

### 1. Discover and Validate Trace Files on the Host
Locate torch profiler trace files in `{{PROFILE_DIR}}/`, filtering out async_llm (frontend-only) traces, docker logs, and benchmark result JSONs. Validate that each candidate file actually contains `traceEvents` (Chrome Trace Event format) — benchmark result JSONs (with keys like `request_throughput`, `model_id`) will cause `KeyError: 'traceEvents'` in TraceLens.

Detect per-rank trace files by matching the `*-rank-N*.json.gz` or `*-rank-N*.json` naming pattern (also match `rank0`, `rank1` without the dash). Fall back to any `.json.gz` / `.json` files if no rank pattern is found.

```bash
python3 -c "
import json, gzip, glob, re, sys, os

trace_dir = '{{PROFILE_DIR}}'
valid_traces = []
rank_map = {}

for f in sorted(glob.glob(os.path.join(trace_dir, '*.json*'))):
    basename = os.path.basename(f)
    if '_docker.log' in basename:
        continue
    if 'async_llm' in basename.lower():
        print(f'SKIPPED (async_llm frontend trace): {f}')
        continue
    try:
        opener = gzip.open if f.endswith('.gz') else open
        with opener(f, 'rt') as fh:
            data = json.load(fh)
        if isinstance(data, dict) and 'traceEvents' in data:
            valid_traces.append(f)
            rank_match = re.search(r'rank[-_]?(\d+)', basename)
            rank = int(rank_match.group(1)) if rank_match else len(valid_traces) - 1
            rank_map[f] = rank
            n_events = len(data['traceEvents'])
            print(f'VALID torch trace (rank {rank}, {n_events} events): {f}')
        else:
            keys = list(data.keys())[:5] if isinstance(data, dict) else type(data).__name__
            print(f'SKIPPED (not a torch trace, keys: {keys}): {f}')
    except Exception as e:
        print(f'ERROR reading {f}: {e}')

print(f'TRACE_COUNT={len(valid_traces)}')
if valid_traces:
    sorted_by_rank = sorted(valid_traces, key=lambda x: rank_map.get(x, 999))
    print(f'RANK0_TRACE={sorted_by_rank[0]}')
    print(f'WORLD_SIZE={len(valid_traces)}')
else:
    print('WARNING: No valid torch profiler traces found. Trace analysis will be skipped.')
"
```

If TRACE_COUNT is 0, print a warning and skip to step 4 (bottleneck analysis using benchmark data only). Do NOT run trace analysis on files that lack `traceEvents`.

### 2. Gap Analysis (Time-Windowed Kernel Profiling) — Primary
Run gap analysis **first** — it uses only standard Python (no external dependencies) and is the primary kernel-level analysis method.

This analyzes a configurable time window of the trace to focus on steady-state inference behavior (skipping warmup and cooldown), producing a ranked list of the most expensive GPU kernels.

The gap analysis pipeline:
1. **Apply time window** — focus on the 50%–80% range of trace duration to capture steady-state behavior (skips warmup at start, cooldown at end)
2. **Filter by category** — include only `kernel` and `gpu` events (case-insensitive substring matching), exclude `gpu_user_annotation`
3. **Aggregate per kernel** — group by kernel name, sum total CUDA time, count calls
4. **Merge across ranks** — combine stats from all rank traces into a single ranking
5. **Rank by total duration** — sort kernels by cumulative GPU time descending

**IMPORTANT**: This script processes large trace files (potentially millions of events). Set a long bash timeout (at least 600 seconds). The trace file loading step alone can take 30+ seconds for a 100MB+ gzipped trace.

The pipeline deploys `trace_analyzer.py` to `{{SCRIPTS_DIR}}/`. Use it for gap analysis:

```bash
python3 "{{SCRIPTS_DIR}}/trace_analyzer.py" "{{PROFILE_DIR}}" \
    --gap-analysis \
    --output-dir "{{OUTPUT_DIR}}/results/gap_analysis" \
    --start-pct 50 --end-pct 80 --top-k 20
```

You can also generate clamped (time-windowed) trace files alongside the analysis:
```bash
python3 "{{SCRIPTS_DIR}}/trace_analyzer.py" "{{PROFILE_DIR}}" \
    --gap-analysis --clamped-traces \
    --output-dir "{{OUTPUT_DIR}}/results/gap_analysis"
```

You can also run a full (non-windowed) kernel summary:
```bash
python3 "{{SCRIPTS_DIR}}/trace_analyzer.py" "{{PROFILE_DIR}}"
```

The gap analysis CSV follows this format (matching InferenceX `gen_kstats_clamped_traces.py` output):
```
Name, Calls, Self CUDA total (us), Avg time (us), % Total
```

This reveals which GPU kernels dominate steady-state inference time. Typical bottleneck categories for vLLM:
- **GEMM kernels** (e.g., `ck_fmha_*`, `hipblas*`) — matrix multiply for attention and FFN layers
- **Communication kernels** (e.g., `ncclAllReduce*`, `allgather*`) — collective ops for tensor parallelism
- **Custom attention** (e.g., `paged_attention_*`, `flash_attn_*`) — KV cache operations
- **Quantization** (e.g., `dequant*`, `mxfp4_*`) — precision conversion overhead

### 3. TraceLens Analysis (Optional Enhancement)
TraceLens provides additional insights beyond gap analysis (GPU timeline, operator-level breakdown, collective communication analysis). It requires external installation and is **optional** — gap analysis (step 2) already covers kernel-level bottleneck identification.

**IMPORTANT**: The `pip install` can take several minutes due to git clone + dependency compilation. Set a bash timeout of at least **300 seconds** for the install command. If it times out or fails, skip TraceLens entirely and proceed to step 4 — the gap analysis from step 2 provides the essential kernel profiling data.

**Check if TraceLens is already installed, then install only if missing:**
```bash
if command -v TraceLens_generate_perf_report_pytorch &>/dev/null; then
    echo "TraceLens CLI already available"
else
    echo "Installing TraceLens (this may take a few minutes)..."
    pip install --no-build-isolation git+https://github.com/AMD-AIG-AIMA/TraceLens.git 2>&1 | tail -10
    if command -v TraceLens_generate_perf_report_pytorch &>/dev/null; then
        echo "TraceLens CLI installed successfully"
    else
        echo "TRACELENS_INSTALL_FAILED=true"
        echo "TraceLens installation failed — skipping TraceLens analysis (gap analysis already completed in step 2)"
    fi
fi
```

If the output contains `TRACELENS_INSTALL_FAILED=true` or the install command timed out, **skip the rest of step 3 entirely** and proceed to step 4. Do NOT retry the installation.

**If TraceLens is available, run the single-rank performance report:**
```bash
RANK0_TRACE="<rank-0 trace path from step 1>"
mkdir -p "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs"

TraceLens_generate_perf_report_pytorch \
    --profile_json_path "$RANK0_TRACE" \
    --output_csvs_dir "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs" \
    --enable_kernel_summary \
    2>&1 | tee "{{OUTPUT_DIR}}/results/tracelens_rank0.log"

echo "Rank-0 report exit code: $?"
ls -lh "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/"
```

This produces CSV files including:
- `gpu_timeline.csv` — GPU activity timeline
- `ops_summary.csv` — Operator-level time breakdown
- `ops_summary_by_category.csv` — Time grouped by op category
- `kernel_summary.csv` — GPU kernel execution statistics
- `coll_analysis.csv` — Collective communication analysis

**If WORLD_SIZE > 1, also run the multi-rank collective report:**
```bash
WORLD_SIZE=<from step 1>
if [ "$WORLD_SIZE" -gt 1 ]; then
    mkdir -p "{{OUTPUT_DIR}}/results/tracelens_collective_csvs"

    TraceLens_generate_multi_rank_collective_report_pytorch \
        --trace_dir "{{PROFILE_DIR}}" \
        --world_size "$WORLD_SIZE" \
        --output_csvs_dir "{{OUTPUT_DIR}}/results/tracelens_collective_csvs" \
        2>&1 | tee "{{OUTPUT_DIR}}/results/tracelens_collective.log"

    echo "Multi-rank collective report exit code: $?"
    ls -lh "{{OUTPUT_DIR}}/results/tracelens_collective_csvs/"
else
    echo "Single rank trace — skipping multi-rank collective report"
fi
```

**Parse TraceLens results** (if the reports were generated successfully):
Read the generated CSV files from `{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/` (and `tracelens_collective_csvs/` if present) and extract key insights:
- From `ops_summary.csv`: top time-consuming operations, their cumulative GPU time
- From `kernel_summary.csv`: most expensive GPU kernels, call counts, average duration
- From `ops_summary_by_category.csv`: time distribution across categories (GEMM, attention, communication, etc.)
- From `coll_analysis.csv`: collective communication overhead and patterns
- From `gpu_timeline.csv`: GPU utilization and idle gaps

Save TraceLens analysis to `{{OUTPUT_DIR}}/results/profile_analysis.json`:
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

### 4. Identify Profile Bottlenecks

From **gap analysis** (step 2 — always available when traces exist):
- Identify the top-K most expensive steady-state kernels from `gap_analysis.csv`
- Compare kernel time distribution across ranks (look for load imbalance)
- Identify whether bottleneck is compute-bound (GEMM-heavy) or communication-bound (collective-heavy)

From **TraceLens** results (step 3 — only if TraceLens was installed successfully):
- Rank GPU kernels by cumulative time from `kernel_summary.csv`
- Quantify collective communication overhead from `coll_analysis.csv` (time spent in AllReduce, AllGather, etc.)
- Detect GPU idle gaps from `gpu_timeline.csv` indicating pipeline bubbles or CPU-bound phases
- Analyze time distribution across op categories from `ops_summary_by_category.csv`

Save profile bottleneck findings to `{{OUTPUT_DIR}}/results/profile_analysis.json` (merge with TraceLens data if already created in step 3).

## Completion
Update progress.json (include "profile" in phases_completed only if profiling was run):
```json
{
  "phase": "profile-analyze",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze", "profile", "profile-analyze"],
  "current_step": "profile analysis complete",
  "details": {
    "gap_analysis": <true if step 2 succeeded, false otherwise>,
    "tracelens_analysis": <true if step 3 succeeded, false otherwise>
  }
}
```
