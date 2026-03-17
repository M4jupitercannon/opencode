# Phase 5: Profile Analysis {{SKIP_LABEL}}

## Objective
Analyze profiling traces to identify GPU kernel-level performance bottlenecks and optimization opportunities.

{{PROFILE_ANALYSIS_NOTE}}

## IMPORTANT: Always Re-run Analysis From Scratch
When this phase is entered (including via `--from-phase profile-analyze`), **always re-run the full analysis pipeline from step 1**, even if previous analysis artifacts already exist. Delete stale results first so every run produces fresh, consistent output.

```bash
echo "Cleaning previous profile analysis artifacts..."
rm -rf "{{OUTPUT_DIR}}/results/gap_analysis"
rm -rf "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs"
rm -rf "{{OUTPUT_DIR}}/results/tracelens_collective_csvs"
rm -rf "{{OUTPUT_DIR}}/results/phase_split"
rm -rf "{{OUTPUT_DIR}}/results/tracelens_prefill_decode_csvs"
rm -rf "{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs"
rm -f  "{{OUTPUT_DIR}}/results/profile_analysis.json"
rm -f  "{{OUTPUT_DIR}}/results/tracelens_rank0.log"
rm -f  "{{OUTPUT_DIR}}/results/tracelens_collective.log"
rm -f  "{{OUTPUT_DIR}}/results/gpu_arch.json"
echo "Cleanup done — starting fresh analysis"
```

After cleanup, verify that profile trace files (the **source** data, not previous analysis output) exist in `{{PROFILE_DIR}}/`. If no trace files exist there, print a warning and skip to step 4.

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

### 3. TraceLens Analysis 
TraceLens provides additional insights beyond gap analysis (GPU timeline, operator-level breakdown, collective communication analysis). It requires external installation but is **required** — it provides GPU timeline, operator-level breakdown, and collective communication analysis that gap analysis alone cannot.

**IMPORTANT**: The `pip install` can take several minutes due to dependency compilation. Set a bash timeout of at least **300 seconds** for the install command. If it times out or fails, **retry the installation once** before reporting an error.

**Check if TraceLens is already installed, then clone and install only if missing:**
```bash
if command -v TraceLens_generate_perf_report_pytorch &>/dev/null; then
    echo "TraceLens CLI already available"
else
    if [ ! -d "$HOME/TraceLens-internal" ]; then
        echo "Cloning TraceLens-internal..."
        git clone git@github.com:AMD-AGI/TraceLens-internal.git "$HOME/TraceLens-internal"
    fi
    echo "Installing TraceLens (this may take a few minutes)..."
    pip install --no-build-isolation "$HOME/TraceLens-internal" 2>&1 | tail -10
    if command -v TraceLens_generate_perf_report_pytorch &>/dev/null; then
        echo "TraceLens CLI installed successfully"
    else
        echo "First install attempt failed — retrying..."
        pip install --no-build-isolation "$HOME/TraceLens-internal" 2>&1 | tail -10
        if command -v TraceLens_generate_perf_report_pytorch &>/dev/null; then
            echo "TraceLens CLI installed successfully on retry"
        else
            echo "TRACELENS_INSTALL_FAILED=true"
            echo "ERROR: TraceLens installation failed after retry"
        fi
    fi
fi
```

If the output contains `TRACELENS_INSTALL_FAILED=true` after the retry, report the installation error but still proceed to step 4 using the gap analysis data. Do NOT skip TraceLens analysis without attempting the retry.

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

**Phase-Split Roofline Analysis (Prefill-Decode vs Decode-Only):**

This sub-step splits the rank-0 trace into prefill-decode and decode-only phases using TraceLens-internal's `split_vllm_trace_annotation.py`, then runs the inference-specific TraceLens report with roofline analysis on each phase. This reveals whether bottlenecks differ between compute-heavy prefill steps and memory-bandwidth-heavy decode steps.

**Auto-detect GPU and create GPU arch JSON** (required for roofline calculations):
```bash
python3 -c "
import json, subprocess, re, os

gpu_arch_path = '{{OUTPUT_DIR}}/results/gpu_arch.json'

PLATFORM_SPECS = {
    'MI300X': {'name': 'MI300X', 'mem_bw_gbps': 5300, 'max_achievable_tflops': {'matrix_fp16': 654, 'matrix_bf16': 708, 'matrix_fp32': 163, 'matrix_fp64': 81, 'matrix_fp8': 1273, 'matrix_int8': 2600, 'vector_fp16': 163, 'vector_bf16': 163, 'vector_fp32': 81, 'vector_fp64': 40}},
    'MI325X': {'name': 'MI325X', 'mem_bw_gbps': 6000, 'max_achievable_tflops': {'matrix_fp16': 794, 'matrix_bf16': 843, 'matrix_fp32': 194, 'matrix_fp64': 97, 'matrix_fp8': 1519, 'matrix_int8': 3094, 'vector_fp16': 194, 'vector_bf16': 194, 'vector_fp32': 97, 'vector_fp64': 48}},
    'MI355X': {'name': 'MI355X', 'mem_bw_gbps': 8000, 'max_achievable_tflops': {'matrix_fp16': 1686, 'matrix_bf16': 1686, 'matrix_fp32': 137, 'matrix_fp64': 68, 'matrix_fp8': 3567, 'matrix_fp6': 4574, 'matrix_fp4': 5663, 'matrix_int8': 7134, 'vector_fp16': 274, 'vector_bf16': 274, 'vector_fp32': 137, 'vector_fp64': 68}},
}

gpu_name = None
try:
    result = subprocess.run(['rocm-smi', '--showproductname'], capture_output=True, text=True, timeout=10)
    for line in result.stdout.splitlines():
        for key in PLATFORM_SPECS:
            if key.lower().replace('x', '') in line.lower().replace('x', ''):
                gpu_name = key
                break
        if gpu_name:
            break
except Exception:
    pass

if not gpu_name:
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=10)
        for line in result.stdout.splitlines():
            if 'gfx' in line.lower():
                if 'gfx942' in line.lower():
                    gpu_name = 'MI300X'
                elif 'gfx950' in line.lower():
                    gpu_name = 'MI355X'
                break
    except Exception:
        pass

if gpu_name and gpu_name in PLATFORM_SPECS:
    spec = PLATFORM_SPECS[gpu_name]
    os.makedirs(os.path.dirname(gpu_arch_path), exist_ok=True)
    with open(gpu_arch_path, 'w') as f:
        json.dump(spec, f, indent=2)
    print(f'GPU_ARCH_DETECTED={gpu_name}')
    print(f'GPU_ARCH_JSON={gpu_arch_path}')
else:
    print(f'GPU_ARCH_DETECTED=unknown')
    print('WARNING: Could not detect GPU model for roofline analysis. Roofline data will be omitted.')
"
```

**Split rank-0 trace into prefill-decode and decode-only phases:**
```bash
RANK0_TRACE="<rank-0 trace path from step 1>"
SPLIT_SCRIPT="$HOME/TraceLens-internal/examples/custom_workflows/split_vllm_trace_annotation.py"
PHASE_SPLIT_DIR="{{OUTPUT_DIR}}/results/phase_split"

if [ -f "$SPLIT_SCRIPT" ]; then
    mkdir -p "$PHASE_SPLIT_DIR"
    echo "Splitting trace into prefill-decode and decode-only phases..."
    python3 "$SPLIT_SCRIPT" "$RANK0_TRACE" \
        -o "$PHASE_SPLIT_DIR" \
        --find-steady-state \
        --num-steps 32 \
        2>&1 | tail -30

    echo "Phase split exit code: $?"
    ls -lh "$PHASE_SPLIT_DIR/"
else
    echo "PHASE_SPLIT_UNAVAILABLE=true"
    echo "WARNING: split_vllm_trace_annotation.py not found at $SPLIT_SCRIPT"
    echo "Skipping phase-split roofline analysis"
fi
```

**Identify the phase-specific trace files** from the split output. The splitter produces files named `prefilldecode_*` and `decode_*`:
```bash
if [ -d "$PHASE_SPLIT_DIR" ] && [ "$(ls -A $PHASE_SPLIT_DIR/*.json.gz 2>/dev/null)" ]; then
    PREFILL_DECODE_TRACE=$(ls "$PHASE_SPLIT_DIR"/prefilldecode_*.json.gz 2>/dev/null | head -1)
    DECODE_ONLY_TRACE=$(ls "$PHASE_SPLIT_DIR"/decode_*.json.gz 2>/dev/null | head -1)

    echo "PREFILL_DECODE_TRACE=$PREFILL_DECODE_TRACE"
    echo "DECODE_ONLY_TRACE=$DECODE_ONLY_TRACE"
else
    echo "No phase-split traces found — skipping per-phase roofline analysis"
fi
```

**Run TraceLens inference report with roofline on the prefill-decode phase:**
```bash
INFERENCE_REPORT_SCRIPT="$HOME/TraceLens-internal/TraceLens/Reporting/generate_perf_report_pytorch_inference.py"
GPU_ARCH_JSON="{{OUTPUT_DIR}}/results/gpu_arch.json"

if [ -n "$PREFILL_DECODE_TRACE" ] && [ -f "$PREFILL_DECODE_TRACE" ] && [ -f "$INFERENCE_REPORT_SCRIPT" ]; then
    mkdir -p "{{OUTPUT_DIR}}/results/tracelens_prefill_decode_csvs"
    echo "Running TraceLens roofline on prefill-decode phase..."
    python3 "$INFERENCE_REPORT_SCRIPT" \
        --profile_json_path "$PREFILL_DECODE_TRACE" \
        --output_csvs_dir "{{OUTPUT_DIR}}/results/tracelens_prefill_decode_csvs" \
        --enable_pseudo_ops \
        --group_by_parent_module \
        --enable_kernel_summary \
        $([ -f "$GPU_ARCH_JSON" ] && echo "--gpu_arch_json_path $GPU_ARCH_JSON") \
        2>&1 | tail -30

    echo "Prefill-decode roofline exit code: $?"
    ls -lh "{{OUTPUT_DIR}}/results/tracelens_prefill_decode_csvs/"
else
    echo "Skipping prefill-decode roofline (trace or script unavailable)"
fi
```

**Run TraceLens inference report with roofline on the decode-only phase:**
```bash
if [ -n "$DECODE_ONLY_TRACE" ] && [ -f "$DECODE_ONLY_TRACE" ] && [ -f "$INFERENCE_REPORT_SCRIPT" ]; then
    mkdir -p "{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs"
    echo "Running TraceLens roofline on decode-only phase..."
    python3 "$INFERENCE_REPORT_SCRIPT" \
        --profile_json_path "$DECODE_ONLY_TRACE" \
        --output_csvs_dir "{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs" \
        --enable_pseudo_ops \
        --group_by_parent_module \
        --enable_kernel_summary \
        $([ -f "$GPU_ARCH_JSON" ] && echo "--gpu_arch_json_path $GPU_ARCH_JSON") \
        2>&1 | tail -30

    echo "Decode-only roofline exit code: $?"
    ls -lh "{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs/"
else
    echo "Skipping decode-only roofline (trace or script unavailable)"
fi
```

**Parse TraceLens results** (if the reports were generated successfully):
Read the generated CSV files from `{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/` (and `tracelens_collective_csvs/` if present) and extract key insights:
- From `ops_summary.csv`: top time-consuming operations, their cumulative GPU time
- From `kernel_summary.csv`: most expensive GPU kernels, call counts, average duration
- From `ops_summary_by_category.csv`: time distribution across categories (GEMM, attention, communication, etc.)
- From `coll_analysis.csv`: collective communication overhead and patterns
- From `gpu_timeline.csv`: GPU utilization and idle gaps

Also read the phase-specific CSV files from `tracelens_prefill_decode_csvs/` and `tracelens_decode_only_csvs/` (if present) and extract:
- From `unified_perf_summary.csv`: per-op roofline analysis (FLOPS/byte, TFLOPS/s, bound type, bound distance)
- From `SDPA_fwd.csv` / `FLASH_ATTN_fwd.csv`: attention roofline metrics per phase
- From `GEMM.csv`: GEMM roofline metrics per phase
- From `gpu_timeline.csv`: GPU utilization comparison between prefill-decode and decode-only phases
- From `ops_summary_by_category.csv`: category time distribution differences between phases

**Display TraceLens results to the console** so the user can see key findings:
```bash
echo ""
echo "============================================"
echo "  TraceLens Analysis Summary (Rank 0)"
echo "============================================"

if [ -f "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/gpu_timeline.csv" ]; then
    echo ""
    echo "--- GPU Timeline ---"
    cat "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/gpu_timeline.csv"
fi

if [ -f "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/ops_summary_by_category.csv" ]; then
    echo ""
    echo "--- Ops Summary by Category ---"
    cat "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/ops_summary_by_category.csv"
fi

if [ -f "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/ops_summary.csv" ]; then
    echo ""
    echo "--- Top Ops Summary (first 25 lines) ---"
    head -25 "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/ops_summary.csv"
fi

if [ -f "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/kernel_summary.csv" ]; then
    echo ""
    echo "--- Top Kernel Summary (first 25 lines) ---"
    head -25 "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/kernel_summary.csv"
fi

if [ -f "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/GEMM.csv" ]; then
    echo ""
    echo "--- GEMM Kernel Summary (first 25 lines) ---"
    head -25 "{{OUTPUT_DIR}}/results/tracelens_rank0_csvs/GEMM.csv"
fi

echo ""
echo "============================================"
echo "  Phase-Split Roofline Analysis"
echo "============================================"

for PHASE_LABEL in "Prefill-Decode" "Decode-Only"; do
    if [ "$PHASE_LABEL" = "Prefill-Decode" ]; then
        PHASE_DIR="{{OUTPUT_DIR}}/results/tracelens_prefill_decode_csvs"
    else
        PHASE_DIR="{{OUTPUT_DIR}}/results/tracelens_decode_only_csvs"
    fi

    if [ -d "$PHASE_DIR" ] && [ "$(ls -A $PHASE_DIR/*.csv 2>/dev/null)" ]; then
        echo ""
        echo "--- $PHASE_LABEL Phase ---"

        if [ -f "$PHASE_DIR/gpu_timeline.csv" ]; then
            echo ""
            echo "  GPU Timeline ($PHASE_LABEL):"
            cat "$PHASE_DIR/gpu_timeline.csv"
        fi

        if [ -f "$PHASE_DIR/ops_summary_by_category.csv" ]; then
            echo ""
            echo "  Ops by Category ($PHASE_LABEL):"
            cat "$PHASE_DIR/ops_summary_by_category.csv"
        fi

        if [ -f "$PHASE_DIR/unified_perf_summary.csv" ]; then
            echo ""
            echo "  Roofline / Unified Perf Summary ($PHASE_LABEL, first 25 lines):"
            head -25 "$PHASE_DIR/unified_perf_summary.csv"
        fi

        if [ -f "$PHASE_DIR/GEMM.csv" ]; then
            echo ""
            echo "  GEMM Roofline ($PHASE_LABEL, first 25 lines):"
            head -25 "$PHASE_DIR/GEMM.csv"
        fi

        for ATTN_CSV in "$PHASE_DIR/SDPA_fwd.csv" "$PHASE_DIR/FLASH_ATTN_fwd.csv"; do
            if [ -f "$ATTN_CSV" ]; then
                echo ""
                echo "  Attention Roofline ($PHASE_LABEL, first 25 lines):"
                head -25 "$ATTN_CSV"
                break
            fi
        done
    else
        echo ""
        echo "  $PHASE_LABEL phase: no roofline data available"
    fi
done

echo ""
echo "============================================"
```

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
  "output_csv_dirs": ["tracelens_rank0_csvs/", "tracelens_collective_csvs/"],
  "gpu_arch": "<detected GPU model or null>",
  "phase_split": {
    "available": true,
    "prefill_decode_trace": "<path to prefill-decode trace or null>",
    "decode_only_trace": "<path to decode-only trace or null>",
    "execution_details": "<contents of phase_split/execution_details.json>"
  },
  "roofline": {
    "prefill_decode": {
      "available": true,
      "csv_dir": "tracelens_prefill_decode_csvs/",
      "gpu_timeline": {"computation_pct": ..., "communication_pct": ..., "idle_pct": ...},
      "category_breakdown": {"gemm": ..., "attention": ..., ...},
      "top_roofline_ops": [{"name": "...", "flops_per_byte": ..., "tflops_s": ..., "bound_type": "memory|compute", "bound_distance_pct": ...}, ...]
    },
    "decode_only": {
      "available": true,
      "csv_dir": "tracelens_decode_only_csvs/",
      "gpu_timeline": {"computation_pct": ..., "communication_pct": ..., "idle_pct": ...},
      "category_breakdown": {"gemm": ..., "attention": ..., ...},
      "top_roofline_ops": [{"name": "...", "flops_per_byte": ..., "tflops_s": ..., "bound_type": "memory|compute", "bound_distance_pct": ...}, ...]
    }
  }
}
```

### 4. Identify Profile Bottlenecks

From **gap analysis** (step 2 — always available when traces exist):
- Identify the top-K most expensive steady-state kernels from `gap_analysis.csv`
- Compare kernel time distribution across ranks (look for load imbalance)
- Identify whether bottleneck is compute-bound (GEMM-heavy) or communication-bound (collective-heavy)

From **TraceLens** results (step 3 — should always be available; fall back to gap analysis only if install failed after retry):
- Rank GPU kernels by cumulative time from `kernel_summary.csv`
- Quantify collective communication overhead from `coll_analysis.csv` (time spent in AllReduce, AllGather, etc.)
- Detect GPU idle gaps from `gpu_timeline.csv` indicating pipeline bubbles or CPU-bound phases
- Analyze time distribution across op categories from `ops_summary_by_category.csv`

From **phase-split roofline analysis** (step 3 — available when trace annotations support phase detection):
- Compare GPU utilization and category breakdown between prefill-decode and decode-only phases
- Identify whether prefill is compute-bound (expected for large-batch GEMM) or decode is memory-bound (expected for single-token attention)
- From `unified_perf_summary.csv`: extract per-op roofline metrics (FLOPS/byte, TFLOPS/s, bound type, distance to roofline)
- From `GEMM.csv` / `SDPA_fwd.csv`: compare GEMM and attention arithmetic intensity between phases
- Flag ops far from the roofline ceiling as optimization opportunities (e.g., low TFLOPS/s relative to achievable peak)

Save profile bottleneck findings to `{{OUTPUT_DIR}}/results/profile_analysis.json` (merge with TraceLens data if already created in step 3).

### 5. Generate / Update Benchmark Report

Generate (or update if it already exists) the benchmark report at `{{REPORT_DIR}}/benchmark_report.md`.

If an existing report from Phase 3 (benchmark-analyze) is present, **append** a "Profile Analysis" section to it. If no report exists yet, create one from scratch using the template below, filling in benchmark data from `{{OUTPUT_DIR}}/results/benchmark_summary.json` if available.

Read `{{OUTPUT_DIR}}/results/profile_analysis.json` and `{{OUTPUT_DIR}}/results/gap_analysis/gap_analysis.json` to populate the profile sections.

The report MUST include the following profile analysis sections (append to existing report or include in new report):

```markdown
## Profile Analysis

### GPU Utilization
| Metric | Value |
|--------|-------|
| Computation Time (%) | <from gpu_timeline.csv> |
| Exposed Communication Time (%) | <from gpu_timeline.csv> |
| Exposed Memcpy Time (%) | <from gpu_timeline.csv> |
| GPU Busy Time (%) | <from gpu_timeline.csv> |

### Top GPU Kernels (Steady-State, 50%-80% Window)

From gap analysis of the steady-state inference window:

| Rank | Kernel Name | Calls | Total Time (us) | Avg (us) | % Total |
|------|-------------|-------|-----------------|----------|---------|
| 1 | ... | ... | ... | ... | ... |

### Kernel Category Breakdown

From TraceLens ops_summary_by_category:

| Category | Count | Total Time (ms) | % of Kernel Time |
|----------|-------|-----------------|------------------|
| ... | ... | ... | ... |

### Phase-Split Roofline Analysis

Traces are split into prefill-decode and decode-only phases using TraceLens-internal's `split_vllm_trace_annotation.py`, then analyzed with `generate_perf_report_pytorch_inference.py` for per-phase roofline insights.

#### Prefill-Decode Phase

| Metric | Value |
|--------|-------|
| Computation Time (%) | <from prefill-decode gpu_timeline.csv> |
| GPU Busy Time (%) | <from prefill-decode gpu_timeline.csv> |
| Dominant Bound Type | <compute or memory, from unified_perf_summary.csv> |

Top roofline ops (prefill-decode):

| Op Name | FLOPS/Byte | TFLOPS/s | Bound Type | Distance to Roofline (%) |
|---------|------------|----------|------------|--------------------------|
| ... | ... | ... | ... | ... |

#### Decode-Only Phase

| Metric | Value |
|--------|-------|
| Computation Time (%) | <from decode-only gpu_timeline.csv> |
| GPU Busy Time (%) | <from decode-only gpu_timeline.csv> |
| Dominant Bound Type | <compute or memory, from unified_perf_summary.csv> |

Top roofline ops (decode-only):

| Op Name | FLOPS/Byte | TFLOPS/s | Bound Type | Distance to Roofline (%) |
|---------|------------|----------|------------|--------------------------|
| ... | ... | ... | ... | ... |

#### Phase Comparison

| Metric | Prefill-Decode | Decode-Only |
|--------|----------------|-------------|
| GPU Computation (%) | ... | ... |
| GEMM Time (%) | ... | ... |
| Attention Time (%) | ... | ... |
| Communication Time (%) | ... | ... |
| Dominant Bound | compute / memory | compute / memory |

### Profile Bottlenecks & Optimization Opportunities
- <bottleneck 1: description and recommendation>
- <bottleneck 2: description and recommendation>
- ...

### Raw Profile Data
- Gap analysis: `results/gap_analysis/`
- TraceLens rank-0 CSVs: `results/tracelens_rank0_csvs/`
- Phase-split traces: `results/phase_split/`
- Prefill-decode roofline CSVs: `results/tracelens_prefill_decode_csvs/`
- Decode-only roofline CSVs: `results/tracelens_decode_only_csvs/`
- GPU arch config: `results/gpu_arch.json`
- Profile analysis JSON: `results/profile_analysis.json`
```

If no benchmark data exists (only profile-analyze was run), create a report with just the Configuration and Profile Analysis sections.

**Print the final report path:**
```bash
echo ""
echo "============================================"
echo "  Benchmark Report Generated"
echo "============================================"
echo "Report: {{REPORT_DIR}}/benchmark_report.md"
echo "============================================"
```

## Completion
Update progress.json (include "profile" in phases_completed only if profiling was run):
```json
{
  "phase": "profile-analyze",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze", "profile", "profile-analyze"],
  "current_step": "profile analysis complete",
  "details": {
    "gap_analysis": <true if step 2 succeeded, false otherwise>,
    "tracelens_analysis": <true if step 3 succeeded, false otherwise>,
    "phase_split_roofline": <true if phase-split roofline analysis succeeded, false otherwise>,
    "gpu_arch_detected": "<GPU model name or null>",
    "report": "{{REPORT_DIR}}/benchmark_report.md"
  }
}
```
