# Phase 5: Profile Analysis {{SKIP_LABEL}}

## Objective
Analyze profiling traces from **both eager and graph modes** to identify GPU kernel-level performance bottlenecks and optimization opportunities. Produces per-mode reports and a final combined report.

{{PROFILE_ANALYSIS_NOTE}}

## IMPORTANT: Always Re-run Analysis From Scratch
When this phase is entered (including via `--from-phase profile-analyze`), **always re-run the full analysis pipeline from step 1**, even if previous analysis artifacts already exist. Delete stale results first so every run produces fresh, consistent output.

```bash
echo "Cleaning previous profile analysis artifacts..."
for MODE in eager graph; do
    rm -rf "{{OUTPUT_DIR}}/results/gap_analysis_${MODE}"
    rm -rf "{{OUTPUT_DIR}}/results/gap_analysis_prefill_decode_${MODE}"
    rm -rf "{{OUTPUT_DIR}}/results/gap_analysis_decode_only_${MODE}"
    rm -rf "{{OUTPUT_DIR}}/results/tracelens_${MODE}"
    rm -rf "{{OUTPUT_DIR}}/results/tracelens_collective_${MODE}"
    rm -rf "{{OUTPUT_DIR}}/results/tracelens_prefill_decode_${MODE}"
    rm -rf "{{OUTPUT_DIR}}/results/tracelens_decode_only_${MODE}"
    rm -rf "{{OUTPUT_DIR}}/results/phase_split_${MODE}"
    rm -f  "{{OUTPUT_DIR}}/results/profile_analysis_${MODE}.json"
done
rm -f  "{{OUTPUT_DIR}}/results/gpu_arch.json"
rm -rf "{{REPORT_DIR}}/profiling_report_eager.md"
rm -rf "{{REPORT_DIR}}/profiling_report_graph.md"
rm -rf "{{REPORT_DIR}}/profiling_report.md"
echo "Cleanup done — starting fresh analysis"
```

## TraceLens Setup

Before running analysis steps, ensure TraceLens is installed and detect the correct Python version.

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
    if ! command -v TraceLens_generate_perf_report_pytorch &>/dev/null; then
        echo "First install attempt failed — retrying..."
        pip install --no-build-isolation "$HOME/TraceLens-internal" 2>&1 | tail -10
    fi
fi
```

**Detect the Python version where TraceLens is installed** (it may differ from the default `python3`):
```bash
TRACELENS_PYTHON=$(python3 -c "import TraceLens; print('python3')" 2>/dev/null \
    || python3.11 -c "import TraceLens; print('python3.11')" 2>/dev/null \
    || echo "python3")
echo "TraceLens Python: $TRACELENS_PYTHON"
```

**Auto-detect GPU and create GPU arch JSON** (required for roofline calculations):
```bash
python3 -c "
import json, subprocess, re, os

gpu_arch_path = '{{OUTPUT_DIR}}/results/gpu_arch.json'

PLATFORM_SPECS = {
    'MI300X': {'name': 'MI300X', 'mem_bw_gbps': 5300, 'memory_gb': 192, 'max_achievable_tflops': {'matrix_fp16': 654, 'matrix_bf16': 708, 'matrix_fp32': 163, 'matrix_fp64': 81, 'matrix_fp8': 1273, 'matrix_int8': 2600, 'vector_fp16': 163, 'vector_bf16': 163, 'vector_fp32': 81, 'vector_fp64': 40}},
    'MI325X': {'name': 'MI325X', 'mem_bw_gbps': 6000, 'memory_gb': 256, 'max_achievable_tflops': {'matrix_fp16': 794, 'matrix_bf16': 843, 'matrix_fp32': 194, 'matrix_fp64': 97, 'matrix_fp8': 1519, 'matrix_int8': 3094, 'vector_fp16': 194, 'vector_bf16': 194, 'vector_fp32': 97, 'vector_fp64': 48}},
    'MI350X': {'name': 'MI350X', 'mem_bw_gbps': 6000, 'memory_gb': 288, 'max_achievable_tflops': {'matrix_fp16': 794, 'matrix_bf16': 843, 'matrix_fp32': 194, 'matrix_fp64': 97, 'matrix_fp8': 1519, 'matrix_int8': 3094, 'vector_fp16': 194, 'vector_bf16': 194, 'vector_fp32': 97, 'vector_fp64': 48}},
    'MI355X': {'name': 'MI355X', 'mem_bw_gbps': 8000, 'memory_gb': 288, 'max_achievable_tflops': {'matrix_fp16': 1686, 'matrix_bf16': 1686, 'matrix_fp32': 137, 'matrix_fp64': 68, 'matrix_fp8': 3567, 'matrix_fp6': 4574, 'matrix_fp4': 5663, 'matrix_int8': 7134, 'vector_fp16': 274, 'vector_bf16': 274, 'vector_fp32': 137, 'vector_fp64': 68}},
    'MI400': {'name': 'MI400', 'mem_bw_gbps': 19600, 'memory_gb': 432, 'max_achievable_tflops': {'matrix_fp16': 2500, 'matrix_bf16': 2500, 'matrix_fp32': 1250, 'matrix_fp64': 625, 'matrix_fp8': 20000, 'matrix_fp4': 40000, 'matrix_int8': 20000, 'vector_fp16': 625, 'vector_bf16': 625, 'vector_fp32': 312, 'vector_fp64': 156}},
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

---

## Steps — Run for EACH Mode (eager, graph)

The following steps 1–4 are executed **twice**: once for `MODE=eager` (traces in `profiles_eager/`) and once for `MODE=graph` (traces in `profiles_graph/`). All output directories are suffixed with `_${MODE}`.

```bash
for MODE in eager graph; do
    TRACE_DIR="{{PROFILE_DIR}}/profiles_${MODE}"
    echo ""
    echo "========================================================"
    echo "  Analyzing $MODE mode traces from $TRACE_DIR"
    echo "========================================================"
```

### Step 1: Discover and Validate Trace Files

Locate torch profiler trace files in `$TRACE_DIR`, filtering out async_llm (frontend-only) traces, docker logs, and benchmark result JSONs. Validate that each candidate file actually contains `traceEvents`.

```bash
python3 -c "
import gzip, glob, re, sys, os

trace_dir = '$TRACE_DIR'
valid_traces = []
rank_map = {}

PEEK_BYTES = 65536

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
            prefix = fh.read(PEEK_BYTES)
        if '\"traceEvents\"' in prefix:
            valid_traces.append(f)
            rank_match = re.search(r'rank[-_]?(\d+)', basename)
            rank = int(rank_match.group(1)) if rank_match else len(valid_traces) - 1
            rank_map[f] = rank
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f'VALID torch trace (rank {rank}, {size_mb:.1f} MB compressed): {f}')
        else:
            print(f'SKIPPED (no traceEvents key in first {PEEK_BYTES} bytes): {f}')
    except Exception as e:
        print(f'ERROR reading {f}: {e}')

print(f'TRACE_COUNT={len(valid_traces)}')
if valid_traces:
    sorted_by_rank = sorted(valid_traces, key=lambda x: rank_map.get(x, 999))
    print(f'RANK0_TRACE={sorted_by_rank[0]}')
    print(f'WORLD_SIZE={len(valid_traces)}')
else:
    print('WARNING: No valid torch profiler traces found for $MODE mode.')
"
```

If TRACE_COUNT is 0, skip this mode and continue to the next.

**Create trace naming symlinks for TraceLens multi-rank tools** (they expect `rank*_trace.json.gz`):
```bash
cd "$TRACE_DIR"
for f in dp0_pp0_tp*_rank*.pt.trace.json.gz; do
    [ -f "$f" ] || continue
    rank=$(echo "$f" | grep -oP 'rank\K\d+')
    ln -sf "$f" "rank${rank}_trace.json.gz"
done
cd -
```

### Step 2: Phase-Split Traces — CRITICAL

Split the rank-0 trace into prefill-decode and decode-only phases. **This step is mandatory and must succeed before proceeding.** Phase-split traces enable per-phase gap analysis and roofline comparison, which are essential for identifying whether bottlenecks differ between compute-heavy prefill and memory-bandwidth-heavy decode.

```bash
SPLIT_SCRIPT="$HOME/TraceLens-internal/examples/custom_workflows/split_vllm_trace_annotation.py"
PHASE_SPLIT_DIR="{{OUTPUT_DIR}}/results/phase_split_${MODE}"
mkdir -p "$PHASE_SPLIT_DIR"

echo "Splitting $MODE rank-0 trace into prefill-decode and decode-only phases..."
$TRACELENS_PYTHON "$SPLIT_SCRIPT" "$RANK0_TRACE" \
    -o "$PHASE_SPLIT_DIR" \
    --find-steady-state \
    --num-steps 32 \
    2>&1 | tee /tmp/phase_split_${MODE}.log | tail -30

SPLIT_EXIT=${PIPESTATUS[0]}
echo "Phase split exit code: $SPLIT_EXIT"
ls -lh "$PHASE_SPLIT_DIR/"
```

**If the split fails, diagnose, fix, and retry. Do NOT proceed without valid phase-split traces:**

1. `ModuleNotFoundError`: The script may need a different Python version. Use the detected `$TRACELENS_PYTHON`.
2. Empty output / no `prefilldecode_*` or `decode_*` files: The profiler `delay_iterations` / `max_iterations` may have missed the transition. **Re-run Phase 4 profiling** with adjusted parameters (e.g., increase `max_iterations` to 512 or reduce `delay_iterations`).
3. Corrupted trace (JSON parse errors): **Re-run Phase 4 profiling** to produce fresh traces.
4. Python version incompatibility (e.g., `match` syntax in Python 3.10): Try `python3.11` explicitly.

After fixing, retry the split. Only proceed once you have valid `prefilldecode_*.json.gz` and `decode_*.json.gz` files in `$PHASE_SPLIT_DIR`.

```bash
PREFILL_DECODE_TRACE=$(ls "$PHASE_SPLIT_DIR"/prefilldecode_*.json.gz 2>/dev/null | head -1)
DECODE_ONLY_TRACE=$(ls "$PHASE_SPLIT_DIR"/decode_*.json.gz 2>/dev/null | head -1)

if [ -z "$PREFILL_DECODE_TRACE" ] || [ -z "$DECODE_ONLY_TRACE" ]; then
    echo "CRITICAL: Phase split failed for $MODE mode. Fix and retry before proceeding."
    # Diagnose and retry here — see guidance above
fi

echo "PREFILL_DECODE_TRACE=$PREFILL_DECODE_TRACE"
echo "DECODE_ONLY_TRACE=$DECODE_ONLY_TRACE"
```

### Step 3: Gap Analysis (Per-Phase + Full Trace)

Run gap analysis on three trace variants to get separate kernel rankings for prefill vs decode:

```bash
# Full trace
python3 "{{SCRIPTS_DIR}}/trace_analyzer.py" "$TRACE_DIR" \
    --gap-analysis \
    --output-dir "{{OUTPUT_DIR}}/results/gap_analysis_${MODE}" \
    --start-pct 0 --end-pct 100 --top-k 20

# Prefill-decode phase
python3 "{{SCRIPTS_DIR}}/trace_analyzer.py" "$PHASE_SPLIT_DIR" \
    --gap-analysis \
    --output-dir "{{OUTPUT_DIR}}/results/gap_analysis_prefill_decode_${MODE}" \
    --start-pct 0 --end-pct 100 --top-k 20 \
    --trace-pattern "prefilldecode_*.json.gz"

# Decode-only phase
python3 "{{SCRIPTS_DIR}}/trace_analyzer.py" "$PHASE_SPLIT_DIR" \
    --gap-analysis \
    --output-dir "{{OUTPUT_DIR}}/results/gap_analysis_decode_only_${MODE}" \
    --start-pct 0 --end-pct 100 --top-k 20 \
    --trace-pattern "decode_*.json.gz"
```

This produces a ranked list of the most expensive GPU kernels for each phase. Key categories:
- **GEMM kernels** (e.g., `ck_fmha_*`, `hipblas*`) — matrix multiply for attention and FFN layers
- **Communication kernels** (e.g., `ncclAllReduce*`, `allgather*`) — collective ops for tensor parallelism
- **Custom attention** (e.g., `paged_attention_*`, `flash_attn_*`) — KV cache operations
- **Quantization** (e.g., `dequant*`, `mxfp4_*`) — precision conversion overhead

### Step 4: TraceLens Analysis

**4a. Single-rank performance report (full trace):**
```bash
mkdir -p "{{OUTPUT_DIR}}/results/tracelens_${MODE}"

TraceLens_generate_perf_report_pytorch \
    --profile_json_path "$RANK0_TRACE" \
    --output_csvs_dir "{{OUTPUT_DIR}}/results/tracelens_${MODE}" \
    --enable_kernel_summary \
    2>&1 | tee "{{OUTPUT_DIR}}/results/tracelens_${MODE}.log"

echo "Rank-0 report exit code: $?"
ls -lh "{{OUTPUT_DIR}}/results/tracelens_${MODE}/"
```

**4b. Multi-rank collective report (if TP > 1):**
```bash
if [ "$WORLD_SIZE" -gt 1 ]; then
    mkdir -p "{{OUTPUT_DIR}}/results/tracelens_collective_${MODE}"

    PANDAS_FUTURE_INFER_STRING=0 TraceLens_generate_multi_rank_collective_report_pytorch \
        --trace_dir "$TRACE_DIR" \
        --world_size "$WORLD_SIZE" \
        --trace_pattern "rank*_trace.json.gz" \
        --output_csvs_dir "{{OUTPUT_DIR}}/results/tracelens_collective_${MODE}" \
        2>&1 | tee "{{OUTPUT_DIR}}/results/tracelens_collective_${MODE}.log"

    echo "Multi-rank collective report exit code: $?"
else
    echo "Single rank trace — skipping multi-rank collective report"
fi
```

**4c. Per-phase roofline analysis:**

Use the correct TraceLens script based on mode:
- Eager mode: `generate_perf_report_pytorch_vllm.py`
- Graph mode: `generate_perf_report_pytorch_vllm_graph.py`

```bash
GPU_ARCH_JSON="{{OUTPUT_DIR}}/results/gpu_arch.json"

if [ "$MODE" = "graph" ]; then
    ROOFLINE_SCRIPT="$HOME/TraceLens-internal/TraceLens/Reporting/generate_perf_report_pytorch_vllm_graph.py"
else
    ROOFLINE_SCRIPT="$HOME/TraceLens-internal/TraceLens/Reporting/generate_perf_report_pytorch_vllm.py"
fi
```

**Prefill-decode phase roofline:**
```bash
if [ -n "$PREFILL_DECODE_TRACE" ] && [ -f "$PREFILL_DECODE_TRACE" ] && [ -f "$ROOFLINE_SCRIPT" ]; then
    mkdir -p "{{OUTPUT_DIR}}/results/tracelens_prefill_decode_${MODE}"
    echo "Running TraceLens roofline on $MODE prefill-decode phase..."

    PANDAS_FUTURE_INFER_STRING=0 $TRACELENS_PYTHON "$ROOFLINE_SCRIPT" \
        --profile_json_path "$PREFILL_DECODE_TRACE" \
        --output_csvs_dir "{{OUTPUT_DIR}}/results/tracelens_prefill_decode_${MODE}" \
        --enable_pseudo_ops \
        --group_by_parent_module \
        --enable_kernel_summary \
        $([ -f "$GPU_ARCH_JSON" ] && echo "--gpu_arch_json_path $GPU_ARCH_JSON") \
        2>&1 | tail -30

    echo "Prefill-decode roofline ($MODE) exit code: $?"
else
    echo "Skipping prefill-decode roofline for $MODE (trace or script unavailable)"
fi
```

**Decode-only phase roofline:**
```bash
if [ -n "$DECODE_ONLY_TRACE" ] && [ -f "$DECODE_ONLY_TRACE" ] && [ -f "$ROOFLINE_SCRIPT" ]; then
    mkdir -p "{{OUTPUT_DIR}}/results/tracelens_decode_only_${MODE}"
    echo "Running TraceLens roofline on $MODE decode-only phase..."

    PANDAS_FUTURE_INFER_STRING=0 $TRACELENS_PYTHON "$ROOFLINE_SCRIPT" \
        --profile_json_path "$DECODE_ONLY_TRACE" \
        --output_csvs_dir "{{OUTPUT_DIR}}/results/tracelens_decode_only_${MODE}" \
        --enable_pseudo_ops \
        --group_by_parent_module \
        --enable_kernel_summary \
        $([ -f "$GPU_ARCH_JSON" ] && echo "--gpu_arch_json_path $GPU_ARCH_JSON") \
        2>&1 | tail -30

    echo "Decode-only roofline ($MODE) exit code: $?"
else
    echo "Skipping decode-only roofline for $MODE (trace or script unavailable)"
fi
```

> **WARNING (graph mode)**: Graph mode roofline analysis can take 20+ minutes with `--enable_pseudo_ops --group_by_parent_module` due to complex `hipGraphLaunch` event hierarchies. Set a timeout of at least 1800 seconds. If it appears stalled, check CPU usage — high CPU means it's processing, not hung.

> **NOTE (graph mode category attribution)**: In graph mode, TraceLens `ops_summary_by_category` sees most ops as "other" under `GraphModule`/`hipGraphLaunch`. Use gap analysis kernel classification from step 3 as the fallback for per-category time breakdown in the graph-mode report.

**4d. Aggregate results into structured JSON:**

Save all analysis data to `{{OUTPUT_DIR}}/results/profile_analysis_${MODE}.json`:
```json
{
  "mode": "<eager|graph>",
  "trace_file": "<rank0 trace path>",
  "num_ranks_analyzed": <N>,
  "gap_analysis": {
    "full_trace": "<gap_analysis_{mode}/gap_analysis.json contents>",
    "prefill_decode": "<gap_analysis_prefill_decode_{mode}/gap_analysis.json contents>",
    "decode_only": "<gap_analysis_decode_only_{mode}/gap_analysis.json contents>"
  },
  "tracelens": {
    "gpu_timeline": "<from tracelens_{mode}/gpu_timeline.csv>",
    "category_breakdown": "<from tracelens_{mode}/ops_summary_by_category.csv>",
    "top_ops": "<from tracelens_{mode}/ops_summary.csv>",
    "top_kernels": "<from tracelens_{mode}/kernel_summary.csv>"
  },
  "collective": "<from tracelens_collective_{mode}/nccl_summary_implicit_sync.csv if available>",
  "gpu_arch": "<detected GPU model or null>",
  "phase_split": {
    "execution_details": "<from phase_split_{mode}/execution_details.json>",
    "prefill_decode_trace": "<path>",
    "decode_only_trace": "<path>"
  },
  "roofline": {
    "prefill_decode": {
      "gpu_timeline": "<from tracelens_prefill_decode_{mode}/gpu_timeline.csv>",
      "category_breakdown": "<from tracelens_prefill_decode_{mode}/ops_summary_by_category.csv>",
      "unified_perf_summary": "<from tracelens_prefill_decode_{mode}/unified_perf_summary.csv>"
    },
    "decode_only": {
      "gpu_timeline": "<from tracelens_decode_only_{mode}/gpu_timeline.csv>",
      "category_breakdown": "<from tracelens_decode_only_{mode}/ops_summary_by_category.csv>",
      "unified_perf_summary": "<from tracelens_decode_only_{mode}/unified_perf_summary.csv>"
    }
  }
}
```

```bash
done  # end of for MODE in eager graph
```

---

## Step 5: Generate Per-Mode Reports

Generate two standalone profiling reports: `profiling_report_eager.md` and `profiling_report_graph.md`.

Read `profile_analysis_eager.json` / `profile_analysis_graph.json` and the corresponding gap analysis JSONs to populate each report.

Each report MUST use the following template:

```markdown
# InferenceX Profiling Report (<MODE> Mode)

## Configuration
- **Config Key**: {{CONFIG_KEY}}
- **Date**: <current date>
- **Docker Image**: <image from config>
- **GPU**: <detected GPU> (<memory_gb> GB HBM, <mem_bw_gbps> GB/s, <peak_bf16_tflops> TFLOPS bf16 peak)
- **Framework**: <framework from config>
- **Model**: <model name> (<N> layers profiled out of <total> if reduced)
- **Precision**: <precision>
- **Tensor Parallelism**: <TP value>
- **Sequence Length**: ISL=<ISL>, OSL=<OSL>
- **Concurrency**: <concurrency used for profiling>

## GPU Utilization

| Metric | Full Trace | Prefill-Decode | Decode-Only |
|--------|------------|----------------|-------------|
| Computation Time (%) | ... | ... | ... |
| Exposed Comm Time (%) | ... | ... | ... |
| Exposed Memcpy Time (%) | ... | ... | ... |
| GPU Busy Time (%) | ... | ... | ... |
| GPU Idle Time (%) | ... | ... | ... |

**Key Finding**: <1-2 sentence summary of utilization pattern, e.g., "CUDA graphs reduced exposed communication from X% to Y%, shifting the bottleneck from AllReduce to MLA FlashAttention.">

## Top GPU Kernels (Steady-State)

From gap analysis of the full profiled steady-state iterations:

| Rank | Kernel Name | Calls | Total Time (us) | Avg (us) | % Total |
|------|-------------|-------|-----------------|----------|---------|
| 1 | <kernel_name> (<functional annotation>) | ... | ... | ... | ... |

Total GPU kernel time: <X> ms across <Y> unique kernel types.

Annotate kernel names with human-readable functional descriptions, e.g.:
- `_fwd_grouped_kernel_stage1` → `(MLA FlashAttention)`
- `Cijk_Ailk_Bljk_HHS_BH_MT128x128x64...` → `(GEMM FusedQkvAProj)`
- `ncclAllReduceRingLL` → `(AllReduce TP sync)`
- `fused_moe_kernel` → `(FusedMoE expert dispatch)`

## Kernel Category Breakdown

From TraceLens ops_summary_by_category (eager mode) or gap analysis kernel classification (graph mode — see note below):

| Category | Count | Total Time (ms) | % of Kernel Time |
|----------|-------|-----------------|------------------|
| GEMM | ... | ... | ... |
| Attention | ... | ... | ... |
| Communication | ... | ... | ... |
| FusedMoE | ... | ... | ... |
| other (<list key subcategories>) | ... | ... | ... |

> **Note (graph mode only)**: TraceLens `ops_summary_by_category` cannot see inside `hipGraphLaunch` — most ops appear as "other" under `GraphModule`. The category breakdown above uses gap analysis kernel name classification as a fallback to attribute time to functional categories.

## Phase-Split Roofline Analysis

Traces are split into prefill-decode and decode-only phases using `split_vllm_trace_annotation.py`, then analyzed with `generate_perf_report_pytorch_vllm.py` (eager) or `generate_perf_report_pytorch_vllm_graph.py` (graph) for per-phase roofline insights against <GPU> specs (<mem_bw> GB/s HBM bandwidth, <peak_tflops> TFLOPS bf16 peak).

### Prefill-Decode Phase (<N> steps, BS=<batch_size>, conc=<concurrency>)

| Metric | Value |
|--------|-------|
| Computation Time (%) | ... |
| GPU Busy Time (%) | ... |
| Dominant Bound Type | compute or memory |

Top roofline ops (prefill-decode):

| Op Name | M×N×K | FLOPS/Byte | TFLOPS/s | Bound Type | Pct Roofline |
|---------|-------|------------|----------|------------|--------------|
| <descriptive name> | ... | ... | ... | ... | ... |

Filter out trivial ops (e.g., `elementwise copy`) to focus on meaningful compute ops.

### Decode-Only Phase (<N> steps, BS=<batch_size>, conc=<concurrency>)

| Metric | Value |
|--------|-------|
| Computation Time (%) | ... |
| GPU Busy Time (%) | ... |
| Dominant Bound Type | compute or memory |

Top roofline ops (decode-only):

| Op Name | M×N×K | FLOPS/Byte | TFLOPS/s | Bound Type | Pct Roofline |
|---------|-------|------------|----------|------------|--------------|
| <descriptive name> | ... | ... | ... | ... | ... |

> **Note (graph mode only)**: In graph mode, most ops inside CUDA graphs appear as a single `hipGraphLaunch` entry. Individual GEMM, attention, and MoE op roofline data is only available in the eager-mode report. See the combined report (`profiling_report.md`) for cross-referenced data.

### Phase Comparison

| Metric | Prefill-Decode | Decode-Only |
|--------|----------------|-------------|
| GPU Computation (%) | ... | ... |
| GPU Busy (%) | ... | ... |
| FusedMoE Time (%) | ... | ... |
| GEMM Time (%) | ... | ... |
| Attention Time (%) | ... | ... |
| RMSNorm Time (%) | ... | ... |
| Communication Time (%) | ... | ... |
| Dominant Bound | compute / memory | compute / memory |
| Batch Size | ... | ... |

## Profile Bottlenecks & Optimization Opportunities
- <bottleneck 1: description and recommendation>
- <bottleneck 2: description and recommendation>
- ...

## Raw Profile Data
- Gap analysis (full): `results/gap_analysis_<mode>/`
- Gap analysis (prefill-decode): `results/gap_analysis_prefill_decode_<mode>/`
- Gap analysis (decode-only): `results/gap_analysis_decode_only_<mode>/`
- TraceLens rank-0 CSVs: `results/tracelens_<mode>/`
- TraceLens collective CSVs: `results/tracelens_collective_<mode>/`
- Phase-split traces: `results/phase_split_<mode>/`
- Prefill-decode roofline CSVs: `results/tracelens_prefill_decode_<mode>/`
- Decode-only roofline CSVs: `results/tracelens_decode_only_<mode>/`
- GPU arch config: `results/gpu_arch.json`
- Profile analysis JSON: `results/profile_analysis_<mode>.json`
- Trace files: `profiles_<mode>/`
- Traces viewable at: https://ui.perfetto.dev/
```

---

## Step 6: Generate Final Combined Report

After generating both per-mode reports, produce a **final combined report** at `{{REPORT_DIR}}/profiling_report.md`. This is the **primary deliverable**. The per-mode reports remain as supporting detail.

### Motivation

Graph mode is the production execution path and has better utilization/timing data. But its roofline table is nearly empty because TraceLens can't see inside `hipGraphLaunch`. Eager mode reveals all the individual GEMMs, attention ops, and MoE kernels with their names, shapes, and roofline metrics — these same ops run inside the graph, they're just not individually measurable in graph mode.

### Rules for the Combined Report

1. **All timing/utilization data** comes from graph mode: GPU utilization table, top kernels (gap analysis), category breakdown, phase comparison, bottleneck analysis.

2. **Roofline table shows both graph and eager data, clearly labeled** with a `Mode` column:
   - Ops individually visible in graph mode (outside `hipGraphLaunch`, e.g., LM head GEMM, BMM attention) show their **graph-mode** roofline metrics, marked `graph`.
   - Ops hidden inside `hipGraphLaunch` in graph mode but individually visible in eager mode show their **eager-mode** roofline metrics (FLOPS/Byte, TFLOPS/s, Bound Type, Pct Roofline), marked `eager`.

3. **Show group relationships** using tree notation under a `**hipGraphLaunch group**` parent:

```markdown
Top roofline ops (decode-only):

| Op Name | M×N×K | FLOPS/Byte | TFLOPS/s | Bound Type | Pct Roofline | Mode |
|---------|-------|------------|----------|------------|--------------|------|
| GEMM LM head (hidden→vocab/TP) | 16×40960×7168 | 16.0 | 90.2 | memory | 70.6% | graph |
| BMM MLA (Q×K attention scores) | 16×512×128 | 13.8 | 0.4 | memory | 0.4% | graph |
| BMM MLA (attn×V context) | 16×128×512 | 13.8 | 0.2 | memory | 0.2% | graph |
| **hipGraphLaunch group** | — | — | — | — | — | graph |
| ├─ GEMM FusedQkvAProj (MLA QKV compress) | 16×9216×7168 | 15.9 | 68.9 | memory | 54.0% | eager |
| ├─ GEMM RowParallel (MLA output proj) | 16×7168×4608 | 15.9 | 61.4 | memory | 48.2% | eager |
| ├─ GEMM RowParallel (MLA KV proj) | 16×2112×7168 | 15.8 | 34.9 | memory | 27.5% | eager |
| ├─ GEMM ColumnParallel (MLA Q absorb) | 16×7168×2048 | 15.8 | 34.1 | memory | 26.9% | eager |
| └─ GEMM MLA (KV latent proj) | 16×7168×512 | 15.5 | 12.2 | memory | 9.9% | eager |
```

> **Note**: Ops under `hipGraphLaunch group` are individually invisible in graph mode. Their roofline data (FLOPS/Byte, TFLOPS/s, Pct Roofline) is measured from **eager mode** and shown here as reference. Eager-mode dispatch overhead differs from graph mode, so these numbers are approximate — actual graph-mode per-op performance may differ due to reduced launch latency and better pipelining.

4. **Matching logic**: Match eager roofline ops to graph by `(op_name, M, N, K)` tuple. Ops already individually visible in graph-mode roofline appear in the top section with graph data. Remaining eager ops that are not individually visible in graph mode are grouped under the `hipGraphLaunch group`.

5. **Graph-mode aggregate for the group**: From gap analysis, compute the total kernel time inside graph launches (sum of all kernels not individually visible in TraceLens roofline). Show this as the `hipGraphLaunch group` row's timing context in the Top GPU Kernels table (not the roofline table).

6. **Output**: `{{REPORT_DIR}}/profiling_report.md` — the primary deliverable.

---

**Print the final report paths:**
```bash
echo ""
echo "============================================"
echo "  Profiling Reports Generated"
echo "============================================"
echo "Per-mode reports:"
echo "  Eager: {{REPORT_DIR}}/profiling_report_eager.md"
echo "  Graph: {{REPORT_DIR}}/profiling_report_graph.md"
echo "Combined report (primary):"
echo "  {{REPORT_DIR}}/profiling_report.md"
echo "============================================"
```

## Completion
Update progress.json:
```json
{
  "phase": "profile-analyze",
  "phases_completed": ["env", "config", "benchmark", "benchmark-analyze", "profile", "profile-analyze"],
  "current_step": "profile analysis complete",
  "details": {
    "modes_analyzed": ["eager", "graph"],
    "gap_analysis": true,
    "tracelens_analysis": true,
    "phase_split_roofline": true,
    "gpu_arch_detected": "<GPU model name or null>",
    "reports": {
      "eager": "{{REPORT_DIR}}/profiling_report_eager.md",
      "graph": "{{REPORT_DIR}}/profiling_report_graph.md",
      "combined": "{{REPORT_DIR}}/profiling_report.md"
    }
  }
}
```
