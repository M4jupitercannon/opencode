# Phase 4: Performance Profiling {{SKIP_LABEL}}

## Goal

Benchmark vLLM serving throughput AND collect GPU kernel trace for bottleneck analysis,
including per-shape kernel time breakdown.

## ⚠️ CRITICAL: ALL vLLM output MUST go to log files

**NEVER let vLLM stdout/stderr appear in bash output.** Always use `&> logfile`.
**For `vllm bench serve`, redirect to file and only extract key metrics.**

## ⚠️ Execution mode (docker vs venv)

Detect once before running this phase:

```bash
ENV_TYPE=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('env_type','venv'))" 2>/dev/null || echo "venv")
CONTAINER_NAME=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('container','vllm_model_opt'))" 2>/dev/null || echo "vllm_model_opt")
BEST_GPU=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('best_gpu',0))" 2>/dev/null || echo 0)
```

If `ENV_TYPE=docker`, run commands with:
`docker exec -e HIP_VISIBLE_DEVICES=$BEST_GPU $CONTAINER_NAME bash -lc "<command>"`.

## Step 1: Baseline Throughput Benchmark

```bash
# venv mode only:
# source {{OUTPUT_DIR}}/venv/bin/activate

# Start vLLM — ALL output to log file
vllm serve {{HF_MODEL}} \
  --dtype auto \
  --max-model-len 4096 \
  --port 8192 \
  --no-enable-log-requests &> {{OUTPUT_DIR}}/vllm_baseline.log &
VLLM_PID=$!
echo "Baseline vLLM PID: $VLLM_PID (log: {{OUTPUT_DIR}}/vllm_baseline.log)"

# Wait silently
for i in $(seq 1 60); do
  curl -s http://localhost:8192/health > /dev/null 2>&1 && break
  sleep 5
done
curl -s http://localhost:8192/health > /dev/null 2>&1 && echo "Server ready" || { echo "FAILED — check vllm_baseline.log"; tail -5 {{OUTPUT_DIR}}/vllm_baseline.log; }

# Run benchmark — output to file, then extract only key metrics
vllm bench serve \
  --model {{HF_MODEL}} --port 8192 \
  --dataset-name random \
  --input-len {{INPUT_LEN}} --output-len {{OUTPUT_LEN}} \
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{CONCURRENCY}} \
  --request-rate inf --save-result \
  --result-dir {{PROFILE_DIR}} --result-filename baseline_benchmark.json \
  --label baseline &> {{PROFILE_DIR}}/bench_baseline.log

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null

# Show ONLY key metrics (not the full benchmark output)
python3 -c "
import json
with open('{{PROFILE_DIR}}/baseline_benchmark.json') as f:
    d = json.load(f)
print('=== Baseline Metrics ===')
for k in ['output_throughput','request_throughput','mean_tpot_ms','mean_ttft_ms','mean_itl_ms','completed']:
    print(f'  {k}: {d.get(k,\"N/A\")}')
"
```

## Step 2: Collect Kernel Trace

### ⛔ HARD REQUIREMENTS — all three are mandatory, do NOT skip any:

1. **`--enforce-eager`** on the `vllm serve` command line — disables CUDA Graphs so every GPU kernel retains its `External id` linkage back to the CPU op that launched it. Without this, the trace has GPU kernels but no way to correlate them to operator shapes.

2. **`--profiler-config`** with a JSON object containing **`torch_profiler_record_shapes: true`** — tells the PyTorch profiler to record `Input Dims` on every CPU operator event. Without this, the trace has CPU ops but their shapes are empty.

3. **`/start_profile` → send requests → `/stop_profile`** API sequence — vLLM does NOT profile from startup. You must explicitly start and stop profiling via the HTTP API. Without this sequence, no trace file is written at all.

**If any of the three is missing, `analyze_kernels.py` WILL produce 0% attributed shapes. You MUST re-collect the trace — do NOT proceed with bad data.**

### Docker path note

When running inside Docker, the profiler writes to the **container-side path**. Use `/workspace/output/profile/traces` (not the host path) inside `--profiler-config`.
Do NOT set the `VLLM_TORCH_PROFILER_DIR` environment variable — it is deprecated (removed in v0.15+). Use `torch_profiler_dir` inside `--profiler-config` instead.

### ⚠️ Two trace files are written

vLLM writes **two** separate trace files per profiling session:

- **`*async_llm*`** — frontend-only trace (CPU activity only, NO GPU kernels, NO shapes). **This file is USELESS for shape analysis.**
- **`*rank-0*`** — worker trace (CPU + CUDA activities, has `cpu_op` events with `Input Dims`, has `kernel` events with `External id`). **This is the file you need.**

Setting `ignore_frontend: true` in the profiler config suppresses the frontend trace entirely, which also reduces profiling overhead.

**When selecting the trace file in subsequent steps, you MUST pick the `rank-0` worker trace, NOT the `async_llm` trace.**

```bash
# venv mode only:
# source {{OUTPUT_DIR}}/venv/bin/activate
mkdir -p {{PROFILE_DIR}}/traces

# Determine the correct trace directory path
# Docker mode: use container-side path; venv mode: use host path
if [ "$ENV_TYPE" = "docker" ]; then
  TRACE_DIR="/workspace/output/profile/traces"
else
  TRACE_DIR=$(realpath {{PROFILE_DIR}}/traces)
fi

# Build profiler config JSON
# ⚠️ torch_profiler_record_shapes: true is the critical flag for shape analysis
# ⚠️ ignore_frontend: true suppresses the useless async_llm trace (CPU-only, no GPU data)
PROFILER_CFG=$(python3 -c "
import json; print(json.dumps({
  'profiler': 'torch',
  'torch_profiler_dir': '$TRACE_DIR',
  'torch_profiler_record_shapes': True,
  'torch_profiler_with_stack': True,
  'torch_profiler_with_flops': True,
  'torch_profiler_with_memory': False,
  'torch_profiler_use_gzip': True,
  'ignore_frontend': True,
}))
")
echo "Profiler config: $PROFILER_CFG"

# VLLM_RPC_TIMEOUT: trace flush after /stop_profile can take minutes for large models.
# vLLM docs recommend 30min for 100 reqs on 70B. Default 10s causes timeouts.
export VLLM_RPC_TIMEOUT=1800000

# Start vLLM WITH profiler + enforce-eager — output to log file
# ⚠️ BOTH --enforce-eager AND --profiler-config are MANDATORY
# ⚠️ Do NOT set VLLM_TORCH_PROFILER_DIR env var — it is deprecated.
#    torch_profiler_dir in --profiler-config is the correct way.
vllm serve {{HF_MODEL}} \
  --dtype auto \
  --max-model-len 4096 \
  --port 8193 \
  --no-enable-log-requests \
  --enforce-eager \
  --profiler-config "$PROFILER_CFG" &> {{OUTPUT_DIR}}/vllm_trace.log &
VLLM_PID=$!
echo "Trace vLLM PID: $VLLM_PID (log: {{OUTPUT_DIR}}/vllm_trace.log)"

# Wait for server ready
for i in $(seq 1 60); do
  curl -s http://localhost:8193/health > /dev/null 2>&1 && break
  sleep 5
done
curl -s http://localhost:8193/health > /dev/null 2>&1 && echo "Trace server ready" || { echo "FAILED"; tail -5 {{OUTPUT_DIR}}/vllm_trace.log; kill $VLLM_PID 2>/dev/null; exit 1; }

# Verify --enforce-eager is active (check log for "enforce_eager")
grep -qi "enforce.eager\|CUDA graphs.*disabled\|eager mode" {{OUTPUT_DIR}}/vllm_trace.log && echo "enforce-eager: confirmed" || echo "WARNING: enforce-eager not confirmed in log"

# ⚠️ CRITICAL: Start profiling via API — without this, NO trace is written
PROFILE_RESP=$(curl -s -X POST http://localhost:8193/start_profile)
echo "start_profile response: $PROFILE_RESP"
# Verify profiling started (response should not be an error)
echo "$PROFILE_RESP" | grep -qi "error" && { echo "⛔ FAILED to start profiler — check vllm_trace.log"; kill $VLLM_PID 2>/dev/null; exit 1; }
sleep 2

# Send requests for trace — output to file
vllm bench serve \
  --model {{HF_MODEL}} --port 8193 \
  --dataset-name random \
  --input-len {{INPUT_LEN}} --output-len {{OUTPUT_LEN}} \
  --num-prompts 30 --max-concurrency {{CONCURRENCY}} \
  --request-rate inf --save-result \
  --result-dir {{PROFILE_DIR}} --result-filename trace_benchmark.json \
  --label trace &> {{PROFILE_DIR}}/bench_trace.log

# ⚠️ CRITICAL: Stop profiling via API — this flushes the trace to disk
STOP_RESP=$(curl -s -X POST http://localhost:8193/stop_profile)
echo "stop_profile response: $STOP_RESP"

# Wait for trace file to be fully written (poll until size stabilizes)
echo "Waiting for trace flush (may take several minutes for large models)..."
PREV_SIZE=0; STABLE=0
for i in $(seq 1 120); do
  TRACE_FILE=$(ls -S {{PROFILE_DIR}}/traces/rank*.gz 2>/dev/null | head -1)
  if [ -n "$TRACE_FILE" ]; then
    CUR_SIZE=$(stat -c%s "$TRACE_FILE" 2>/dev/null || echo 0)
    if [ "$CUR_SIZE" -eq "$PREV_SIZE" ] && [ "$CUR_SIZE" -gt 0 ]; then
      STABLE=$((STABLE + 1))
      [ $STABLE -ge 3 ] && echo "Trace stabilized at $(du -h "$TRACE_FILE" | cut -f1)" && break
    else
      STABLE=0
    fi
    PREV_SIZE=$CUR_SIZE
  fi
  sleep 5
done
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null

echo "Trace files:"
ls -lh {{PROFILE_DIR}}/traces/ 2>/dev/null | head -5
```

### ⛔ Step 2b: Verify Trace Has Shape Data (MANDATORY before proceeding)

**Do NOT skip this step.** If this verification fails, you MUST go back and re-collect the trace with the correct flags.

```bash
# Run the analyzer in validation-only mode.
# It auto-selects the worker trace (rank-0) from the traces/ directory,
# validates that it has GPU kernels + CPU ops with shapes, and exits non-zero if invalid.
cp {{OUTPUT_DIR}}/scripts/analyze_kernels.py {{PROFILE_DIR}}/
python3 {{PROFILE_DIR}}/analyze_kernels.py -i {{PROFILE_DIR}}/traces/ --validate-only 2>&1 | head -20

# If the above exits non-zero, the trace is invalid for shape analysis.
# Check the output for the specific failure reason and re-collect with:
#   1. --enforce-eager on vllm serve
#   2. --profiler-config with torch_profiler_record_shapes: true AND ignore_frontend: true
#   3. /start_profile API call BEFORE requests, /stop_profile AFTER
```

## Step 3: Split Trace & Run TraceLens Performance Analysis

The `analyze_kernels.py` script handles trace splitting and TraceLens analysis in one command. It:

1. Splits the trace into prefill-decode and decode-only phases via TraceLens
2. Runs standalone performance analysis on each phase trace
3. Produces `analysis_summary.json` and per-phase CSV reports

```bash
cd {{PROFILE_DIR}}
cp {{OUTPUT_DIR}}/scripts/analyze_kernels.py .

# TraceLens is auto-discovered or cloned from GitHub if not found.
# Use --tracelens-dir to point to a local copy, or set TRACELENS_DIR env var.
# --skip-validation skips re-loading the full trace (already validated in Step 2b).
python3 analyze_kernels.py -i traces/ -o . --skip-validation
```

This produces:

- `phase_traces/` — split trace files (prefill-decode, decode-only)
- `prefilldecode_report/` — TraceLens CSVs (ops_summary.csv, unified_perf_summary.csv, etc.)
- `decode_report/` — TraceLens CSVs for decode-only phase
- `analysis_summary.json` — machine-readable summary with roofline data

## Step 4: Generate bottlenecks.json from TraceLens Results

See `model-optimize.md` Phase 4, Step 4 for the full inline script that:

- Reads TraceLens `ops_summary.csv` and `unified_perf_summary.csv`
- Calculates roofline efficiency for GEMM kernels
- Breaks down Attention (SDPA) sub-kernels
- Produces `bottlenecks.json` with per-shape roofline data

⚠️ **CRITICAL**: The shapes are real traced shapes from the profiled workload. Use this data in Phase 5 to pick exact dimensions for problem files and prioritize which (operator, shape) combinations to optimize first.

## Step 6: Save model shapes

```bash
# venv mode only:
# source {{OUTPUT_DIR}}/venv/bin/activate
python3 -c "
import json
from transformers import AutoConfig
c = AutoConfig.from_pretrained('{{HF_MODEL}}', trust_remote_code=True)
shapes = {
    'hidden_size': getattr(c, 'hidden_size', None),
    'intermediate_size': getattr(c, 'intermediate_size', None),
    'num_attention_heads': getattr(c, 'num_attention_heads', None),
    'num_key_value_heads': getattr(c, 'num_key_value_heads', None),
    'head_dim': getattr(c, 'hidden_size', 0) // max(getattr(c, 'num_attention_heads', 1), 1),
    'num_hidden_layers': getattr(c, 'num_hidden_layers', None),
    'vocab_size': getattr(c, 'vocab_size', None),
}
with open('{{PROFILE_DIR}}/model_shapes.json', 'w') as f:
    json.dump(shapes, f, indent=2)
print(json.dumps(shapes, indent=2))
"
```

Update progress.json: phases_completed.append("profile")
