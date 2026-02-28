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
  --disable-log-requests &> {{OUTPUT_DIR}}/vllm_baseline.log &
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

⚠️ **CRITICAL**: Two flags are mandatory for kernel shape analysis:
- `--enforce-eager` — disables CUDA Graphs so GPU kernels retain their `External id` linkage to CPU ops
- `--profiler-config` with `torch_profiler_record_shapes: true` — records tensor `Input Dims` on every CPU op

Without both, `analyze_kernel_shapes.py` will produce only `(unattributed)` shapes.

```bash
# venv mode only:
# source {{OUTPUT_DIR}}/venv/bin/activate
mkdir -p {{PROFILE_DIR}}/traces
TRACE_DIR=$(realpath {{PROFILE_DIR}}/traces)

# Build profiler config JSON (record_shapes is the key flag)
PROFILER_CFG=$(python3 -c "
import json; print(json.dumps({
  'profiler': 'torch',
  'torch_profiler_dir': '$TRACE_DIR',
  'torch_profiler_record_shapes': True,
  'torch_profiler_with_stack': True,
  'torch_profiler_with_flops': True,
  'torch_profiler_with_memory': False,
  'torch_profiler_use_gzip': True,
}))
")

# Start vLLM WITH profiler + enforce-eager — output to log file
VLLM_TORCH_PROFILER_DIR="$TRACE_DIR" \
vllm serve {{HF_MODEL}} \
  --dtype auto \
  --max-model-len 4096 \
  --port 8193 \
  --disable-log-requests \
  --enforce-eager \
  --profiler-config "$PROFILER_CFG" &> {{OUTPUT_DIR}}/vllm_trace.log &
VLLM_PID=$!
echo "Trace vLLM PID: $VLLM_PID (log: {{OUTPUT_DIR}}/vllm_trace.log)"

# Wait silently
for i in $(seq 1 60); do
  curl -s http://localhost:8193/health > /dev/null 2>&1 && break
  sleep 5
done
curl -s http://localhost:8193/health > /dev/null 2>&1 && echo "Trace server ready" || { echo "FAILED"; tail -5 {{OUTPUT_DIR}}/vllm_trace.log; }

# Start profiling via API
curl -s -X POST http://localhost:8193/start_profile && echo "Profiler started"

# Send requests for trace — output to file
vllm bench serve \
  --model {{HF_MODEL}} --port 8193 \
  --dataset-name random \
  --input-len {{INPUT_LEN}} --output-len {{OUTPUT_LEN}} \
  --num-prompts 30 --max-concurrency {{CONCURRENCY}} \
  --request-rate inf --save-result \
  --result-dir {{PROFILE_DIR}} --result-filename trace_benchmark.json \
  --label trace &> {{PROFILE_DIR}}/bench_trace.log

# Stop profiling and flush trace
curl -s -X POST http://localhost:8193/stop_profile && echo "Profiler stopped"
sleep 15
kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null

echo "Trace files:"
ls -lh {{PROFILE_DIR}}/traces/ 2>/dev/null | head -5
```

## Step 3: Extract Kernel Bottlenecks from Trace

```bash
cd {{PROFILE_DIR}}
cp {{OUTPUT_DIR}}/scripts/vllm_trace_extractor.py .

TRACE_FILE=$(ls -t traces/*.json traces/*.json.gz 2>/dev/null | head -1)
echo "Analyzing: $TRACE_FILE"

python3 vllm_trace_extractor.py -i "$TRACE_FILE" \
  --full-csv kernel_full.csv \
  --unique-csv kernel_unique.csv
```

## Step 4: Generate bottlenecks.json

```bash
cd {{PROFILE_DIR}}
python3 -c "
import csv, json

kernels = []
with open('kernel_unique.csv') as f:
    for row in csv.DictReader(f):
        kernels.append({
            'name': row['name'], 'count': int(row['count']),
            'total_dur_us': float(row['total_dur']),
            'avg_dur_us': float(row['avg_dur']),
            'median_dur_us': float(row['median_dur']),
        })

total = sum(k['total_dur_us'] for k in kernels)
bottlenecks = []
for k in kernels[:30]:
    pct = k['total_dur_us'] / total * 100 if total > 0 else 0
    name = k['name']
    optimizable = True
    reason = ''
    if 'Cijk_' in name or 'gemm' in name.lower() or 'hipblas' in name.lower():
        reason = 'GEMM/rocBLAS'; optimizable = False
    elif 'attn' in name.lower() or 'flash' in name.lower() or 'mha' in name.lower():
        reason = 'Attention'; optimizable = True
    elif 'norm' in name.lower() or 'rms' in name.lower():
        reason = 'Normalization'; optimizable = True
    elif 'elementwise' in name.lower() or 'vectorized' in name.lower():
        reason = 'Elementwise'; optimizable = True
    elif 'silu' in name.lower() or 'gelu' in name.lower():
        reason = 'Activation'; optimizable = True
    elif 'copy' in name.lower() or 'Cat' in name:
        reason = 'Memory op'; optimizable = False
    bottlenecks.append({**k, 'cuda_time_percent': pct, 'optimizable': optimizable, 'reason': reason})

print('Total GPU time: %.2fms, Top 10 kernels:' % (total/1000))
for i, b in enumerate(bottlenecks[:10], 1):
    print('  %d. %-45s %5.1f%%' % (i, b['name'][:45], b['cuda_time_percent']))

with open('bottlenecks.json', 'w') as f:
    json.dump(bottlenecks, f, indent=2)
print('Saved bottlenecks.json (%d kernels)' % len(bottlenecks))
"
```

## Step 5: Per-Shape Kernel Time Analysis

Analyze time proportion of **each shape** for each operator category.
This correlates GPU kernel durations with CPU-side operator shapes from the trace.

```bash
cd {{PROFILE_DIR}}
cp {{OUTPUT_DIR}}/scripts/analyze_kernel_shapes.py .

TRACE_FILE=$(ls -t traces/*.json traces/*.json.gz 2>/dev/null | head -1)
echo "Shape analysis on: $TRACE_FILE"

python3 analyze_kernel_shapes.py -i "$TRACE_FILE" -o .
```

This produces:
- `kernel_shape_analysis.json` — structured per-category, per-shape breakdown
- `kernel_shape_analysis.csv` — flat CSV for inspection

Example output:
```
  GEMM — 94.1% of total (15488.61ms, 10 shapes, 100% attributed)
  ──────────────────────────────────────────────────────────────────────────────────────
    Shape                                               %Total  %InCat  Time(ms)  Count  Avg(us)
    [4,4096]x[4096,24576]                                30.5%   32.4%  5023.08   9072    553.7
    [2,4096]x[4096,24576]                                15.4%   16.4%  2533.19   4644    545.5
    [4,12288]x[12288,4096]                               12.8%   13.6%  2112.64   9072    232.9
    [4,4096]x[4096,6144]                                  7.0%    7.4%  1144.21   9072    126.1
    ...

  Attention — 2.4% of total (390.56ms, 4 shapes, 100% attributed)
  ──────────────────────────────────────────────────────────────────────────────────────
    Shape                                               %Total  %InCat  Time(ms)  Count  Avg(us)
    [4,32,128]x[4,8,128]x[4,8,128]x[4,32,128]             1.5%   64.4%   251.52  27216      9.2
    ...
```

⚠️ **CRITICAL**: The shapes must be real traced shapes. Use this data in Phase 5 to pick exact dimensions for problem files and prioritize which (operator, shape) combinations to optimize first.

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
