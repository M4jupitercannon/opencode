# Phase 4: Performance Profiling {{SKIP_LABEL}}

## Goal
Benchmark vLLM serving throughput AND collect GPU kernel trace for bottleneck analysis.

## ⚠️ CRITICAL: ALL vLLM output MUST go to log files
**NEVER let vLLM stdout/stderr appear in bash output.** Always use `&> logfile`.
**For `vllm bench serve`, redirect to file and only extract key metrics.**

## Step 1: Baseline Throughput Benchmark

```bash
source {{OUTPUT_DIR}}/venv/bin/activate

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
curl -s http://localhost:8192/health > /dev/null 2>&1 && echo "✓ Server ready" || { echo "✗ Failed — check vllm_baseline.log"; tail -5 {{OUTPUT_DIR}}/vllm_baseline.log; }

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

```bash
source {{OUTPUT_DIR}}/venv/bin/activate
mkdir -p {{PROFILE_DIR}}/traces

# Start vLLM WITH profiler — output to log file
VLLM_TORCH_PROFILER_DIR={{PROFILE_DIR}}/traces \
vllm serve {{HF_MODEL}} \
  --dtype auto \
  --max-model-len 4096 \
  --port 8193 \
  --disable-log-requests &> {{OUTPUT_DIR}}/vllm_trace.log &
VLLM_PID=$!
echo "Trace vLLM PID: $VLLM_PID (log: {{OUTPUT_DIR}}/vllm_trace.log)"

# Wait silently
for i in $(seq 1 60); do
  curl -s http://localhost:8193/health > /dev/null 2>&1 && break
  sleep 5
done
curl -s http://localhost:8193/health > /dev/null 2>&1 && echo "✓ Trace server ready" || { echo "✗ Failed"; tail -5 {{OUTPUT_DIR}}/vllm_trace.log; }

# Send requests for trace — output to file
vllm bench serve \
  --model {{HF_MODEL}} --port 8193 \
  --dataset-name random \
  --input-len {{INPUT_LEN}} --output-len {{OUTPUT_LEN}} \
  --num-prompts 30 --max-concurrency {{CONCURRENCY}} \
  --request-rate inf --save-result \
  --result-dir {{PROFILE_DIR}} --result-filename trace_benchmark.json \
  --label trace &> {{PROFILE_DIR}}/bench_trace.log

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

# Print compact summary (not full data)
print(f'Total GPU time: {total/1000:.2f}ms, Top 10 kernels:')
for i, b in enumerate(bottlenecks[:10], 1):
    print(f\"  {i}. {b['name'][:45]:45s} {b['cuda_time_percent']:5.1f}%\")

with open('bottlenecks.json', 'w') as f:
    json.dump(bottlenecks, f, indent=2)
print(f'Saved bottlenecks.json ({len(bottlenecks)} kernels)')
"
```

## Step 5: Save model shapes

```bash
source {{OUTPUT_DIR}}/venv/bin/activate
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
