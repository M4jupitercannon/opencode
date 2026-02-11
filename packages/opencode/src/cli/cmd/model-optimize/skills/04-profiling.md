# Phase 4: Performance Profiling {{SKIP_LABEL}}

## Goal
Benchmark vLLM serving throughput AND collect GPU kernel trace for bottleneck analysis.

## ⚠️ CORRECT Profiling Approach for vLLM

**DO NOT** use `rocprof` or single-request profiling.
**DO** use `vllm bench serve` with realistic concurrency to capture production-like behavior.

The profiling has TWO parts:
1. **Throughput benchmark**: Measure baseline ITPS/OTPS/TPOT/TTFT at target concurrency
2. **Kernel trace**: Collect torch profiler trace via `VLLM_TORCH_PROFILER_DIR` for kernel analysis

## Step 1: Baseline Throughput Benchmark

```bash
source {{OUTPUT_DIR}}/venv/bin/activate

# Start vLLM serve (baseline, no profiler)
vllm serve {{HF_MODEL}} \
  --dtype auto \
  --max-model-len 4096 \
  --port 8192 \
  --disable-log-requests &
VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM to be ready..."
timeout 300 bash -c 'until curl -s http://localhost:8192/health > /dev/null 2>&1; do sleep 5; done'
echo "Server ready!"

# Run benchmark: 1K input / 1K output, concurrency={{CONCURRENCY}}
vllm bench serve \
  --model {{HF_MODEL}} \
  --port 8192 \
  --dataset-name random \
  --input-len {{INPUT_LEN}} \
  --output-len {{OUTPUT_LEN}} \
  --num-prompts {{NUM_PROMPTS}} \
  --max-concurrency {{CONCURRENCY}} \
  --request-rate inf \
  --save-result \
  --result-dir {{PROFILE_DIR}} \
  --result-filename baseline_benchmark.json \
  --label baseline

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
```

Parse and save the key metrics:
```bash
python3 -c "
import json
with open('{{PROFILE_DIR}}/baseline_benchmark.json') as f:
    data = json.load(f)
print('=== Baseline Throughput ===')
for key in ['total_input_tokens', 'total_output_tokens', 'request_throughput',
            'input_throughput', 'output_throughput',
            'mean_ttft_ms', 'median_ttft_ms', 'p99_ttft_ms',
            'mean_tpot_ms', 'median_tpot_ms', 'p99_tpot_ms',
            'mean_itl_ms', 'median_itl_ms', 'p99_itl_ms']:
    val = data.get(key, 'N/A')
    print(f'  {key}: {val}')
"
```

## Step 2: Collect Kernel Trace

```bash
source {{OUTPUT_DIR}}/venv/bin/activate
mkdir -p {{PROFILE_DIR}}/traces

# Start vLLM WITH profiler enabled
VLLM_TORCH_PROFILER_DIR={{PROFILE_DIR}}/traces \
vllm serve {{HF_MODEL}} \
  --dtype auto \
  --max-model-len 4096 \
  --port 8193 \
  --disable-log-requests &
VLLM_PID=$!

echo "Waiting for vLLM (profiler) to be ready..."
timeout 300 bash -c 'until curl -s http://localhost:8193/health > /dev/null 2>&1; do sleep 5; done'

# Send requests for trace collection (fewer prompts, same concurrency)
vllm bench serve \
  --model {{HF_MODEL}} \
  --port 8193 \
  --dataset-name random \
  --input-len {{INPUT_LEN}} \
  --output-len {{OUTPUT_LEN}} \
  --num-prompts 30 \
  --max-concurrency {{CONCURRENCY}} \
  --request-rate inf \
  --save-result \
  --result-dir {{PROFILE_DIR}} \
  --result-filename trace_benchmark.json \
  --label trace

# Wait for profiler to flush
sleep 15

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null

echo "Trace files:"
ls -lh {{PROFILE_DIR}}/traces/
```

## Step 3: Extract Kernel Bottlenecks from Trace

```bash
cd {{PROFILE_DIR}}
cp {{OUTPUT_DIR}}/scripts/vllm_trace_extractor.py .

# Find latest trace file
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
        reason = 'Normalization - Triton fusable'; optimizable = True
    elif 'elementwise' in name.lower() or 'vectorized' in name.lower():
        reason = 'Elementwise - Triton fusable'; optimizable = True
    elif 'silu' in name.lower() or 'gelu' in name.lower() or 'act' in name.lower():
        reason = 'Activation - fusable'; optimizable = True
    elif 'rope' in name.lower() or 'rotary' in name.lower():
        reason = 'RoPE - Triton fusable'; optimizable = True
    elif 'copy' in name.lower() or 'Cat' in name:
        reason = 'Memory op'; optimizable = False
    bottlenecks.append({**k, 'cuda_time_percent': pct, 'optimizable': optimizable, 'reason': reason})

print(f'Total GPU time: {total/1000:.2f}ms')
for i, b in enumerate(bottlenecks[:15], 1):
    opt = '✓' if b['optimizable'] else '✗'
    print(f\"{i:2d}. {b['name'][:50]:50s} {b['cuda_time_percent']:5.1f}% ({b['total_dur_us']/1000:.2f}ms) x{b['count']} {opt} {b['reason']}\")

with open('bottlenecks.json', 'w') as f:
    json.dump(bottlenecks, f, indent=2)
"
```

## Step 5: Save model shapes for problem file generation

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
