# Phase 4: Performance Profiling {{SKIP_LABEL}}

## Goal
Profile the vLLM serving to identify bottleneck GPU kernels.

## ⚠️ CRITICAL: Use torch profiler trace via vLLM, NOT rocprof

vLLM has built-in torch profiler support via `VLLM_TORCH_PROFILER_DIR`.
This produces `.pt.trace.json` files that contain GPU kernel events.

### Step 1: Collect Torch Profiler Trace

```bash
source {{OUTPUT_DIR}}/venv/bin/activate
mkdir -p {{PROFILE_DIR}}/traces

# Start vLLM with profiler enabled
VLLM_TORCH_PROFILER_DIR={{PROFILE_DIR}}/traces \
vllm serve {{HF_MODEL}} \
  --dtype auto \
  --max-model-len 2048 \
  --port 8192 \
  --disable-log-requests &

VLLM_PID=$!
echo "vLLM started (PID=$VLLM_PID), waiting for model to load..."
sleep 60  # Wait for model to fully load

# Send profiling requests (short prompts to capture decode kernels)
for i in $(seq 1 5); do
  curl -s http://localhost:8192/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"{{HF_MODEL}}\", \"prompt\": \"The future of AI is\", \"max_tokens\": 20}" > /dev/null
  echo "Request $i sent"
  sleep 2
done

# Wait a moment for profiler to flush
sleep 10

# Stop vLLM
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "Trace files collected:"
ls -lh {{PROFILE_DIR}}/traces/
```

### Step 2: Extract GPU Kernel Events from Trace

Use the provided `vllm_trace_extractor.py` script (ALKA-style trace analysis):

```bash
cd {{PROFILE_DIR}}
cp {{OUTPUT_DIR}}/scripts/vllm_trace_extractor.py .

# Find the trace file
TRACE_FILE=$(ls -t traces/*.json traces/*.json.gz 2>/dev/null | head -1)
echo "Analyzing trace: $TRACE_FILE"

python3 vllm_trace_extractor.py -i "$TRACE_FILE" \
  --full-csv kernel_full.csv \
  --unique-csv kernel_unique.csv
```

### Step 3: Generate Bottleneck Analysis

```bash
cd {{PROFILE_DIR}}
python3 -c "
import csv, json

# Read unique kernels (already sorted by total_dur descending)
kernels = []
with open('kernel_unique.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        kernels.append({
            'name': row['name'],
            'count': int(row['count']),
            'total_dur_us': float(row['total_dur']),
            'avg_dur_us': float(row['avg_dur']),
            'median_dur_us': float(row['median_dur']),
        })

total_gpu_time = sum(k['total_dur_us'] for k in kernels)

bottlenecks = []
for k in kernels[:30]:  # Top 30
    pct = k['total_dur_us'] / total_gpu_time * 100 if total_gpu_time > 0 else 0
    # Classify kernel type
    name = k['name']
    optimizable = True
    reason = ''
    if 'Cijk_' in name or 'gemm' in name.lower():
        reason = 'GEMM - rocBLAS/Tensile optimized'
        optimizable = False
    elif 'attn' in name.lower() or 'flash' in name.lower():
        reason = 'Attention kernel - can try AITER'
        optimizable = True
    elif 'norm' in name.lower() or 'reduce' in name.lower():
        reason = 'Normalization/Reduce - Triton fusable'
        optimizable = True
    elif 'elementwise' in name.lower() or 'vectorized' in name.lower():
        reason = 'Elementwise - Triton fusable'
        optimizable = True
    elif 'copy' in name.lower() or 'Cat' in name:
        reason = 'Memory op - bandwidth limited'
        optimizable = False
    else:
        reason = 'Other kernel'
        optimizable = True

    bottlenecks.append({
        'name': name,
        'count': k['count'],
        'total_dur_us': k['total_dur_us'],
        'avg_dur_us': k['avg_dur_us'],
        'cuda_time_percent': pct,
        'optimizable': optimizable,
        'reason': reason,
    })

print(f'Total GPU time: {total_gpu_time/1000:.2f}ms')
print(f'\\n=== Top Bottleneck Kernels ===')
for i, b in enumerate(bottlenecks[:15], 1):
    opt = '✓' if b['optimizable'] else '✗'
    print(f\"{i:2d}. {b['name'][:50]:50s} {b['cuda_time_percent']:5.1f}% ({b['total_dur_us']/1000:.2f}ms) x{b['count']} {opt}\")

with open('bottlenecks.json', 'w') as f:
    json.dump(bottlenecks, f, indent=2)
print(f'\\nSaved bottlenecks.json ({len(bottlenecks)} kernels)')
"
```

### Step 4: Capture model config shapes

```bash
source {{OUTPUT_DIR}}/venv/bin/activate
python3 -c "
import json
from transformers import AutoConfig
config = AutoConfig.from_pretrained('{{HF_MODEL}}', trust_remote_code=True)
shapes = {
    'hidden_size': getattr(config, 'hidden_size', None),
    'intermediate_size': getattr(config, 'intermediate_size', None),
    'num_attention_heads': getattr(config, 'num_attention_heads', None),
    'num_key_value_heads': getattr(config, 'num_key_value_heads', None),
    'head_dim': getattr(config, 'hidden_size', 0) // max(getattr(config, 'num_attention_heads', 1), 1),
    'num_hidden_layers': getattr(config, 'num_hidden_layers', None),
    'vocab_size': getattr(config, 'vocab_size', None),
    'max_position_embeddings': getattr(config, 'max_position_embeddings', None),
    'typical_batch_size': 1,
    'typical_seq_len': 64,
}
with open('{{PROFILE_DIR}}/model_shapes.json', 'w') as f:
    json.dump(shapes, f, indent=2)
print(json.dumps(shapes, indent=2))
"
```

## Steps Summary
1. Run vLLM with `VLLM_TORCH_PROFILER_DIR` to collect torch trace
2. Extract kernel events using `vllm_trace_extractor.py`
3. Generate bottlenecks.json with kernel classification
4. Capture model config shapes for problem file generation
5. Update progress.json
