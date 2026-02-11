# Phase 7: Integration & Final Testing {{SKIP_LABEL}}

## Goal
Integrate optimized kernels into vLLM serving and **MEASURE ACTUAL end-to-end performance**.

## ⚠️ CRITICAL REQUIREMENTS
1. **MEASURE ACTUAL end-to-end speedup** using vLLM benchmark tools
2. **Compare original vs optimized** vLLM serving throughput
3. **Run the SAME workload** with and without optimizations

## Use Project venv
```bash
source {{OUTPUT_DIR}}/venv/bin/activate
```

## vLLM Integration Strategy

For vLLM, optimizations are integrated via:
1. **Custom attention backends** (e.g., AITER Flash Attention)
2. **Monkey-patching** vLLM's internal modules at startup
3. **Environment variables** to select optimized paths

### Create Integration Script: `{{OPTIMIZED_DIR}}/integrate_vllm.py`

Write an `integrate_vllm.py` that:
1. Imports optimized Triton kernels from `*_opt.py` files
2. Monkey-patches vLLM's model modules (e.g., RMSNorm, attention)
3. Can be loaded before vLLM serve starts

```python
"""
Usage: Import this before starting vLLM to apply optimized kernels.
  python -c "import integrate_vllm; integrate_vllm.apply_patches()" && vllm serve ...
OR:
  VLLM_PLUGINS=integrate_vllm vllm serve ...
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Import optimized kernels
_optimized = {}
for name in ["fused_rmsnorm", "fused_residual_rmsnorm", "fused_rope", "fused_swiglu"]:
    try:
        mod = __import__(f"problem_{name}_opt")
        if hasattr(mod, 'ModelNew'):
            _optimized[name] = mod.ModelNew
            print(f"  [OK] Loaded: {name}")
    except Exception as e:
        print(f"  [SKIP] {name}: {e}")

def apply_patches():
    """Apply optimized kernels to vLLM model layers."""
    # Monkey-patch approach depends on which kernels succeeded
    # Example: patch RMSNorm in vllm.model_executor.layers
    pass
```

## MANDATORY: End-to-End Performance Measurement

### Baseline (original vLLM)
```bash
source {{OUTPUT_DIR}}/venv/bin/activate

# Start original vLLM
vllm serve {{HF_MODEL}} --dtype auto --max-model-len 2048 --port 8192 &
VLLM_PID=$!
sleep 60

# Benchmark with vLLM's built-in benchmark
python3 -m vllm.entrypoints.openai.run_batch_benchmark \
  --model {{HF_MODEL}} \
  --endpoint /v1/completions \
  --num-prompts 50 \
  --prompt-len 128 \
  --output-len 64 \
  --port 8192 \
  2>&1 | tee {{REPORT_DIR}}/baseline_benchmark.txt

# Or use simple timing
python3 -c "
import time, requests, json
url = 'http://localhost:8192/v1/completions'
prompts = ['The future of AI is'] * 20
times = []
for p in prompts:
    t0 = time.perf_counter()
    r = requests.post(url, json={'model': '{{HF_MODEL}}', 'prompt': p, 'max_tokens': 64})
    times.append(time.perf_counter() - t0)
avg_ms = sum(times)/len(times)*1000
print(f'Baseline avg latency: {avg_ms:.1f}ms')
with open('{{REPORT_DIR}}/baseline_latency.json', 'w') as f:
    json.dump({'avg_ms': avg_ms, 'times_ms': [t*1000 for t in times]}, f, indent=2)
"

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
```

### Optimized (with kernel patches)
```bash
# Start vLLM with optimized kernels
# Method 1: Pre-import patches
python3 -c "
import sys; sys.path.insert(0, '{{OPTIMIZED_DIR}}')
import integrate_vllm; integrate_vllm.apply_patches()
import vllm
# ... continue with vLLM serve
"

# Method 2: If using AITER or env-based optimizations
# Set appropriate environment variables and re-run benchmark
```

### Compare Results
```bash
python3 -c "
import json
with open('{{REPORT_DIR}}/baseline_latency.json') as f:
    baseline = json.load(f)
with open('{{REPORT_DIR}}/optimized_latency.json') as f:
    optimized = json.load(f)
speedup = baseline['avg_ms'] / optimized['avg_ms']
print(f'Baseline:  {baseline[\"avg_ms\"]:.1f}ms')
print(f'Optimized: {optimized[\"avg_ms\"]:.1f}ms')
print(f'Speedup:   {speedup:.2f}x')
results = {
    'baseline_ms': baseline['avg_ms'],
    'optimized_ms': optimized['avg_ms'],
    'speedup': speedup,
}
with open('{{REPORT_DIR}}/comparison_outputs/comparison_results.json', 'w') as f:
    json.dump(results, f, indent=2)
"
```

## Steps
1. Create integrate_vllm.py with kernel patches
2. Benchmark original vLLM (baseline)
3. Apply patches and benchmark optimized vLLM
4. Record ACTUAL speedup in progress.json
5. Update progress.json: phases_completed.append("integrate")
