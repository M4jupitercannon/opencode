# Phase 7: Integration & End-to-End Testing {{SKIP_LABEL}}

## Goal
Apply optimized kernels to vLLM and measure ACTUAL serving throughput improvement.

## ⚠️ CRITICAL: The ONLY meaningful metric is end-to-end serving throughput
- Kernel-level speedup numbers are reference only
- **MUST compare vLLM bench serve results: baseline vs patched**
- Same concurrency, same input/output lengths, same number of prompts

## Overview: Before/After Comparison

```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│   BASELINE vLLM serve       │     │   PATCHED vLLM serve        │
│   (no modifications)        │     │   (with optimized kernels)  │
│                             │     │                             │
│   vllm bench serve          │ vs  │   vllm bench serve          │
│   → throughput, TPOT, TTFT  │     │   → throughput, TPOT, TTFT  │
└─────────────────────────────┘     └─────────────────────────────┘
```

## Step 1: Prepare Patched vLLM Launcher

A `patch_vllm.py` script is provided at `{{OUTPUT_DIR}}/scripts/patch_vllm.py`.
It monkey-patches vLLM's internal layers (RMSNorm, activations) with your optimized Triton kernels.

Create a wrapper script that:
1. Imports `patch_vllm` to apply monkey-patches
2. Then starts vLLM serve normally

```bash
source {{OUTPUT_DIR}}/venv/bin/activate

cat > {{OPTIMIZED_DIR}}/run_patched_vllm.py << 'LAUNCHER'
#!/usr/bin/env python3
"""Launch vLLM with optimized kernel patches applied."""
import sys
import os

# Add paths for optimized kernels and patch script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "scripts"))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "problems"))

# Apply patches BEFORE vLLM loads the model
os.environ["PATCH_DIR"] = SCRIPT_DIR
import patch_vllm
stats = patch_vllm.apply_all(SCRIPT_DIR)

print(f"\nPatches applied: {stats}")
print("Starting vLLM serve with optimized kernels...\n")

# Now start vLLM - it will use patched modules
from vllm.entrypoints.openai.api_server import run_server
import asyncio
# Pass through command line args
asyncio.run(run_server())
LAUNCHER

chmod +x {{OPTIMIZED_DIR}}/run_patched_vllm.py
```

## Step 2: Copy optimized kernels to integration directory

```bash
# Copy all successful *_opt.py to optimized/
cd {{PROBLEMS_DIR}}
for f in *_opt.py; do
  if [ -f "$f" ]; then
    cp "$f" {{OPTIMIZED_DIR}}/
    echo "Copied: $f"
  fi
done

# Also copy the patch script
cp {{OUTPUT_DIR}}/scripts/patch_vllm.py {{OPTIMIZED_DIR}}/
```

## Step 3: Benchmark BASELINE (original vLLM)

This should already be done in Phase 4. If not:

```bash
source {{OUTPUT_DIR}}/venv/bin/activate

vllm serve {{HF_MODEL}} --dtype auto --max-model-len 4096 --port 8192 --disable-log-requests &
VLLM_PID=$!
timeout 300 bash -c 'until curl -s http://localhost:8192/health > /dev/null 2>&1; do sleep 5; done'

vllm bench serve \
  --model {{HF_MODEL}} --port 8192 \
  --dataset-name random \
  --input-len {{INPUT_LEN}} --output-len {{OUTPUT_LEN}} \
  --num-prompts {{NUM_PROMPTS}} \
  --max-concurrency {{CONCURRENCY}} \
  --request-rate inf \
  --result-dir {{REPORT_DIR}} \
  --result-filename baseline_serving.json \
  --label baseline

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
```

## Step 4: Benchmark PATCHED vLLM

```bash
source {{OUTPUT_DIR}}/venv/bin/activate

# Start patched vLLM
python3 {{OPTIMIZED_DIR}}/run_patched_vllm.py \
  --model {{HF_MODEL}} --dtype auto --max-model-len 4096 \
  --port 8193 --disable-log-requests &
PATCHED_PID=$!

echo "Waiting for patched vLLM to be ready..."
timeout 300 bash -c 'until curl -s http://localhost:8193/health > /dev/null 2>&1; do sleep 5; done'
echo "Patched server ready!"

# Same benchmark parameters as baseline
vllm bench serve \
  --model {{HF_MODEL}} --port 8193 \
  --dataset-name random \
  --input-len {{INPUT_LEN}} --output-len {{OUTPUT_LEN}} \
  --num-prompts {{NUM_PROMPTS}} \
  --max-concurrency {{CONCURRENCY}} \
  --request-rate inf \
  --result-dir {{REPORT_DIR}} \
  --result-filename optimized_serving.json \
  --label optimized

kill $PATCHED_PID 2>/dev/null; wait $PATCHED_PID 2>/dev/null
```

## Step 5: Compare Results

```bash
python3 -c "
import json, os

report_dir = '{{REPORT_DIR}}'
os.makedirs(os.path.join(report_dir, 'comparison_outputs'), exist_ok=True)

with open(os.path.join(report_dir, 'baseline_serving.json')) as f:
    baseline = json.load(f)
with open(os.path.join(report_dir, 'optimized_serving.json')) as f:
    optimized = json.load(f)

def safe_get(d, key, default=0):
    v = d.get(key, default)
    return float(v) if v is not None else default

metrics = [
    ('request_throughput', 'req/s', True),       # higher is better
    ('output_throughput', 'tok/s', True),         # higher is better (OTPS)
    ('input_throughput', 'tok/s', True),          # higher is better (ITPS)
    ('mean_tpot_ms', 'ms', False),               # lower is better
    ('median_tpot_ms', 'ms', False),             # lower is better
    ('p99_tpot_ms', 'ms', False),                # lower is better
    ('mean_ttft_ms', 'ms', False),               # lower is better
    ('median_ttft_ms', 'ms', False),             # lower is better
    ('mean_itl_ms', 'ms', False),                # lower is better
]

print('=' * 70)
print(f'{\"Metric\":<25} {\"Baseline\":>12} {\"Optimized\":>12} {\"Change\":>12}')
print('=' * 70)

comparison = {}
for metric, unit, higher_better in metrics:
    b = safe_get(baseline, metric)
    o = safe_get(optimized, metric)
    if b > 0:
        if higher_better:
            change = (o - b) / b * 100
        else:
            change = (b - o) / b * 100  # positive = improvement for latency
        symbol = '+' if change > 0 else ''
        print(f'{metric:<25} {b:>10.2f}{unit:>2} {o:>10.2f}{unit:>2} {symbol}{change:>8.1f}%')
    else:
        print(f'{metric:<25} {b:>10.2f}{unit:>2} {o:>10.2f}{unit:>2} {\"N/A\":>9}')
    comparison[metric] = {'baseline': b, 'optimized': o}

# Calculate overall speedup based on output throughput
b_otps = safe_get(baseline, 'output_throughput')
o_otps = safe_get(optimized, 'output_throughput')
speedup = o_otps / b_otps if b_otps > 0 else 1.0
print(f'\\n>>> OVERALL SERVING SPEEDUP (OTPS): {speedup:.3f}x <<<')

result = {
    'baseline': {k: safe_get(baseline, k) for k, _, _ in metrics},
    'optimized': {k: safe_get(optimized, k) for k, _, _ in metrics},
    'speedup_otps': speedup,
    'concurrency': {{CONCURRENCY}},
    'input_len': {{INPUT_LEN}},
    'output_len': {{OUTPUT_LEN}},
    'num_prompts': {{NUM_PROMPTS}},
}
with open(os.path.join(report_dir, 'comparison_outputs', 'comparison_results.json'), 'w') as f:
    json.dump(result, f, indent=2)
print(f'\\nResults saved to {report_dir}/comparison_outputs/comparison_results.json')
"
```

## Steps Summary
1. Create patched vLLM launcher with `run_patched_vllm.py`
2. Copy optimized kernels to integration directory
3. Benchmark baseline vLLM (`vllm bench serve`)
4. Benchmark patched vLLM (same parameters)
5. Compare throughput metrics (OTPS, TPOT, TTFT)
6. Record results in progress.json

Update progress.json: phases_completed.append("integrate")
