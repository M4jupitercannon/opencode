# Phase 7: Integration & End-to-End Testing {{SKIP_LABEL}}

## Goal
Apply optimized kernels to vLLM and measure ACTUAL serving throughput improvement.

## ⚠️ CRITICAL: The ONLY meaningful metric is end-to-end serving throughput
- Kernel-level speedup numbers are reference only
- **MUST compare `vllm bench serve` results: baseline vs patched**
- Same concurrency, same input/output lengths, same number of prompts

## Step 1: Discover vLLM Module Structure

Before creating any patch, inspect the actual vLLM internal modules to find the correct patch targets.

```bash
source {{OUTPUT_DIR}}/venv/bin/activate

python3 -c "
import inspect

# Inspect layernorm module
import vllm.model_executor.layers.layernorm as ln
print('=== layernorm module ===')
for name in dir(ln):
    obj = getattr(ln, name)
    if isinstance(obj, type):
        methods = [m for m in dir(obj) if 'forward' in m.lower()]
        print(f'  {name}: {methods}')

# Inspect activation module
import vllm.model_executor.layers.activation as act
print('\\n=== activation module ===')
for name in dir(act):
    obj = getattr(act, name)
    if isinstance(obj, type):
        methods = [m for m in dir(obj) if 'forward' in m.lower()]
        if methods:
            print(f'  {name}: {methods}')

# Inspect rotary embedding
import vllm.model_executor.layers.rotary_embedding as rope
print('\\n=== rotary_embedding module ===')
for name in dir(rope):
    obj = getattr(rope, name)
    if isinstance(obj, type):
        methods = [m for m in dir(obj) if 'forward' in m.lower()]
        if methods:
            print(f'  {name}: {methods}')

# Check what the model actually uses
print('\\n=== Model-specific layers ===')
from transformers import AutoConfig
config = AutoConfig.from_pretrained('{{HF_MODEL}}', trust_remote_code=True)
print(f'model_type: {config.model_type}')
try:
    import vllm.model_executor.models as models
    model_module = getattr(models, config.model_type, None)
    if model_module:
        print(f'vLLM model module: {model_module}')
except: pass
"
```

## Step 2: Map Each Optimized Kernel to a vLLM Target

For EACH `*_opt.py` file from Phase 6, determine:
- **What does the kernel do?** (Read the `Model` class in the corresponding problem file)
- **Which vLLM internal class/method does the same thing?**
- **What init args does the vLLM class need?** (e.g., `weight.shape[0]` for hidden_size)

Examine each kernel's interface:
```bash
cd {{PROBLEMS_DIR}}
for f in *_opt.py; do
  echo "=== $f ==="
  python3 -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('m', '$f')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
if hasattr(mod, 'ModelNew'):
    cls = mod.ModelNew
    import inspect
    sig = inspect.signature(cls.__init__) if hasattr(cls.__init__, '__wrapped__') else inspect.signature(cls.__init__)
    print(f'  ModelNew.__init__ signature: {sig}')
    print(f'  ModelNew.forward signature: {inspect.signature(cls.forward)}')
" 2>/dev/null
done
```

## Step 3: Generate patch_manifest.json

Based on the discovery from Steps 1 and 2, create `{{OPTIMIZED_DIR}}/patch_manifest.json`.

**The manifest describes WHERE to apply each kernel — no hardcoded assumptions.**

```json
{
  "model": "{{HF_MODEL}}",
  "created": "<ISO date>",
  "description": "Auto-generated patch manifest mapping optimized kernels to vLLM internals",
  "patches": [
    {
      "kernel_file": "problem_fused_rmsnorm_opt.py",
      "target_module": "<discovered vLLM module path>",
      "target_class": "<discovered class name>",
      "target_method": "<discovered forward method>",
      "init_args_from": "<expression to get init args from self, e.g. weight.shape[0]>",
      "description": "<what this patch does>"
    }
  ]
}
```

**Rules for generating the manifest:**
- Only include kernels that were verified faster in Phase 6
- The `target_module`, `target_class`, `target_method` MUST match actual vLLM code (from Step 1)
- The `init_args_from` is a Python expression evaluated as `self.<expr>` on the target instance
- If you can't determine a safe patch target for a kernel, SKIP it — don't guess

## Step 4: Validate Manifest (dry run)

```bash
source {{OUTPUT_DIR}}/venv/bin/activate
cd {{OPTIMIZED_DIR}}

# Copy patch script
cp {{OUTPUT_DIR}}/scripts/patch_vllm.py .

# Copy all *_opt.py from problems/
cp {{PROBLEMS_DIR}}/*_opt.py .

# Dry run to validate
python3 patch_vllm.py --manifest patch_manifest.json --dry-run
```

## Step 5: Create Patched vLLM Launcher

```bash
cat > {{OPTIMIZED_DIR}}/run_patched_vllm.py << 'LAUNCHER'
#!/usr/bin/env python3
"""Launch vLLM with optimized kernel patches from manifest."""
import sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "scripts"))

from patch_vllm import apply_manifest
manifest = os.path.join(SCRIPT_DIR, "patch_manifest.json")
results = apply_manifest(manifest, search_dirs=[SCRIPT_DIR])

# Start vLLM (pass through CLI args)
from vllm.entrypoints.openai.api_server import run_server
import asyncio
asyncio.run(run_server())
LAUNCHER

chmod +x {{OPTIMIZED_DIR}}/run_patched_vllm.py
```

## Step 6: Benchmark Baseline vs Patched

### Baseline (should already exist from Phase 4, re-run if needed)
```bash
source {{OUTPUT_DIR}}/venv/bin/activate

vllm serve {{HF_MODEL}} --dtype auto --max-model-len 4096 --port 8192 --disable-log-requests &
VLLM_PID=$!
timeout 300 bash -c 'until curl -s http://localhost:8192/health > /dev/null 2>&1; do sleep 5; done'

vllm bench serve \
  --model {{HF_MODEL}} --port 8192 \
  --dataset-name random \
  --input-len {{INPUT_LEN}} --output-len {{OUTPUT_LEN}} \
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{CONCURRENCY}} \
  --request-rate inf \
  --save-result \
  --result-dir {{REPORT_DIR}} --result-filename baseline_serving.json --label baseline

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null
```

### Patched
```bash
source {{OUTPUT_DIR}}/venv/bin/activate

python3 {{OPTIMIZED_DIR}}/run_patched_vllm.py \
  --model {{HF_MODEL}} --dtype auto --max-model-len 4096 --port 8193 --disable-log-requests &
PATCHED_PID=$!
timeout 300 bash -c 'until curl -s http://localhost:8193/health > /dev/null 2>&1; do sleep 5; done'

vllm bench serve \
  --model {{HF_MODEL}} --port 8193 \
  --dataset-name random \
  --input-len {{INPUT_LEN}} --output-len {{OUTPUT_LEN}} \
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{CONCURRENCY}} \
  --request-rate inf \
  --save-result \
  --result-dir {{REPORT_DIR}} --result-filename optimized_serving.json --label optimized

kill $PATCHED_PID 2>/dev/null; wait $PATCHED_PID 2>/dev/null
```

## Step 7: Compare Results

```bash
python3 -c "
import json, os

report_dir = '{{REPORT_DIR}}'
os.makedirs(os.path.join(report_dir, 'comparison_outputs'), exist_ok=True)

with open(os.path.join(report_dir, 'baseline_serving.json')) as f:
    baseline = json.load(f)
with open(os.path.join(report_dir, 'optimized_serving.json')) as f:
    optimized = json.load(f)

def g(d, k):
    v = d.get(k, 0)
    return float(v) if v is not None else 0

metrics = [
    ('request_throughput', 'req/s', True),
    ('output_throughput', 'tok/s', True),
    ('input_throughput', 'tok/s', True),
    ('mean_tpot_ms', 'ms', False),
    ('median_tpot_ms', 'ms', False),
    ('p99_tpot_ms', 'ms', False),
    ('mean_ttft_ms', 'ms', False),
    ('median_ttft_ms', 'ms', False),
    ('mean_itl_ms', 'ms', False),
]

print('=' * 70)
print(f'{\"Metric\":<25} {\"Baseline\":>12} {\"Optimized\":>12} {\"Change\":>12}')
print('=' * 70)

comparison = {}
for m, unit, hb in metrics:
    b, o = g(baseline, m), g(optimized, m)
    if b > 0:
        chg = (o-b)/b*100 if hb else (b-o)/b*100
        print(f'{m:<25} {b:>10.2f}{unit:>2} {o:>10.2f}{unit:>2} {\"+\" if chg>0 else \"\"}{chg:>8.1f}%')
    comparison[m] = {'baseline': b, 'optimized': o}

otps_b, otps_o = g(baseline, 'output_throughput'), g(optimized, 'output_throughput')
speedup = otps_o / otps_b if otps_b > 0 else 1.0
print(f'\\n>>> OVERALL SERVING SPEEDUP (OTPS): {speedup:.3f}x <<<')

with open(os.path.join(report_dir, 'comparison_outputs', 'comparison_results.json'), 'w') as f:
    json.dump({
        'baseline': {m: g(baseline, m) for m, _, _ in metrics},
        'optimized': {m: g(optimized, m) for m, _, _ in metrics},
        'speedup_otps': speedup,
        'concurrency': {{CONCURRENCY}}, 'input_len': {{INPUT_LEN}},
        'output_len': {{OUTPUT_LEN}}, 'num_prompts': {{NUM_PROMPTS}},
    }, f, indent=2)
"
```

Update progress.json: phases_completed.append("integrate")
