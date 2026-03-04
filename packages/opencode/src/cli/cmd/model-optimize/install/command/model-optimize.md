---
description: "vLLM kernel optimization and integration pipeline (Phases 5-8). Usage: /model-optimize <model_name> [output_dir]"
agent: model-opt
---

# vLLM Kernel Optimization & Integration Pipeline (Phases 5–8)

## Target
- **HuggingFace Model**: $1
- **Output Directory**: $2 (if not specified, use `/tmp/model_opt_<model_short_name>`)

## Prerequisites
Phases 0–4 (environment setup, model serving, profiling, bottleneck analysis) must be completed first.
See `model-analyzer.md` for those phases. The following artifacts from Phase 4 are required:
- `<output_dir>/profile/bottlenecks.json`
- `<output_dir>/profile/analysis_summary.json`
- `<output_dir>/profile/model_shapes.json`
- `<output_dir>/profile/decode_report/unified_perf_summary.csv`
- `<output_dir>/profile/prefilldecode_report/unified_perf_summary.csv`

## ⚠️ CRITICAL RULES
- **Docker mode** (preferred): If `env_info.json` has `env_type: "docker"`, prefix commands with `docker exec $CONTAINER_NAME bash -c "..."`. Set `HIP_VISIBLE_DEVICES=$BEST_GPU`.
- **venv mode** (fallback): If `env_type: "venv"`, activate venv: `source <output_dir>/venv/bin/activate`
- **ALL vLLM commands MUST redirect output to log files** (`&> logfile`) — NEVER dump vLLM logs into bash output
- **ALL decisions MUST be data-driven** — read shapes from TraceLens `analysis_summary.json` / `unified_perf_summary.csv`, not hardcoded
- **Optimized kernels MUST use @triton.jit** — torch rewrites are FORBIDDEN
- **Integrate via vLLM CustomOp.register_oot()** — NEVER modify installed packages
- **Serving benchmarks MUST use `vllm bench serve --save-result`**

## ⛔ MANDATORY VALIDATION
After Phase 6 and Phase 7, run:
```bash
python <output_dir>/scripts/validate_pipeline.py --project-dir <output_dir> --phase all
```

---

# Phase 5: Generate Problem Files for Kernel Optimization 

## Goal
Convert bottleneck operators into Problem files for kernel-optimize.
**IMPORTANT**: Analyze operators for fusion opportunities BEFORE creating individual problem files.

## STEP 0: Review TraceLens Analysis (from Phase 4)

Before creating problem files, review the TraceLens analysis results to understand:
- Which **operator categories** dominate GPU time (GEMM, Attention, Norm, Activation, ...)
- For each category, which **specific shapes** are the hottest (from `unified_perf_summary.csv`)
- Whether each op is **memory-bound** or **compute-bound** (from roofline analysis)
- The difference between **prefill** and **decode** phase characteristics

```bash
# Review decode phase (dominant in serving workloads)
python3 -c "
import json
with open('<output_dir>/profile/analysis_summary.json') as f:
    summary = json.load(f)
for phase_name, phase in summary['phases'].items():
    print(f'\n=== {phase_name} ===')
    for cat in phase.get('categories', []):
        print(f'  {cat[\"category\"]:20s} {cat[\"percentage\"]:5.1f}%  ({cat[\"total_kernel_time_ms\"]:.2f}ms)')
    print(f'  Top ops:')
    for op in phase.get('top_ops', [])[:8]:
        print(f'    {op[\"name\"]:45s} {op[\"percentage\"]:5.1f}%')
"

# Review roofline data for decode phase (shapes + memory/compute bound)
head -20 <output_dir>/profile/decode_report/unified_perf_summary.csv
```

## STEP 1: Operator Fusion Analysis (CRITICAL)

A standalone `analyze_fusion.py` script is provided at `<output_dir>/scripts/analyze_fusion.py`.
Use it to detect fusable operator patterns:

```bash
cp <output_dir>/scripts/analyze_fusion.py <output_dir>/profile/
cd <output_dir>/profile
python analyze_fusion.py
cat fusion_opportunities.json
```

### Common Fusion Opportunities in LLMs

| Pattern | Operators to Fuse | Fused Name | Expected Speedup |
|---------|-------------------|------------|------------------|
| **ResidualNorm** | add + rmsnorm/layernorm | fused_residual_norm | 1.2-1.5x |
| **SwiGLU/GeGLU** | silu/gelu + mul | fused_swiglu | 1.3-1.8x |
| **BiasAdd** | matmul + add (bias) | fused_linear_bias | 1.1-1.3x |
| **RotaryEmbed** | rope_cos + rope_sin + cat | fused_rope | 1.2-1.5x |
| **QKV Projection** | 3x linear (q,k,v) | fused_qkv_proj | 1.2-1.4x |
| **MLP Block** | linear + activation + linear | fused_mlp | 1.3-2.0x |

## STEP 2: Create FUSED Problem Files (Priority)

**Create fused kernels BEFORE individual kernels!**

Use ACTUAL shapes from TraceLens `unified_perf_summary.csv` in each phase report directory
(e.g. `<output_dir>/profile/decode_report/unified_perf_summary.csv`) and `<output_dir>/profile/model_shapes.json`.
Focus on the ops with the highest `Percentage (%)` and use roofline data to decide optimization strategy.

### Example: Fused Residual + RMSNorm
```python
# problem_fused_residual_rmsnorm.py
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))
        self.eps = eps
    def forward(self, hidden_states, residual):
        hidden_states = hidden_states + residual
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states

# ⚠️ Use shapes from shape_ranges.json!
batch_size = 1
seq_len = 64       # typical from profiling
hidden_size = 4096 # from model config

def get_inputs():
    return [
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda'),
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda'),
    ]
def get_init_inputs():
    return [hidden_size]
```

### Example: Fused SwiGLU
```python
# problem_fused_swiglu.py
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, gate, up):
        return torch.nn.functional.silu(gate) * up

batch_size, seq_len, intermediate_size = 1, 64, 11008
def get_inputs():
    return [
        torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.float16, device='cuda'),
        torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.float16, device='cuda'),
    ]
def get_init_inputs():
    return []
```

## STEP 3: Create Individual Problem Files (Lower Priority)

Only for operators that: cannot be fused, take > 5% time, and are not already optimized by vendor libs (rocBLAS GEMM).

### Common Operators

- **RMSNorm/LayerNorm**: `class Model` with weight param, forward does variance + rsqrt + mul
- **Attention**: `class Model` wrapping `torch.nn.functional.scaled_dot_product_attention`
- **SiLU/GELU**: `class Model` with activation function
- **RoPE**: `class Model` with cos/sin rotation

## STEP 4: Generate Optimization Manifest

Create `<output_dir>/problems/optimization_manifest.json`:

```json
{
  "model": "$1",
  "description": "Edit 'enabled' to control which optimizations to apply",
  "optimizations": [
    {"name": "fused_residual_rmsnorm", "file": "problem_fused_residual_rmsnorm.py",
     "type": "fused", "priority": "HIGH", "enabled": true},
    {"name": "linear_gemm", "file": "problem_linear.py",
     "type": "individual", "priority": "LOW", "enabled": false,
     "notes": "rocBLAS usually optimal"}
  ]
}
```

## Steps
1. Run fusion analysis
2. Create fused problem files (HIGH priority)
3. Create individual problem files (MEDIUM/LOW)
4. Generate optimization_manifest.json
5. Update progress.json



---

# Phase 6: Kernel Optimization 

## Goal
Write optimized Triton kernels for each problem file and verify speedup.

## ⚠️ NO external `opencode` command needed
Optimize kernels DIRECTLY in this session using the test scripts provided.

## Scripts Available
- `<output_dir>/scripts/kernel_test_runner.py` — test accuracy + benchmark
- `<output_dir>/scripts/kernel_finalize.py` — save best result to target file

## Workflow for EACH Problem File

For each `problem_*.py` file in `<output_dir>/problems/`:

### 1. Read the source file to understand the PyTorch operator
```bash
cat <output_dir>/problems/problem_XXX.py
```

### 2. Check GPU architecture
```bash
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}, Arch: {torch.cuda.get_device_capability()}')"
```

### 3. Write the optimized Triton kernel
Create `<output_dir>/problems/problem_XXX_opt.py` with:
- `class ModelNew(nn.Module)` using `@triton.jit` Triton kernels
- Same `__init__` signature as `Model`
- Use `@triton.autotune` with 10-20 diverse configs

### 4. Test accuracy + benchmark
```bash
source <output_dir>/venv/bin/activate
python3 <output_dir>/scripts/kernel_test_runner.py \
  --src <output_dir>/problems/problem_XXX.py \
  --target <output_dir>/problems/problem_XXX_opt.py
```

The script prints: `RESULT_JSON: {"speedup": 1.5, "accuracy": "PASSED", ...}`

### 5. Iterate if needed
- Accuracy FAILED → fix kernel, re-run step 4
- Speedup too low → adjust block sizes, fusion strategy, re-run step 4

### 6. Finalize when satisfied
```bash
python3 <output_dir>/scripts/kernel_finalize.py \
  --target <output_dir>/problems/problem_XXX_opt.py
```

## Priority Order

| Priority | Kernel Type | Goal | Reason |
|----------|-------------|------|--------|
| **HIGH** | Fused Residual+RMSNorm | 1.5x | Memory traffic reduction |
| **HIGH** | Fused SwiGLU | 1.5x | Activation fusion |
| **HIGH** | Fused RoPE | 1.5x | Custom optimization |
| MEDIUM | Individual norms | 1.3x | If not covered by fused version |
| LOW | Linear/GEMM | 1.1x | rocBLAS usually optimal |
| **SKIP** | Simple add/copy | — | Overhead > benefit |

## When to SKIP a kernel
- If it's part of a fused kernel you already optimized
- If rocBLAS/vendor lib is already near-optimal
- If after 3 attempts speedup is < 1.0x at actual shapes

## Triton Optimization Guide

### Autotune Strategy
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        # ... add 10-20 configs covering the space
    ],
    key=['N'],
)
```

### Common Patterns
- **Memory-bound**: Optimize access patterns, vectorization
- **Compute-bound**: Larger tiles, more arithmetic per memory access
- **Fused kernels**: Combine multiple ops to reduce memory traffic
- **FP32 accumulation**: Use `tl.float32` for acc, cast output at end

## After All Kernels Done

```bash
echo "=== Kernel Optimization Results ==="
cd <output_dir>/problems
for f in *_opt.py; do
  if [ -f "$f" ]; then
    tracker="${f%.py}_best.json"
    if [ -f "$tracker" ]; then
      speedup=$(python3 -c "import json; print(json.load(open('$tracker')).get('best_speedup', 0))")
      echo "  $f: ${speedup}x"
    fi
  fi
done
```

Copy successful optimizations to `<output_dir>/optimized/`:
```bash
cd <output_dir>/problems
for f in *_opt.py; do
  tracker="${f%.py}_best.json"
  if [ -f "$tracker" ]; then
    speedup=$(python3 -c "import json; d=json.load(open('$tracker')); print(d.get('best_speedup',0))")
    if python3 -c "exit(0 if $speedup > 1.0 else 1)"; then
      cp "$f" <output_dir>/optimized/
      echo "Copied $f (${speedup}x)"
    fi
  fi
done
```

Update progress.json: phases_completed.append("optimize")


---

# Phase 7: Integration & End-to-End Testing 

## Goal
Apply optimized kernels to vLLM via CustomOp and measure ACTUAL serving throughput.

## ⛔ MANDATORY: This phase REQUIRES real measured data

**This phase is NOT complete until:**
1. A patched vLLM server has ACTUALLY been started and served requests
2. `vllm bench serve` has been run against the patched server
3. `optimized_serving.json` has `"label": "optimized"` (NOT "baseline")
4. The validation script passes

**FORBIDDEN:**
- Estimating speedup with Amdahl's law
- Copying baseline numbers and modifying them
- Reporting "estimated" or "conservative" speedup
- Skipping the patched server benchmark

---

## Integration Mechanism: vLLM CustomOp.register_oot()

We use vLLM's OFFICIAL extension mechanism (not monkey-patching):
- Docs: https://docs.vllm.ai/en/latest/design/custom_op/
- Each optimized kernel is wrapped as a vLLM CustomOp subclass
- `CustomOp.register_oot()` replaces the default op at instantiation time
- If the optimized kernel fails, vLLM falls back to the default

---

## Step 1: Generate vLLM Plugin

The `generate_vllm_plugin.py` script auto-creates a plugin from `*_opt.py` files:

```bash
source <output_dir>/venv/bin/activate
cd <output_dir>/optimized

# Copy all *_opt.py from problems
cp <output_dir>/problems/*_opt.py . 2>/dev/null

# Generate the plugin
python3 <output_dir>/scripts/generate_vllm_plugin.py \
  --kernel-dir <output_dir>/optimized

# Verify generated files
ls -la vllm_plugin/
cat vllm_plugin/manifest.json
```

This generates:
- `<output_dir>/optimized/vllm_plugin/__init__.py` — registers CustomOps
- `<output_dir>/optimized/run_patched_vllm.py` — launcher script
- `<output_dir>/optimized/vllm_plugin/manifest.json` — registration summary

## Step 2: Test Plugin Registration (dry run)

Verify that the plugin loads without errors:

```bash
source <output_dir>/venv/bin/activate
python3 -c "
import sys; sys.path.insert(0, '<output_dir>/optimized')
import vllm_plugin
print('Plugin loaded successfully')
"
```

## Step 3: ⛔ MANDATORY — Benchmark Baseline

Phase 4 produced `baseline_benchmark.json` in `profile/` (different workload parameters). Run a fresh baseline here in `report/` with the same parameters as the optimized run for a fair comparison:

```bash
source <output_dir>/venv/bin/activate

# ALL vLLM output to log files — NEVER to stdout
vllm serve $1 --dtype auto --max-model-len 4096 --port 8192 --disable-log-requests &> <output_dir>/vllm_baseline_e2e.log &
VLLM_PID=$!
echo "Baseline PID: $VLLM_PID"
for i in $(seq 1 60); do curl -s http://localhost:8192/health > /dev/null 2>&1 && break; sleep 5; done
curl -s http://localhost:8192/health > /dev/null 2>&1 && echo "✓ Ready" || { echo "✗ Failed"; tail -3 <output_dir>/vllm_baseline_e2e.log; }

vllm bench serve \
  --model $1 --port 8192 \
  --dataset-name random \
  --input-len 1024 --output-len 1024 \
  --num-prompts 100 --max-concurrency 16 \
  --request-rate inf --save-result \
  --result-dir <output_dir>/report --result-filename baseline_serving.json --label baseline \
  &> <output_dir>/report/bench_baseline.log

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null

# Show only key metrics
python3 -c "
import json
with open('<output_dir>/report/baseline_serving.json') as f: d=json.load(f)
print('=== Baseline ===')
for k in ['output_throughput','mean_tpot_ms','mean_ttft_ms','completed']:
    print(f'  {k}: {d.get(k,\"N/A\")}')
"
```

## Step 4: ⛔ MANDATORY — Start Patched vLLM and Benchmark

```bash
source <output_dir>/venv/bin/activate

# Start patched vLLM — ALL output to log file
python3 <output_dir>/optimized/run_patched_vllm.py serve \
  --model $1 --dtype auto --max-model-len 4096 \
  --port 8193 --disable-log-requests &> <output_dir>/vllm_patched.log &
PATCHED_PID=$!
echo "Patched PID: $PATCHED_PID (log: <output_dir>/vllm_patched.log)"

# Wait silently
for i in $(seq 1 60); do curl -s http://localhost:8193/health > /dev/null 2>&1 && break; sleep 5; done
curl -s http://localhost:8193/health > /dev/null 2>&1 && echo "✓ Patched server ready" || { echo "✗ Failed"; tail -5 <output_dir>/vllm_patched.log; }

# Verify correct model (compact output)
curl -s http://localhost:8193/v1/models | python3 -c "
import json,sys; d=json.load(sys.stdin)
models=[m['id'] for m in d.get('data',[])]
print(f'Models: {models}')
assert '$1' in models, f'Wrong model!'
"

# Quick correctness test
curl -s http://localhost:8193/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"$1","prompt":"Hello","max_tokens":5}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('✓ OK' if 'choices' in d else f'✗ {d}')"

# Benchmark — output to file
vllm bench serve \
  --model $1 --port 8193 \
  --dataset-name random \
  --input-len 1024 --output-len 1024 \
  --num-prompts 100 --max-concurrency 16 \
  --request-rate inf --save-result \
  --result-dir <output_dir>/report --result-filename optimized_serving.json --label optimized \
  &> <output_dir>/report/bench_optimized.log

kill $PATCHED_PID 2>/dev/null; wait $PATCHED_PID 2>/dev/null

# Show only key metrics
python3 -c "
import json
with open('<output_dir>/report/optimized_serving.json') as f: d=json.load(f)
print('=== Optimized ===')
for k in ['output_throughput','mean_tpot_ms','mean_ttft_ms','completed']:
    print(f'  {k}: {d.get(k,\"N/A\")}')
"
```

**If the patched server fails to start or crashes:**
1. Check `run_patched_vllm.py` output for registration errors
2. Try removing problematic kernels from `vllm_plugin/` and regenerate
3. If ALL patches fail, run benchmark anyway (it measures "no-change" as the honest result)

## Step 5: ⛔ MANDATORY — Validate Results

```bash
source <output_dir>/venv/bin/activate
python3 << 'VALIDATE'
import json, sys, os

report_dir = "<output_dir>/report"
baseline_path = os.path.join(report_dir, "baseline_serving.json")
optimized_path = os.path.join(report_dir, "optimized_serving.json")
errors = []

for path, name, expected_label in [
    (baseline_path, "baseline", "baseline"),
    (optimized_path, "optimized", "optimized"),
]:
    if not os.path.exists(path):
        errors.append(f"MISSING: {name}_serving.json — you must run vllm bench serve")
        continue
    with open(path) as f:
        data = json.load(f)
    label = data.get("label", "")
    if label != expected_label:
        errors.append(f"{name}_serving.json label='{label}', expected '{expected_label}'")
    completed = data.get("completed", 0)
    if completed == 0 and name == "optimized":
        errors.append(f"optimized_serving.json has completed=0 — patched server did not work")
    if name == "optimized" and os.path.exists(baseline_path):
        with open(baseline_path) as f:
            bl = json.load(f)
        if data.get("date") == bl.get("date"):
            errors.append("SUSPICIOUS: same date on baseline and optimized — were these separate runs?")

if errors:
    print("⛔ VALIDATION FAILED:")
    for e in errors:
        print(f"  - {e}")
    print("\nYou must fix the issues above. Phase 7 is NOT complete.")
    sys.exit(1)

with open(baseline_path) as f: baseline = json.load(f)
with open(optimized_path) as f: optimized = json.load(f)

b_otps = baseline.get("output_throughput", 0)
o_otps = optimized.get("output_throughput", 0)
speedup = o_otps / b_otps if b_otps > 0 else 1.0

print("✅ VALIDATION PASSED — Real measurements confirmed")
print(f"  Baseline OTPS:  {b_otps:.2f} tok/s (completed={baseline.get('completed',0)})")
print(f"  Optimized OTPS: {o_otps:.2f} tok/s (completed={optimized.get('completed',0)})")
print(f"  Speedup:        {speedup:.3f}x")

os.makedirs(os.path.join(report_dir, "comparison_outputs"), exist_ok=True)
with open(os.path.join(report_dir, "comparison_outputs", "comparison_results.json"), "w") as f:
    json.dump({
        "validated": True,
        "baseline_otps": b_otps, "optimized_otps": o_otps, "speedup_otps": speedup,
        "baseline_tpot_ms": baseline.get("mean_tpot_ms", 0),
        "optimized_tpot_ms": optimized.get("mean_tpot_ms", 0),
        "baseline_ttft_ms": baseline.get("mean_ttft_ms", 0),
        "optimized_ttft_ms": optimized.get("mean_ttft_ms", 0),
        "concurrency": 16, "input_len": 1024, "output_len": 1024,
    }, f, indent=2)
VALIDATE
```

**If validation fails, fix the issue and re-run from the failing step.**

Update progress.json: phases_completed.append("integrate")


---

# Phase 8: Generate Final Report 

## Goal
Create a comprehensive optimization report.

## Create Report: `<output_dir>/report/optimization_report.md`

**⚠️ CRITICAL**: Include **ACTUAL MEASURED** end-to-end speedup from comparison_results.json.

```markdown
# Model Optimization Report

## Model Information
- **Model**: $1
- **Optimization Date**: [DATE]

## Summary
- **ACTUAL End-to-End Speedup**: X.Xx (measured, NOT estimated)
- **Kernels Optimized**: N
- **Baseline Inference Time**: X.Xs
- **Optimized Inference Time**: X.Xs

## Bottleneck Analysis
| Operator | Original Time (ms) | % of Total | Optimized | Speedup |
|----------|-------------------|------------|-----------|---------|
| ...      | ...               | ...        | ...       | ...     |

## Performance Results (ACTUAL MEASURED)
| Metric | Original | Optimized | Speedup |
|--------|----------|-----------|---------|
| End-to-End Inference Time | X.Xs | X.Xs | **X.Xx** |

## Comparison Outputs (Seed=42)
Outputs generated with fixed random seed for verification.

### Text Models:
| Original | Optimized |
|----------|-----------|
| [text]   | [text]    |

### Image Models (if applicable):
> Include this section only for vision/multimodal models that produce image outputs.

| Original | Optimized |
|:--------:|:---------:|
| ![Original](comparison_outputs/original_output.png) | ![Optimized](comparison_outputs/optimized_output.png) |

## Files Generated
- problems/ - Problem files + optimized kernels
- optimized/vllm_plugin/ - vLLM CustomOp integration plugin
- report/baseline_serving.json - Baseline benchmark results
- report/optimized_serving.json - Optimized benchmark results
- report/optimization_report.md - This report

## Recommendations
1. ...
```

## Steps
1. Gather all results from previous phases
2. Generate the comprehensive report
3. Update progress.json: phase="complete", phases_completed.append("report")



---

# EXECUTION INSTRUCTIONS
Execute phases: 5 → 6 → 7 → 8.
**ALL vLLM output to log files. Run validate_pipeline.py after Phase 6 and 7.**
Begin with Phase 5.
