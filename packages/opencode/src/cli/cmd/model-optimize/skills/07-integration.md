# Phase 7: Integration & End-to-End Testing {{SKIP_LABEL}}

## Goal

Apply optimized kernels to vLLM via CustomOp and measure ACTUAL serving throughput.

## ⛔ MANDATORY: This phase REQUIRES real measured data

**This phase is NOT complete until:**

1. `baseline_serving.json` exists (reused from Phase 4 `baseline_benchmark.json`)
2. A patched vLLM server has ACTUALLY been started and served requests
3. `vllm bench serve` has been run in **both compiled and eager modes** for the optimized server
4. `optimized_serving.json` (compiled) and `optimized_eager_serving.json` (eager) exist with correct labels
5. The validation script passes with `comparison_results.json` containing both modes

**FORBIDDEN:**

- Estimating speedup with Amdahl's law
- Copying baseline numbers and modifying them
- Reporting "estimated" or "conservative" speedup
- Skipping the patched server benchmark

---

## ⚠️ Docker vs venv

If Phase 0 created a Docker container (`env_type: "docker"` in `env_info.json`), prefix all commands with `docker exec $CONTAINER_NAME bash -c "..."` and use `HIP_VISIBLE_DEVICES=$BEST_GPU`.

Detect once before running this phase:

```bash
ENV_TYPE=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('env_type','venv'))" 2>/dev/null || echo "venv")
CONTAINER_NAME=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('container','vllm_model_opt'))" 2>/dev/null || echo "vllm_model_opt")
BEST_GPU=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('best_gpu',0))" 2>/dev/null || echo 0)
```

---

## Integration Mechanisms

### 1. vLLM CustomOp.register_oot() (primary)

For kernels that map to vLLM ops (RMSNorm, SiluAndMul, attention, etc.):

- Docs: https://docs.vllm.ai/en/latest/design/custom_op/
- Each optimized kernel is wrapped as a vLLM CustomOp subclass
- `CustomOp.register_oot()` replaces the default op at instantiation time
- If the optimized kernel fails, vLLM falls back to the default

### 2. torch.mm override (for GEMM kernels)

For GEMM kernels (aten::mm) that don't map to a vLLM CustomOp, the plugin patches
`torch.mm` to dispatch to the optimized Triton GEMM for decode-sized inputs:

```python
_original_mm = torch.mm

def _patched_mm(a, b, **kwargs):
    if (a.ndim == 2 and b.ndim == 2
        and a.dtype == torch.bfloat16
        and a.shape[0] <= 16
        and a.shape[1] in MODEL_K_DIMS and b.shape[1] in MODEL_N_DIMS):
        return optimized_gemm(a, b)
    return _original_mm(a, b, **kwargs)

torch.mm = _patched_mm
```

The `generate_vllm_plugin.py` script automatically adds this override for GEMM `_opt.py` files
that don't have a CustomOp mapping. The shape conditions are derived from the model config
(hidden_size, intermediate_size, num_kv_heads * head_dim, vocab_size).

---

## Step 1: Generate vLLM Plugin

The `generate_vllm_plugin.py` script auto-creates a plugin from `*_opt.py` files:

```bash
# venv mode only:
# source {{OUTPUT_DIR}}/venv/bin/activate
cd {{OPTIMIZED_DIR}}

# Copy all *_opt.py from problems
cp {{PROBLEMS_DIR}}/*_opt.py . 2>/dev/null

# Generate the plugin
python3 {{OUTPUT_DIR}}/scripts/generate_vllm_plugin.py \
  --kernel-dir {{OPTIMIZED_DIR}}

# Verify generated files
ls -la vllm_plugin/
cat vllm_plugin/manifest.json
```

This generates:

- `{{OPTIMIZED_DIR}}/vllm_plugin/__init__.py` — registers CustomOps
- `{{OPTIMIZED_DIR}}/run_patched_vllm.py` — launcher script
- `{{OPTIMIZED_DIR}}/vllm_plugin/manifest.json` — registration summary

## Step 2: Test Plugin Registration (dry run)

Verify that the plugin loads without errors:

```bash
# venv mode only:
# source {{OUTPUT_DIR}}/venv/bin/activate
python3 -c "
import sys; sys.path.insert(0, '{{OPTIMIZED_DIR}}')
import vllm_plugin
print('Plugin loaded successfully')
"
```

## ⚠️ Before EVERY benchmark run: Clean GPU state

Before starting each vLLM server (Steps 3a, 3b, 4a, 4b), **always** kill all leftover processes and verify GPUs are clean. Stale processes cause memory contention, OOM errors, and unreliable benchmark numbers.

```bash
# Kill ALL vllm/geak/python GPU processes
pkill -9 -f 'vllm serve' 2>/dev/null
pkill -9 -f 'run_patched_vllm' 2>/dev/null
pkill -9 -f 'geak' 2>/dev/null
sleep 3

# Verify target GPU is free (should show 0% utilization and ~0 MB VRAM used)
rocm-smi --showuse --showmemuse 2>/dev/null | grep -A2 "GPU\[$BEST_GPU\]"
# If GPU still shows memory in use, wait or pick a different GPU
```

Run this cleanup **before every server start** — not just once at the beginning.

## Step 3: Reuse Phase 4 Baseline

Phase 4 already collected a baseline throughput benchmark (`baseline_benchmark.json`) with the same model, parameters, and container. **Do NOT re-run the baseline** — copy it to the report directory instead.

```bash
cp {{PROFILE_DIR}}/baseline_benchmark.json {{REPORT_DIR}}/baseline_serving.json
echo "Baseline reused from Phase 4"

python3 -c "
import json
d = json.load(open('{{REPORT_DIR}}/baseline_serving.json'))
print('=== Baseline (from Phase 4) ===')
for k in ['output_throughput','mean_tpot_ms','mean_ttft_ms','completed']:
    print(f'  {k}: {d.get(k,\"N/A\")}')
"
```

> **Why reuse?** The baseline environment (same container, same GPU, same model weights) hasn't changed.
> Re-running wastes ~10 minutes of server startup + benchmark time.

## Step 4: ⛔ MANDATORY — Benchmark Optimized vLLM (both modes)

Benchmark the **optimized** vLLM server in both modes. For Triton CustomOp-based optimizations, use `run_patched_vllm.py`. For in-place HIP kernel optimizations (installed via `AITER_REBUILD`), use the standard `vllm serve` command — the optimized kernel is already loaded.

### 4a: Optimized — compiled mode (production)

```bash
# If using CustomOp plugin:
python3 {{OPTIMIZED_DIR}}/run_patched_vllm.py serve \
  --model {{HF_MODEL}} --dtype auto --max-model-len 4096 \
  --port 8193 --no-enable-log-requests --gpu-memory-utilization 0.85 &> {{OUTPUT_DIR}}/vllm_optimized_compiled.log &
# If using in-place HIP optimization (AITER_REBUILD), use standard vllm serve instead:
# vllm serve {{HF_MODEL}} --dtype auto --max-model-len 4096 --port 8193 --no-enable-log-requests --gpu-memory-utilization 0.85 &> {{OUTPUT_DIR}}/vllm_optimized_compiled.log &
PATCHED_PID=$!

for i in $(seq 1 60); do curl -s http://localhost:8193/health > /dev/null 2>&1 && break; sleep 5; done
curl -s http://localhost:8193/health > /dev/null 2>&1 && echo "Optimized server ready" || { echo "FAILED"; tail -5 {{OUTPUT_DIR}}/vllm_optimized_compiled.log; }

# Quick correctness test
curl -s http://localhost:8193/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"{{HF_MODEL}}","prompt":"Hello","max_tokens":5}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('OK' if 'choices' in d else f'Error: {d}')"

vllm bench serve \
  --model {{HF_MODEL}} --port 8193 \
  --dataset-name random \
  --input-len {{INPUT_LEN}} --output-len {{OUTPUT_LEN}} \
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{CONCURRENCY}} \
  --request-rate inf --save-result \
  --result-dir {{REPORT_DIR}} --result-filename optimized_serving.json --label optimized \
  &> {{REPORT_DIR}}/bench_optimized.log

kill $PATCHED_PID 2>/dev/null; wait $PATCHED_PID 2>/dev/null
```

### 4b: Optimized — eager mode (diagnostic)

```bash
# Same as 4a but with --enforce-eager
# If using CustomOp plugin:
python3 {{OPTIMIZED_DIR}}/run_patched_vllm.py serve \
  --model {{HF_MODEL}} --dtype auto --max-model-len 4096 \
  --port 8193 --enforce-eager --no-enable-log-requests --gpu-memory-utilization 0.85 &> {{OUTPUT_DIR}}/vllm_optimized_eager.log &
# If using in-place HIP optimization:
# vllm serve {{HF_MODEL}} --dtype auto --max-model-len 4096 --port 8193 --enforce-eager --no-enable-log-requests --gpu-memory-utilization 0.85 &> {{OUTPUT_DIR}}/vllm_optimized_eager.log &
PATCHED_PID=$!

for i in $(seq 1 60); do curl -s http://localhost:8193/health > /dev/null 2>&1 && break; sleep 5; done
curl -s http://localhost:8193/health > /dev/null 2>&1 && echo "Ready" || { echo "FAILED"; tail -5 {{OUTPUT_DIR}}/vllm_optimized_eager.log; }

vllm bench serve \
  --model {{HF_MODEL}} --port 8193 \
  --dataset-name random \
  --input-len {{INPUT_LEN}} --output-len {{OUTPUT_LEN}} \
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{CONCURRENCY}} \
  --request-rate inf --save-result \
  --result-dir {{REPORT_DIR}} --result-filename optimized_eager_serving.json --label optimized_eager \
  &> {{REPORT_DIR}}/bench_optimized_eager.log

kill $PATCHED_PID 2>/dev/null; wait $PATCHED_PID 2>/dev/null
```

Print all four results:

```bash
python3 -c "
import json
results = {}
for fname, label in [
    ('baseline_serving.json', 'Baseline (compiled)'),
    ('baseline_eager_serving.json', 'Baseline (eager)'),
    ('optimized_serving.json', 'Optimized (compiled)'),
    ('optimized_eager_serving.json', 'Optimized (eager)')
]:
    try:
        d = json.load(open('{{REPORT_DIR}}/' + fname))
        results[label] = d
        print(f'=== {label} ===')
        for k in ['output_throughput','mean_tpot_ms','mean_ttft_ms','completed']:
            print(f'  {k}: {d.get(k,\"N/A\")}')
    except FileNotFoundError:
        print(f'=== {label} === MISSING')

# Comparison summary
if 'Baseline (compiled)' in results and 'Optimized (compiled)' in results:
    b = results['Baseline (compiled)']['output_throughput']
    o = results['Optimized (compiled)']['output_throughput']
    print(f'\n  Compiled speedup: {o/b:.3f}x ({b:.0f} -> {o:.0f} tok/s)')
if 'Baseline (eager)' in results and 'Optimized (eager)' in results:
    b = results['Baseline (eager)']['output_throughput']
    o = results['Optimized (eager)']['output_throughput']
    print(f'  Eager speedup:    {o/b:.3f}x ({b:.0f} -> {o:.0f} tok/s)')
"
```

**If the patched server fails to start or crashes:**

1. Check `run_patched_vllm.py` output for registration errors
2. Try removing problematic kernels from `vllm_plugin/` and regenerate
3. If ALL patches fail, run benchmark anyway (it measures "no-change" as the honest result)
4. If eager mode works but compiled mode fails, the optimization may be incompatible with torch.compile

## Step 5: ⛔ MANDATORY — Validate Results

Validates both compiled and eager results. The compiled-mode comparison is the **primary** result (production performance). Eager-mode is **diagnostic** (isolates kernel-level impact).

```bash
python3 << 'VALIDATE'
import json, sys, os

report_dir = "{{REPORT_DIR}}"
errors = []

required_pairs = [
    ("baseline_serving.json", "baseline", "optimized_serving.json", "optimized", "compiled"),
]
optional_pairs = [
    ("baseline_eager_serving.json", "baseline_eager", "optimized_eager_serving.json", "optimized_eager", "eager"),
]

results = {}

for bl_file, bl_label, opt_file, opt_label, mode in required_pairs + optional_pairs:
    bl_path = os.path.join(report_dir, bl_file)
    opt_path = os.path.join(report_dir, opt_file)
    is_required = (bl_file, bl_label, opt_file, opt_label, mode) in required_pairs

    for path, name, expected_label in [(bl_path, bl_label, bl_label), (opt_path, opt_label, opt_label)]:
        if not os.path.exists(path):
            if is_required:
                errors.append(f"MISSING: {name} — you must run vllm bench serve ({mode} mode)")
            continue
        with open(path) as f:
            data = json.load(f)
        if is_required and data.get("completed", 0) == 0 and "optimized" in name:
            errors.append(f"{name} has completed=0 — patched server did not work ({mode})")

    if os.path.exists(bl_path) and os.path.exists(opt_path):
        bl_data = json.load(open(bl_path))
        opt_data = json.load(open(opt_path))
        b_otps = bl_data.get("output_throughput", 0)
        o_otps = opt_data.get("output_throughput", 0)
        speedup = o_otps / b_otps if b_otps > 0 else 1.0
        results[mode] = {
            "baseline_otps": b_otps, "optimized_otps": o_otps, "speedup": speedup,
            "baseline_tpot_ms": bl_data.get("mean_tpot_ms", 0),
            "optimized_tpot_ms": opt_data.get("mean_tpot_ms", 0),
            "baseline_ttft_ms": bl_data.get("mean_ttft_ms", 0),
            "optimized_ttft_ms": opt_data.get("mean_ttft_ms", 0),
        }

if errors:
    print("VALIDATION FAILED:")
    for e in errors:
        print(f"  - {e}")
    print("\nYou must fix the issues above. Phase 7 is NOT complete.")
    sys.exit(1)

print("VALIDATION PASSED — Real measurements confirmed")
print()
for mode, r in results.items():
    tag = "PRIMARY" if mode == "compiled" else "DIAGNOSTIC"
    print(f"  [{tag}] {mode.upper()} mode:")
    print(f"    Baseline:  {r['baseline_otps']:.1f} tok/s  TPOT={r['baseline_tpot_ms']:.2f}ms")
    print(f"    Optimized: {r['optimized_otps']:.1f} tok/s  TPOT={r['optimized_tpot_ms']:.2f}ms")
    print(f"    Speedup:   {r['speedup']:.3f}x")
    print()

if "compiled" in results and "eager" in results:
    print("  INSIGHT: If eager speedup >> compiled speedup, the kernel optimization")
    print("  is real but torch.compile/CUDAGraphs masks it in production mode.")
    print("  Consider disabling torch.compile for the optimized kernel path.")

comparison = {"validated": True, "concurrency": {{CONCURRENCY}}, "input_len": {{INPUT_LEN}}, "output_len": {{OUTPUT_LEN}}}
for mode, r in results.items():
    prefix = mode + "_"
    for k, v in r.items():
        comparison[prefix + k] = v

os.makedirs(os.path.join(report_dir, "comparison_outputs"), exist_ok=True)
with open(os.path.join(report_dir, "comparison_outputs", "comparison_results.json"), "w") as f:
    json.dump(comparison, f, indent=2)
print(f"Saved comparison_results.json with {len(results)} mode(s)")
VALIDATE
```

**If validation fails, fix the issue and re-run from the failing step.**

Update progress.json: phases_completed.append("integrate")
