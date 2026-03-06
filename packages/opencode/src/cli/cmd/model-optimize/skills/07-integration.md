# Phase 7: Integration & End-to-End Testing {{SKIP_LABEL}}

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

## ⚠️ Docker vs venv

If Phase 0 created a Docker container (`env_type: "docker"` in `env_info.json`), prefix all commands with `docker exec $CONTAINER_NAME bash -c "..."` and use `HIP_VISIBLE_DEVICES=$BEST_GPU`.

Detect once before running this phase:

```bash
ENV_TYPE=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('env_type','venv'))" 2>/dev/null || echo "venv")
CONTAINER_NAME=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('container','vllm_model_opt'))" 2>/dev/null || echo "vllm_model_opt")
BEST_GPU=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('best_gpu',0))" 2>/dev/null || echo 0)
```

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

## Step 3: ⛔ MANDATORY — Benchmark Baseline

Phase 4 produced `baseline_benchmark.json` in `profile/` (different workload). Run a fresh baseline here in `report/` with the same parameters as the optimized run for a fair comparison:

```bash
# venv mode only:
# source {{OUTPUT_DIR}}/venv/bin/activate

# ALL vLLM output to log files — NEVER to stdout
vllm serve {{HF_MODEL}} --dtype auto --max-model-len 4096 --port 8192 --no-enable-log-requests &> {{OUTPUT_DIR}}/vllm_baseline_e2e.log &
VLLM_PID=$!
echo "Baseline PID: $VLLM_PID"
for i in $(seq 1 60); do curl -s http://localhost:8192/health > /dev/null 2>&1 && break; sleep 5; done
curl -s http://localhost:8192/health > /dev/null 2>&1 && echo "Ready" || { echo "FAILED"; tail -3 {{OUTPUT_DIR}}/vllm_baseline_e2e.log; }

vllm bench serve \
  --model {{HF_MODEL}} --port 8192 \
  --dataset-name random \
  --input-len {{INPUT_LEN}} --output-len {{OUTPUT_LEN}} \
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{CONCURRENCY}} \
  --request-rate inf --save-result \
  --result-dir {{REPORT_DIR}} --result-filename baseline_serving.json --label baseline \
  &> {{REPORT_DIR}}/bench_baseline.log

kill $VLLM_PID 2>/dev/null; wait $VLLM_PID 2>/dev/null

# Show only key metrics
python3 -c "
import json
with open('{{REPORT_DIR}}/baseline_serving.json') as f: d=json.load(f)
print('=== Baseline ===')
for k in ['output_throughput','mean_tpot_ms','mean_ttft_ms','completed']:
    print(f'  {k}: {d.get(k,\"N/A\")}')
"
```

## Step 4: ⛔ MANDATORY — Start Patched vLLM and Benchmark

```bash
# venv mode only:
# source {{OUTPUT_DIR}}/venv/bin/activate

# Start patched vLLM — ALL output to log file
python3 {{OPTIMIZED_DIR}}/run_patched_vllm.py serve \
  --model {{HF_MODEL}} --dtype auto --max-model-len 4096 \
  --port 8193 --no-enable-log-requests &> {{OUTPUT_DIR}}/vllm_patched.log &
PATCHED_PID=$!
echo "Patched PID: $PATCHED_PID (log: {{OUTPUT_DIR}}/vllm_patched.log)"

# Wait silently
for i in $(seq 1 60); do curl -s http://localhost:8193/health > /dev/null 2>&1 && break; sleep 5; done
curl -s http://localhost:8193/health > /dev/null 2>&1 && echo "Patched server ready" || { echo "FAILED"; tail -5 {{OUTPUT_DIR}}/vllm_patched.log; }

# Verify correct model (compact output)
curl -s http://localhost:8193/v1/models | python3 -c "
import json,sys; d=json.load(sys.stdin)
models=[m['id'] for m in d.get('data',[])]
print(f'Models: {models}')
assert '{{HF_MODEL}}' in models, f'Wrong model!'
"

# Quick correctness test
curl -s http://localhost:8193/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"{{HF_MODEL}}","prompt":"Hello","max_tokens":5}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print('OK' if 'choices' in d else f'Error: {d}')"

# Benchmark — output to file
vllm bench serve \
  --model {{HF_MODEL}} --port 8193 \
  --dataset-name random \
  --input-len {{INPUT_LEN}} --output-len {{OUTPUT_LEN}} \
  --num-prompts {{NUM_PROMPTS}} --max-concurrency {{CONCURRENCY}} \
  --request-rate inf --save-result \
  --result-dir {{REPORT_DIR}} --result-filename optimized_serving.json --label optimized \
  &> {{REPORT_DIR}}/bench_optimized.log

kill $PATCHED_PID 2>/dev/null; wait $PATCHED_PID 2>/dev/null

# Show only key metrics
python3 -c "
import json
with open('{{REPORT_DIR}}/optimized_serving.json') as f: d=json.load(f)
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
# venv mode only:
# source {{OUTPUT_DIR}}/venv/bin/activate
python3 << 'VALIDATE'
import json, sys, os

report_dir = "{{REPORT_DIR}}"
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
    print("VALIDATION FAILED:")
    for e in errors:
        print(f"  - {e}")
    print("\nYou must fix the issues above. Phase 7 is NOT complete.")
    sys.exit(1)

with open(baseline_path) as f: baseline = json.load(f)
with open(optimized_path) as f: optimized = json.load(f)

b_otps = baseline.get("output_throughput", 0)
o_otps = optimized.get("output_throughput", 0)
speedup = o_otps / b_otps if b_otps > 0 else 1.0

print("VALIDATION PASSED — Real measurements confirmed")
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
        "concurrency": {{CONCURRENCY}}, "input_len": {{INPUT_LEN}}, "output_len": {{OUTPUT_LEN}},
    }, f, indent=2)
VALIDATE
```

**If validation fails, fix the issue and re-run from the failing step.**

Update progress.json: phases_completed.append("integrate")
