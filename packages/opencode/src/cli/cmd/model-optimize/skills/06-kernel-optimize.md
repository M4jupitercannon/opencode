# Phase 6: Kernel Optimization via GEAK {{SKIP_LABEL}}

## Goal
Optimize each problem file using GEAK (GPU Evolutionary Agent for Kernels) to produce
Triton kernels that beat the PyTorch baseline.

## Prerequisites
- GEAK must be installed in the Docker container (`geak --help` works)
- `AMD_LLM_API_KEY` must be set in `/root/.config/mini-swe-agent/.env`
- Problem files from Phase 5 in `{{PROBLEMS_DIR}}/`
- `optimization_manifest.json` in `{{PROBLEMS_DIR}}/`

## Docker vs venv
If Phase 0 created a Docker container (`env_type: "docker"` in `env_info.json`), run GEAK inside it.

```bash
ENV_TYPE=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('env_type','venv'))" 2>/dev/null || echo "venv")
CONTAINER_NAME=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('container','vllm_model_opt'))" 2>/dev/null || echo "vllm_model_opt")
GPU_COUNT=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('gpu_count',1))" 2>/dev/null || echo 1)
```

## Step 1: Verify GEAK availability

```bash
docker exec $CONTAINER_NAME bash -c "geak --help >/dev/null 2>&1 && echo 'GEAK available' || echo 'GEAK NOT available -- install it first'"
```

If GEAK is not available, install it:
```bash
docker exec $CONTAINER_NAME bash -c "cd /workspace/GEAK && pip install -e . 2>/dev/null"
```

## Step 2: Read optimization manifest and detect GPU architecture

```bash
GPU_ARCH=$(docker exec $CONTAINER_NAME bash -c "rocminfo 2>/dev/null | grep -oP 'gfx\w+' | head -1 || echo 'unknown'")
GPU_NAME=$(docker exec $CONTAINER_NAME bash -c "rocm-smi --showproductname 2>/dev/null | grep -oP 'MI\w+' | head -1 || echo 'AMD GPU'")
```

## Step 3: Run GEAK on each enabled problem file

For each optimization in `optimization_manifest.json` where `enabled: true`:

```bash
# Read manifest and launch GEAK for each enabled problem
python3 -c "
import json
manifest = json.load(open('{{PROBLEMS_DIR}}/optimization_manifest.json'))
gpu_idx = 0
for opt in manifest['optimizations']:
    if not opt.get('enabled', False) or not opt.get('file'):
        continue
    print(f'GEAK: {opt[\"name\"]} -> {opt[\"file\"]} (GPU {gpu_idx})')
    gpu_idx = (gpu_idx + 1) % $GPU_COUNT
"

# Launch GEAK for each problem (parallel across GPUs)
# For each enabled problem file:
docker exec -d -e HIP_VISIBLE_DEVICES=$GPU_ID $CONTAINER_NAME bash -c "
  cd {{PROBLEMS_DIR}}
  geak -m claude-4.6-opus \
    -t 'Optimize $PROBLEM_FILE: write a Triton kernel for $DESCRIPTION on AMD $GPU_NAME ($GPU_ARCH). Shape: $SHAPE. The Model class is the baseline. Create a ModelNew class with @triton.jit kernel that beats it.' \
    --gpu-ids 0 --yolo &> {{PROBLEMS_DIR}}/geak_${NAME}.log
"
```

Run one GEAK agent per problem file, each on a different GPU. Monitor progress:

```bash
# Check status
docker exec $CONTAINER_NAME bash -c "
  for f in {{PROBLEMS_DIR}}/geak_*.log; do
    name=\$(basename \$f .log)
    lines=\$(wc -l < \$f 2>/dev/null || echo 0)
    done=\$(grep -c 'Selected best patch' \$f 2>/dev/null || echo 0)
    echo \"\$name: \$lines lines, completed=\$done\"
  done
"
```

## Step 4: Verify results

After GEAK completes, each problem file should contain a `ModelNew` class with an optimized Triton kernel.

```bash
docker exec $CONTAINER_NAME bash -c "
  cd {{PROBLEMS_DIR}}
  for f in problem_*.py; do
    has_new=\$(grep -c 'class ModelNew' \$f 2>/dev/null || echo 0)
    has_triton=\$(grep -c '@triton.jit' \$f 2>/dev/null || echo 0)
    echo \"\$f: ModelNew=\$has_new triton=\$has_triton\"
  done
"
```

For each problem file with `ModelNew`, run correctness + benchmark:

```bash
docker exec -e HIP_VISIBLE_DEVICES=0 $CONTAINER_NAME bash -c "
  cd {{PROBLEMS_DIR}}
  python3 -c '
import importlib, torch, time, sys
sys.path.insert(0, \".\")
results = []
import glob
for f in sorted(glob.glob(\"problem_*.py\")):
    mod_name = f.replace(\".py\", \"\")
    try:
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, \"ModelNew\"): continue
        m_base = mod.Model(*mod.get_init_inputs()).cuda().eval()
        m_new = mod.ModelNew(*mod.get_init_inputs()).cuda().eval()
        inputs = mod.get_inputs()
        with torch.no_grad():
            out_b = m_base(*inputs)
            out_n = m_new(*inputs)
        correct = torch.allclose(out_b, out_n, atol=1e-2, rtol=1e-2)
        # Benchmark
        for _ in range(20):
            with torch.no_grad(): m_base(*inputs); m_new(*inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(200):
            with torch.no_grad(): m_base(*inputs)
        torch.cuda.synchronize()
        base_t = (time.perf_counter() - t0) / 200
        t0 = time.perf_counter()
        for _ in range(200):
            with torch.no_grad(): m_new(*inputs)
        torch.cuda.synchronize()
        new_t = (time.perf_counter() - t0) / 200
        speedup = base_t / new_t if new_t > 0 else 0
        status = \"PASS\" if correct else \"FAIL\"
        print(f\"  {mod_name}: {status} baseline={base_t*1e6:.1f}us opt={new_t*1e6:.1f}us speedup={speedup:.2f}x\")
        results.append({\"name\": mod_name, \"correct\": correct, \"speedup\": speedup, \"baseline_us\": base_t*1e6, \"optimized_us\": new_t*1e6})
    except Exception as e:
        print(f\"  {mod_name}: ERROR {e}\")

import json
with open(\"geak_results.json\", \"w\") as f:
    json.dump(results, f, indent=2)
print(f\"Saved geak_results.json ({len(results)} kernels)\")
'
"
```

## Step 5: Copy winning kernels to optimized/

```bash
mkdir -p {{OPTIMIZED_DIR}}
docker exec $CONTAINER_NAME bash -c "
  cd {{PROBLEMS_DIR}}
  python3 -c '
import json, shutil, os
results = json.load(open(\"geak_results.json\"))
os.makedirs(\"{{OPTIMIZED_DIR}}\", exist_ok=True)
for r in results:
    if r[\"correct\"] and r[\"speedup\"] > 1.0:
        src = r[\"name\"] + \".py\"
        shutil.copy2(src, \"{{OPTIMIZED_DIR}}/\")
        print(f\"Copied {src} ({r[\"speedup\"]:.2f}x)\")
    elif not r[\"correct\"]:
        print(f\"Skipped {r[\"name\"]} (correctness FAILED)\")
    else:
        print(f\"Skipped {r[\"name\"]} (speedup {r[\"speedup\"]:.2f}x < 1.0x)\")
'
"
```

Update progress.json: phases_completed.append("optimize")

## When GEAK is not available (fallback)

If GEAK cannot be installed or the API key is not set, fall back to manual optimization:
for each problem file, write a `ModelNew` class with `@triton.jit` kernels + `@triton.autotune`,
test with `kernel_test_runner.py`, iterate, and finalize with `kernel_finalize.py`.
See the previous version of this skill for the manual workflow details.
