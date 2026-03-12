# Phase 6: Kernel Optimization via GEAK {{SKIP_LABEL}}

## Goal
Optimize each bottleneck kernel using the appropriate GEAK mode based on `kernel_type`:
- **C++ kernels** (`hip`, `ck`, `asm`, `triton_composite`): use `geak --kernel-url` to optimize the source in-place
- **Python/Triton kernels** (`triton`, `aten_gemm`, `aten_elementwise`): use `geak -t` (simple mode) to write optimized Triton replacements

## Prerequisites
- GEAK must be installed in the Docker container (`geak --help` works)
- `AMD_LLM_API_KEY` must be set in `/root/.config/mini-swe-agent/.env`
- For `hip`/`ck`/`asm` kernels: `geak-oe` must be installed (`/opt/geak-oe`) -- see Phase 0
- Problem files from Phase 5 in `{{PROBLEMS_DIR}}/`
- `optimization_manifest.json` in `{{PROBLEMS_DIR}}/` with `kernel_type` and `source_file` metadata

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

## Step 3a: Optimize C++ kernels via `geak --kernel-url`

For each manifest entry where `kernel_type` is `hip`, `ck`, `asm`, or `triton_composite` AND `source_file` is available:

1. **Prepare workspace**: copy source file and dependencies, init git repo, set up build system
2. **Launch**: `geak --kernel-url <source_file>#L<line> --workspace ... --repo ... --gpu-ids 0,1 --yolo`
3. **Monitor**: check `geak_output/results/round_*/*/task_*.log` for patches and speedups

For the full procedure, see `hip-kernel-optimize-geak.md`.

```bash
docker exec -d -e HIP_VISIBLE_DEVICES=$GPU_IDS -e GEAK_OE_ROOT=/opt/geak-oe $CONTAINER_NAME bash -c "
  cd /workspace/${NAME}_opt
  geak -m claude-opus-4.6 \
    --kernel-url /workspace/${NAME}_opt/csrc/kernels/${SOURCE_FILE}#L${LINE} \
    --workspace /workspace/${NAME}_opt \
    --repo /workspace/${NAME}_opt \
    --gpu-ids 0,1 \
    -o /workspace/${NAME}_opt/geak_output \
    --yolo &> /workspace/${NAME}_opt/geak.log
"
```

If a `geak --kernel-url` optimization finds speedup > 1.0x, install the winning source via the appropriate rebuild mechanism (e.g., `AITER_REBUILD=1` for aiter kernels, `pip install -e .` for others).

## Step 3b: Optimize Triton/ATen kernels via `geak -t` (simple mode)

For each manifest entry where `kernel_type` is `triton`, `aten_gemm`, or `aten_elementwise`:

Build kernel-type-aware task descriptions and launch one GEAK agent per problem, parallel across GPUs. Run in **priority order** (HIGH first).

```python
import json, os

gpu_arch = os.environ.get('GPU_ARCH', 'gfx942')
manifest = json.load(open('{{PROBLEMS_DIR}}/optimization_manifest.json'))

priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
simple_types = {'triton', 'aten_gemm', 'aten_elementwise', 'unknown'}
enabled = [o for o in manifest['optimizations']
           if o.get('enabled') and o.get('file') and o.get('kernel_type', 'unknown') in simple_types]
enabled.sort(key=lambda o: priority_order.get(o.get('priority', 'LOW'), 3))

gpu_idx = 0
for opt in enabled:
    kt = opt.get('kernel_type', 'unknown')
    src = opt.get('source_file', '')
    name = opt['name']
    f = opt['file']

    if kt == 'triton':
        task = f"Optimize {f}: the baseline is a Triton kernel (source: {src}). Write a faster ModelNew with @triton.jit + @triton.autotune for AMD {gpu_arch}. The Model class is the baseline."
    elif kt == 'aten_gemm':
        task = f"Optimize {f}: the baseline uses torch.mm dispatched to rocBLAS. Write a Triton GEMM ModelNew with @triton.jit that beats rocBLAS on AMD {gpu_arch}. The Model class is the baseline."
    elif kt == 'aten_elementwise':
        task = f"Optimize {f}: the baseline is a PyTorch ATen op. Write a Triton kernel ModelNew with @triton.jit on AMD {gpu_arch}. The Model class is the baseline."
    else:
        task = f"Optimize {f}: write a Triton ModelNew with @triton.jit that beats the baseline Model class on AMD {gpu_arch}."

    print(f"GEAK [{opt.get('priority','?')}]: {name} -> {f} (GPU {gpu_idx}) | kernel_type={kt}")
    gpu_idx = (gpu_idx + 1) % int(os.environ.get('GPU_COUNT', '1'))
```

```bash
docker exec -d -e HIP_VISIBLE_DEVICES=$GPU_ID $CONTAINER_NAME bash -c "
  cd {{PROBLEMS_DIR}}
  geak -m claude-opus-4.6 \
    -t '$TASK' \
    --gpu-ids 0 --yolo &> {{PROBLEMS_DIR}}/geak_${NAME}.log
"
```

## Step 3.5: Collect GEAK patches

GEAK stores optimized kernels in `optimization_logs/<name>_<timestamp>/patch_N.patch` (simple mode) or `geak_output/results/round_N/worktrees/slot_N/` (kernel-url mode). These are NOT applied to the original problem files automatically.

**For simple-mode patches**: extract the `kernel.py` from the best patch (lowest `GEAK_RESULT_LATENCY_MS` in `patch_N_test.txt`) and copy to `{{OPTIMIZED_DIR}}/`.

**For kernel-url patches**: the winning source is already in the worktree. Copy it and install via the appropriate rebuild mechanism.

```bash
# Simple mode: find best patch per kernel
for dir in {{PROBLEMS_DIR}}/optimization_logs/*/; do
  best=$(for t in $dir/patch_*_test.txt; do
    [ -f "$t" ] && lat=$(grep -oP "GEAK_RESULT_LATENCY_MS=([0-9.]+)" "$t" | tail -1 | cut -d= -f2) && echo "$lat $t"
  done | sort -n | head -1 | awk '{print $2}')
  [ -n "$best" ] && echo "Best: $best"
done

# Kernel-url mode: find best patch
for p in /workspace/*_opt/geak_output/results/round_*/*/patch_*_test.txt; do
  lat=$(grep -oP "GEAK_RESULT_LATENCY_MS=([0-9.]+)" "$p" | tail -1 | cut -d= -f2)
  echo "$lat $p"
done | sort -n | head -1
```

## Step 4: Verify results

After GEAK completes, verify optimized kernels pass correctness and benchmark faster than baseline.

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
for each problem file (regardless of `kernel_type`), write a `ModelNew` class with
`@triton.jit` kernels + `@triton.autotune`, test with `kernel_test_runner.py`, iterate,
and finalize with `kernel_finalize.py`.
See the previous version of this skill for the manual workflow details.
