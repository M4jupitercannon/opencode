# Phase 6: Kernel Optimization via GEAK {{SKIP_LABEL}}

## Goal
Optimize each bottleneck kernel using the appropriate GEAK mode based on `kernel_type`:
- **C++ kernels** (`hip`, `ck`, `asm`, `triton_composite`): use `mini --config mini_kernel.yaml` to optimize the source in-place
- **Python/Triton kernels** (`triton`, `aten_gemm`, `aten_elementwise`): use `mini -t` (simple mode) to write optimized Triton replacements

## Prerequisites
- GEAK (mini CLI) must be installed in the Docker container (`mini --help` works) — installed from `main` branch in Phase 0
- An LLM API key must be configured in `/root/.config/mini-swe-agent/.env` (set during Phase 0 Step 6)
- Check `env_info.json` for `geak_available: true` — if `false`, skip to the manual fallback section
- For `hip`/`ck`/`asm` kernels: `geak-oe` must be installed (`/opt/geak-oe`) — see Phase 0
- Problem files from Phase 5 in `{{PROBLEMS_DIR}}/`
- `optimization_manifest.json` in `{{PROBLEMS_DIR}}/` with `kernel_type` and `source_file` metadata

## Docker vs venv
If Phase 0 created a Docker container (`env_type: "docker"` in `env_info.json`), run GEAK inside it.

```bash
ENV_TYPE=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('env_type','venv'))" 2>/dev/null || echo "venv")
CONTAINER_NAME=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('container','vllm_model_opt'))" 2>/dev/null || echo "vllm_model_opt")
GPU_COUNT=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('gpu_count',1))" 2>/dev/null || echo 1)
```

## Step 0: Verify GEAK availability and API key

⛔ **Check BEFORE doing anything else.** If either check fails, skip to the manual fallback section at the bottom.

```bash
# Check GEAK (mini CLI) is installed
docker exec $CONTAINER_NAME bash -c "python3 -c 'from minisweagent.run.mini import app; print(\"mini: OK\")'" 2>&1 | tail -1

# Check API key is configured
docker exec $CONTAINER_NAME bash -c "cat /root/.config/mini-swe-agent/.env 2>/dev/null | grep -qE 'AMD_LLM_API_KEY|ANTHROPIC_API_KEY|OPENAI_API_KEY' && echo 'API key: OK' || echo 'API key: MISSING'"

# Check env_info.json
python3 -c "import json; d=json.load(open('{{OUTPUT_DIR}}/env_info.json')); print(f'geak_available: {d.get(\"geak_available\", False)}')"
```

If `geak_available` is `false` or the API key is missing, **ask the user for an API key** (AMD_LLM_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY). If the user cannot provide one, skip to the manual fallback section.

## Step 2: Read optimization manifest and detect GPU architecture

```bash
GPU_ARCH=$(docker exec $CONTAINER_NAME bash -c "rocminfo 2>/dev/null | grep -oP 'gfx\w+' | head -1 || echo 'unknown'")
GPU_NAME=$(docker exec $CONTAINER_NAME bash -c "rocm-smi --showproductname 2>/dev/null | grep -oP 'MI\w+' | head -1 || echo 'AMD GPU'")
```

## Step 3a: Optimize C++ kernels via `mini --config mini_kernel.yaml`

For each manifest entry where `kernel_type` is `hip`, `ck`, `asm`, or `triton_composite` AND `source_file` is available:

1. **Prepare workspace**: copy source file and dependencies, init git repo, set up build system
2. **Launch**: `mini -m claude-opus-4.6 --config mini_kernel.yaml --repo <workspace> --gpu-ids 0,1 --yolo -t "Optimize <source_file>"`
3. **Monitor**: check `optimization_logs/<name>_<timestamp>/mini_agent.log` for progress and patches

For the full procedure, see `hip-kernel-optimize-geak.md`.

```bash
docker exec -d -e HIP_VISIBLE_DEVICES=$GPU_IDS -e GEAK_OE_ROOT=/opt/geak-oe $CONTAINER_NAME bash -c "
  cd /workspace/${NAME}_opt
  mini -m claude-opus-4.6 \
    --config mini_kernel.yaml \
    --repo /workspace/${NAME}_opt \
    --gpu-ids 0,1 \
    -o /workspace/${NAME}_opt/traj.json \
    --yolo \
    -t 'Optimize /workspace/${NAME}_opt/csrc/kernels/${SOURCE_FILE}: improve kernel performance on AMD ${GPU_ARCH}. Test: python3 test_harness.py --benchmark' \
    &> /workspace/${NAME}_opt/mini.log
"
```

If a kernel-url optimization finds speedup > 1.0x, install the winning source via the appropriate rebuild mechanism (e.g., `AITER_REBUILD=1` for aiter kernels, `pip install -e .` for others).

## Step 3b: Optimize Triton/ATen kernels via `mini -t` (simple mode)

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

⚠️ **CRITICAL**: The `-o` output path must be short (e.g. `traj_${NAME}.json`). Long task descriptions used as filenames cause `OSError: File name too long`.

⚠️ **CRITICAL**: Initialize git in the working directory before launching mini, otherwise patch generation fails:
```bash
docker exec $CONTAINER_NAME bash -c "cd {{PROBLEMS_DIR}} && git init && git add -A && git commit -m init" 2>/dev/null
```

```bash
docker exec -d -e HIP_VISIBLE_DEVICES=$GPU_ID $CONTAINER_NAME bash -c "
  cd {{PROBLEMS_DIR}}
  mini -m claude-opus-4.6 \
    --config geak.yaml \
    --gpu-ids 0 --yolo \
    -o {{PROBLEMS_DIR}}/traj_${NAME}.json \
    -t '$TASK' \
    &> {{PROBLEMS_DIR}}/log_${NAME}.txt
"
```

## Step 3.5: Collect GEAK patches and recover best kernels

GEAK stores results in `optimization_logs/<name>_<timestamp>/`:
- `patch_N.patch` — git diff patches
- `patch_N_test.txt` — test results with `RESULT_JSON: {...}` or `GEAK_RESULT_LATENCY_MS=...`
- `mini_agent.log` — full agent log

⚠️ **CRITICAL**: GEAK's `[SelectPatch]` may fail to apply the best patch (e.g. git working directory conflicts). When this happens, the `_best.json` tracker will have a **lower speedup** than the actual best GEAK patch. You MUST recover the best kernel from the patch diff.

### Recovery procedure

For each kernel in `optimization_logs/`:

1. **Check if `[SelectPatch]` succeeded** — look for `"Best patch applied successfully"` in the log. If yes, the `_opt.py` file already has the best code.

2. **If `[SelectPatch] Failed to apply`** — extract the optimized `_opt.py` from the best patch:

```bash
cd {{PROBLEMS_DIR}}
for dir in optimization_logs/*/; do
  name=$(basename $dir | sed 's/_[0-9]*$//')

  # Find the best patch by RESULT_JSON speedup
  best_patch=""
  best_speedup="0"
  for t in $dir/patch_*_test.txt; do
    [ -f "$t" ] || continue
    speedup=$(grep 'RESULT_JSON' "$t" | tail -1 | python3 -c "import sys,json; d=json.loads(sys.stdin.read().split('RESULT_JSON: ')[1]); print(d.get('speedup',0))" 2>/dev/null || echo 0)
    if python3 -c "exit(0 if float('$speedup') > float('$best_speedup') else 1)"; then
      best_speedup="$speedup"
      best_patch="${t%_test.txt}.patch"
    fi
  done

  if [ -n "$best_patch" ] && [ -f "$best_patch" ]; then
    echo "$name: best patch=$best_patch speedup=${best_speedup}x"

    # Extract the _opt.py content from the patch diff
    OPT_FILE=$(grep -oP 'problems/problem_\S+_opt\.py' "$best_patch" | head -1)
    if [ -n "$OPT_FILE" ]; then
      # Apply just this file from the patch
      git apply --include="$OPT_FILE" "$best_patch" 2>/dev/null || \
        git apply --include="$OPT_FILE" --3way "$best_patch" 2>/dev/null || \
        echo "  WARNING: could not apply patch for $OPT_FILE — extract manually"
    fi
  fi
done
```

3. **Verify recovered code** — re-run `kernel_test_runner.py` to confirm the recovered kernel matches the GEAK patch speedup:

```bash
python3 {{OUTPUT_DIR}}/scripts/kernel_test_runner.py --src $PROBLEM.py --target ${PROBLEM}_opt.py
```

If the `_best.json` speedup is significantly lower than the patch speedup, the recovery was needed.

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

If `env_info.json` has `geak_available: false` or the LLM API key is missing after asking the user, fall back to manual optimization:

1. For each problem file (prioritized by `optimization_manifest.json` priority: HIGH first), write a `ModelNew` class with `@triton.jit` kernels + `@triton.autotune`
2. Test with `kernel_test_runner.py --src <problem>.py --target <problem>_opt.py`
3. Iterate on the kernel (adjust block sizes, num_warps, memory access patterns) until speedup > 1.0x or 5 attempts exhausted
4. Finalize with `kernel_finalize.py --target <problem>_opt.py` (writes the BEST code, not last)

Focus on the highest-impact kernels first: fused ops (residual+RMSNorm, SwiGLU) typically give the best speedups because they reduce memory traffic.
