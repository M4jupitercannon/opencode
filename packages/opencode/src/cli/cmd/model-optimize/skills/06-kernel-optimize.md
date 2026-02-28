# Phase 6: Kernel Optimization {{SKIP_LABEL}}

## Goal
Write optimized Triton kernels for each problem file and verify speedup.

## ⚠️ Docker vs venv
If Phase 0 created a Docker container (`env_type: "docker"` in `env_info.json`), prefix all commands with `docker exec $CONTAINER_NAME bash -c "..."` and use `HIP_VISIBLE_DEVICES=$BEST_GPU`.

Detect once before running this phase:
```bash
ENV_TYPE=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('env_type','venv'))" 2>/dev/null || echo "venv")
CONTAINER_NAME=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('container','vllm_model_opt'))" 2>/dev/null || echo "vllm_model_opt")
BEST_GPU=$(python3 -c "import json; print(json.load(open('{{OUTPUT_DIR}}/env_info.json')).get('best_gpu',0))" 2>/dev/null || echo 0)
```

## ⚠️ NO external `opencode` command needed
Optimize kernels DIRECTLY in this session using the test scripts provided.

## Scripts Available
- `{{OUTPUT_DIR}}/scripts/kernel_test_runner.py` — test accuracy + benchmark
- `{{OUTPUT_DIR}}/scripts/kernel_finalize.py` — save best result to target file

## Workflow for EACH Problem File

For each `problem_*.py` file in `{{PROBLEMS_DIR}}/`:

### 1. Read the source file to understand the PyTorch operator
```bash
cat {{PROBLEMS_DIR}}/problem_XXX.py
```

### 2. Check GPU architecture
```bash
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}, Arch: {torch.cuda.get_device_capability()}')"
```

### 3. Write the optimized Triton kernel
Create `{{PROBLEMS_DIR}}/problem_XXX_opt.py` with:
- `class ModelNew(nn.Module)` using `@triton.jit` Triton kernels
- Same `__init__` signature as `Model`
- Use `@triton.autotune` with 10-20 diverse configs

### 4. Test accuracy + benchmark
```bash
# venv mode only:
# source {{OUTPUT_DIR}}/venv/bin/activate
python3 {{OUTPUT_DIR}}/scripts/kernel_test_runner.py \
  --src {{PROBLEMS_DIR}}/problem_XXX.py \
  --target {{PROBLEMS_DIR}}/problem_XXX_opt.py
```

The script prints: `RESULT_JSON: {"speedup": 1.5, "accuracy": "PASSED", ...}`

### 5. Iterate if needed
- Accuracy FAILED → fix kernel, re-run step 4
- Speedup too low → adjust block sizes, fusion strategy, re-run step 4

### 6. Finalize when satisfied
```bash
python3 {{OUTPUT_DIR}}/scripts/kernel_finalize.py \
  --target {{PROBLEMS_DIR}}/problem_XXX_opt.py
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
cd {{PROBLEMS_DIR}}
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

Copy successful optimizations to `{{OPTIMIZED_DIR}}/`:
```bash
cd {{PROBLEMS_DIR}}
for f in *_opt.py; do
  tracker="${f%.py}_best.json"
  if [ -f "$tracker" ]; then
    speedup=$(python3 -c "import json; d=json.load(open('$tracker')); print(d.get('best_speedup',0))")
    if python3 -c "exit(0 if $speedup > 1.0 else 1)"; then
      cp "$f" {{OPTIMIZED_DIR}}/
      echo "Copied $f (${speedup}x)"
    fi
  fi
done
```

Update progress.json: phases_completed.append("optimize")
