# Phase 6: Run Kernel Optimization {{SKIP_LABEL}}

## Goal
Optimize bottleneck kernels using kernel-optimize, running **in parallel** for speed.

## ⚠️ PARALLEL OPTIMIZATION

```bash
cd {{PROBLEMS_DIR}}

# HIGH priority (fused kernels) - run in parallel
opencode kernel-optimize --src problem_fused_residual_rmsnorm.py --goal 1.5 &
opencode kernel-optimize --src problem_fused_rmsnorm.py --goal 1.5 &
opencode kernel-optimize --src problem_fused_rope.py --goal 1.5 &
wait

# MEDIUM priority
opencode kernel-optimize --src problem_rope.py --goal 1.3 &
opencode kernel-optimize --src problem_silu_mul.py --goal 1.3 &
wait
```

## Priority Order
| Priority | Kernel Type | Goal | Reason |
|----------|-------------|------|--------|
| **HIGH** | Fused Residual+RMSNorm | 1.5x | Memory traffic reduction |
| **HIGH** | Fused RMSNorm | 1.5x | Repeated many times |
| **HIGH** | Fused RoPE | 1.5x | Custom AMD optimization |
| MEDIUM | SwiGLU/GELU | 1.3x | Activation functions |
| LOW | Linear/GEMM | 1.1x | rocBLAS usually optimal |
| **SKIP** | Simple add/copy | - | Overhead > benefit |

## When to Skip
- If operator is part of a fused kernel you already optimized
- If rocBLAS is already near-optimal (GEMM/Linear)
- If simple elementwise (add, copy) - overhead exceeds benefit

## Verify Speedup at ACTUAL Inference Shapes

```bash
cd {{PROBLEMS_DIR}}
python -c "
import torch, time
from problem_XXX import Model as RefModel, get_inputs, get_init_inputs
from problem_XXX_opt import ModelNew as OptModel
inputs = get_inputs()
ref = RefModel(*get_init_inputs()).cuda().eval()
opt = OptModel(*get_init_inputs()).cuda().eval()
for _ in range(20): ref(*inputs); opt(*inputs)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(500): ref(*inputs)
torch.cuda.synchronize()
t_ref = (time.perf_counter()-t0)/500*1000
t0 = time.perf_counter()
for _ in range(500): opt(*inputs)
torch.cuda.synchronize()
t_opt = (time.perf_counter()-t0)/500*1000
print(f'Speedup: {t_ref/t_opt:.2f}x')
"
```

Only copy kernels faster at ACTUAL shapes to `{{OPTIMIZED_DIR}}/`.

---

# Phase 6.5: Additional Optimization Opportunities

## Data-Driven Decision Making
**Before enabling ANY optimization**: check profiling, benchmark at actual shapes, only apply if > 1.0x.

### AITER Flash Attention
Consider if attention > 10% runtime AND seq > 64. AITER is slower for short sequences.
```bash
source {{OUTPUT_DIR}}/venv/bin/activate
pip install -e /sgl-workspace/aiter/ 2>/dev/null || echo "AITER not available"
```

### Additional AMD Optimizations
- AITER GEMM for MoE/FP8 workloads
- hipBLASLt for specific GEMM shapes
- ROCm environment tuning

**If an optimization doesn't help at actual shapes, simply don't apply it.**

