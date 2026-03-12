# Phase 5: Generate Problem Files for Kernel Optimization {{SKIP_LABEL}}

## Goal
Convert bottleneck operators into problem files for kernel optimization (Phase 6).
**IMPORTANT**: Analyze operators for fusion opportunities BEFORE creating individual problem files.

## STEP 0: Review Per-Shape Kernel Analysis (from Phase 4)

```bash
python3 {{OUTPUT_DIR}}/scripts/generate_problems.py --review --profile-dir {{PROFILE_DIR}}
```

Review which operator categories dominate GPU time and which specific shapes are hottest.

## STEP 1: Operator Fusion Analysis

Run `analyze_fusion.py` to detect fusable operator patterns (ResidualNorm, SwiGLU, RoPE, QKV, MLP):

```bash
cp {{OUTPUT_DIR}}/scripts/analyze_fusion.py {{PROFILE_DIR}}/
cd {{PROFILE_DIR}}
python3 analyze_fusion.py --bottlenecks decode_bottlenecks.json
cat fusion_opportunities.json
```

## STEP 1.5: Trace and Classify Kernel Types

For each bottleneck op with >= 1% GPU time (from `decode_bottlenecks.json` and `prefilldecode_bottlenecks.json`), determine the **actual kernel implementation type** by tracing the op to its source file.

### Classification procedure

1. **For `aten::*` ops**: skip source tracing — classify by op name directly:
   - `aten::mm`, `aten::addmm` -> `kernel_type: "aten_gemm"` (dispatched to rocBLAS/hipBLASlt)
   - All other `aten::*` -> `kernel_type: "aten_elementwise"`

2. **For custom ops** (anything not `aten::*`): strip the namespace prefix to get the bare function name (e.g., `_rocm_C::wvSplitK` -> `wvSplitK`, `vllm::gdn_attention_core` -> `gdn_attention_core`), then search the registering package:

```bash
# Find which package registers the op namespace
python3 -c "import torch; print(torch.ops._rocm_C)" 2>/dev/null
# Search only that package for the bare function name
grep -r "BARE_NAME" /path/to/package/ --include="*.py" --include="*.cu" --include="*.hip" -l
```

3. **Read the source file** and classify:
   - Source is `.cu` / `.hip` / `.cpp` with HIP kernel functions (`__global__`, `__device__`) -> `kernel_type: "hip"`
   - Source is `.cu` / `.cpp` with CK template instantiation (`#include <ck/tensor_operation/...>`) -> `kernel_type: "ck"`
   - Source loads HSACO binary (`hipModuleLoad`, `.hsaco`) -> `kernel_type: "asm"`
   - Source is `.py` containing `@triton.jit` -> `kernel_type: "triton"`
   - Source is `.py` with `torch.autograd.Function` that dispatches to other functions -> trace those callees (max depth 2). If callees are `@triton.jit` -> `kernel_type: "triton_composite"`

4. **For `hip`/`ck`/`asm` kernels**, find the Python binding: search `torch.ops.<namespace>.*` in the package's Python files to find the callable signature.

5. **If source not found** within 30 seconds or the op is untraceable (e.g., `hipModuleLaunchKernel`):
   - Set `kernel_type: "unknown"`
   - Set `priority: "LOW"`, add `notes` explaining the trace failed

6. **Record** `kernel_type`, `source_file`, and `python_binding` for each op — these are used in STEP 4 (manifest) and by Phase 6 (GEAK optimization mode selection).

**IMPORTANT**: `source_file` is required for all types that use `geak --kernel-url` in Phase 6: `hip`, `ck`, `asm`, and `triton_composite`. Without `source_file`, Phase 6 falls back to simple mode.

### Classification types and Phase 6 GEAK modes

| Type | Source | Phase 6 GEAK Mode | Optimization Strategy |
|------|--------|-------------------|----------------------|
| `triton` | `.py` with `@triton.jit` | `geak -t` (simple) | Write faster Triton kernel (ModelNew) |
| `triton_composite` | Python wrapper -> multiple `@triton.jit` | `geak --kernel-url` | Optimize inner Triton sub-kernels in-place |
| `hip` | `.cu` / `.hip` / `.cpp` with HIP kernels | `geak --kernel-url` | Optimize HIP source in-place (see `hip-kernel-optimize-geak.md`) |
| `ck` | CK template instantiation (`ck/tensor_operation/`) | `geak --kernel-url` | Tune CK template parameters and pipeline |
| `asm` | Pre-compiled HSACO (`hipModuleLoad`) | `geak --kernel-url` | Optimize launch config and wrapper only (binary not editable) |
| `aten_gemm` | `aten::mm` / `aten::addmm` (rocBLAS) | `geak -t` (simple) | Write Triton GEMM to beat rocBLAS |
| `aten_elementwise` | `aten::*` elementwise/reduce | `geak -t` (simple) | Fuse with neighbors or write standalone Triton |
| `unknown` | Not found / binary-only | skip or `geak -t` (LOW) | PyTorch-equivalent baseline, low priority |

**Rule of thumb**: C++ source (`.cu`/`.hip`/`.cpp`) -> `geak --kernel-url`. Python/Triton source -> `geak -t` simple mode.

## STEP 2: Generate All Problem Files

Run `generate_problems.py` to auto-generate problem files from Phase 4 analysis:

```bash
python3 {{OUTPUT_DIR}}/scripts/generate_problems.py \
  --profile-dir {{PROFILE_DIR}} \
  --problems-dir {{PROBLEMS_DIR}}
```

This generates three categories of problem files with PyTorch baselines:

- **Fusion problems** (from `fusion_opportunities.json`) -- fused_residual_rmsnorm, fused_swiglu, etc.
- **GEMM problems** (from `{phase}_analysis.json`) -- shapes with roofline efficiency < 80%, ranked worst-first
- **Attention problems** (from `{phase}_bottlenecks.json`) -- all attention kernel types (SDPA, flash, linear, GDN) >= 1% GPU time

Each problem file contains `class Model(nn.Module)` (PyTorch baseline), `get_inputs()`, and `get_init_inputs()` using actual shapes from profiling.

## STEP 3: Create Additional Problem Files

Create problem files for bottleneck ops NOT already covered by STEP 2. This now includes two categories:

### 3a: Individual ops (as before)
For operators that cannot be fused, take > 5% time, and are not covered by STEP 2 (RMSNorm, SiLU/GELU, RoPE, standalone attention). Use PyTorch baseline in `class Model`.

### 3b: HIP kernel ops (`kernel_type: "hip"`)
For ops classified as `hip` in STEP 1.5, create problem files where `class Model` calls the **actual HIP kernel** via its Python binding. This makes the baseline the real HIP kernel, not a PyTorch equivalent.

The agent must:
1. Read the function signature from the `source_file` found in STEP 1.5
2. Map profiled tensor dimensions (from `unified_perf_summary.csv` `Input Dims` column) to the function parameters
3. **Import the binding loader** in the problem file — the `torch.ops` namespace for the HIP kernel only exists after the registering package is imported

Example structure:

```python
import torch
import torch.nn as nn
import vllm._custom_ops as ops  # registers torch.ops._rocm_C.*

class Model(nn.Module):
    def forward(self, a, b):
        return ops.wvSplitK(a, b, cu_count=304)

# shapes from profiling Input Dims column
def get_inputs():
    return [torch.randn(1, 4096, dtype=torch.bfloat16, device="cuda"),
            torch.randn(4096, 14336, dtype=torch.bfloat16, device="cuda")]
def get_init_inputs():
    return []
```

### 3c: Triton composite ops (`kernel_type: "triton_composite"`)
For ops classified as `triton_composite` in STEP 1.5, create problem files where `class Model` calls the composite function directly. Include a comment listing the source file paths of the Triton sub-kernels for GEAK's reference.

## STEP 4: Generate Optimization Manifest

Create `{{PROBLEMS_DIR}}/optimization_manifest.json`. Every entry MUST include the `kernel_type` metadata from STEP 1.5. All problems are `enabled: true` by default — the speedup > 1.0x correctness filter in Phase 6 Step 5 is the proper gate.

```json
{
  "model": "{{HF_MODEL}}",
  "description": "Optimization manifest with kernel type metadata. All enabled by default.",
  "optimizations": [
    {
      "name": "fused_residual_rmsnorm",
      "file": "problem_fused_residual_rmsnorm.py",
      "type": "fused",
      "priority": "HIGH",
      "kernel_type": "triton",
      "original_kernel": "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0",
      "source_file": "/path/to/fused_kernel.py",
      "enabled": true
    },
    {
      "name": "skinny_gemm",
      "file": "problem_skinny_gemm.py",
      "type": "hip_kernel",
      "priority": "HIGH",
      "kernel_type": "hip",
      "original_kernel": "_rocm_C::wvSplitK",
      "source_file": "/path/to/custom_kernels.cu",
      "python_binding": "torch.ops._rocm_C.wvSplitK(a, b, bias, cu_count)",
      "enabled": true
    },
    {
      "name": "gemm_256x4096x12288",
      "file": "problem_gemm_prefilldecode_256x4096x12288.py",
      "type": "gemm",
      "priority": "MEDIUM",
      "kernel_type": "aten_gemm",
      "original_kernel": "aten::mm",
      "enabled": true
    }
  ]
}
```

## Steps Summary
1. Review analysis_summary.json for shape priorities
2. Run fusion analysis (`analyze_fusion.py`)
3. Trace and classify kernel types for all bottleneck ops >= 1% GPU time (source tracing)
4. Generate standard problem files (`generate_problems.py` -- fusion + GEMM roofline + attention)
5. Create additional problem files: individual ops, HIP kernel ops (real baseline), triton_composite ops
6. Generate optimization_manifest.json with kernel_type metadata, all enabled
7. Update progress.json
