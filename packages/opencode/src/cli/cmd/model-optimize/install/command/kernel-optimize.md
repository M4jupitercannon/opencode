---
description: "Optimize a PyTorch operator to Triton kernel. Usage: /kernel-optimize <src_file> [goal]"
agent: model-opt
---

# GPU Kernel Optimization Task

- **Source**: `$1` (must contain `class Model` + `def get_inputs()` + `def get_init_inputs()`)
- **Target**: `${SRC%.py}_opt.py` (auto-generated from source path)
- **Goal**: $2 (default: maximize speedup)
- **Scripts**: `~/.config/opencode/scripts/` (`kernel_test_runner.py`, `kernel_finalize.py`)

## Hard-Stop Rules (Instruction Strictness)
- If source format is invalid (missing `Model`/`get_inputs`/`get_init_inputs`), **STOP**.
- If test output is not `accuracy=PASSED`, **do not finalize**.
- Do not claim speedup without runner output evidence (`RESULT_JSON` and tracker file).
- Never replace compute with PyTorch ops to “pass quickly”; keep compute in Triton kernels.
- If performance regresses after 3 iterations, report failure and likely bottleneck reason instead of forcing finalize.

## Preflight Checks (Mandatory)
```bash
SRC="$1"
TARGET="${SRC%.py}_opt.py"
SCRIPTS_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/scripts"

test -f "$SRC" || { echo "ERROR: source file not found: $SRC"; exit 1; }
test -f "$SCRIPTS_DIR/kernel_test_runner.py" || { echo "ERROR: missing kernel_test_runner.py"; exit 1; }
test -f "$SCRIPTS_DIR/kernel_finalize.py" || { echo "ERROR: missing kernel_finalize.py"; exit 1; }

SRC="$SRC" python3 - << 'PY'
from pathlib import Path
import os
src = Path(os.environ["SRC"]).read_text()
required = ["class Model", "def get_inputs", "def get_init_inputs"]
missing = [x for x in required if x not in src]
if missing:
    raise SystemExit(f"ERROR: source missing required blocks: {missing}")
print("Preflight OK")
PY
```

## Step 1: Read and Understand the Source

Read the source file. Identify:
- The PyTorch operation(s) in `Model.forward()`
- Input shapes and dtypes from `get_inputs()`
- Whether it's memory-bound (elementwise, norm) or compute-bound (matmul, conv)
- Fusion opportunities (multiple ops that share intermediate tensors)

## Step 2: Detect GPU Architecture

```bash
python3 -c "
import torch
name = torch.cuda.get_device_name()
cap = torch.cuda.get_device_capability()
print(f'GPU: {name}, Capability: {cap}')
"
```

Adapt strategy to the target GPU:

| GPU | ISA | Wave/Warp Size | Key Notes |
|-----|-----|---------------|-----------|
| AMD MI300X/MI325X | CDNA 3 | 64 | Large LDS (64KB), high HBM BW |
| AMD MI355X | CDNA 4 | 64 | Next-gen CDNA |
| AMD gfx1200/gfx1201 | RDNA 4 | 32 | Smaller LDS, different occupancy model |
| NVIDIA A100/H100 | Ampere/Hopper | 32 | Native TF32, larger shared memory on H100 |

## Step 3: Write Optimized Triton Kernel

Create the target file with `class ModelNew(nn.Module)`:

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        # Add 10-20 configs spanning the search space
    ],
    key=['M', 'N'],
)
@triton.jit
def kernel_fn(
    x_ptr, y_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn, stride_ym, stride_yn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    ...

class ModelNew(nn.Module):
    def __init__(self, *args, **kwargs):  # keep same signature as Model
        super().__init__()
        ...
    def forward(self, x, y):  # keep same signature as Model.forward
        # Allocate output, launch kernel, return
        ...
```

### Strict Rules
- **MUST** use `@triton.jit` Triton kernels for ALL compute
- **FORBIDDEN**: `torch.matmul`, `torch.mm`, `torch.bmm`, `torch.add`, `torch.mul`, `torch.relu`, `torch.sigmoid`, `torch.softmax`, `torch.layer_norm`, `F.linear`, `F.relu`, etc.
- **ALLOWED**: `.view()`, `.reshape()`, `.contiguous()`, `.to()`, `torch.empty()`, `torch.zeros()` (for output buffers and shape manipulation only)
- `ModelNew.__init__` must accept same parameters as `Model.__init__`
- Output must match `Model` within tolerance (fp16: rel < 1e-2, fp32: rel < 1e-5)

### Optimization Strategies

**Memory-bound ops** (elementwise, norm, activation):
- Maximize memory throughput via coalesced access and vectorized loads
- Fuse multiple ops into a single kernel to avoid round-trips to global memory
- Use `tl.load(..., mask=mask)` to handle boundary elements

**Compute-bound ops** (matmul, conv):
- Use larger tile sizes and multi-stage pipelining
- Accumulate in FP32 (`acc = tl.zeros(..., dtype=tl.float32)`), cast at end
- Consider split-K or persistent kernel strategies for small-batch GEMMs

**General**:
- `@triton.autotune` with diverse configs (vary BLOCK sizes, num_warps, num_stages)
- Profile with `triton.testing.do_bench()` if needed to isolate bottlenecks
- On AMD GPUs, prefer `num_warps` that are multiples of wavefront size (2, 4, 8 for CDNA; 1, 2, 4 for RDNA)

## Step 4: Test

```bash
SRC="$1"
TARGET="${SRC%.py}_opt.py"
SCRIPTS_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/scripts"

python3 "$SCRIPTS_DIR/kernel_test_runner.py" --src "$SRC" --target "$TARGET"
```

Output: `RESULT_JSON: {"speedup": 1.5, "accuracy": "PASSED", ...}`

## Step 5: Iterate

- **Accuracy FAILED** → fix kernel logic, re-run Step 4
- **Speedup below goal** → try different block sizes, num_warps, num_stages; consider algorithm changes (tiling, split-K, persistent kernels); check if memory-bound or compute-bound

Re-run Step 4 after each change.

## Step 6: Finalize

```bash
SRC="$1"
TARGET="${SRC%.py}_opt.py"
SCRIPTS_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/scripts"

python3 "$SCRIPTS_DIR/kernel_finalize.py" --target "$TARGET"
```

This writes the BEST result (not necessarily the last) to the target file.

## Finalization Gate (Mandatory)
- `RESULT_JSON.accuracy == "PASSED"`
- `RESULT_JSON.speedup` is present and numeric
- Tracker file for target exists and records the same/better best result
- If goal `$2` is provided, `speedup >= goal`; otherwise require non-regression (`speedup >= 1.0`) unless the user explicitly approves a regression (e.g., accuracy-preserving refactor with minor slowdown)
