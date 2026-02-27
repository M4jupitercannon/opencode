---
description: "Optimize a PyTorch operator to Triton kernel. Usage: /kernel-optimize <src_file> [goal]"
agent: model-opt
---

# GPU Kernel Optimization Task

## Source File: $1
## Target File: (auto: $1 with _opt.py suffix)
## Goal: $2 (if not specified, optimize for best possible speedup)

## Setup

Determine paths:
- Source: `$1` (must contain `class Model` + `def get_inputs()`)
- Target: replace `.py` with `_opt.py` in the source path
- Scripts: `~/.config/opencode/scripts/` (kernel_test_runner.py, kernel_finalize.py)

```bash
SRC="$1"
TARGET="${SRC%.py}_opt.py"
SCRIPTS_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/scripts"
echo "Source: $SRC"
echo "Target: $TARGET"
echo "Scripts: $SCRIPTS_DIR"
```

## Step 1: Read and Understand the Source

Read the source file. It contains:
- `class Model(nn.Module)` — the reference PyTorch implementation
- `def get_inputs()` — returns input tensors for testing
- `def get_init_inputs()` — returns args for `Model.__init__`

## Step 2: Get GPU Information

```bash
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}'); print(f'Arch: {torch.cuda.get_device_capability()}')"
```
or 
```bash
 rocm-smi --showproductname || nvidia-smi
```
Adjust optimization strategy based on GPU:
- AMD MI300X/MI325X: wave size 64, CDNA3
- AMD MI355X: wave size 64, CDNA4
- NVIDIA A100/H100: warp size 32
- AMD gfx1201/ gfx1200：wave size 32, RDNA4

## Step 3: Write Optimized Triton Kernel

Create the target file with `class ModelNew(nn.Module)` that:
- Produces identical output to `Model` within tolerance
- Uses `@triton.jit` decorated Triton kernels for ALL compute
- **FORBIDDEN**: torch.matmul, torch.mm, torch.relu, torch.softmax, etc.
- **ALLOWED**: .view(), .reshape(), .contiguous(), torch.empty() for output buffers

### Optimization Techniques
1. Use `@triton.autotune` with 10-20 diverse configs
2. Ensure coalesced memory access
3. Use FP32 accumulation for stability
4. Consider operation fusion to reduce memory traffic

## Step 4: Test with kernel_test_runner.py

```bash
python3 "$SCRIPTS_DIR/kernel_test_runner.py" --src "$SRC" --target "$TARGET"
```

The script:
- Tests accuracy (Model vs ModelNew)
- Benchmarks performance (median of 5 rounds)
- Automatically tracks best result
- Prints `RESULT_JSON: {...}` with speedup

## Step 5: Iterate

If accuracy fails → fix the kernel, go to Step 4.
If speedup is below goal → optimize further:
- Try different block sizes, num_warps, num_stages
- Consider algorithm changes (tiling, split-K, persistent kernels)
- Check if memory-bound or compute-bound

Re-run: `python3 "$SCRIPTS_DIR/kernel_test_runner.py" --src "$SRC" --target "$TARGET"`

## Step 6: Finalize

When satisfied with the result:

```bash
python3 "$SCRIPTS_DIR/kernel_finalize.py" --target "$TARGET"
```

This writes the BEST code (not last!) to the target file.

## STRICT REQUIREMENTS

### ModelNew Rules
- **MUST use @triton.jit Triton kernels for ALL compute**
- **FORBIDDEN**: torch.matmul, torch.mm, torch.bmm, torch.add, torch.mul, torch.relu, torch.sigmoid, torch.softmax, torch.layer_norm, F.linear, F.relu, etc.
- **ALLOWED**: Triton kernels, .view(), .reshape(), .contiguous(), .to(), torch.empty(), torch.zeros()
- ModelNew must accept same `__init__` parameters as Model
- Output must match Model within tolerance (fp16: rel < 1e-2, fp32: < 1e-5)

