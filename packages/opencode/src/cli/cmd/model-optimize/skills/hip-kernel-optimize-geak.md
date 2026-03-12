# HIP Kernel Optimization via GEAK kernel-url Pipeline

## Summary

This instruction covers optimizing HIP C++ kernels (`.cu` files) using GEAK's `--kernel-url` pipeline mode. Unlike Triton kernels which GEAK handles natively, HIP kernels require a build step (hipcc compilation) that the pipeline automatically manages through the preprocessor -> orchestrator flow.

### When to Use

- The profiled bottleneck kernel is implemented in HIP C++ (`.cu`/`.hip`/`.cpp`)
- Source code is accessible (not a precompiled binary)
- The kernel is registered through a JIT build system (aiter, torch cpp_extension) or a standard build system (CMake, Makefile)
- You want to optimize the kernel body (memory access patterns, loop structure, register pressure, dispatch logic) rather than replace it with Triton

### What GEAK kernel-url Does

The `--kernel-url` mode runs a 7-step pipeline:

1. **Resolve kernel URL** -- locates the source file and target function
2. **Codebase context** -- scans repo structure and dependencies
3. **Test discovery + harness creation** -- creates a test harness with `--correctness`, `--benchmark`, `--profile` modes, including the hipcc build step
4. **Kernel profiling** -- profiles the baseline kernel with Metrix/rocprofv3
5. **Baseline metrics** -- records latency/bandwidth numbers
6. **COMMANDMENT generation** -- creates the evaluation contract (SETUP/CORRECTNESS/BENCHMARK sections with build commands)
7. **Orchestrator** -- dispatches strategy_agent(s) that edit the .cu, compile, test, and benchmark iteratively

## Prerequisites

- GEAK installed in the container (`geak --help` works)
- `AMD_LLM_API_KEY` set in `/root/.config/mini-swe-agent/.env`
- geak-oe installed (`/opt/geak-oe` or set `GEAK_OE_ROOT`):
  ```bash
  git clone --depth 1 --branch optimizer-geak-openevolve-benchmark git@github.com:AMD-AGI/GEAK.git /opt/geak-oe
  cd /opt/geak-oe && pip install -e . --ignore-installed blinker
  ```
- Python 3.12 regex fix for `commandment_evaluator.py` (required for Python 3.12+):
  ```bash
  # In /opt/geak-oe/openevolve/commandment_evaluator.py, change:
  # _BENIGN_RE = _re.compile("|".join(_BENIGN_STDERR_PATTERNS))
  # to:
  # _BENIGN_RE = _re.compile("|".join(_BENIGN_STDERR_PATTERNS), _re.IGNORECASE)
  # and remove all inline (?i) flags from _BENIGN_STDERR_PATTERNS entries
  ```

## Procedure

### Step 1: Prepare a Workspace with the Kernel Source

Copy the target `.cu` file and its dependencies into a standalone workspace. Initialize a git repo so GEAK can create worktrees for isolated agent workspaces.

```bash
mkdir -p /workspace/<kernel_name>_opt/csrc/kernels
cp /path/to/target_kernel.cu /workspace/<kernel_name>_opt/csrc/kernels/
cp /path/to/other_required_sources.cu /workspace/<kernel_name>_opt/csrc/
cd /workspace/<kernel_name>_opt
git init && git add -A && git commit -m "Initial kernel source"
```

### Step 2: Set Up the Build System

Copy or create a `build.ninja` (or Makefile) that compiles the kernel. For aiter-managed kernels, the ninja file can be found at:

```
/usr/local/lib/python3.12/dist-packages/aiter/jit/build/<module_name>/build/build.ninja
```

Copy it and rewrite source paths to point to the workspace:

```bash
mkdir -p /workspace/<kernel_name>_opt/build
cp /path/to/aiter/jit/build/<module>/build/build.ninja /workspace/<kernel_name>_opt/build/
# Rewrite paths
sed -i 's|/original/path/to/source|/workspace/<kernel_name>_opt/csrc|g' build/build.ninja
```

Verify the build works:

```bash
cd /workspace/<kernel_name>_opt/build && ninja
# Should produce module_<name>.so in ~20-30 seconds
```

### Step 3: Create a Test Harness (optional -- GEAK creates one automatically)

If you want to provide a pre-built harness, create `test_harness.py` with:
- `--correctness`: loads the compiled .so, runs the kernel, compares against a reference (e.g., `torch.mm`)
- `--benchmark`: benchmarks across representative shapes, outputs `GEAK_RESULT_LATENCY_MS=<value>`
- `--profile`: runs the kernel for profiling tools

GEAK's UnitTestAgent will create this automatically if not provided.

### Step 4: Launch GEAK kernel-url

```bash
geak -m claude-opus-4.6 \
  --kernel-url /workspace/<kernel_name>_opt/csrc/kernels/<target>.cu#L<line> \
  --workspace /workspace/<kernel_name>_opt \
  --repo /workspace/<kernel_name>_opt \
  --gpu-ids 0,1 \
  -o /workspace/<kernel_name>_opt/geak_output \
  --yolo &> /workspace/<kernel_name>_opt/geak.log
```

Key flags:
- `--kernel-url`: path to the .cu file, optionally with `#L<line>` to point at a specific kernel function
- `--workspace`: working directory for the agent
- `--repo`: git repo root (for worktree creation)
- `--gpu-ids`: GPUs for parallel agents (each agent gets one GPU)
- `-o`: output directory for results, patches, logs
- `--yolo`: run without confirmation prompts

### Step 5: Monitor Progress

The pipeline progresses through 7 steps, then the orchestrator launches strategy_agent(s):

```bash
# Check pipeline progress
grep -E "Step [0-9]|Round|speedup|PASS|FAIL" geak.log

# Check agent progress
for f in geak_output/results/round_*/*/task_*.log; do
  name=$(basename $f .log)
  step=$(grep -oP "step \d+" $f | tail -1)
  cost=$(grep -oP '\$[0-9.]+' $f | tail -1)
  echo "$name: $step, cost=$cost"
done

# Check patches (sorted by latency)
for p in geak_output/results/round_*/*/patch_*_test.txt; do
  lat=$(grep -oP "GEAK_RESULT_LATENCY_MS=([0-9.]+)" $p | tail -1)
  echo "$lat $(basename $p)"
done | sort -n
```

### Step 6: Collect the Winning Kernel

After the agents finish, the best patch can be applied:

```bash
# Find the best patch
BEST=$(for p in geak_output/results/round_*/*/patch_*_test.txt; do
  lat=$(grep -oP "GEAK_RESULT_LATENCY_MS=([0-9.]+)" $p | tail -1 | cut -d= -f2)
  echo "$lat $p"
done | sort -n | head -1 | awk '{print $2}')
echo "Best patch: $BEST"

# Apply the winning .cu to the installed package (aiter example)
PATCH_DIR=$(dirname $BEST)
PATCH_NUM=$(basename $BEST _test.txt)
# The patched source is in the worktree
# Copy it over the installed source and force JIT rebuild
cp /path/to/patched/custom_kernels.cu /usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/kernels/
rm -f /usr/local/lib/python3.12/dist-packages/aiter/jit/module_custom.so
AITER_REBUILD=1 python3 -c "from aiter.ops.custom import wvSpltK; print('Rebuilt OK')"
```

## What the Strategy Agent Does

The GEAK strategy_agent has bash + editor access and full control over the compilation pipeline. It typically:

1. **Reads and analyzes** the kernel source, understanding the algorithm, data flow, and hardware constraints
2. **Creates an optimization strategy list** with 3-5 targeted approaches:
   - Dispatch parameter tuning (tile sizes, unroll factors, threshold routing)
   - Memory access optimization (coalescing, LDS staging, vectorized loads)
   - Loop restructuring (software pipelining, K-dimension splitting)
   - Register pressure optimization (recompute vs store tradeoffs)
3. **Iterates**: edits .cu -> compiles with ninja/hipcc -> runs correctness -> benchmarks -> compares against baseline -> records results
4. **Tracks strategies** in `.optimization_strategies.md` to avoid repeating failed approaches

## Example: wvSplitK Optimization Results

Target: `wvSplitK_hf_sml_` (skinny GEMM for linear attention decode, 66.6% of decode GPU time)

| Metric | Value |
|--------|-------|
| Kernel | `_rocm_C::wvSplitK` in `custom_kernels.cu` (2480 lines) |
| Architecture | AMD gfx950 (MI350X, 192 CUs) |
| Baseline latency | 0.0550 ms (geo mean, 25 shapes) |
| Best optimized | 0.0420 ms (patch_8) |
| **Speedup** | **1.31x** |
| Agent cost | ~$0.90 total (2 agents, ~110 steps each) |
| Wall time | ~45 minutes |
| Patches generated | 10 (7 faster than baseline) |

The agents identified that tuning dispatch parameters (YTILE, UNRL thresholds for different N values) and optimizing the medium-kernel routing logic were the most effective strategies.

## Known Issues (from Qwen3.5-9B pipeline run)

- **Triton version conflict**: `geak-oe` installs `triton>=3.3.0` which may break the container's triton (e.g., `module 'triton.language' has no attribute 'constexpr_function'`). This causes vLLM memory access faults. **Fix**: pin triton to the container's version after geak-oe install: `pip3 install triton==$ORIGINAL_VERSION`.
- **numpy version conflict**: `geak-oe` installs `numpy>=2.4` which is incompatible with `numba` and `amd-quark`. **Fix**: `pip3 install 'numpy<2.2'` after geak-oe install.
- **BlockSize patch lost**: Some models (Qwen3.5 hybrid attention) require custom `BlockSize` values (e.g., 528) patched into `vllm/config/cache.py`. Package reinstalls (pip) silently revert this patch. Always verify `BlockSize = Literal[...]` includes required values before starting vLLM.
- **AITER_REBUILD caching**: After installing a modified `.cu`, BOTH the cached `.so` AND the `build/` directory must be deleted to force a full hipcc rebuild. Deleting only the `.so` may cause aiter to skip recompilation if the build directory still contains stale `.o` files.
- **E2E vs kernel-level gap**: The 1.31x kernel-level speedup on wvSplitK (0.0550ms -> 0.0420ms, 25-shape microbenchmark) produced only 1.003x E2E serving improvement (1532 -> 1537 tok/s). `torch.compile` + CUDAGraphs optimize the execution graph at a higher level, potentially masking individual kernel changes. Set realistic E2E expectations.
- **OpenEvolve limitations for HIP**: OpenEvolve's evolutionary mutations are less effective than strategy_agent's targeted edits for well-optimized HIP kernels. The `geak --kernel-url` pipeline (which uses strategy_agent) consistently outperformed OpenEvolve in our testing.

## Troubleshooting

- **UnitTestAgent stuck creating harness**: The C++ kernel test harness is complex. If it takes >10 min, the agent may be struggling with the build system. Provide a pre-built `test_harness.py` in the workspace.
- **Compilation errors**: Ensure all include paths and linked libraries are correct in `build.ninja`. Missing headers from composable_kernel or pybind11 are common issues.
- **Speedup always 1.0x**: The benchmark output must contain `GEAK_RESULT_LATENCY_MS=<value>` for the evaluator to parse. Without this, speedup defaults to 1.0.
- **Python 3.12 regex error**: `"global flags not at the start of the expression"` -- apply the `commandment_evaluator.py` fix described in Prerequisites.

## Model Compatibility

This procedure is kernel-type-driven, not model-specific. It works for any model with HIP C++ kernels:

- **Standard transformers** (Llama, Mistral): typically no HIP kernels -- use simple mode for Triton/ATen instead
- **Hybrid attention** (Qwen3.5, Mamba-based): `_rocm_C::wvSplitK` and similar aiter kernels -- primary use case
- **MoE models** (Mixtral, DeepSeek): may have Composable Kernel (CK) GEMM implementations -- same workflow, different build system
- **Custom architectures**: any model with custom `.cu` kernels registered via `torch.ops`
