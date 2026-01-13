import { cmd } from "./cmd"
import { bootstrap } from "../bootstrap"
import { Server } from "../../server/server"
import { createOpencodeClient } from "@opencode-ai/sdk"
import * as fs from "fs"
import * as path from "path"
import { UI } from "../ui"

export const KernelOptimizeCommand = cmd({
  command: "kernel-optimize",
  describe: "optimize PyTorch code to Triton kernels",
  builder: (yargs) =>
    yargs
      .option("src", {
        type: "string",
        describe: "source PyTorch implementation file",
        demandOption: true,
      })
      .option("target", {
        type: "string",
        describe: "target Triton implementation output (default: <src>_opt.py)",
      })
      .option("goal", {
        type: "number",
        describe: "target speedup ratio (e.g., 2.0 for 2x speedup)",
      }),
  async handler(args) {
    // Check if LLM_GATEWAY_KEY is set (use process.env directly, not Env.get which requires bootstrap context)
    const gatewayKey = process.env.LLM_GATEWAY_KEY
    if (!gatewayKey) {
      UI.error("LLM_GATEWAY_KEY environment variable is not set")
      UI.println("Please set it with: export LLM_GATEWAY_KEY=your-api-key")
      process.exit(1)
    }

    // Validate source file
    const srcFile = args.src as string
    if (!fs.existsSync(srcFile)) {
      UI.error(`Source file '${srcFile}' not found`)
      process.exit(1)
    }

    // Check source file contains Model class and get_inputs()
    const srcContent = fs.readFileSync(srcFile, "utf-8")
    const hasModel = srcContent.includes("class Model(") || srcContent.includes("class Model:")
    const hasModelNew = srcContent.includes("class ModelNew(") || srcContent.includes("class ModelNew:")
    const hasGetInputs = srcContent.includes("def get_inputs")

    if (!hasModel && hasModelNew) {
      // User passed an _opt.py file as source
      UI.error(`Source file contains 'ModelNew' but not 'Model'`)
      UI.println("")
      UI.println("The --src file must contain 'class Model' as the accuracy reference.")
      UI.println("")
      UI.println("Usage options:")
      UI.println("  1. torch2triton: opencode kernel-optimize --src original.py")
      UI.println("  2. triton2triton: opencode kernel-optimize --src optimized.py")
      UI.println("     (where optimized.py has 'class Model' using Triton)")
      UI.println("  3. Continue from target: opencode kernel-optimize --src original.py --target optimized.py")
      UI.println("")
      UI.println("If you want to continue optimizing, rename 'ModelNew' to 'Model' in your source,")
      UI.println("or use option 3 above.")
      process.exit(1)
    }
    if (!hasModel) {
      UI.error(`Source file must contain 'class Model' (reference implementation)`)
      UI.println("")
      UI.println("The Model class can use torch operators OR triton kernels.")
      UI.println("It serves as the accuracy baseline for ModelNew.")
      process.exit(1)
    }
    if (!hasGetInputs) {
      UI.error(`Source file must contain 'def get_inputs()' function`)
      process.exit(1)
    }

    // Get absolute paths
    const srcPath = path.resolve(srcFile)
    const srcDir = path.dirname(srcPath)
    const srcBasename = path.basename(srcFile, ".py")

    // Default target file if not specified
    const targetFile = (args.target as string) || path.join(srcDir, `${srcBasename}_opt.py`)
    const targetPath = path.resolve(targetFile)
    const targetDir = path.dirname(targetPath)

    // Create target directory if needed
    if (!fs.existsSync(targetDir)) {
      fs.mkdirSync(targetDir, { recursive: true })
    }

    // Log file for optimization history
    const logFile = targetPath.replace(/\.py$/, "_history.log")

    const goal = args.goal as number | undefined

    UI.println("============================================")
    UI.println("GPU Kernel Optimization Tool")
    UI.println("============================================")
    UI.println(`Source file:  ${srcPath}`)
    UI.println(`Target file:  ${targetPath}`)
    UI.println(`History log:  ${logFile}`)
    if (goal) {
      UI.println(`Goal:         ${goal}x speedup`)
    }
    UI.println(`Working dir:  ${srcDir}`)
    UI.println("============================================")
    UI.println("")

    // Initialize log file
    const logHeader = `# Kernel Optimization History Log
# Source: ${srcPath}
# Target: ${targetPath}
# Started: ${new Date().toISOString()}
${goal ? `# Goal: ${goal}x speedup` : ""}
# ============================================

`
    fs.writeFileSync(logFile, logHeader)

    // Create best tracker file (JSON)
    const bestTrackerFile = targetPath.replace(/\.py$/, "_best.json")
    const initialTracker = { best_speedup: 0, best_code: "", attempt: 0 }
    fs.writeFileSync(bestTrackerFile, JSON.stringify(initialTracker, null, 2))

    // Create automated test script that handles tracking
    const testScriptPath = path.join(srcDir, ".kernel_test_runner.py")
    const testScript = `#!/usr/bin/env python3
"""
Automated kernel test runner with best result tracking.
Usage: python .kernel_test_runner.py
"""
import torch
import time
import json
import os
import sys
import statistics

SRC = '${srcPath}'
TGT = '${targetPath}'
LOG = '${logFile}'
TRACKER = '${bestTrackerFile}'

def main():
    # Load tracker
    with open(TRACKER) as f:
        tracker = json.load(f)
    
    attempt = tracker['attempt'] + 1
    
    print(f"\\n{'='*50}")
    print(f"Testing Attempt {attempt}")
    print(f"{'='*50}\\n")
    
    # Import modules using importlib (required for Triton to get source)
    import importlib.util
    
    def load_module(filepath, module_name):
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    
    # Load source
    try:
        src_module = load_module(SRC, 'src_module')
        # Import all from source module to globals
        for name in dir(src_module):
            if not name.startswith('_'):
                globals()[name] = getattr(src_module, name)
    except Exception as e:
        print(f"ERROR loading source: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Read target code for snapshot, then import
    try:
        target_code = open(TGT).read()
        tgt_module = load_module(TGT, 'tgt_module')
        # Import ModelNew from target
        for name in dir(tgt_module):
            if not name.startswith('_'):
                globals()[name] = getattr(tgt_module, name)
    except Exception as e:
        print(f"ERROR loading target: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Check if get_init_inputs exists
    try:
        init_inputs = get_init_inputs()
    except:
        init_inputs = []
    
    # Initialize models
    try:
        if init_inputs:
            model_ref = Model(*init_inputs).cuda().eval()
            model_new = ModelNew(*init_inputs).cuda().eval()
        else:
            model_ref = Model().cuda().eval()
            model_new = ModelNew().cuda().eval()
    except Exception as e:
        print(f"ERROR initializing models: {e}")
        sys.exit(1)
    
    # Sync weights from Model to ModelNew (for fair comparison)
    # This handles cases where ModelNew has trainable parameters (e.g., fused nn.Linear)
    try:
        ref_state = model_ref.state_dict()
        new_state = model_new.state_dict()
        if ref_state and new_state:
            # Try to match parameters by name and shape
            matched = 0
            for name, param in ref_state.items():
                if name in new_state and new_state[name].shape == param.shape:
                    new_state[name].copy_(param)
                    matched += 1
                else:
                    # Try common mappings (linear.weight -> weight, etc.)
                    for new_name in new_state:
                        if new_state[new_name].shape == param.shape:
                            # Check if names are similar (e.g., 'gemm.weight' vs 'weight')
                            ref_parts = name.split('.')
                            new_parts = new_name.split('.')
                            if ref_parts[-1] == new_parts[-1]:  # Same parameter type
                                new_state[new_name].copy_(param)
                                matched += 1
                                break
            if matched > 0:
                model_new.load_state_dict(new_state)
                print(f"[Note: Synced {matched} parameters from Model to ModelNew]")
    except Exception as e:
        # Weight sync is optional, continue if it fails
        pass
    
    # Get inputs
    inputs = [x.cuda() if hasattr(x, 'cuda') else x for x in get_inputs()]
    
    # Auto-convert model dtype to match input dtype (for models with nn.Linear etc.)
    # Find the primary input dtype (first floating point tensor)
    input_dtype = None
    for x in inputs:
        if hasattr(x, 'dtype') and x.dtype in [torch.float16, torch.bfloat16]:
            input_dtype = x.dtype
            break
    
    if input_dtype is not None:
        # Check if model has parameters (nn.Module with layers)
        if hasattr(model_ref, 'parameters') and any(True for _ in model_ref.parameters()):
            model_ref = model_ref.to(input_dtype)
            print(f"[Note: Model converted to {input_dtype} to match input dtype]")
        if hasattr(model_new, 'parameters') and any(True for _ in model_new.parameters()):
            model_new = model_new.to(input_dtype)
    
    # Check if inputs contain low-precision types (int8/float8)
    def is_low_precision(dtype):
        dtype_str = str(dtype)
        return any(x in dtype_str for x in ['int8', 'float8', 'uint8', 'qint8'])
    
    has_quantized_input = any(
        is_low_precision(x.dtype) for x in inputs if hasattr(x, 'dtype')
    )
    
    # Test accuracy
    with torch.no_grad():
        try:
            out_ref = model_ref(*inputs)
            out_new = model_new(*inputs)
        except Exception as e:
            print(f"ERROR in forward pass: {e}")
            sys.exit(1)
    
    # Determine tolerance based on input/output precision
    # - Quantized inputs (int8/float8): very relaxed (5e-2, 5e-2)
    # - fp16/bf16 output: relaxed (1e-2, 1e-3)
    # - fp32 output: strict (1e-5, 1e-6)
    if has_quantized_input:
        rtol, atol = (5e-2, 5e-2)  # Quantized: ~5% tolerance
        print(f"[Note: Using relaxed tolerance for quantized inputs]")
    elif out_ref.dtype in [torch.float16, torch.bfloat16]:
        rtol, atol = (1e-2, 1e-3)  # fp16/bf16
    else:
        rtol, atol = (1e-5, 1e-6)  # fp32
    
    max_diff = (out_ref - out_new).abs().max().item()
    denom = out_ref.abs() + 1e-8
    rel_diff = ((out_ref - out_new).abs() / denom).max().item()
    is_close = torch.allclose(out_ref, out_new, rtol=rtol, atol=atol)
    
    print(f"=== Accuracy ===")
    print(f"Max abs error: {max_diff:.2e}, Max rel error: {rel_diff:.2e}")
    print(f"Accuracy: {'PASSED' if is_close else 'FAILED'}")
    
    if not is_close:
        print("\\nERROR: Accuracy test failed!")
        # Log the failed attempt
        log_entry = f"""
## Attempt {attempt} - {time.strftime('%Y-%m-%dT%H:%M:%S')} - FAILED (accuracy)
Max abs error: {max_diff:.2e}, Max rel error: {rel_diff:.2e}

"""
        with open(LOG, 'a') as f:
            f.write(log_entry)
        tracker['attempt'] = attempt
        with open(TRACKER, 'w') as f:
            json.dump(tracker, f, indent=2)
        sys.exit(1)
    
    # Benchmark
    print("\\n=== Benchmarking ===")
    
    # Warmup
    for _ in range(20):
        model_ref(*inputs)
        model_new(*inputs)
    torch.cuda.synchronize()
    
    # Multiple rounds for stability
    NUM_ROUNDS = 5
    N_PER_ROUND = 100
    ref_times = []
    new_times = []
    
    for r in range(NUM_ROUNDS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_PER_ROUND):
            model_ref(*inputs)
        torch.cuda.synchronize()
        ref_times.append((time.perf_counter() - t0) / N_PER_ROUND * 1000)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_PER_ROUND):
            model_new(*inputs)
        torch.cuda.synchronize()
        new_times.append((time.perf_counter() - t0) / N_PER_ROUND * 1000)
    
    t_ref = statistics.median(ref_times)
    t_new = statistics.median(new_times)
    speedup = t_ref / t_new
    
    print(f"\\n=== Performance (median of {NUM_ROUNDS} rounds) ===")
    print(f"PyTorch (ref): {t_ref:.4f} ms (std: {statistics.stdev(ref_times):.4f})")
    print(f"Triton (opt):  {t_new:.4f} ms (std: {statistics.stdev(new_times):.4f})")
    print(f"Speedup: {speedup:.2f}x")
    
    # Check if this is the best result
    is_best = speedup > tracker['best_speedup']
    
    if is_best:
        print(f"\\n*** NEW BEST! {speedup:.2f}x > {tracker['best_speedup']:.2f}x ***")
        tracker['best_speedup'] = speedup
        tracker['best_code'] = target_code
        tracker['best_ref_time'] = t_ref
        tracker['best_opt_time'] = t_new
        tracker['best_attempt'] = attempt
    else:
        print(f"\\nNot best. Current best: {tracker['best_speedup']:.2f}x (Attempt {tracker.get('best_attempt', '?')})")
    
    tracker['attempt'] = attempt
    
    # Save tracker
    with open(TRACKER, 'w') as f:
        json.dump(tracker, f, indent=2)
    
    # Log the attempt
    log_entry = f"""
## Attempt {attempt} - {time.strftime('%Y-%m-%dT%H:%M:%S')}{'  *** BEST ***' if is_best else ''}
Speedup: {speedup:.2f}x
Ref time: {t_ref:.4f} ms
Opt time: {t_new:.4f} ms

### Code Snapshot
\`\`\`python
{target_code}
\`\`\`

"""
    with open(LOG, 'a') as f:
        f.write(log_entry)
    
    print(f"\\nResults logged to: {LOG}")
    print(f"Tracker updated: {TRACKER}")
    
    return speedup

if __name__ == '__main__':
    main()
`
    fs.writeFileSync(testScriptPath, testScript)

    // Create finalize script that writes best result to target
    const finalizeScriptPath = path.join(srcDir, ".kernel_finalize.py")
    const finalizeScript = `#!/usr/bin/env python3
"""
Finalize optimization: write best result to target file.
Usage: python .kernel_finalize.py
"""
import json
import os

TGT = '${targetPath}'
TRACKER = '${bestTrackerFile}'
LOG = '${logFile}'
SRC = '${srcPath}'

def main():
    with open(TRACKER) as f:
        tracker = json.load(f)
    
    if tracker['best_speedup'] == 0:
        print("ERROR: No successful optimization found!")
        return
    
    best_code = tracker['best_code']
    best_speedup = tracker['best_speedup']
    best_ref = tracker.get('best_ref_time', 0)
    best_opt = tracker.get('best_opt_time', 0)
    best_attempt = tracker.get('best_attempt', '?')
    
    # Update header in best code
    lines = best_code.split('\\n')
    new_lines = []
    in_header = False
    header_done = False
    
    for line in lines:
        if line.startswith('# Speedup:') and not header_done:
            new_lines.append(f'# Speedup: {best_speedup:.2f}x')
        elif line.startswith('# Ref time:') and not header_done:
            new_lines.append(f'# Ref time: {best_ref:.4f} ms (PyTorch)')
        elif line.startswith('# Opt time:') and not header_done:
            new_lines.append(f'# Opt time: {best_opt:.4f} ms (Triton)')
            header_done = True
        else:
            new_lines.append(line)
    
    final_code = '\\n'.join(new_lines)
    
    # Write to target
    with open(TGT, 'w') as f:
        f.write(final_code)
    
    # Append final result to log
    log_entry = f"""
# ============================================
# FINAL BEST RESULT
# ============================================
# Best Speedup: {best_speedup:.2f}x (from Attempt {best_attempt})
# Ref time: {best_ref:.4f} ms
# Opt time: {best_opt:.4f} ms
# ============================================

"""
    with open(LOG, 'a') as f:
        f.write(log_entry)
    
    print(f"\\n{'='*50}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*50}")
    print(f"Best Speedup: {best_speedup:.2f}x (Attempt {best_attempt})")
    print(f"Ref time: {best_ref:.4f} ms")
    print(f"Opt time: {best_opt:.4f} ms")
    print(f"\\nBest code written to: {TGT}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
`
    fs.writeFileSync(finalizeScriptPath, finalizeScript)

    // Build goal text
    const goalText = goal
      ? `Achieve ${goal}x speedup. Stop optimization when this goal is reached.`
      : "Optimize for best possible speedup."

    const goalSection = goal
      ? `## Performance Goal
- Target speedup: ${goal}x
- Once you achieve >= ${goal}x speedup with correct accuracy, STOP optimization
- If after multiple attempts you cannot reach the goal, save your BEST result`
      : `## Performance Goal
- Optimize for best possible speedup
- Keep iterating until performance plateaus`

    // Check if target already exists (continue mode)
    const targetExists = fs.existsSync(targetPath)
    const continueSection = targetExists
      ? `
## CONTINUE MODE
The target file already exists! This means you should:
1. Read the EXISTING target file first: \`${targetPath}\`
2. Use it as your starting point for further optimization
3. Try to improve upon the existing implementation
`
      : ""

    // Check if source already uses triton (triton2triton mode)
    const srcUsesTriton = srcContent.includes("@triton.jit") || srcContent.includes("import triton")

    // Build the prompt
    const prompt = `# GPU Kernel Optimization Task

## Files
- **Source**: \`${srcPath}\` (Reference implementation - contains Model class and get_inputs())
- **Target**: \`${targetPath}\` (Optimized implementation - contains ModelNew class)
- **History Log**: \`${logFile}\` (Log all attempts here)

## Mode: ${srcUsesTriton ? "Triton-to-Triton (continue optimizing existing Triton)" : "Torch-to-Triton (convert PyTorch to Triton)"}
${targetExists ? "**CONTINUE MODE**: Target file exists, use it as starting point!" : ""}

**IMPORTANT**: 
- Source file provides \`Model\` class + \`get_inputs()\` as ACCURACY BASELINE
- Target file provides \`ModelNew\` class (your optimized implementation)
- Model and ModelNew must produce identical outputs within tolerance

${goalSection}
${continueSection}

## Task
1. Read the source file to understand the PyTorch implementation (Model class)
2. ${targetExists ? "Read the EXISTING target file as your starting point" : "Implement an optimized Triton kernel"}
3. Check if get_init_inputs() exists - use it for Model/ModelNew initialization
4. Write the implementation to the TARGET file: \`${targetPath}\`
5. **RUN THE TEST SCRIPT**: \`python .kernel_test_runner.py\`
6. The test script AUTOMATICALLY tracks and saves the best result!
7. Iterate until goal is achieved or performance plateaus
8. **FINALIZE**: Run \`python .kernel_finalize.py\` to write best result to target

## AUTOMATED TEST SYSTEM

### Test Script: \`.kernel_test_runner.py\`
This script is pre-configured and handles everything:
- Tests accuracy (exits with error if failed)
- Benchmarks performance (median of 5 rounds)
- **AUTOMATICALLY tracks best result** in \`${bestTrackerFile}\`
- **AUTOMATICALLY logs** each attempt to \`${logFile}\`

**After EVERY code change, run:**
\`\`\`bash
python .kernel_test_runner.py
\`\`\`

The script will print:
- "*** NEW BEST! X.XXx > Y.YYx ***" if this is the best result
- "Not best. Current best: X.XXx" otherwise

### Finalize Script: \`.kernel_finalize.py\`
When optimization is complete, run:
\`\`\`bash
python .kernel_finalize.py
\`\`\`
This writes the BEST code (not last!) to the target file with correct header.

## Target File Header Requirements
The target file MUST include this header (finalize script updates values):
\`\`\`
# ============================================
# Kernel Optimization Result
# ============================================
# Source: ${srcPath}
# Speedup: X.XXx
# Ref time: X.XXXX ms (PyTorch)
# Opt time: X.XXXX ms (Triton)
# ============================================
# Optimization notes:
# - <key optimization techniques used>
# ============================================
\`\`\`

## WORKFLOW (FOLLOW EXACTLY!)

1. Read source file
2. Write initial Triton implementation to target file
3. Run: \`python .kernel_test_runner.py\`
4. If accuracy fails: fix the kernel, goto 3
5. Check speedup, optimize code
6. Run: \`python .kernel_test_runner.py\` 
7. Repeat 5-6 until goal reached or plateau
8. **CRITICAL FINAL STEP**: Run \`python .kernel_finalize.py\`

## Triton Optimization Guide

### STEP 0: Get GPU Information (REQUIRED!)
Before writing any kernel, run this to get actual GPU architecture:
\`\`\`bash
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}'); print(f'Arch: {torch.cuda.get_device_capability()}')"
\`\`\`
This tells you:
- AMD MI300X/MI355X: Use wave size 64, CDNA3 architecture
- NVIDIA A100/H100: Use warp size 32, different optimal configs
Adjust your optimization strategy based on actual hardware!

### STEP 1: Analyze the Problem
Before optimizing, understand the operation:
1. **What type of operation?**
   - Matrix multiplication (GEMM): Focus on tiling, L2 cache reuse
   - Elementwise: Focus on vectorization, memory bandwidth
   - Reduction: Focus on parallel reduction, shared memory
   - Attention: Consider Flash Attention patterns
   
2. **What are the tensor shapes?**
   - Square vs rectangular matrices need different strategies
   - Small vs large tensors: small may not benefit from complex tiling
   - Power-of-2 vs arbitrary dimensions affect boundary handling

3. **Compute-bound or Memory-bound?**
   - Compute-bound: Increase arithmetic intensity, use larger tiles
   - Memory-bound: Optimize memory access patterns, coalescing

### STEP 2: Autotune Strategy (CRITICAL!)
Use @triton.autotune to let Triton find optimal configurations:

\`\`\`python
@triton.autotune(
    configs=[
        # Generate configs with VARYING block sizes
        # Start with powers of 2: 32, 64, 128, 256
        # Vary num_warps: 2, 4, 8
        # Vary num_stages: 1, 2, 3, 4
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8, num_stages=2),
        # ... add 10-20 configs covering the parameter space
    ],
    key=['N'],  # Key dimensions that affect optimal config
)
\`\`\`

**Key Principles:**
- Generate 10-20 diverse configs, not just 2-3
- Cover different block sizes (32 to 256)
- Vary num_warps (2, 4, 8) and num_stages (1, 2, 3, 4)
- Include both conservative and aggressive configs
- Let autotune find what works for YOUR specific problem

### STEP 3: Common Optimization Techniques

**Memory Access:**
- Ensure coalesced memory access (threads access consecutive memory)
- Use \`tl.load\` with appropriate masks for boundary handling
- Consider removing masks when dimensions are divisible by block size

**Computation:**
- Use FP32 accumulation for numerical stability, cast output at the end
- For matrix ops: \`tl.dot(a, b, acc)\` accumulates in acc's dtype
- Consider operation fusion to reduce memory traffic

**Cache Optimization:**
- For 2D operations: consider tile reordering (swizzle patterns)
- Adjust block sizes to fit in L2 cache when beneficial
- Use persistent kernels for small, repeated operations

### STEP 4: Iterative Optimization Process
1. Start with a CORRECT implementation (even if slow)
2. Run autotune with diverse configs
3. Analyze which config wins and why
4. If performance plateaus, try:
   - Different algorithm approach
   - Fusing operations
   - Adjusting for memory vs compute bound
5. Use profiling tools when needed:
   \`\`\`bash
   # PyTorch profiler
   python -c "import torch; from torch.profiler import profile, ProfilerActivity; ..."
   # ROCm profiler (if available)
   rocprof --stats python your_script.py
   \`\`\`

## STRICT KERNEL REQUIREMENTS

### Source File (Model)
- Can use any operators: torch native operators, triton kernels, or custom CUDA kernels
- This is the REFERENCE implementation - do NOT modify it

### Target File (ModelNew)
- **MUST use Triton kernels for ALL compute operations**
- **FORBIDDEN**: Direct calls to torch operators like:
  - torch.matmul, torch.mm, torch.bmm
  - torch.add, torch.mul, torch.sub, torch.div
  - torch.relu, torch.sigmoid, torch.tanh, torch.gelu, torch.silu
  - torch.softmax, torch.layer_norm, torch.batch_norm
  - torch.conv1d, torch.conv2d, torch.conv3d
  - F.linear, F.relu, F.softmax, etc.
  - Any torch.nn.functional operations
- **ALLOWED** in ModelNew:
  - Triton kernels (@triton.jit decorated functions)
  - Data movement: .view(), .reshape(), .contiguous(), .to(), .cuda()
  - Shape operations: .size(), .shape, .stride()
  - Memory allocation: torch.empty(), torch.zeros() (for output buffers only)
  - Indexing and slicing

### Example - WRONG (uses torch.matmul):
\`\`\`python
class ModelNew(nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)  # FORBIDDEN!
\`\`\`

### Example - CORRECT (uses Triton kernel):
\`\`\`python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, ...):
    # Triton implementation
    ...

class ModelNew(nn.Module):
    def forward(self, a, b):
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        matmul_kernel[grid](a, b, c, ...)  # Use Triton kernel!
        return c
\`\`\`

## CRITICAL INSTRUCTIONS
- Source file: ${srcPath}
- Target file: ${targetPath}
- History log: ${logFile}
- ALWAYS write code to ${targetPath}, never to any other file
- ALWAYS log each attempt to ${logFile}
- ALWAYS track the BEST result and ensure it's saved to target
- Use get_init_inputs() for Model/ModelNew initialization if it exists
- ModelNew must accept same __init__ parameters as Model
- **ModelNew MUST use Triton kernels - NO torch operators for compute!**

Start by reading the source file.`

    // Create temporary .opencode config
    const opencodeDir = path.join(srcDir, ".opencode")
    const agentDir = path.join(opencodeDir, "agent")
    fs.mkdirSync(agentDir, { recursive: true })

    // Create agent config
    const agentConfig = `---
model: amd-anthropic/claude-opus-4-5
temperature: 0.3
steps: 50
---

# GPU Kernel Development Expert

You are a professional GPU Kernel development expert, specializing in accelerating PyTorch operators using Triton.

## Core Objectives

1. **Functional Correctness**: ModelNew output must match Model output (fp16: rel error < 1e-2, fp32: < 1e-5)
2. **Performance Optimization**: ${goalText}
3. **Code Quality**: Readable and maintainable Triton Kernel code

${goalSection}

## STRICT KERNEL REQUIREMENTS

### Source File Rules
- Source file can use ANY operators (torch, triton, CUDA)
- This is the REFERENCE - do NOT modify it

### Target File Rules (ModelNew)
- **MUST use @triton.jit decorated Triton kernels for ALL compute**
- **FORBIDDEN in ModelNew**:
  - torch.matmul, torch.mm, torch.bmm, torch.addmm
  - torch.add, torch.mul, torch.sub, torch.div (element-wise)
  - torch.relu, torch.sigmoid, torch.tanh, torch.gelu, torch.silu
  - torch.softmax, torch.layer_norm, torch.batch_norm
  - torch.conv1d, torch.conv2d, torch.conv3d
  - torch.nn.functional.* compute operations
  - Any native torch compute operators
- **ALLOWED in ModelNew**:
  - Triton kernels (@triton.jit)
  - Shape/memory: .view(), .reshape(), .contiguous(), .to()
  - Allocation: torch.empty(), torch.zeros() for outputs
  - Indexing and slicing

## AUTOMATED BEST RESULT TRACKING

**The test system handles tracking automatically!**

After each code change:
1. Run: \`python .kernel_test_runner.py\`
2. Script automatically:
   - Tests accuracy and performance
   - Tracks best result in JSON file
   - Logs each attempt with code snapshot
   - Prints whether this is NEW BEST or not

When done optimizing:
3. Run: \`python .kernel_finalize.py\`
4. This writes the BEST code to target file

**YOU DON'T NEED TO MANUALLY TRACK BEST RESULT!**
Just run the test script after each change, and finalize at the end.

- **VERIFY: ModelNew uses ONLY Triton kernels, NO torch operators!**

## Optimization Workflow

### Phase 1: Understand & Baseline
1. **Get GPU info first**: Run \`python3 -c "import torch; print(torch.cuda.get_device_name())"\`
2. **Analyze the operation**: What type? What shapes? Compute or memory bound?
3. Implement a CORRECT baseline kernel with safe boundary checks
4. Test accuracy before any optimization

### Phase 2: Autotune Exploration
1. Add @triton.autotune with **10-20 diverse configurations**
2. Cover the parameter space:
   - Block sizes: 32, 64, 128, 256 (and combinations for 2D)
   - num_warps: 2, 4, 8
   - num_stages: 1, 2, 3, 4
3. Let autotune find optimal config for YOUR specific problem
4. Analyze winning config to understand what works

### Phase 3: Targeted Optimization
Based on autotune results and problem analysis:
- **If memory-bound**: Optimize access patterns, coalescing, vectorization
- **If compute-bound**: Larger tiles, more arithmetic per memory access
- **If boundaries hurt**: Check if dims are divisible, remove masks if safe
- **If cache misses**: Try tile reordering, different block shapes

### Phase 4: Advanced Techniques (if needed)
- Algorithm changes (e.g., Flash Attention for attention)
- Operation fusion (combine multiple kernels)
- Persistent kernels for small, repeated operations
- Split-K parallelism for very large reduction dimensions

### Key Principles
- **Correctness first**: A fast wrong answer is useless
- **Let autotune work**: Don't over-constrain, explore the space
- **Profile when stuck**: Use torch.profiler or rocprof to find bottlenecks
- **Understand YOUR problem**: Generic tips may not apply to your specific case
`
    fs.writeFileSync(path.join(agentDir, "kernel-dev.md"), agentConfig)

    // Create opencode.jsonc
    const opencodeConfig = `{
  "$schema": "https://opencode.ai/config.json",
  "model": "amd-anthropic/claude-opus-4-5",
  "default_agent": "kernel-dev",
  "provider": {
    "amd-anthropic": {
      "options": {
        "timeout": 600000
      }
    }
  },
  "permission": {
    "bash": "allow",
    "edit": "allow",
    "read": "allow",
    "write": "allow",
    "external_directory": "allow"
  }
}
`
    fs.writeFileSync(path.join(opencodeDir, "opencode.jsonc"), opencodeConfig)

    // Run opencode
    await bootstrap(srcDir, async () => {
      const server = Server.listen({ port: 0, hostname: "127.0.0.1" })
      const sdk = createOpencodeClient({ baseUrl: `http://${server.hostname}:${server.port}` })

      try {
        const sessionResult = await sdk.session.create({})
        const sessionID = sessionResult.data?.id
        if (!sessionID) {
          UI.error("Failed to create session")
          process.exit(1)
        }

        // Subscribe to events
        const events = await sdk.event.subscribe()

        // Event processor
        const eventProcessor = (async () => {
          for await (const event of events.stream) {
            // Print tool use events
            if (event.type === "message.part.updated") {
              const part = event.properties.part
              if (part.sessionID !== sessionID) continue
              if (part.type === "tool" && part.state.status === "completed") {
                const title = part.state.title || JSON.stringify(part.state.input)
                UI.println(UI.Style.TEXT_INFO_BOLD + `|`, UI.Style.TEXT_DIM + ` ${part.tool.padEnd(7)}`, title)
                if (part.tool === "bash" && part.state.output?.trim()) {
                  UI.println()
                  UI.println(part.state.output)
                }
              }
              if (part.type === "text") {
                const textPart = part as any
                if (textPart.state?.done) {
                  UI.println(textPart.state.content)
                }
              }
            }
            if (event.type === "session.idle" || event.type === "session.error") {
              break
            }
          }
        })()

        // Send the prompt
        await sdk.session.prompt({
          path: { id: sessionID },
          body: {
            parts: [{ type: "text", text: prompt }],
          },
        })

        // Wait for completion
        await eventProcessor

        // Auto-run finalize to ensure best result is saved
        UI.println("")
        UI.println("Running finalize script to save best result...")
        const { execSync } = await import("child_process")
        try {
          const output = execSync(`python "${finalizeScriptPath}"`, {
            cwd: srcDir,
            encoding: "utf-8",
            stdio: ["pipe", "pipe", "pipe"],
          })
          UI.println(output)
        } catch (e: any) {
          UI.println("Finalize script output:")
          if (e.stdout) UI.println(e.stdout)
          if (e.stderr) UI.println(e.stderr)
        }
      } finally {
        // Cleanup
        try {
          fs.rmSync(opencodeDir, { recursive: true })
          fs.unlinkSync(testScriptPath)
          fs.unlinkSync(finalizeScriptPath)
        } catch {
          // Ignore cleanup errors
        }
        server.stop()
      }
    })

    UI.println("")
    UI.println("============================================")
    UI.println("Optimization Complete")
    UI.println("============================================")
    UI.println(`Target file: ${targetPath}`)
    UI.println(`History log: ${logFile}`)
    UI.println("============================================")
  },
})

