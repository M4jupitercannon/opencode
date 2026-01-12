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

    // Build the prompt
    const prompt = `# GPU Kernel Optimization Task

## Files
- **Source**: \`${srcPath}\` (PyTorch implementation to optimize)
- **Target**: \`${targetPath}\` (Write your Triton implementation HERE)
- **History Log**: \`${logFile}\` (Log all attempts here)

${goalSection}

## Task
1. Read and understand the PyTorch implementation in the source file
2. Check if get_init_inputs() exists - use it for Model/ModelNew initialization
3. Implement an optimized Triton kernel
4. Write the complete implementation to the TARGET file: \`${targetPath}\`
5. Test accuracy and performance (uses median of 5 rounds for stable measurement)
6. **Log the result** to: \`${logFile}\`
7. **Track BEST speedup**: If current > best, SAVE current code and update best
8. Iterate until goal is achieved or performance plateaus
9. **CRITICAL**: At the end, write the code with HIGHEST speedup to target (NOT the last attempt!)

**Best Result Tracking**:
- Maintain: best_speedup variable and best_code snapshot
- After each test: if speedup > best_speedup, save code to best_code
- Final step: write best_code to target file with correct speedup in header

## Validation Test Script

IMPORTANT: Use get_init_inputs() if it exists to initialize Model and ModelNew!

\`\`\`python
import torch
import time

SRC = '${srcPath}'
TGT = '${targetPath}'

exec(open(SRC).read())
exec(open(TGT).read())

# Check if get_init_inputs exists for Model initialization
try:
    init_inputs = get_init_inputs()
except:
    init_inputs = []

# Initialize models with init_inputs if provided
if init_inputs:
    model_ref = Model(*init_inputs).cuda().eval()
    model_new = ModelNew(*init_inputs).cuda().eval()
else:
    model_ref = Model().cuda().eval()
    model_new = ModelNew().cuda().eval()

# Get forward inputs
inputs = [x.cuda() if hasattr(x, 'cuda') else x for x in get_inputs()]

with torch.no_grad():
    out_ref = model_ref(*inputs)
    out_new = model_new(*inputs)

rtol, atol = (1e-2, 1e-3) if out_ref.dtype == torch.float16 else (1e-5, 1e-6)
max_diff = (out_ref - out_new).abs().max().item()
rel_diff = ((out_ref - out_new).abs() / (out_ref.abs() + 1e-8)).max().item()
is_close = torch.allclose(out_ref, out_new, rtol=rtol, atol=atol)

print(f'=== Accuracy ===')
print(f'Max abs error: {max_diff:.2e}, Max rel error: {rel_diff:.2e}')
print(f'Accuracy: {"PASSED" if is_close else "FAILED"}')

if not is_close:
    print('ERROR: Accuracy test failed!')
    exit(1)

# Benchmark with multiple rounds for stable measurement
import statistics

# Warmup
for _ in range(20): model_ref(*inputs); model_new(*inputs)
torch.cuda.synchronize()

# Run 5 rounds, take median for stability
NUM_ROUNDS = 5
N_PER_ROUND = 100
ref_times = []
new_times = []

for round_idx in range(NUM_ROUNDS):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(N_PER_ROUND): model_ref(*inputs)
    torch.cuda.synchronize(); ref_times.append((time.perf_counter() - t0) / N_PER_ROUND * 1000)
    
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(N_PER_ROUND): model_new(*inputs)
    torch.cuda.synchronize(); new_times.append((time.perf_counter() - t0) / N_PER_ROUND * 1000)

t_ref = statistics.median(ref_times)
t_new = statistics.median(new_times)
speedup = t_ref / t_new

print(f'=== Performance (median of {NUM_ROUNDS} rounds) ===')
print(f'PyTorch (ref): {t_ref:.4f} ms (std: {statistics.stdev(ref_times):.4f})')
print(f'Triton (opt):  {t_new:.4f} ms (std: {statistics.stdev(new_times):.4f})')
print(f'Speedup: {speedup:.2f}x')
\`\`\`

## Target File Header Requirements
The target file MUST include this header with actual measured values:
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

## Logging Results
After EVERY test, append results to the history log:
\`\`\`bash
echo "## Attempt N - $(date -Iseconds)" >> ${logFile}
echo "Speedup: X.XXx" >> ${logFile}
echo "Ref time: X.XX ms" >> ${logFile}
echo "Opt time: X.XX ms" >> ${logFile}
echo "Optimization: <brief description>" >> ${logFile}
echo "" >> ${logFile}
\`\`\`

## Triton Tips (AMD GPU)
- BLOCK_SIZE: multiples of 64 (wave size)
- num_warps: 2-8 based on block size
- Use tl.load with mask for boundaries
- fp16->fp32 for intermediate calculations

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

## CRITICAL: Track Best Result

You MUST maintain a "best result" tracker throughout optimization:

1. **Initialize**: best_speedup = 0, best_code = None
2. **After each test**: 
   - If speedup > best_speedup: save current code as best_code, update best_speedup
   - Log attempt to history file with speedup value
3. **At the end**: 
   - Write best_code (not last code!) to target file
   - Update header with best_speedup value
   - Add "=== BEST RESULT ===" section to history log

**IMPORTANT**: The measurement now uses median of 5 rounds for stability.
Do NOT save the last attempt - save the attempt with HIGHEST speedup!

Example: If you get 2.67x on attempt 6, then 2.62x on attempt 7,
the target file MUST contain the code from attempt 6 (2.67x).

- **VERIFY: ModelNew uses ONLY Triton kernels, NO torch operators!**
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
      } finally {
        // Cleanup
        try {
          fs.rmSync(opencodeDir, { recursive: true })
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

