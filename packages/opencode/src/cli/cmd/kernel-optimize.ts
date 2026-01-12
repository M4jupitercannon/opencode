import { cmd } from "./cmd"
import { bootstrap } from "../bootstrap"
import { Server } from "../../server/server"
import { createOpencodeClient } from "@opencode-ai/sdk"
import * as fs from "fs"
import * as path from "path"
import { UI } from "../ui"
import { Env } from "../../env"

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
    // Check if LLM_GATEWAY_KEY is set
    const gatewayKey = Env.get("LLM_GATEWAY_KEY")
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
5. Test accuracy and performance
6. **Log the result** to: \`${logFile}\`
7. Track the BEST speedup achieved across all attempts
8. Iterate until goal is achieved or performance plateaus
9. **IMPORTANT**: At the end, ensure the TARGET file contains the BEST performing code

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

# Benchmark
for _ in range(10): model_ref(*inputs); model_new(*inputs)
torch.cuda.synchronize()

N = 100
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(N): model_ref(*inputs)
torch.cuda.synchronize(); t_ref = (time.perf_counter() - t0) / N * 1000

torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(N): model_new(*inputs)
torch.cuda.synchronize(); t_new = (time.perf_counter() - t0) / N * 1000

speedup = t_ref / t_new
print(f'=== Performance ===')
print(f'PyTorch (ref): {t_ref:.4f} ms')
print(f'Triton (opt):  {t_new:.4f} ms')
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

## CRITICAL INSTRUCTIONS
- Source file: ${srcPath}
- Target file: ${targetPath}
- History log: ${logFile}
- ALWAYS write code to ${targetPath}, never to any other file
- ALWAYS log each attempt to ${logFile}
- ALWAYS track the BEST result and ensure it's saved to target
- Use get_init_inputs() for Model/ModelNew initialization if it exists
- ModelNew must accept same __init__ parameters as Model

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

## CRITICAL: Track Best Result
- After each successful test, record the speedup
- Keep track of the BEST speedup achieved
- At the end, write the BEST performing code to target
- Log all attempts to the history log file
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

