import { cmd } from "./cmd"
import { bootstrap } from "../bootstrap"
import { Server } from "../../server/server"
import { createOpencodeClient } from "@opencode-ai/sdk/v2"
import * as fs from "fs"
import * as path from "path"
import { UI } from "../ui"
import { Provider } from "../../provider/provider"
import { select } from "@clack/prompts"

export const ModelOptimizeCommand = cmd({
  command: "model-optimize",
  describe: "end-to-end HuggingFace model optimization pipeline",
  builder: (yargs) =>
    yargs
      .option("model", {
        type: "string",
        alias: "m",
        describe: "HuggingFace model name (e.g., Qwen/Qwen3-8B)",
        demandOption: true,
      })
      .option("output", {
        type: "string",
        alias: "o",
        describe: "output directory for optimized model (default: ./model_opt_<model_name>)",
      })
      .option("llm", {
        type: "string",
        describe: "LLM model to use for optimization (e.g., opencode/glm-4.7-free)",
      })
      .option("skip-download", {
        type: "boolean",
        describe: "skip model download if already exists",
        default: false,
      })
      .option("resume", {
        type: "boolean",
        describe: "resume from last completed phase in existing project",
        default: false,
      })
      .option("from-phase", {
        type: "string",
        alias: "f",
        describe: "start from specific phase (demo, profile, problems, optimize, integrate, report)",
      }),
  async handler(args) {
    const llmArg = args.llm as string | undefined

    // Check if LLM_GATEWAY_KEY is set for amd providers
    const gatewayKey = process.env.LLM_GATEWAY_KEY
    const needsGatewayKey = !llmArg || llmArg.startsWith("amd-")
    if (needsGatewayKey && !gatewayKey) {
      UI.error("LLM_GATEWAY_KEY environment variable is not set")
      UI.println("Please set it with: export LLM_GATEWAY_KEY=your-api-key")
      UI.println("")
      UI.println("Or use a free model that doesn't require a key:")
      UI.println("  opencode model-optimize -m Qwen/Qwen3-8B --llm opencode/glm-4.7-free")
      process.exit(1)
    }

    const hfModel = args.model as string
    const modelName = hfModel.split("/").pop() || hfModel
    const modelNameSafe = modelName.replace(/[^a-zA-Z0-9_-]/g, "_")

    // Output directory
    const outputDir =
      (args.output as string) || path.resolve(`./model_opt_${modelNameSafe}`)

    UI.println("============================================")
    UI.println("Model Optimization Pipeline")
    UI.println("============================================")
    UI.println(`HuggingFace Model: ${hfModel}`)
    UI.println(`Output Directory:  ${outputDir}`)
    if (llmArg) {
      UI.println(`LLM Model:         ${llmArg}`)
    }
    UI.println("============================================")
    UI.println("")

    // Create output directory
    fs.mkdirSync(outputDir, { recursive: true })

    // Create subdirectories
    const dirs = {
      model: path.join(outputDir, "model"),
      demo: path.join(outputDir, "demo"),
      profile: path.join(outputDir, "profile"),
      problems: path.join(outputDir, "problems"),
      optimized: path.join(outputDir, "optimized"),
      report: path.join(outputDir, "report"),
    }
    Object.values(dirs).forEach((d) => fs.mkdirSync(d, { recursive: true }))

    // Handle resume / from-phase options
    const resumeMode = args.resume as boolean
    const fromPhase = args["from-phase"] as string | undefined
    const progressFile = path.join(outputDir, "progress.json")
    
    let existingProgress: any = null
    let startPhase = "download"
    
    if (resumeMode || fromPhase) {
      // Check for existing progress
      if (fs.existsSync(progressFile)) {
        try {
          existingProgress = JSON.parse(fs.readFileSync(progressFile, "utf-8"))
          UI.println(UI.Style.TEXT_INFO_BOLD + "Found existing project progress")
          
          if (fromPhase) {
            // Validate phase name
            const validPhases = ["env", "download", "demo", "compatibility", "profile", "problems", "optimize", "integrate", "report"]
            if (!validPhases.includes(fromPhase)) {
              UI.error(`Invalid phase: ${fromPhase}. Valid phases: ${validPhases.join(", ")}`)
              process.exit(1)
            }
            startPhase = fromPhase
            UI.println(`Starting from phase: ${startPhase}`)
          } else if (resumeMode && existingProgress.phases_completed) {
            // Resume from last completed phase
            const phasesOrder = ["env", "download", "demo", "compatibility", "profile", "problems", "optimize", "integrate", "report"]
            const completed = existingProgress.phases_completed as string[]
            for (let i = phasesOrder.length - 1; i >= 0; i--) {
              if (completed.includes(phasesOrder[i])) {
                startPhase = phasesOrder[i + 1] || "report"
                break
              }
            }
            UI.println(`Resuming from phase: ${startPhase}`)
          }
        } catch (e) {
          UI.println(UI.Style.TEXT_WARNING + "Could not parse existing progress.json, starting fresh")
        }
      } else {
        UI.println(UI.Style.TEXT_WARNING + "No existing progress.json found, starting from beginning")
      }
    }

    // Create config file for the agent
    const configFile = path.join(outputDir, "config.json")
    const config = {
      hf_model: hfModel,
      model_name: modelName,
      dirs: dirs,
      created: new Date().toISOString(),
      skip_download: args["skip-download"],
      start_phase: startPhase,
      resume_mode: resumeMode || !!fromPhase,
    }
    fs.writeFileSync(configFile, JSON.stringify(config, null, 2))

    // Create or update progress tracker
    const progress = existingProgress || {
      phase: "init",
      phases_completed: [] as string[],
      current_step: "",
      errors: [] as string[],
      optimizations: [] as { kernel: string; speedup: number }[],
      final_speedup: 0,
    }
    if (!existingProgress) {
      fs.writeFileSync(progressFile, JSON.stringify(progress, null, 2))
    }

    // Create the main prompt for the agent
    const prompt = buildAgentPrompt(hfModel, modelName, outputDir, dirs, args["skip-download"] as boolean, startPhase, existingProgress)

    // Create temporary .opencode config
    const opencodeDir = path.join(outputDir, ".opencode")
    const agentDir = path.join(opencodeDir, "agent")
    fs.mkdirSync(agentDir, { recursive: true })

    // Create agent config
    const agentConfig = buildAgentConfig(hfModel)
    fs.writeFileSync(path.join(agentDir, "model-opt.md"), agentConfig)

    // Create opencode.jsonc - use claude-opus-4-5 which is available
    const opencodeConfig = `{
  "$schema": "https://opencode.ai/config.json",
  "model": "amd-anthropic/claude-opus-4-5",
  "default_agent": "model-opt",
  "provider": {
    "amd-anthropic": {
      "options": {
        "timeout": 1200000
      }
    }
  },
  "permission": {
    "*": "allow",
    "bash": "allow",
    "edit": {
      "*": "allow",
      "/opt/*": "deny",
      "/usr/*": "deny"
    },
    "read": "allow",
    "write": {
      "*": "allow",
      "/opt/*": "deny",
      "/usr/*": "deny"
    },
    "glob": "allow",
    "grep": "allow",
    "list": "allow",
    "task": "allow",
    "external_directory": "allow",
    "todowrite": "allow",
    "todoread": "allow",
    "question": "allow",
    "webfetch": "allow",
    "websearch": "allow",
    "codesearch": "allow",
    "lsp": "allow",
    "doom_loop": "allow"
  }
}
`
    fs.writeFileSync(path.join(opencodeDir, "opencode.jsonc"), opencodeConfig)

    // Run opencode
    await bootstrap(outputDir, async () => {
      const server = Server.listen({ port: 0, hostname: "127.0.0.1" })
      const sdk = createOpencodeClient({ baseUrl: `http://${server.hostname}:${server.port}` })

      try {
        const sessionResult = await sdk.session.create()
        const sessionID = sessionResult.data?.id
        if (!sessionID) {
          UI.error("Failed to create session")
          process.exit(1)
        }

        // Subscribe to events
        const events = await sdk.event.subscribe()
        UI.println("Session created, sending prompt...")

        // Create detailed log file
        const logFilePath = path.join(outputDir, "optimization.log")
        const logStream = fs.createWriteStream(logFilePath, { flags: "a" })
        const log = (msg: string) => {
          const timestamp = new Date().toISOString()
          logStream.write(`[${timestamp}] ${msg}\n`)
        }
        log("=" .repeat(60))
        log(`Model Optimization Started: ${modelName}`)
        log(`Output Directory: ${outputDir}`)
        log(`LLM Model: ${llmArg || "default"}`)
        log("=" .repeat(60))
        UI.println(UI.Style.TEXT_DIM + `Detailed log: ${logFilePath}`)

        // Event processor with improved logging
        let currentPhase = ""
        const eventProcessor = (async () => {
          for await (const event of events.stream) {
            // Only log meaningful events to file (skip raw JSON noise)
            if (event.type === "message.part.updated") {
              const part = event.properties.part
              if (part.sessionID !== sessionID) continue
              
              // Log agent's thinking/text to file
              if (part.type === "text") {
                const textPart = part as any
                if (textPart.state?.done && textPart.state.content?.trim()) {
                  log(`\n[AGENT THINKING]\n${textPart.state.content.trim()}\n`)
                }
              }
              
              if (part.type === "tool" && part.state.status === "completed") {
                const tool = part.tool
                const title = part.state.title || ""
                const input = (part.state.input || {}) as Record<string, any>
                
                // Format based on tool type for better readability
                if (tool === "bash") {
                  const cmd = input.command || title
                  log(`\n$ ${cmd}`)
                  UI.println(UI.Style.TEXT_INFO_BOLD + "$ " + UI.Style.TEXT_DIM + title)
                  if (part.state.output?.trim()) {
                    const output = part.state.output.trim()
                    // Log full output to file, but truncate if very long
                    if (output.length > 2000) {
                      log(`${output.slice(0, 2000)}\n... (truncated, ${output.length} chars total)`)
                    } else {
                      log(output)
                    }
                    // Only show first few lines on console
                    const lines = output.split("\n")
                    if (lines.length > 10) {
                      UI.println(lines.slice(0, 8).join("\n"))
                      UI.println(UI.Style.TEXT_DIM + `... (${lines.length - 8} more lines)`)
                    } else {
                      UI.println(output)
                    }
                  }
                } else if (tool === "write" || tool === "edit") {
                  const filePath = input.target_file || input.file_path || title
                  const shortPath = filePath.replace(outputDir + "/", "")
                  log(`\n[FILE ${tool.toUpperCase()}] ${shortPath}`)
                  UI.println(UI.Style.TEXT_SUCCESS + `✎ ${tool === "write" ? "Creating" : "Editing"}: ` + UI.Style.TEXT_DIM + shortPath)
                } else if (tool === "read") {
                  // Skip read logs - too noisy
                } else if (tool === "todowrite") {
                  // Parse todo updates to show progress
                  const todos = input.todos || []
                  const inProgress = todos.filter((t: any) => t.status === "in_progress")
                  const completed = todos.filter((t: any) => t.status === "completed")
                  if (inProgress.length > 0) {
                    UI.println(UI.Style.TEXT_INFO + `▶ In Progress: ` + inProgress.map((t: any) => t.content).join(", "))
                  }
                  if (completed.length > 0) {
                    UI.println(UI.Style.TEXT_SUCCESS + `✓ Completed: ` + completed.map((t: any) => t.content).join(", "))
                  }
                } else {
                  // Other tools - show if non-empty title
                  if (title) {
                    UI.println(UI.Style.TEXT_DIM + `[${tool}] ${title}`)
                  }
                }
              }
              // Phase detection is handled in the text logging above
              if (part.type === "text") {
                const textPart = part as any
                if (textPart.state?.done && textPart.state.content?.trim()) {
                  // Detect phase changes from agent text for console output
                  const content = textPart.state.content
                  if (content.includes("Phase 0") || content.includes("Environment Setup")) {
                    if (currentPhase !== "env") {
                      currentPhase = "env"
                      log("\n" + "=".repeat(50) + "\n  Phase 0: Environment Setup\n" + "=".repeat(50))
                      UI.println()
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "  Phase 0: Environment Setup")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                    }
                  } else if (content.includes("Phase 1") || content.includes("Model Download")) {
                    if (currentPhase !== "download") {
                      currentPhase = "download"
                      UI.println()
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "  Phase 1: Model Download")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                    }
                  } else if (content.includes("Phase 2") || content.includes("Demo Script")) {
                    if (currentPhase !== "demo") {
                      currentPhase = "demo"
                      UI.println()
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "  Phase 2: Generate Demo Script")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                    }
                  } else if (content.includes("Phase 3") || content.includes("Compatibility")) {
                    if (currentPhase !== "compatibility") {
                      currentPhase = "compatibility"
                      UI.println()
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "  Phase 3: Fix Compatibility Issues")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                    }
                  } else if (content.includes("Phase 4") || content.includes("Profiling")) {
                    if (currentPhase !== "profile") {
                      currentPhase = "profile"
                      UI.println()
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "  Phase 4: Performance Profiling")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                    }
                  } else if (content.includes("Phase 5") || content.includes("Problem Files")) {
                    if (currentPhase !== "problems") {
                      currentPhase = "problems"
                      UI.println()
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "  Phase 5: Generate Problem Files")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                    }
                  } else if (content.includes("Phase 6") || content.includes("Kernel Optimization")) {
                    if (currentPhase !== "optimize") {
                      currentPhase = "optimize"
                      UI.println()
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "  Phase 6: Kernel Optimization")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                    }
                  } else if (content.includes("Phase 7") || content.includes("Integration")) {
                    if (currentPhase !== "integrate") {
                      currentPhase = "integrate"
                      UI.println()
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "  Phase 7: Integration & Testing")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                    }
                  } else if (content.includes("Phase 8") || content.includes("Final Report")) {
                    if (currentPhase !== "report") {
                      currentPhase = "report"
                      UI.println()
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "  Phase 8: Generate Final Report")
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                    }
                  }
                }
              }
            }
            if (event.type === "session.error") {
              UI.error(`Session error: ${JSON.stringify(event.properties || event)}`)
              break
            }
            if (event.type === "session.idle") {
              break
            }
            // Handle permission requests
            if (event.type === "permission.asked") {
              const permission = event.properties as any
              if (permission.sessionID !== sessionID) continue
              
              const permType = permission.permission || ""
              const patterns = (permission.patterns || []).join(", ")
              
              // Auto-approve read operations, bash, and external directory access
              // Only write/edit operations to system paths should require confirmation
              const isAutoApprove = ["read", "external_directory", "glob", "grep", "list", "codesearch", "lsp", "bash", "task", "todowrite", "todoread", "webfetch", "websearch", "question"].includes(permType)
              
              if (isAutoApprove) {
                // Auto-approve read operations
                UI.println(UI.Style.TEXT_DIM + `[Auto-approved: ${permType}] ${patterns}`)
                await sdk.permission.respond({
                  sessionID,
                  permissionID: permission.id,
                  response: "always",
                })
              } else {
                // Prompt for write/edit operations
                UI.println()
                UI.println(UI.Style.TEXT_WARNING_BOLD + "⚠ Permission required:")
                UI.println(`  Type: ${permType}`)
                UI.println(`  Patterns: ${patterns}`)
                const result = await select({
                  message: `Allow this action?`,
                  options: [
                    { value: "once", label: "Allow once" },
                    { value: "always", label: `Always allow: ${(permission.always || []).join(", ")}` },
                    { value: "reject", label: "Reject" },
                  ],
                  initialValue: "once",
                }).catch(() => "reject")
                const response = (result.toString().includes("cancel") ? "reject" : result) as "once" | "always" | "reject"
                await sdk.permission.respond({
                  sessionID,
                  permissionID: permission.id,
                  response,
                })
              }
            }
          }
        })()

        // Send the prompt
        // If using AMD gateway (LLM_GATEWAY_KEY set) and no --llm specified, default to claude-opus-4-5
        let modelParam
        if (llmArg) {
          modelParam = Provider.parseModel(llmArg)
        } else if (gatewayKey) {
          // Default to claude-opus-4-5 for AMD gateway
          modelParam = Provider.parseModel("amd-anthropic/claude-opus-4-5")
          UI.println(`Using default AMD gateway model: amd-anthropic/claude-opus-4-5`)
        }
        UI.println(`Sending prompt to LLM...`)
        await sdk.session.prompt({
          sessionID,
          model: modelParam,
          parts: [{ type: "text", text: prompt }],
        })
        UI.println("Prompt sent, waiting for completion...")

        // Wait for completion
        await eventProcessor
        log("Optimization completed")
        logStream.end()
        UI.println("Event processor completed")
        UI.println(UI.Style.TEXT_SUCCESS + `Full log saved to: ${logFilePath}`)
      } finally {
        // Cleanup opencode config (keep other files)
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
    UI.println("Model Optimization Pipeline Complete")
    UI.println("============================================")
    UI.println(`Output directory: ${outputDir}`)
    UI.println(`Report: ${path.join(dirs.report, "optimization_report.md")}`)
    UI.println("============================================")
  },
})

function buildAgentPrompt(
  hfModel: string,
  modelName: string,
  outputDir: string,
  dirs: Record<string, string>,
  skipDownload: boolean,
  startPhase: string = "download",
  existingProgress: any = null
): string {
  // Build resume context if applicable
  const resumeContext = startPhase !== "download" ? `
## RESUME MODE ACTIVE
**Starting from Phase: ${startPhase}**
${existingProgress ? `
### Previous Progress
- Phases completed: ${existingProgress.phases_completed?.join(", ") || "none"}
- Previous optimizations: ${JSON.stringify(existingProgress.details?.optimization?.kernels_optimized || [], null, 2)}
` : ""}

**IMPORTANT**: Skip phases before "${startPhase}" - their artifacts already exist.
Review existing files before proceeding to understand current state.

---
` : ""

  return `# End-to-End Model Optimization Pipeline

## Target Model
- **HuggingFace Model**: ${hfModel}
- **Model Name**: ${modelName}
${resumeContext}
## Output Directory Structure
\`\`\`
${outputDir}/
├── venv/           # Project-specific Python virtual environment
├── model/          # Downloaded model files
├── demo/           # Demo scripts for running the model
├── profile/        # Profiling results
├── problems/       # Kernel problems for optimization
├── optimized/      # Optimized kernels
├── report/         # Final optimization report
├── config.json     # Configuration
└── progress.json   # Progress tracking
\`\`\`

## ⚠️ CRITICAL: Use Project venv for ALL Python Operations
- **ALWAYS activate venv before running Python**: \`source ${outputDir}/venv/bin/activate\`
- **NEVER modify system Python packages** in /opt/, /usr/, or site-packages/
- **Install all dependencies in project venv** - this allows safe modifications

## 🎯 CRITICAL PRINCIPLE: Data-Driven Decisions Only

**ALL optimization decisions MUST be based on actual profiling data, NOT assumptions or hardcoded rules.**

- ✅ **DO**: Read shapes from \`inference_shapes.json\`, analyze \`bottlenecks.json\`
- ✅ **DO**: Benchmark each optimization at actual inference shapes before applying
- ✅ **DO**: Skip optimizations that don't show improvement - no explanation needed
- ❌ **DON'T**: Use hardcoded shape values like "batch=8" or "hidden=4096"
- ❌ **DON'T**: Add comments like "DISABLED for ModelX" - just don't apply it
- ❌ **DON'T**: Assume any optimization will help without measuring

**Code Quality**: Generate clean, reusable code without model-specific hacks.

## IMPORTANT FILES
- **Config**: ${path.join(outputDir, "config.json")}
- **Progress**: ${path.join(outputDir, "progress.json")}

Update progress.json after completing each phase!

## YOUR TASK: Complete phases starting from "${startPhase}"

---

# Phase 0: Environment Setup ${startPhase !== "download" && startPhase !== "env" ? "[SKIP - ALREADY DONE]" : ""}

## Goal
Create an isolated Python virtual environment for the project with all required dependencies.

## Steps

### 1. Detect ROCm Version
\`\`\`bash
# Get ROCm version from rocminfo or /opt/rocm/.info/version
ROCM_VERSION=$(cat /opt/rocm/.info/version 2>/dev/null | head -1 | cut -d'-' -f1 || echo "6.0")
ROCM_MAJOR=$(echo $ROCM_VERSION | cut -d'.' -f1)
ROCM_MINOR=$(echo $ROCM_VERSION | cut -d'.' -f2)
echo "Detected ROCm version: $ROCM_VERSION (major=$ROCM_MAJOR, minor=$ROCM_MINOR)"
\`\`\`

### 2. Create venv with system site-packages access (FAST - no install needed!)
\`\`\`bash
cd ${outputDir}

# Create venv with --system-site-packages to access system torch/triton directly
if [ ! -d "venv" ]; then
  python3 -m venv venv --system-site-packages
  echo "Created venv with system site-packages access"
fi

source venv/bin/activate

# Verify system packages are accessible
python3 -c "import torch; print(f'PyTorch {torch.__version__} available')"
python3 -c "import triton; print('Triton available')"
\`\`\`

### 3. Install Only Missing Small Packages (if needed)
\`\`\`bash
source ${outputDir}/venv/bin/activate
# Most packages should be available from system. Only install if missing:
python3 -c "import transformers" 2>/dev/null || pip install transformers
python3 -c "import diffusers" 2>/dev/null || pip install diffusers
python3 -c "import accelerate" 2>/dev/null || pip install accelerate
\`\`\`

### 5. Verify Installation
\`\`\`bash
source ${outputDir}/venv/bin/activate
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
\`\`\`

### 6. Update progress.json
Update progress.json: phase="env", phases_completed.append("env")

---

# Phase 1: Model Download ${skipDownload ? "(SKIP if exists)" : ""} ${startPhase !== "download" ? "[SKIP - ALREADY DONE]" : ""}

## Goal
Download the HuggingFace model to \`${dirs.model}\`

## Steps
1. Check if model already exists (look for config.json in model dir)
2. ${skipDownload ? "If --skip-download is set AND model exists, skip to Phase 2" : "Download using huggingface_hub or transformers"}

\`\`\`python
from huggingface_hub import snapshot_download
# OR
from transformers import AutoModel, AutoTokenizer
\`\`\`

3. Update progress.json: phase="download", phases_completed.append("download")

---

# Phase 2: Generate Demo Script ${startPhase === "download" || startPhase === "env" ? "" : (startPhase === "demo" ? "" : "[SKIP - ALREADY DONE]")}

## Goal
Create a working demo script that runs inference on the model.

## IMPORTANT: Always Activate venv
\`\`\`bash
source ${outputDir}/venv/bin/activate
\`\`\`

## Steps
1. **Detect model type** by reading model's config.json:
   - Check \`model_type\` field (e.g., "llama", "qwen2", "stable-diffusion", etc.)
   - Check \`architectures\` field
   - Check task in model card (text-generation, text2image, etc.)

2. **Generate appropriate demo script** in \`${dirs.demo}/demo.py\`:

### For Text Generation Models (Qwen, Llama, etc.)
\`\`\`python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "${dirs.model}"

def run_inference():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True
    )
    
    prompt = "Hello, I am"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    
    print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    run_inference()
\`\`\`

### For Image Generation Models
\`\`\`python
import torch
from diffusers import DiffusionPipeline

MODEL_PATH = "${dirs.model}"

def run_inference():
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16
    ).to("cuda")
    
    prompt = "A cat sitting on a couch"
    image = pipe(prompt).images[0]
    image.save("output.png")

if __name__ == "__main__":
    run_inference()
\`\`\`

3. **Test the demo script**:
\`\`\`bash
cd ${dirs.demo}
python demo.py
\`\`\`

4. Update progress.json

---

# Phase 3: Fix Compatibility Issues ${["env", "download", "demo"].includes(startPhase) ? "" : (startPhase === "compatibility" ? "" : "[SKIP - ALREADY DONE]")}

## Goal
If demo.py fails, diagnose and fix issues using monkey-patching.

## IMPORTANT: Always Activate venv
\`\`\`bash
source ${outputDir}/venv/bin/activate
\`\`\`

## ⚠️ Fixing Dependencies
Since we have our OWN venv, we CAN safely modify packages within it:
\`\`\`bash
# Safe to do in project venv:
pip install some-missing-package
pip install --upgrade diffusers
pip install git+https://github.com/xxx/fix.git

# STILL NEVER modify /opt/ or /usr/ system directories!
\`\`\`

## Common Issues & Fixes

### Issue: Missing custom operators
\`\`\`python
# Create patches/custom_ops.py
import torch

def my_custom_op(x):
    # Implement missing op
    return x

# Monkey patch
import transformers.models.xxx as module
module.custom_op = my_custom_op
\`\`\`

### Issue: Unsupported attention implementation
\`\`\`python
# patches/attention_fix.py
def patched_attention(self, query, key, value, **kwargs):
    # Fallback implementation
    return torch.nn.functional.scaled_dot_product_attention(query, key, value)

ModelClass._attention = patched_attention
\`\`\`

### Issue: RoPE/positional encoding errors
\`\`\`python
# patches/rope_fix.py
def fixed_rope(x, seq_len):
    # Fixed implementation
    ...
\`\`\`

## Steps
1. Run demo.py and capture the error
2. Analyze the error traceback
3. Create fix in \`${dirs.demo}/patches/\` directory
4. Update demo.py to import patches first
5. Re-run demo.py until it works
6. Update progress.json

---

# Phase 4: Performance Profiling ${["env", "download", "demo", "compatibility"].includes(startPhase) ? "" : (startPhase === "profile" ? "" : "[SKIP - ALREADY DONE]")}

## Goal
Profile the model to identify bottleneck operators/kernels.

## ⚠️ CRITICAL: Keep Profiling Lightweight
- **For image/video generation models**: Use ONLY 2-3 diffusion steps (NOT the full 20-50 steps)
- **For LLMs**: Generate only 10-20 tokens
- **Trace file should be < 100MB** - if larger, reduce steps/tokens
- Skip warmup iterations in profiler schedule to avoid huge traces
- Focus on capturing operator patterns, not full inference

## ⚠️ CRITICAL: Use rocprof for GPU profiling on ROCm
**PyTorch's profiler.key_averages().table() does NOT show CUDA/GPU times on ROCm!**
You MUST use rocprof for accurate GPU kernel timing.

## Create Profiling Script: \`${dirs.profile}/profile_model.py\`

\`\`\`python
"""
Profile model using rocprof for accurate GPU kernel times on ROCm.
PyTorch's profiler table() doesn't show GPU times on ROCm - only CPU times!
"""
import torch
import json
import os
import sys
import time
import subprocess
import csv

sys.path.insert(0, "${dirs.demo}")

# Import patches from demo
try:
    from patches import apply_all_patches
    apply_all_patches()
except ImportError:
    pass

PROFILE_DIR = "${dirs.profile}"
DEMO_DIR = "${dirs.demo}"
SEED = 42
NUM_PROFILE_STEPS = 3  # Keep small for diffusion models!

def get_gpu_time(func, *args, **kwargs):
    """Measure GPU execution time with proper synchronization."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = func(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return result, (end - start) * 1000  # ms

def run_rocprof():
    """Run rocprof to get actual GPU kernel times."""
    rocprof_output = os.path.join(PROFILE_DIR, "rocprof_results.csv")
    demo_script = os.path.join(DEMO_DIR, "demo.py")
    
    # Create a minimal profiling script that runs the demo with limited steps
    profile_script = os.path.join(PROFILE_DIR, "rocprof_run.py")
    with open(profile_script, "w") as f:
        f.write(f'''
import sys
sys.path.insert(0, "{DEMO_DIR}")
try:
    from patches import apply_all_patches
    apply_all_patches()
except ImportError:
    pass

# Import and run demo with minimal steps
exec(open("{demo_script}").read())
''')
    
    try:
        cmd = ["rocprof", "--stats", "-o", rocprof_output, sys.executable, profile_script]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            stats_file = rocprof_output.replace(".csv", ".stats.csv")
            if os.path.exists(stats_file):
                return parse_rocprof_stats(stats_file)
        else:
            print(f"rocprof error: {result.stderr[:500]}")
    except FileNotFoundError:
        print("rocprof not found, using manual timing fallback")
    except subprocess.TimeoutExpired:
        print("rocprof timed out")
    
    return None

def parse_rocprof_stats(stats_file):
    """Parse rocprof stats CSV and generate bottlenecks."""
    print(f"\\n=== Parsing GPU kernel stats from {stats_file} ===")
    
    kernels = []
    with open(stats_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get('Name', row.get('KernelName', ''))
            time_ns = float(row.get('TotalDurationNs', row.get('DurationNs', 0)))
            count = int(row.get('Calls', row.get('Count', 1)))
            kernels.append({
                'name': name,
                'total_ms': time_ns / 1e6,
                'count': count,
                'avg_ms': (time_ns / 1e6) / count if count > 0 else 0
            })
    
    kernels = sorted(kernels, key=lambda x: x['total_ms'], reverse=True)
    total_gpu_time = sum(k['total_ms'] for k in kernels)
    
    # Aggregate by kernel type for bottleneck analysis
    bottlenecks = []
    gemm_time = sum(k['total_ms'] for k in kernels if 'Cijk_' in k['name'])
    attn_time = sum(k['total_ms'] for k in kernels if 'attn' in k['name'].lower())
    elem_time = sum(k['total_ms'] for k in kernels if 'elementwise' in k['name'] or 'vectorized' in k['name'])
    reduce_time = sum(k['total_ms'] for k in kernels if 'reduce' in k['name'])
    mem_time = sum(k['total_ms'] for k in kernels if 'Cat' in k['name'] or 'copy' in k['name'].lower())
    
    bottlenecks = [
        {"name": "GEMM (Cijk_* kernels)", "cuda_time_ms": gemm_time, 
         "cuda_time_percent": gemm_time/total_gpu_time*100 if total_gpu_time > 0 else 0,
         "count": sum(k['count'] for k in kernels if 'Cijk_' in k['name']),
         "input_shapes": "[batch, seq, hidden] x [hidden, hidden]",
         "optimizable": False, "reason": "Already optimized by rocBLAS/Tensile"},
        {"name": "Attention (attn_fwd)", "cuda_time_ms": attn_time,
         "cuda_time_percent": attn_time/total_gpu_time*100 if total_gpu_time > 0 else 0,
         "count": sum(k['count'] for k in kernels if 'attn' in k['name'].lower()),
         "input_shapes": "[batch, heads, seq, head_dim]",
         "optimizable": True, "reason": "Can use AITER Flash Attention"},
        {"name": "Elementwise operations", "cuda_time_ms": elem_time,
         "cuda_time_percent": elem_time/total_gpu_time*100 if total_gpu_time > 0 else 0,
         "count": sum(k['count'] for k in kernels if 'elementwise' in k['name'] or 'vectorized' in k['name']),
         "input_shapes": "various",
         "optimizable": True, "reason": "Can be fused using Triton"},
        {"name": "Reduce operations (LayerNorm)", "cuda_time_ms": reduce_time,
         "cuda_time_percent": reduce_time/total_gpu_time*100 if total_gpu_time > 0 else 0,
         "count": sum(k['count'] for k in kernels if 'reduce' in k['name']),
         "input_shapes": "[batch, seq, hidden]",
         "optimizable": True, "reason": "Can be fused with residual add"},
        {"name": "Memory operations (copy, concat)", "cuda_time_ms": mem_time,
         "cuda_time_percent": mem_time/total_gpu_time*100 if total_gpu_time > 0 else 0,
         "count": sum(k['count'] for k in kernels if 'Cat' in k['name'] or 'copy' in k['name'].lower()),
         "input_shapes": "various",
         "optimizable": False, "reason": "Memory bandwidth limited"},
    ]
    
    # Sort by time
    bottlenecks = sorted(bottlenecks, key=lambda x: x['cuda_time_ms'], reverse=True)
    
    print(f"\\nTotal GPU time: {total_gpu_time:.2f}ms")
    print("\\n=== Bottleneck Analysis ===")
    for i, b in enumerate(bottlenecks, 1):
        opt = "✓" if b.get("optimizable") else "✗"
        print(f"{i}. {b['name'][:40]:40s} {b['cuda_time_percent']:5.1f}% ({b['cuda_time_ms']:.1f}ms) {opt}")
    
    # Save results
    with open(os.path.join(PROFILE_DIR, "bottlenecks.json"), "w") as f:
        json.dump(bottlenecks, f, indent=2)
    
    # Also save raw kernel data for detailed analysis
    with open(os.path.join(PROFILE_DIR, "gpu_kernels.json"), "w") as f:
        json.dump(kernels[:50], f, indent=2)
    
    return bottlenecks

if __name__ == "__main__":
    bottlenecks = run_rocprof()
    if bottlenecks is None:
        print("Failed to get GPU profiling data. Check if rocprof is installed.")
\`\`\`

## Steps
1. Create and run the profiling script
2. Analyze bottlenecks.json to identify top time-consuming operators
3. Focus on operators that take > 5% of total CUDA time
4. Update progress.json with bottleneck list

---

# Phase 5: Generate Problem Files for Kernel Optimization ${["env", "download", "demo", "compatibility", "profile"].includes(startPhase) ? "" : "[SKIP - ALREADY DONE]"}

## Goal
Convert bottleneck operators into Problem files for kernel-optimize.
**IMPORTANT**: Analyze operators for fusion opportunities BEFORE creating individual problem files.

## ⚠️ CRITICAL: Capture Dynamic Shape Ranges

**PROBLEM**: Kernel speedups at fixed shapes often DON'T translate to actual inference which has VARIABLE shapes.

**SOLUTION**: Use lightweight hooks to capture shape ranges during actual inference, then optimize for the RANGE.

### Create Shape Capture System: \`${dirs.profile}/shape_capture.py\`

\`\`\`python
"""
Dynamic Shape Capture System
Hooks into PyTorch operators to capture actual shapes during inference.
Generates shape_ranges.json with min/typical/max for each dimension.
"""
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
import json
import numpy as np

class ShapeCapture:
    """Lightweight hook system to capture operator shapes during inference."""
    
    def __init__(self):
        self.shape_records: Dict[str, List[Dict]] = defaultdict(list)
        self.hooks = []
        self.op_counts: Dict[str, int] = defaultdict(int)
        
    def _create_hook(self, name: str, op_type: str):
        """Create a forward hook that records shapes."""
        def hook(module, inputs, output):
            record = {
                "op_type": op_type,
                "input_shapes": [],
                "output_shape": None,
                "dtype": None
            }
            
            # Capture input shapes
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    record["input_shapes"].append(list(inp.shape))
                    if record["dtype"] is None:
                        record["dtype"] = str(inp.dtype)
                elif inp is None:
                    record["input_shapes"].append(None)
            
            # Capture output shape
            if isinstance(output, torch.Tensor):
                record["output_shape"] = list(output.shape)
            elif isinstance(output, tuple) and len(output) > 0:
                if isinstance(output[0], torch.Tensor):
                    record["output_shape"] = list(output[0].shape)
            
            self.shape_records[name].append(record)
            self.op_counts[name] += 1
            
        return hook
    
    def register_hooks(self, model: nn.Module, target_ops: Optional[List[str]] = None):
        """
        Register hooks on model modules.
        
        Args:
            model: PyTorch model
            target_ops: List of op types to capture (e.g., ["LayerNorm", "Linear", "Attention"])
                       If None, captures common ops
        """
        if target_ops is None:
            target_ops = ["LayerNorm", "RMSNorm", "Linear", "Attention", "Conv", "Embedding"]
        
        for name, module in model.named_modules():
            class_name = module.__class__.__name__
            for op in target_ops:
                if op in class_name:
                    hook = module.register_forward_hook(self._create_hook(name, class_name))
                    self.hooks.append(hook)
                    break
        
        print(f"Registered {len(self.hooks)} shape capture hooks")
        return self
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def compute_shape_ranges(self) -> Dict[str, Any]:
        """
        Compute shape ranges (min/typical/max) for each operator.
        
        Returns:
            Dict with shape ranges for each captured operator
        """
        ranges = {}
        
        for name, records in self.shape_records.items():
            if not records:
                continue
                
            op_type = records[0]["op_type"]
            dtype = records[0]["dtype"]
            
            # Collect all input shapes
            input_shapes_list = [r["input_shapes"] for r in records if r["input_shapes"]]
            output_shapes_list = [r["output_shape"] for r in records if r["output_shape"]]
            
            if not input_shapes_list:
                continue
            
            # Compute ranges for each input
            input_ranges = []
            num_inputs = len(input_shapes_list[0])
            
            for i in range(num_inputs):
                shapes_i = [s[i] for s in input_shapes_list if s[i] is not None]
                if not shapes_i:
                    input_ranges.append(None)
                    continue
                
                # Compute per-dimension ranges
                ndim = len(shapes_i[0])
                dim_ranges = []
                for d in range(ndim):
                    dims = [s[d] for s in shapes_i]
                    dim_ranges.append({
                        "min": int(min(dims)),
                        "max": int(max(dims)),
                        "typical": int(np.median(dims)),
                        "values": sorted(list(set(dims)))[:10]  # Top 10 unique values
                    })
                input_ranges.append(dim_ranges)
            
            # Compute output ranges
            output_range = None
            if output_shapes_list:
                ndim = len(output_shapes_list[0])
                output_range = []
                for d in range(ndim):
                    dims = [s[d] for s in output_shapes_list]
                    output_range.append({
                        "min": int(min(dims)),
                        "max": int(max(dims)),
                        "typical": int(np.median(dims))
                    })
            
            ranges[name] = {
                "op_type": op_type,
                "dtype": dtype,
                "call_count": len(records),
                "input_shape_ranges": input_ranges,
                "output_shape_range": output_range
            }
        
        return ranges
    
    def save_shape_ranges(self, filepath: str):
        """Save shape ranges to JSON file."""
        ranges = self.compute_shape_ranges()
        
        # Add metadata
        output = {
            "metadata": {
                "total_ops_captured": sum(self.op_counts.values()),
                "unique_ops": len(self.shape_records)
            },
            "shape_ranges": ranges
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Saved shape ranges to {filepath}")
        return output


def capture_shapes_during_inference(model, run_inference_fn, num_runs: int = 10) -> Dict:
    """
    Convenience function to capture shapes during inference.
    
    Args:
        model: The model to profile
        run_inference_fn: A function that runs one inference pass
        num_runs: Number of inference passes to capture
        
    Returns:
        Shape ranges dictionary
    """
    capture = ShapeCapture()
    capture.register_hooks(model)
    
    print(f"Running {num_runs} inference passes to capture shape ranges...")
    for i in range(num_runs):
        with torch.no_grad():
            run_inference_fn()
    
    capture.remove_hooks()
    return capture.compute_shape_ranges()
\`\`\`

### Use Shape Capture in Profiling: \`${dirs.profile}/profile_with_shapes.py\`

\`\`\`python
"""Profile model AND capture dynamic shape ranges."""
import torch
import json
import sys
sys.path.insert(0, "${dirs.demo}")
sys.path.insert(0, "${dirs.profile}")

from shape_capture import ShapeCapture

def profile_with_shape_capture():
    # Load model (adapt to your model type)
    # ... model loading code from demo.py ...
    
    # Create shape capture
    capture = ShapeCapture()
    capture.register_hooks(model)
    
    # Run multiple inference passes with different inputs
    print("Capturing shapes during inference...")
    
    # For text models - vary prompt lengths
    prompts = [
        "Hello",  # Short
        "The quick brown fox jumps over the lazy dog",  # Medium
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole...",  # Long
    ]
    
    for prompt in prompts:
        for _ in range(3):  # Multiple runs per prompt
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=20)
    
    # For image models - vary resolutions
    # resolutions = [(256, 256), (512, 512), (768, 768)]
    # for h, w in resolutions:
    #     for _ in range(3):
    #         pipe(prompt, height=h, width=w, num_inference_steps=5)
    
    capture.remove_hooks()
    
    # Save shape ranges
    capture.save_shape_ranges("${dirs.profile}/shape_ranges.json")
    
    print("\\n=== Shape Ranges Summary ===")
    ranges = capture.compute_shape_ranges()
    for name, info in list(ranges.items())[:10]:
        print(f"\\n{name} ({info['op_type']}):")
        print(f"  Calls: {info['call_count']}")
        for i, inp_range in enumerate(info['input_shape_ranges']):
            if inp_range:
                dims_str = ", ".join([f"[{r['min']}-{r['max']}]" for r in inp_range])
                print(f"  Input {i}: ({dims_str})")

if __name__ == "__main__":
    profile_with_shape_capture()
\`\`\`

### Shape Ranges Output Format: \`shape_ranges.json\`

\`\`\`json
{
  "metadata": {
    "total_ops_captured": 15000,
    "unique_ops": 150
  },
  "shape_ranges": {
    "model.layers.0.self_attn.q_proj": {
      "op_type": "Linear",
      "dtype": "torch.bfloat16",
      "call_count": 100,
      "input_shape_ranges": [
        [
          {"min": 1, "max": 1, "typical": 1},
          {"min": 1, "max": 512, "typical": 64},
          {"min": 4096, "max": 4096, "typical": 4096}
        ]
      ],
      "output_shape_range": [
        {"min": 1, "max": 1, "typical": 1},
        {"min": 1, "max": 512, "typical": 64},
        {"min": 4096, "max": 4096, "typical": 4096}
      ]
    },
    "model.layers.0.input_layernorm": {
      "op_type": "RMSNorm",
      "dtype": "torch.bfloat16", 
      "call_count": 100,
      "input_shape_ranges": [
        [
          {"min": 1, "max": 1, "typical": 1, "values": [1]},
          {"min": 1, "max": 512, "typical": 64, "values": [1, 32, 64, 128, 256, 512]},
          {"min": 4096, "max": 4096, "typical": 4096, "values": [4096]}
        ]
      ]
    }
  }
}
\`\`\`

**USE shape_ranges.json TO CREATE PROBLEM FILES** with dynamic shape support!

## STEP 1: Operator Fusion Analysis (CRITICAL)

Before creating individual problem files, analyze the profiling data for **fusable operator patterns**:

### Common Fusion Opportunities in LLMs

| Pattern | Operators to Fuse | Fused Name | Expected Speedup |
|---------|-------------------|------------|------------------|
| **ResidualNorm** | add + rmsnorm/layernorm | fused_residual_norm | 1.2-1.5x |
| **SwiGLU/GeGLU** | silu/gelu + mul | fused_swiglu | 1.3-1.8x |
| **BiasAdd** | matmul + add (bias) | fused_linear_bias | 1.1-1.3x |
| **RotaryEmbed** | rope_cos + rope_sin + cat | fused_rope | 1.2-1.5x |
| **QKV Projection** | 3x linear (q,k,v) | fused_qkv_proj | 1.2-1.4x |
| **MLP Block** | linear + activation + linear | fused_mlp | 1.3-2.0x |

### Fusion Detection Script: \`${dirs.profile}/analyze_fusion.py\`

\`\`\`python
"""Analyze operator patterns for fusion opportunities."""
import json

def analyze_fusion_opportunities(bottlenecks_file):
    with open(bottlenecks_file) as f:
        bottlenecks = json.load(f)
    
    # Extract operator names and percentages
    ops = [(b["name"], b["cuda_time_percent"]) for b in bottlenecks]
    
    fusion_opportunities = []
    
    # Check for residual + norm pattern
    has_add = any("add" in op[0].lower() for op in ops)
    has_norm = any("norm" in op[0].lower() or "mean" in op[0].lower() for op in ops)
    if has_add and has_norm:
        add_pct = sum(op[1] for op in ops if "add" in op[0].lower())
        norm_pct = sum(op[1] for op in ops if "norm" in op[0].lower() or "mean" in op[0].lower())
        fusion_opportunities.append({
            "name": "fused_residual_rmsnorm",
            "operators": ["aten::add", "RMSNorm (aten::mean, aten::rsqrt, aten::mul)"],
            "combined_percent": add_pct + norm_pct,
            "expected_speedup": "1.3-1.5x",
            "priority": "HIGH" if add_pct + norm_pct > 10 else "MEDIUM"
        })
    
    # Check for SiLU + mul pattern (SwiGLU)
    has_silu = any("silu" in op[0].lower() for op in ops)
    has_mul = any("mul" in op[0].lower() and "norm" not in op[0].lower() for op in ops)
    if has_silu and has_mul:
        fusion_opportunities.append({
            "name": "fused_swiglu",
            "operators": ["aten::silu", "aten::mul"],
            "combined_percent": sum(op[1] for op in ops if "silu" in op[0].lower() or ("mul" in op[0].lower() and "norm" not in op[0].lower())),
            "expected_speedup": "1.3-1.8x",
            "priority": "MEDIUM"
        })
    
    # Check for consecutive linear layers (QKV projection)
    mm_count = sum(1 for op in ops if "mm" in op[0].lower() or "linear" in op[0].lower())
    if mm_count >= 3:
        fusion_opportunities.append({
            "name": "fused_qkv_proj",
            "operators": ["3x aten::mm for Q, K, V"],
            "combined_percent": sum(op[1] for op in ops if "mm" in op[0].lower()) / 3 * 1.5,
            "expected_speedup": "1.2-1.4x (batch the projections)",
            "priority": "LOW"  # rocBLAS already fast
        })
    
    print("\\n=== Fusion Opportunities ===")
    for f in sorted(fusion_opportunities, key=lambda x: x["combined_percent"], reverse=True):
        print(f"\\n{f['name']} [{f['priority']}]")
        print(f"  Operators: {', '.join(f['operators'])}")
        print(f"  Combined time: {f['combined_percent']:.1f}%")
        print(f"  Expected speedup: {f['expected_speedup']}")
    
    # Save to file
    with open("${dirs.profile}/fusion_opportunities.json", "w") as f:
        json.dump(fusion_opportunities, f, indent=2)
    
    return fusion_opportunities

if __name__ == "__main__":
    analyze_fusion_opportunities("${dirs.profile}/bottlenecks.json")
\`\`\`

### Run Fusion Analysis FIRST
\`\`\`bash
cd ${dirs.profile}
python analyze_fusion.py
cat fusion_opportunities.json
\`\`\`

## STEP 2: Create FUSED Problem Files (Priority)

**Create fused kernels BEFORE individual kernels!**

### ⚠️ NEW: Problem Files with Dynamic Shape Ranges

Problem files now support **shape ranges** for dynamic input sizes. This allows kernel-optimize to generate kernels that work efficiently across the entire shape range observed during inference.

### Example: Fused Residual + RMSNorm with Shape Ranges
\`\`\`python
# problem_fused_residual_rmsnorm.py
import torch
import torch.nn as nn
import json

class Model(nn.Module):
    """Fused residual add + RMSNorm for LLM transformer layers."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))
        self.eps = eps
    
    def forward(self, hidden_states, residual):
        hidden_states = hidden_states + residual
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states

# ============================================================
# DYNAMIC SHAPE CONFIGURATION (from shape_ranges.json)
# ============================================================

# Load shape ranges from profiling
SHAPE_RANGES = {
    "batch_size": {"min": 1, "max": 1, "typical": 1},
    "seq_len": {"min": 1, "max": 512, "typical": 64},      # Dynamic!
    "hidden_size": {"min": 4096, "max": 4096, "typical": 4096}
}

# For backward compatibility with fixed-shape kernel-optimize
batch_size = SHAPE_RANGES["batch_size"]["typical"]
seq_len = SHAPE_RANGES["seq_len"]["typical"]
hidden_size = SHAPE_RANGES["hidden_size"]["typical"]

def get_inputs():
    """Return inputs at typical shape for baseline benchmarking."""
    return [
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda'),
        torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16, device='cuda'),
    ]

def get_inputs_for_shape(batch, seq, hidden):
    """Return inputs at specified shape for dynamic benchmarking."""
    return [
        torch.randn(batch, seq, hidden, dtype=torch.float16, device='cuda'),
        torch.randn(batch, seq, hidden, dtype=torch.float16, device='cuda'),
    ]

def get_shape_ranges():
    """Return shape ranges for dynamic kernel optimization."""
    return SHAPE_RANGES

def get_benchmark_shapes():
    """Return list of shapes to benchmark for dynamic optimization."""
    return [
        # (batch, seq, hidden) - cover the range
        (1, 1, hidden_size),      # Autoregressive decoding
        (1, 64, hidden_size),     # Short prompt
        (1, 256, hidden_size),    # Medium prompt
        (1, 512, hidden_size),    # Long prompt (max)
    ]

def get_init_inputs():
    return [hidden_size]
\`\`\`

### Example: Fused SwiGLU with Shape Ranges
\`\`\`python
# problem_fused_swiglu.py
import torch
import torch.nn as nn

class Model(nn.Module):
    """Fused SiLU(gate) * up for SwiGLU MLP."""
    def forward(self, gate, up):
        return torch.nn.functional.silu(gate) * up

# Dynamic shape configuration
SHAPE_RANGES = {
    "batch_size": {"min": 1, "max": 1, "typical": 1},
    "seq_len": {"min": 1, "max": 512, "typical": 64},
    "intermediate_size": {"min": 11008, "max": 11008, "typical": 11008}
}

batch_size = SHAPE_RANGES["batch_size"]["typical"]
seq_len = SHAPE_RANGES["seq_len"]["typical"]
intermediate_size = SHAPE_RANGES["intermediate_size"]["typical"]

def get_inputs():
    return [
        torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.float16, device='cuda'),
        torch.randn(batch_size, seq_len, intermediate_size, dtype=torch.float16, device='cuda'),
    ]

def get_init_inputs():
    return []
\`\`\`

## STEP 3: Create Individual Problem Files (Lower Priority)

Only create individual problem files for operators that:
1. Cannot be fused with neighbors
2. Take > 5% of total time individually
3. Are not already optimized by vendor libraries (e.g., rocBLAS GEMM)

## Problem File Format
Each problem file in \`${dirs.problems}/\` must have:

**⚠️ CRITICAL**: Use ACTUAL inference shapes from \`inference_shapes.json\`, NOT arbitrary shapes!

\`\`\`python
# problem_<operator_name>.py
import torch
import torch.nn as nn

class Model(nn.Module):
    """Reference implementation using PyTorch."""
    def __init__(self):
        super().__init__()
        # Initialize any needed parameters
    
    def forward(self, *inputs):
        # PyTorch implementation of the operator
        return output

# ⚠️ CRITICAL: Use shapes from actual inference profiling!
# Check ${dirs.profile}/inference_shapes.json for real shapes
# DO NOT use large batch sizes (8, 16) if inference uses batch=1

batch_size = 1      # ALWAYS 1 for image generation inference
seq_len = 4096      # Check actual latent size (e.g., 64x64=4096 for 512x512 images)
hidden_size = 4096  # From model config

def get_inputs():
    """Return list of input tensors with ACTUAL INFERENCE shapes."""
    return [torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16, device='cuda')]

def get_init_inputs():
    """Return list of arguments for Model.__init__"""
    return []
\`\`\`

### Shape Validation
Before running kernel-optimize, verify:
\`\`\`bash
# Run a quick benchmark at actual shapes vs benchmark shapes
python -c "
import torch
import time

# Actual inference shape
x_real = torch.randn(1, 4096, 4096, dtype=torch.bfloat16, device='cuda')
# Benchmark shape (often wrong!)
x_bench = torch.randn(8, 4096, 4096, dtype=torch.bfloat16, device='cuda')

# If kernel is slower at x_real but faster at x_bench, the problem file has WRONG shapes!
"
\`\`\`

## Common Operators to Optimize

### 1. Linear/GEMM
\`\`\`python
# problem_linear.py
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16))
    
    def forward(self, x):
        return x @ self.weight.T
\`\`\`

### 2. Attention (Self-Attention)
\`\`\`python
# problem_attention.py
class Model(nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)
\`\`\`

### 3. RMSNorm / LayerNorm
\`\`\`python
# problem_rmsnorm.py
class Model(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float16))
        self.eps = 1e-6
    
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x
\`\`\`

### 4. SiLU/SwiGLU/GELU
\`\`\`python
# problem_activation.py
class Model(nn.Module):
    def forward(self, x, gate):
        return torch.nn.functional.silu(gate) * x
\`\`\`

### 5. Rotary Position Embedding (RoPE)
\`\`\`python
# problem_rope.py
class Model(nn.Module):
    def forward(self, x, cos, sin):
        # RoPE implementation
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
\`\`\`

## Steps
1. Read bottlenecks.json
2. For each major bottleneck (>5% time), create a Problem file
3. Use shapes from profiling data
4. Ensure get_inputs() returns realistic input tensors
5. Update progress.json with list of created problems
6. **Generate optimization_manifest.json** (see below)

## ⚠️ CRITICAL: Generate Optimization Manifest

Create \`${dirs.problems}/optimization_manifest.json\` - allows users to enable/disable specific optimizations:

\`\`\`json
{
  "model": "${hfModel}",
  "generated_at": "[ISO DATE]",
  "description": "Edit 'enabled' to true/false to control which optimizations to apply",
  "optimizations": [
    {
      "name": "fused_residual_rmsnorm",
      "file": "problem_fused_residual_rmsnorm.py",
      "type": "fused",
      "priority": "HIGH",
      "cuda_time_percent": 12.5,
      "expected_speedup": "1.3-1.5x",
      "enabled": true,
      "notes": "Fuses residual add + RMSNorm - high impact"
    },
    {
      "name": "fused_rmsnorm",
      "file": "problem_fused_rmsnorm.py",
      "type": "fused",
      "priority": "HIGH",
      "cuda_time_percent": 7.28,
      "expected_speedup": "1.2-1.5x",
      "enabled": true,
      "notes": "Standalone RMSNorm optimization"
    },
    {
      "name": "fused_rope",
      "file": "problem_fused_rope.py",
      "type": "fused",
      "priority": "MEDIUM",
      "cuda_time_percent": 2.91,
      "expected_speedup": "1.2-1.5x",
      "enabled": true,
      "notes": "Rotary Position Embedding"
    },
    {
      "name": "linear_gemm",
      "file": "problem_linear.py",
      "type": "individual",
      "priority": "LOW",
      "cuda_time_percent": 42.46,
      "expected_speedup": "1.0-1.1x",
      "enabled": false,
      "notes": "rocBLAS usually optimal - skip unless specific issues"
    },
    {
      "name": "aiter_flash_attention",
      "file": null,
      "type": "aggressive",
      "priority": "HIGH",
      "cuda_time_percent": 35.0,
      "expected_speedup": "1.5-2.0x",
      "enabled": false,
      "notes": "Use AITER Flash Attention instead of PyTorch SDPA (experimental)"
    }
  ],
  "integration_options": {
    "patch_transformer_rmsnorm": true,
    "patch_text_encoder_rmsnorm": false,
    "use_aiter_attention": false,
    "run_correctness_test": true,
    "generate_comparison_outputs": true
  }
}
\`\`\`

### Using the Manifest
Users can edit \`optimization_manifest.json\` to:
- Set \`"enabled": false\` to skip specific optimizations
- Set \`"enabled": true\` on experimental optimizations like \`aiter_flash_attention\`
- Configure \`integration_options\` for fine-grained control

Then re-run: \`opencode model-optimize ... --from-phase optimize\`

---

# Phase 6: Run Kernel Optimization ${["env", "download", "demo", "compatibility", "profile", "problems"].includes(startPhase) ? "" : "[SKIP - ALREADY DONE]"}

## Goal
Optimize bottleneck kernels using kernel-optimize, running **in parallel** for speed.

## ⚠️ PARALLEL OPTIMIZATION
Run multiple kernel-optimize processes simultaneously to speed up optimization:

\`\`\`bash
cd ${dirs.problems}

# Run optimizations in PARALLEL using background processes
# HIGH priority (fused kernels) - run these in parallel
opencode kernel-optimize --src problem_fused_residual_rmsnorm.py --goal 1.5 &
opencode kernel-optimize --src problem_fused_rmsnorm.py --goal 1.5 &
opencode kernel-optimize --src problem_fused_rope.py --goal 1.5 &

# Wait for all background jobs to complete
wait

# MEDIUM priority - run in parallel
opencode kernel-optimize --src problem_rope.py --goal 1.3 &
opencode kernel-optimize --src problem_silu_mul.py --goal 1.3 &
wait

# Check results and copy successful ones
\`\`\`

## Optimization Priority Order
| Priority | Kernel Type | Goal | Reason |
|----------|-------------|------|--------|
| **HIGH** | Fused Residual+RMSNorm | 1.5x | Memory traffic reduction |
| **HIGH** | Fused RMSNorm | 1.5x | Repeated many times |
| **HIGH** | Fused RoPE | 1.5x | Custom AMD optimization |
| MEDIUM | Individual RoPE | 1.3x | If not using fused version |
| MEDIUM | SwiGLU/GELU | 1.3x | Activation functions |
| LOW | Linear/GEMM | 1.1x | rocBLAS usually optimal |
| **SKIP** | Simple add/copy | - | Overhead > benefit |

## Decision: When to Skip Optimization
- **SKIP** if operator is part of a fused kernel you already optimized
- **SKIP** GEMM/Linear if profiling shows rocBLAS is already near-optimal
- **SKIP** simple elementwise (add, copy) - overhead exceeds benefit

## After Optimization

**⚠️ CRITICAL: Verify speedup at ACTUAL inference shapes!**

\`\`\`bash
# After kernel-optimize produces *_opt.py, verify it's faster at real shapes:
cd ${dirs.problems}
python -c "
import torch, time
from problem_XXX import Model as RefModel, get_inputs
from problem_XXX_opt import ModelNew as OptModel

# Get inputs at ACTUAL inference shape
inputs = get_inputs()  # Should be batch=1!
ref = RefModel(*get_init_inputs()).cuda().eval()
opt = OptModel(*get_init_inputs()).cuda().eval()

# Warmup
for _ in range(20):
    ref(*inputs); opt(*inputs)
torch.cuda.synchronize()

# Benchmark
t0 = time.perf_counter()
for _ in range(500): ref(*inputs)
torch.cuda.synchronize()
t_ref = (time.perf_counter()-t0)/500*1000

t0 = time.perf_counter()
for _ in range(500): opt(*inputs)
torch.cuda.synchronize()
t_opt = (time.perf_counter()-t0)/500*1000

speedup = t_ref/t_opt
print(f'Speedup: {speedup:.2f}x')
if speedup < 1.0:
    print('WARNING: Kernel is SLOWER at inference shapes! Do not integrate!')
"
\`\`\`

1. **Only copy kernels that are faster at ACTUAL inference shapes** to \`${dirs.optimized}/\`
2. Record speedup for each kernel in progress.json
3. Note which kernels failed or were **slower at inference shapes**

---

# Phase 6.5: Explore Additional Optimization Opportunities

## Goal
Evaluate additional optimizations based on profiling data - apply ONLY if they provide measurable benefit.

## ⚠️ CRITICAL: Data-Driven Decision Making

**Before enabling ANY optimization:**
1. Check profiling data to see if the operation is actually a bottleneck
2. Benchmark the optimization at ACTUAL inference shapes
3. Only apply if measured speedup > 1.0x

## Potential Optimization Areas (Evaluate Based on Profiling)

### 1. AITER Flash Attention
**When to consider**: If attention operations are > 10% of total runtime AND sequence length is typically > 64.

**Location**: \`/sgl-workspace/aiter/\` or via pip

**Important**: AITER benefits depend on sequence length:
- Short sequences (seq < 64): Often SLOWER due to transpose overhead
- Long sequences (seq > 512): Often FASTER

**ALWAYS benchmark before applying**:
\`\`\`python
# Benchmark AITER vs PyTorch SDPA at YOUR actual shapes
# Only use AITER if it's faster at your actual inference shapes
\`\`\`

## Installation (if profiling suggests benefit)

\`\`\`bash
source ${outputDir}/venv/bin/activate

# Try to install AITER
pip install -e /sgl-workspace/aiter/ 2>/dev/null || echo "AITER not available"

# Verify
python -c "from aiter.ops.mha import flash_attn_func; print('AITER available')" 2>&1
\`\`\`

### 2. Additional AMD Optimizations (Evaluate Based on Need)

If profiling shows specific bottlenecks, consider:
- AITER GEMM operations for MoE or FP8 workloads
- hipBLASLt for specific GEMM shapes
- ROCm environment tuning

## Decision Framework

**ALWAYS measure before deciding**:
1. Profile to identify actual bottlenecks
2. Benchmark candidate optimizations at ACTUAL inference shapes
3. Only apply optimizations that show > 1.0x speedup
4. Document actual measured speedup, not theoretical estimates

**If an optimization doesn't help at actual shapes, simply don't apply it.**

---

# Phase 7: Integration & Final Testing ${["env", "download", "demo", "compatibility", "profile", "problems", "optimize"].includes(startPhase) ? "" : "[SKIP - ALREADY DONE]"}

## Goal
Integrate optimized kernels into the model using monkey-patching and **MEASURE ACTUAL end-to-end performance**.

## ⚠️ CRITICAL REQUIREMENTS - THIS PHASE MUST:
1. **MEASURE ACTUAL end-to-end speedup** - NOT estimated speedup using Amdahl's law
2. **Generate BOTH original AND optimized outputs** with same random seed
3. **Run the SAME inference with and without optimizations** to get real numbers

## ⚠️ CRITICAL: Use Project venv
\`\`\`bash
source ${outputDir}/venv/bin/activate
\`\`\`

### What You CAN Do:
- Edit/install packages in project venv: \`${outputDir}/venv/lib/python*/site-packages/\`
- Create files in project directory: \`${outputDir}/\`
- Use monkey-patching to override behavior at runtime
- **Install AITER in the project venv** if needed:
  \`\`\`bash
  pip install aiter  # or install from source if needed
  \`\`\`

### What You CANNOT Do:
- **NEVER edit /opt/, /usr/, or system site-packages**
- **NEVER modify the host Python environment**
- **NEVER report only "estimated" speedup - MUST measure actual performance**

## IMPORTANT: Integration Strategy for Fused Kernels

Fused kernels require careful integration as they replace MULTIPLE operations:

### Fused Residual + RMSNorm Integration Pattern
\`\`\`python
# Locate the RMSNorm class in transformers
# Original pattern in forward():
#   residual = hidden_states
#   hidden_states = self.input_layernorm(hidden_states)
#
# Replace with fused version:
#   hidden_states, residual = fused_residual_rmsnorm(hidden_states, residual)
\`\`\`

## Create Integration Script: \`${dirs.optimized}/integrate.py\`

\`\`\`python
"""
Monkey-patch optimized Triton kernels into the model.
Supports both individual and fused kernels.
CRITICAL: Must provide apply_all_patches() function for e2e measurement.
"""
import torch
import sys
import os

# Add paths
sys.path.insert(0, "${dirs.optimized}")
sys.path.insert(0, "${dirs.problems}")

# Import optimized kernels (check which ones exist)
_optimized_kernels = {}
_patch_stats = {}

def try_import(name, module_name):
    try:
        mod = __import__(module_name)
        if hasattr(mod, 'ModelNew'):
            _optimized_kernels[name] = mod.ModelNew
            print(f"  [OK] Loaded optimized kernel: {name}")
            return True
    except Exception as e:
        print(f"  [SKIP] {name}: {e}")
    return False

print("Loading optimized Triton kernels...")
# Try to import all available optimized kernels
# Fused kernels (higher priority)
try_import("fused_residual_rmsnorm", "problem_fused_residual_rmsnorm_opt")
try_import("fused_residual_layernorm", "problem_fused_residual_layernorm_opt")
try_import("fused_swiglu", "problem_fused_swiglu_opt")
try_import("fused_silu_mul", "problem_fused_silu_mul_opt")
try_import("fused_gelu_mul", "problem_fused_gelu_mul_opt")
try_import("fused_rmsnorm", "problem_fused_rmsnorm_opt")
try_import("fused_rope", "problem_fused_rope_opt")
# Individual kernels
try_import("rmsnorm", "problem_rmsnorm_opt")
try_import("layernorm", "problem_layer_norm_opt")
try_import("rope", "problem_rope_opt")
try_import("gelu", "problem_gelu_opt")

print(f"Loaded kernels: {list(_optimized_kernels.keys())}")

def patch_normalization_layers(model, hidden_size=None):
    """Patch RMSNorm/LayerNorm layers with optimized versions."""
    patched = 0
    
    # Detect hidden_size from model
    if hidden_size is None:
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if len(module.weight.shape) == 1 and module.weight.shape[0] > 256:
                    hidden_size = module.weight.shape[0]
                    break
        if hidden_size is None:
            print("WARNING: Could not auto-detect hidden_size, check model config")
            return 0  # Don't apply without knowing correct size
    
    # Get appropriate kernel
    kernel_cls = (_optimized_kernels.get("fused_residual_rmsnorm") or 
                  _optimized_kernels.get("fused_residual_layernorm") or
                  _optimized_kernels.get("fused_rmsnorm") or
                  _optimized_kernels.get("rmsnorm") or
                  _optimized_kernels.get("layernorm"))
    
    if kernel_cls is None:
        print("No normalization kernel available")
        return 0
    
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if "RMSNorm" in class_name or "LayerNorm" in class_name:
            try:
                opt_module = kernel_cls(hidden_size)
                # Copy weights
                if hasattr(module, 'weight') and hasattr(opt_module, 'weight'):
                    opt_module.weight.data = module.weight.data.clone()
                if hasattr(module, 'bias') and hasattr(opt_module, 'bias') and module.bias is not None:
                    opt_module.bias.data = module.bias.data.clone()
                
                # Replace forward
                original_forward = module.forward
                def make_opt_forward(opt_mod):
                    def opt_forward(x, *args, **kwargs):
                        return opt_mod(x)
                    return opt_forward
                module.forward = make_opt_forward(opt_module)
                patched += 1
            except Exception as e:
                pass  # Skip layers that don't match
    
    return patched

def patch_activations(model):
    """Patch activation functions with fused versions."""
    patched = 0
    # Implement based on model architecture
    return patched

def apply_all_patches(model_or_pipe):
    """
    CRITICAL: Main entry point for applying all optimizations.
    Works with both raw models and diffusers pipelines.
    Returns: (patched_model_or_pipe, stats_dict)
    """
    stats = {
        "normalization_layers": 0,
        "activations": 0,
        "attention": False,
        "kernels_loaded": list(_optimized_kernels.keys())
    }
    
    # Handle diffusers pipelines
    if hasattr(model_or_pipe, 'transformer'):
        stats["normalization_layers"] = patch_normalization_layers(model_or_pipe.transformer)
    elif hasattr(model_or_pipe, 'unet'):
        stats["normalization_layers"] = patch_normalization_layers(model_or_pipe.unet)
    elif hasattr(model_or_pipe, 'model'):
        stats["normalization_layers"] = patch_normalization_layers(model_or_pipe.model)
    else:
        stats["normalization_layers"] = patch_normalization_layers(model_or_pipe)
    
    print(f"\\nPatch stats: {stats}")
    return model_or_pipe, stats

# Alias for backward compatibility
patch_model = apply_all_patches
\`\`\`

## Create Test Script: \`${dirs.optimized}/test_integration.py\`

\`\`\`python
"""
Test optimized model for correctness and performance.
"""
import torch
import time
import sys
sys.path.insert(0, "${dirs.demo}")
sys.path.insert(0, "${dirs.optimized}")

from transformers import AutoModelForCausalLM, AutoTokenizer
from integrate import patch_model

MODEL_PATH = "${dirs.model}"

def test_integration():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Load original model
    model_original = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    
    # Load and patch optimized model
    model_optimized = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    model_optimized = patch_model(model_optimized)
    
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Test correctness
    print("=== Testing Correctness ===")
    with torch.no_grad():
        out_original = model_original(**inputs)
        out_optimized = model_optimized(**inputs)
    
    logits_diff = (out_original.logits - out_optimized.logits).abs().max().item()
    print(f"Max logits difference: {logits_diff:.6f}")
    print(f"Correctness: {'PASSED' if logits_diff < 0.1 else 'FAILED'}")
    
    # Test performance
    print("\\n=== Testing Performance ===")
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            model_original.generate(**inputs, max_new_tokens=10)
            model_optimized.generate(**inputs, max_new_tokens=10)
    torch.cuda.synchronize()
    
    # Benchmark original
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            model_original.generate(**inputs, max_new_tokens=20)
    torch.cuda.synchronize()
    t_original = (time.perf_counter() - t0) / 10
    
    # Benchmark optimized
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            model_optimized.generate(**inputs, max_new_tokens=20)
    torch.cuda.synchronize()
    t_optimized = (time.perf_counter() - t0) / 10
    
    speedup = t_original / t_optimized
    
    print(f"Original:  {t_original*1000:.2f} ms")
    print(f"Optimized: {t_optimized*1000:.2f} ms")
    print(f"Speedup:   {speedup:.2f}x")
    
    return {
        "correctness": logits_diff < 0.1,
        "original_ms": t_original * 1000,
        "optimized_ms": t_optimized * 1000,
        "speedup": speedup
    }

if __name__ == "__main__":
    results = test_integration()
    
    # Save results
    import json
    with open("${dirs.report}/integration_results.json", "w") as f:
        json.dump(results, f, indent=2)
\`\`\`

## ⚠️ CRITICAL: Generate Comparison Outputs for User Verification

After testing correctness and performance, you MUST generate comparison outputs with **fixed random seed** so users can visually verify the optimization didn't break anything.

### For Text Generation Models:
\`\`\`python
import torch
import os

SEED = 42
COMPARISON_DIR = "${dirs.report}/comparison_outputs"
os.makedirs(COMPARISON_DIR, exist_ok=True)

# Generate with fixed seed - ORIGINAL
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
with torch.no_grad():
    output_original = model_original.generate(**inputs, max_new_tokens=100, do_sample=True)
text_original = tokenizer.decode(output_original[0], skip_special_tokens=True)

# Generate with same seed - OPTIMIZED  
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
with torch.no_grad():
    output_optimized = model_optimized.generate(**inputs, max_new_tokens=100, do_sample=True)
text_optimized = tokenizer.decode(output_optimized[0], skip_special_tokens=True)

# Save comparison
with open(f"{COMPARISON_DIR}/original_output.txt", "w") as f:
    f.write(f"Prompt: {prompt}\\n\\nGenerated:\\n{text_original}")
with open(f"{COMPARISON_DIR}/optimized_output.txt", "w") as f:
    f.write(f"Prompt: {prompt}\\n\\nGenerated:\\n{text_optimized}")
print(f"Comparison outputs saved to {COMPARISON_DIR}/")
\`\`\`

### For Image Generation Models (diffusers):
\`\`\`python
import torch
import os
from PIL import Image

SEED = 42
COMPARISON_DIR = "${dirs.report}/comparison_outputs"
os.makedirs(COMPARISON_DIR, exist_ok=True)

prompt = "A beautiful sunset over the ocean, photorealistic"

# Generate with fixed seed - ORIGINAL
generator_orig = torch.Generator(device="cuda").manual_seed(SEED)
image_original = pipe_original(prompt, generator=generator_orig, num_inference_steps=20).images[0]
image_original.save(f"{COMPARISON_DIR}/original_output.png")

# Generate with same seed - OPTIMIZED
generator_opt = torch.Generator(device="cuda").manual_seed(SEED)
image_optimized = pipe_optimized(prompt, generator=generator_opt, num_inference_steps=20).images[0]
image_optimized.save(f"{COMPARISON_DIR}/optimized_output.png")

# Create side-by-side comparison
combined_width = image_original.width * 2 + 20
combined = Image.new('RGB', (combined_width, image_original.height + 30), (255, 255, 255))
combined.paste(image_original, (0, 30))
combined.paste(image_optimized, (image_original.width + 20, 30))
# Add labels
from PIL import ImageDraw
draw = ImageDraw.Draw(combined)
draw.text((10, 5), "Original", fill=(0,0,0))
draw.text((image_original.width + 30, 5), "Optimized", fill=(0,0,0))
combined.save(f"{COMPARISON_DIR}/comparison.png")

print(f"Images saved to {COMPARISON_DIR}/")
print(f"  - original_output.png: Original model output")
print(f"  - optimized_output.png: Optimized model output")
print(f"  - comparison.png: Side-by-side comparison")
\`\`\`

## ⚠️ MANDATORY: End-to-End Performance Measurement

You MUST create and run a script that measures ACTUAL end-to-end performance:

\`\`\`python
"""
CRITICAL: measure_actual_e2e.py
Measures ACTUAL end-to-end performance with and without optimizations.
"""
import torch
import time
import json
import os
import sys

sys.path.insert(0, "${dirs.optimized}")
sys.path.insert(0, "${dirs.demo}")

COMPARISON_DIR = "${dirs.report}/comparison_outputs"
os.makedirs(COMPARISON_DIR, exist_ok=True)

SEED = 42
NUM_WARMUP = 2
NUM_RUNS = 5

def measure_inference():
    results = {}
    
    # Load model/pipeline (adapt for your model type)
    # For diffusers image models:
    from diffusers import AutoPipelineForText2Image
    pipe = AutoPipelineForText2Image.from_pretrained(
        "${dirs.model}",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )
    
    prompt = "A beautiful sunset over the ocean, photorealistic, high quality"
    gen_kwargs = {
        "prompt": prompt,
        "height": 512,
        "width": 512, 
        "num_inference_steps": 20,
        "guidance_scale": 1.5,
    }
    
    # ========== ORIGINAL (baseline) ==========
    print("\\n=== Measuring ORIGINAL (baseline) performance ===")
    
    # Warmup
    for _ in range(NUM_WARMUP):
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        _ = pipe(**gen_kwargs, generator=generator)
    torch.cuda.synchronize()
    
    # Generate original output for comparison
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    t0 = time.perf_counter()
    image_original = pipe(**gen_kwargs, generator=generator).images[0]
    torch.cuda.synchronize()
    t_original_single = time.perf_counter() - t0
    image_original.save(f"{COMPARISON_DIR}/original_output.png")
    
    # Multiple runs for timing
    times_original = []
    for i in range(NUM_RUNS):
        generator = torch.Generator(device="cuda").manual_seed(SEED + i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = pipe(**gen_kwargs, generator=generator)
        torch.cuda.synchronize()
        times_original.append(time.perf_counter() - t0)
    
    t_original = sum(times_original) / len(times_original)
    print(f"Original avg time: {t_original:.3f}s ({NUM_RUNS} runs)")
    
    # ========== APPLY OPTIMIZATIONS ==========
    print("\\n=== Applying optimized kernels ===")
    from integrate import patch_model, apply_all_patches
    pipe, patch_stats = apply_all_patches(pipe)
    print(f"Patches applied: {patch_stats}")
    
    # ========== OPTIMIZED ==========
    print("\\n=== Measuring OPTIMIZED performance ===")
    
    # Warmup with optimizations
    for _ in range(NUM_WARMUP):
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        _ = pipe(**gen_kwargs, generator=generator)
    torch.cuda.synchronize()
    
    # Generate optimized output for comparison (same seed!)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    t0 = time.perf_counter()
    image_optimized = pipe(**gen_kwargs, generator=generator).images[0]
    torch.cuda.synchronize()
    t_optimized_single = time.perf_counter() - t0
    image_optimized.save(f"{COMPARISON_DIR}/optimized_output.png")
    
    # Multiple runs for timing
    times_optimized = []
    for i in range(NUM_RUNS):
        generator = torch.Generator(device="cuda").manual_seed(SEED + i)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = pipe(**gen_kwargs, generator=generator)
        torch.cuda.synchronize()
        times_optimized.append(time.perf_counter() - t0)
    
    t_optimized = sum(times_optimized) / len(times_optimized)
    print(f"Optimized avg time: {t_optimized:.3f}s ({NUM_RUNS} runs)")
    
    # Calculate speedup
    speedup = t_original / t_optimized
    print(f"\\n=== ACTUAL END-TO-END SPEEDUP: {speedup:.2f}x ===")
    
    # Save results
    results = {
        "original_time_s": t_original,
        "optimized_time_s": t_optimized,
        "speedup": speedup,
        "prompt": prompt,
        "seed": SEED,
        "num_runs": NUM_RUNS,
        "patches_applied": patch_stats,
        "files": {
            "original": f"{COMPARISON_DIR}/original_output.png",
            "optimized": f"{COMPARISON_DIR}/optimized_output.png"
        }
    }
    
    with open(f"{COMPARISON_DIR}/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\\nResults saved to {COMPARISON_DIR}/comparison_results.json")
    return results

if __name__ == "__main__":
    measure_inference()
\`\`\`

## Steps
1. Create integrate.py with monkey-patches for optimized kernels  
2. Create the **CRITICAL** measure_actual_e2e.py script above
3. Run the measurement script to get ACTUAL end-to-end timing
4. **Generate BOTH original AND optimized outputs** with same seed
5. If correctness fails, debug and fix
6. Record ACTUAL (not estimated) speedup in progress.json
7. Update progress.json with actual_speedup (not estimated)

---

# Phase 8: Generate Final Report ${startPhase === "report" ? "" : (["env", "download", "demo", "compatibility", "profile", "problems", "optimize", "integrate"].includes(startPhase) ? "" : "[SKIP - ALREADY DONE]")}

## Goal
Create a comprehensive optimization report.

## Create Report: \`${dirs.report}/optimization_report.md\`

**⚠️ CRITICAL**: The report MUST include **ACTUAL MEASURED** end-to-end speedup from the comparison_results.json, NOT estimated/theoretical speedup.

\`\`\`markdown
# Model Optimization Report

## Model Information
- **Model**: ${hfModel}
- **Optimization Date**: [DATE]

## Summary
- **Total Optimization Time**: X hours
- **ACTUAL End-to-End Speedup**: X.Xx (measured, NOT estimated)
- **Kernels Optimized**: N
- **Baseline Inference Time**: X.Xs
- **Optimized Inference Time**: X.Xs

## Bottleneck Analysis

| Operator | Original Time (ms) | % of Total | Optimized | Speedup |
|----------|-------------------|------------|-----------|---------|
| Linear   | XX.X              | XX%        | Yes       | X.Xx    |
| Attention| XX.X              | XX%        | Yes       | X.Xx    |
| ...      | ...               | ...        | ...       | ...     |

## Optimized Kernels

### 1. Linear (GEMM)
- **Problem File**: problem_linear.py
- **Optimization Techniques**: [list techniques]
- **Speedup**: X.Xx

### 2. Attention
- **Problem File**: problem_attention.py
- **Optimization Techniques**: [list techniques]
- **Speedup**: X.Xx

## Integration Details

### Monkey Patches Applied
1. Replaced Linear layers with Triton GEMM
2. Replaced attention with Flash Attention implementation
3. ...

### Correctness Verification
- Max logits difference: X.XXXXXX
- Status: PASSED/FAILED

## Performance Results (ACTUAL MEASURED - NOT ESTIMATED)

**⚠️ These numbers MUST come from measure_actual_e2e.py output, NOT from kernel-level estimates!**

| Metric | Original | Optimized | Speedup |
|--------|----------|-----------|---------|
| End-to-End Inference Time | X.Xs | X.Xs | **X.Xx** |
| Per-Token Latency (if applicable) | XXms | XXms | X.Xx |

### Kernel-Level Benchmarks (for reference)

| Kernel | Baseline | Optimized | Speedup |
|--------|----------|-----------|---------|
| Fused Residual+Norm | X.XX ms | X.XX ms | X.Xx |
| ... | ... | ... | ... |

## Comparison Outputs (Seed=42)

Outputs generated with **fixed random seed** for verification that optimization preserves model behavior.

### For Text Generation Models:

**Prompt**: "[INSERT PROMPT HERE]"

<table>
<tr><th>Original Model</th><th>Optimized Model</th></tr>
<tr>
<td>

[INSERT ORIGINAL OUTPUT TEXT HERE]

</td>
<td>

[INSERT OPTIMIZED OUTPUT TEXT HERE]

</td>
</tr>
</table>

**Verification**: [IDENTICAL / SIMILAR / DIFFERENT - explain if different]

### For Image Generation Models:

**Prompt**: "[INSERT PROMPT HERE]"

| Original | Optimized |
|:--------:|:---------:|
| ![Original](comparison_outputs/original_output.png) | ![Optimized](comparison_outputs/optimized_output.png) |

**Side-by-side Comparison**:

![Comparison](comparison_outputs/comparison.png)

**Visual Verification**: [IDENTICAL / SIMILAR / DIFFERENT - explain if different]

> **Note**: Small numerical differences are expected due to bf16/fp16 precision, but outputs should be visually/textually nearly identical.

## Files Generated

\\\`\\\`\\\`
${outputDir}/
├── venv/               # Project Python virtual environment
├── model/              # Downloaded model
├── demo/
│   ├── demo.py         # Working demo script
│   └── patches/        # Compatibility patches
├── profile/
│   ├── trace.json      # Chrome trace
│   ├── bottlenecks.json
│   └── operator_summary.txt
├── problems/
│   ├── problem_linear.py
│   ├── problem_linear_opt.py
│   └── ...
├── optimized/
│   ├── integrate.py    # Integration script
│   └── test_integration.py
└── report/
    ├── optimization_report.md
    ├── integration_results.json
    └── comparison_outputs/
        ├── original_output.*   # Original model output
        ├── optimized_output.*  # Optimized model output
        └── comparison.*        # Side-by-side (images only)
\\\`\\\`\\\`

## Recommendations for Further Optimization
1. ...
2. ...

## Known Issues
1. ...
\`\`\`

## Steps
1. Gather all results from previous phases
2. Generate the comprehensive report
3. Update progress.json: phase="complete", phases_completed.append("report")

---

# EXECUTION INSTRUCTIONS

${startPhase === "download" ? `
## Fresh Start Mode
1. **Execute phases in order**: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
2. Begin with Phase 1: Model Download
` : `
## Resume Mode Active - Starting from "${startPhase}"
1. **Skip phases before "${startPhase}"** - their artifacts already exist
2. **Review existing files first** to understand current state
3. **Continue from Phase: ${startPhase}**

### Quick Start Checklist
- [ ] Read existing progress.json
- [ ] Verify artifacts from previous phases exist
- [ ] Start working on Phase: ${startPhase}
`}

## General Rules
1. **Update progress.json after each phase**
2. **If a phase fails, debug and fix before proceeding**
3. **For kernel-optimize, use the existing opencode command** (it inherits LLM_GATEWAY_KEY from the environment)
4. **All monkey patches go in ${dirs.demo}/patches/ or ${dirs.optimized}/**
5. **NEVER modify system libraries - only use monkey patching**

## Optimization Priority (Phase 5-6)
1. **FIRST**: Create and optimize FUSED kernels (residual+norm, swiglu)
2. **THEN**: Optimize remaining individual kernels
3. **SKIP**: Operators already optimized by vendor libs (rocBLAS GEMM)
4. **SKIP**: Simple elementwise ops covered by fused kernels

## Start Now
Begin with Phase: ${startPhase.charAt(0).toUpperCase() + startPhase.slice(1)}
`
}

function buildAgentConfig(hfModel: string): string {
  return `---
model: amd-anthropic/claude-opus-4-5
temperature: 0.2
steps: 100
---

# Model Optimization Expert

You are an expert in end-to-end deep learning model optimization. Your task is to optimize the HuggingFace model "${hfModel}" for maximum inference performance.

## Core Competencies

1. **Model Understanding**: Analyze transformer architectures, identify compute patterns
2. **Profiling**: Use PyTorch Profiler and AMD ROCm tools to identify bottlenecks
3. **Kernel Development**: Write high-performance Triton kernels
4. **Integration**: Safely integrate optimizations via monkey-patching

## Key Principles

1. **Correctness First**: Never sacrifice correctness for speed
2. **Systematic Approach**: Follow the phases in order, update progress
3. **No System Modifications**: Use monkey-patching, never modify installed packages
4. **Document Everything**: Keep detailed notes in progress.json and final report

## Workflow

Execute each phase completely before moving to the next:

1. Download model
2. Generate and test demo script
3. Fix compatibility issues (if any)
4. Profile and identify bottlenecks
5. Generate Problem files for top bottlenecks
6. Run kernel-optimize on each Problem
7. Integrate optimized kernels via monkey-patching
8. Generate final report

## Tools You'll Use

- \`huggingface_hub\` / \`transformers\`: Model download and loading
- \`torch.profiler\`: Performance profiling
- \`opencode kernel-optimize\`: Kernel optimization
- Python: Scripting, monkey-patching, testing

## Error Handling

- If model download fails: Check model name, authentication
- If demo fails: Analyze error, create monkey-patch fix
- If profiling fails: Simplify the test case
- If kernel optimization fails: Document and skip that kernel
- If integration fails: Debug monkey-patch, test incrementally

Start by reading the task prompt and executing Phase 1.
`
}

