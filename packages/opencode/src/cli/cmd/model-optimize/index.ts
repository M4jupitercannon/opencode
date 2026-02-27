/**
 * CLI command for end-to-end HuggingFace model optimization.
 *
 * Orchestrates the pipeline by composing independent skill modules
 * and managing the opencode server lifecycle.
 */
import { cmd } from "../cmd"
import { bootstrap } from "../../bootstrap"
import { Server } from "../../../server/server"
import { createOpencodeClient } from "@opencode-ai/sdk/v2"
import * as fs from "fs"
import * as path from "path"
import { UI } from "../../ui"
import { Provider } from "../../../provider/provider"
import { select } from "@clack/prompts"

import type { ModelOptConfig, ModelOptDirs } from "./types"
import { PHASE_ORDER } from "./types"
import { buildAgentPrompt, buildAgentConfig } from "./prompt"

/** Directory containing reusable Python scripts */
const SCRIPTS_DIR = path.join(import.meta.dir, "scripts")

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
      })
      .option("concurrency", {
        type: "number",
        describe: "max concurrent requests for benchmarking (default: 16)",
        default: 16,
      })
      .option("input-len", {
        type: "number",
        describe: "input sequence length for benchmarking (default: 1024)",
        default: 1024,
      })
      .option("output-len", {
        type: "number",
        describe: "output sequence length for benchmarking (default: 1024)",
        default: 1024,
      })
      .option("num-prompts", {
        type: "number",
        describe: "number of prompts for benchmarking (default: 100)",
        default: 100,
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

    // Output directory — default to /tmp to avoid polluting the working directory
    const outputDir =
      (args.output as string) || path.resolve(`/tmp/model_opt_${modelNameSafe}`)

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
    const dirs: ModelOptDirs = {
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
            const validPhases = [...PHASE_ORDER]
            if (!validPhases.includes(fromPhase as any)) {
              UI.error(`Invalid phase: ${fromPhase}. Valid phases: ${validPhases.join(", ")}`)
              process.exit(1)
            }
            startPhase = fromPhase
            UI.println(`Starting from phase: ${startPhase}`)
          } else if (resumeMode && existingProgress.phases_completed) {
            // Resume from last completed phase
            const phasesOrder = [...PHASE_ORDER]
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
    const configData = {
      hf_model: hfModel,
      model_name: modelName,
      dirs: dirs,
      created: new Date().toISOString(),
      skip_download: args["skip-download"],
      start_phase: startPhase,
      resume_mode: resumeMode || !!fromPhase,
    }
    fs.writeFileSync(configFile, JSON.stringify(configData, null, 2))

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

    // Build the pipeline config
    const pipelineConfig: ModelOptConfig = {
      hfModel,
      modelName,
      outputDir,
      dirs,
      skipDownload: args["skip-download"] as boolean,
      startPhase,
      existingProgress,
      concurrency: args.concurrency as number,
      inputLen: args["input-len"] as number,
      outputLen: args["output-len"] as number,
      numPrompts: args["num-prompts"] as number,
    }

    // Copy reusable Python scripts to output directory
    const scriptsOutputDir = path.join(outputDir, "scripts")
    fs.mkdirSync(scriptsOutputDir, { recursive: true })
    for (const file of fs.readdirSync(SCRIPTS_DIR)) {
      if (file.endsWith(".py")) {
        fs.copyFileSync(path.join(SCRIPTS_DIR, file), path.join(scriptsOutputDir, file))
      }
    }

    // Create the main prompt for the agent
    const prompt = buildAgentPrompt(pipelineConfig)

    // Create temporary .opencode config
    const opencodeDir = path.join(outputDir, ".opencode")
    const agentDir = path.join(opencodeDir, "agent")
    fs.mkdirSync(agentDir, { recursive: true })

    // Exclude large directories (venv, model, __pycache__) from opencode indexing
    fs.writeFileSync(path.join(outputDir, ".gitignore"), [
      "venv/",
      "model/",
      "__pycache__/",
      "*.safetensors",
      "*.bin",
      "*.pt",
      "*.gguf",
      "*.trace.json",
      "*.trace.json.gz",
    ].join("\n") + "\n")

    // Create agent config from .md skill file
    const agentConfig = buildAgentConfig(pipelineConfig)
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
              // Phase detection from agent text for console output
              if (part.type === "text") {
                const textPart = part as any
                if (textPart.state?.done && textPart.state.content?.trim()) {
                  const content = textPart.state.content
                  const phaseDetections: Array<{ match: string; phase: string; title: string }> = [
                    { match: "Phase 0", phase: "env", title: "Phase 0: Environment Setup" },
                    { match: "Environment Setup", phase: "env", title: "Phase 0: Environment Setup" },
                    { match: "Phase 1", phase: "download", title: "Phase 1: Model Download" },
                    { match: "Model Download", phase: "download", title: "Phase 1: Model Download" },
                    { match: "Phase 2", phase: "demo", title: "Phase 2: Generate Demo Script" },
                    { match: "Demo Script", phase: "demo", title: "Phase 2: Generate Demo Script" },
                    { match: "Phase 3", phase: "compatibility", title: "Phase 3: Fix Compatibility Issues" },
                    { match: "Compatibility", phase: "compatibility", title: "Phase 3: Fix Compatibility Issues" },
                    { match: "Phase 4", phase: "profile", title: "Phase 4: Performance Profiling" },
                    { match: "Profiling", phase: "profile", title: "Phase 4: Performance Profiling" },
                    { match: "Phase 5", phase: "problems", title: "Phase 5: Generate Problem Files" },
                    { match: "Problem Files", phase: "problems", title: "Phase 5: Generate Problem Files" },
                    { match: "Phase 6", phase: "optimize", title: "Phase 6: Kernel Optimization" },
                    { match: "Kernel Optimization", phase: "optimize", title: "Phase 6: Kernel Optimization" },
                    { match: "Phase 7", phase: "integrate", title: "Phase 7: Integration & Testing" },
                    { match: "Integration", phase: "integrate", title: "Phase 7: Integration & Testing" },
                    { match: "Phase 8", phase: "report", title: "Phase 8: Generate Final Report" },
                    { match: "Final Report", phase: "report", title: "Phase 8: Generate Final Report" },
                  ]
                  for (const detection of phaseDetections) {
                    if (content.includes(detection.match) && currentPhase !== detection.phase) {
                      currentPhase = detection.phase
                      log("\n" + "=".repeat(50) + `\n  ${detection.title}\n` + "=".repeat(50))
                      UI.println()
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                      UI.println(UI.Style.TEXT_INFO_BOLD + `  ${detection.title}`)
                      UI.println(UI.Style.TEXT_INFO_BOLD + "═══════════════════════════════════════")
                      break
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

