/**
 * CLI command for InferenceX benchmarking and profiling.
 *
 * Orchestrates the pipeline by composing independent skill modules
 * and managing the opencode server lifecycle. All benchmark and profiling
 * steps are derived from run_local.sh.
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

import type { InferenceXConfig, InferenceXDirs } from "./types"
import { PHASE_ORDER } from "./types"
import { buildAgentPrompt, buildAgentConfig } from "./prompt"

const DEFAULT_REPO_URL = "https://github.com/SemiAnalysisAI/InferenceX.git"

export const InferenceXOptimizeCommand = cmd({
  command: "inferencex-optimize",
  describe: "InferenceX benchmark & profiling pipeline",
  builder: (yargs) =>
    yargs
      .option("config-key", {
        type: "string",
        alias: "k",
        describe: "Config key from master YAML (e.g., kimik2.5-int4-mi355x-vllm)",
        demandOption: true,
      })
      .option("output", {
        type: "string",
        alias: "o",
        describe: "output directory for results (default: ./inferencex_<config-key>)",
      })
      .option("llm", {
        type: "string",
        describe: "LLM model to use for orchestration (e.g., opencode/glm-4.7-free)",
      })
      .option("repo-dir", {
        type: "string",
        describe: "path to existing InferenceX repo clone",
      })
      .option("repo-url", {
        type: "string",
        describe: "InferenceX git repo URL",
        default: DEFAULT_REPO_URL,
      })
      .option("hf-cache", {
        type: "string",
        describe: "HuggingFace cache directory (default: $HF_HUB_CACHE or ~/.cache/huggingface)",
      })
      .option("conc", {
        type: "string",
        describe: "filter to specific concurrency level",
      })
      .option("seq-len", {
        type: "string",
        describe: "filter to specific sequence length (1k1k, 1k8k, 8k1k)",
      })
      .option("dry-run", {
        type: "boolean",
        describe: "preview Docker commands without running them",
        default: false,
      })
      .option("profile", {
        type: "boolean",
        describe: "enable profiling (trace saved to profiles/)",
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
        describe: "start from specific phase (env, config, benchmark, profile, analyze, report)",
      }),
  async handler(args) {
    const llmArg = args.llm as string | undefined

    const gatewayKey = process.env.LLM_GATEWAY_KEY
    const needsGatewayKey = !llmArg || llmArg.startsWith("amd-")
    if (needsGatewayKey && !gatewayKey) {
      UI.error("LLM_GATEWAY_KEY environment variable is not set")
      UI.println("Please set it with: export LLM_GATEWAY_KEY=your-api-key")
      UI.println("")
      UI.println("Or use a free model that doesn't require a key:")
      UI.println("  opencode inferencex-optimize -k <config-key> --llm opencode/glm-4.7-free")
      process.exit(1)
    }

    const configKey = args["config-key"] as string
    const configKeySafe = configKey.replace(/[^a-zA-Z0-9_-]/g, "_")

    const outputDir =
      (args.output as string) || path.resolve(`./inferencex_${configKeySafe}`)

    const hfCache =
      (args["hf-cache"] as string) ||
      process.env.HF_HUB_CACHE ||
      path.join(process.env.HOME || "~", ".cache/huggingface")

    const repoDir = (args["repo-dir"] as string) || path.join(outputDir, "repo")

    UI.println("============================================")
    UI.println("InferenceX Benchmark Pipeline")
    UI.println("============================================")
    UI.println(`Config Key:        ${configKey}`)
    UI.println(`Output Directory:  ${outputDir}`)
    UI.println(`Repo Directory:    ${repoDir}`)
    UI.println(`HF Cache:          ${hfCache}`)
    UI.println(`Dry Run:           ${args["dry-run"]}`)
    UI.println(`Profiling:         ${args.profile}`)
    if (args.conc) UI.println(`Filter Conc:       ${args.conc}`)
    if (args["seq-len"]) UI.println(`Filter Seq Len:    ${args["seq-len"]}`)
    if (llmArg) UI.println(`LLM Model:         ${llmArg}`)
    UI.println("============================================")
    UI.println("")

    fs.mkdirSync(outputDir, { recursive: true })

    const dirs: InferenceXDirs = {
      repo: repoDir,
      results: path.join(outputDir, "results"),
      profiles: path.join(outputDir, "profiles"),
      report: path.join(outputDir, "report"),
    }
    Object.values(dirs).forEach((d) => fs.mkdirSync(d, { recursive: true }))

    const resumeMode = args.resume as boolean
    const fromPhase = args["from-phase"] as string | undefined
    const progressFile = path.join(outputDir, "progress.json")

    let existingProgress: any = null
    let startPhase = "env"

    if (resumeMode || fromPhase) {
      if (fs.existsSync(progressFile)) {
        try {
          existingProgress = JSON.parse(fs.readFileSync(progressFile, "utf-8"))
          UI.println(UI.Style.TEXT_INFO_BOLD + "Found existing project progress")

          if (fromPhase) {
            const validPhases = [...PHASE_ORDER]
            if (!validPhases.includes(fromPhase as any)) {
              UI.error(`Invalid phase: ${fromPhase}. Valid phases: ${validPhases.join(", ")}`)
              process.exit(1)
            }
            startPhase = fromPhase
            UI.println(`Starting from phase: ${startPhase}`)
          } else if (resumeMode && existingProgress.phases_completed) {
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
        } catch {
          UI.println(UI.Style.TEXT_WARNING + "Could not parse existing progress.json, starting fresh")
        }
      } else {
        UI.println(UI.Style.TEXT_WARNING + "No existing progress.json found, starting from beginning")
      }
    }

    const configFile = path.join(outputDir, "config.json")
    const configData = {
      config_key: configKey,
      dirs,
      created: new Date().toISOString(),
      start_phase: startPhase,
      resume_mode: resumeMode || !!fromPhase,
      dry_run: args["dry-run"],
      profile: args.profile,
      filter_conc: args.conc || "",
      filter_seq: args["seq-len"] || "",
      repo_url: args["repo-url"],
      hf_cache: hfCache,
    }
    fs.writeFileSync(configFile, JSON.stringify(configData, null, 2))

    const progress = existingProgress || {
      phase: "init",
      phases_completed: [] as string[],
      current_step: "",
      errors: [] as string[],
    }
    if (!existingProgress) {
      fs.writeFileSync(progressFile, JSON.stringify(progress, null, 2))
    }

    const pipelineConfig: InferenceXConfig = {
      configKey,
      outputDir,
      dirs,
      repoUrl: args["repo-url"] as string,
      repoDir,
      hfCache,
      filterConc: (args.conc as string) || "",
      filterSeq: (args["seq-len"] as string) || "",
      dryRun: args["dry-run"] as boolean,
      profile: args.profile as boolean,
      startPhase,
      existingProgress,
    }

    const prompt = buildAgentPrompt(pipelineConfig)

    const opencodeDir = path.join(outputDir, ".opencode")
    const agentDir = path.join(opencodeDir, "agent")
    fs.mkdirSync(agentDir, { recursive: true })

    const agentConfig = buildAgentConfig(pipelineConfig)
    fs.writeFileSync(path.join(agentDir, "inferencex-opt.md"), agentConfig)

    const opencodeConfigObj = {
      "$schema": "https://opencode.ai/config.json",
      model: "amd-anthropic/claude-opus-4-5",
      default_agent: "inferencex-opt",
      provider: {
        "amd-anthropic": { options: { timeout: 1200000 } },
      },
      permission: {
        "*": "allow",
        bash: "allow",
        edit: { "*": "allow", "/opt/*": "deny", "/usr/*": "deny" },
        read: "allow",
        write: { "*": "allow", "/opt/*": "deny", "/usr/*": "deny" },
        glob: "allow",
        grep: "allow",
        list: "allow",
        task: "allow",
        external_directory: "allow",
        todowrite: "allow",
        todoread: "allow",
        question: "allow",
        webfetch: "allow",
        websearch: "allow",
        codesearch: "allow",
        lsp: "allow",
        doom_loop: "allow",
      },
    }
    fs.writeFileSync(
      path.join(opencodeDir, "opencode.jsonc"),
      JSON.stringify(opencodeConfigObj, null, 2),
    )

    await bootstrap(outputDir, async () => {
      const server = Server.listen({ port: 0, hostname: "127.0.0.1" })
      const sdk = createOpencodeClient({
        baseUrl: `http://${server.hostname}:${server.port}`,
      })

      try {
        const sessionResult = await sdk.session.create()
        const sessionID = sessionResult.data?.id
        if (!sessionID) {
          UI.error("Failed to create session")
          process.exit(1)
        }

        const events = await sdk.event.subscribe()
        UI.println("Session created, sending prompt...")

        const logFilePath = path.join(outputDir, "pipeline.log")
        const logStream = fs.createWriteStream(logFilePath, { flags: "a" })
        const log = (msg: string) => {
          const timestamp = new Date().toISOString()
          logStream.write(`[${timestamp}] ${msg}\n`)
        }
        log("=".repeat(60))
        log(`InferenceX Pipeline Started: ${configKey}`)
        log(`Output Directory: ${outputDir}`)
        log(`LLM Model: ${llmArg || "default"}`)
        log("=".repeat(60))
        UI.println(UI.Style.TEXT_DIM + `Detailed log: ${logFilePath}`)

        let currentPhase = ""
        const eventProcessor = (async () => {
          for await (const event of events.stream) {
            if (event.type === "message.part.updated") {
              const part = event.properties.part
              if (part.sessionID !== sessionID) continue

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

                if (tool === "bash") {
                  const shellCmd = input.command || title
                  log(`\n$ ${shellCmd}`)
                  UI.println(UI.Style.TEXT_INFO_BOLD + "$ " + UI.Style.TEXT_DIM + title)
                  if (part.state.output?.trim()) {
                    const output = part.state.output.trim()
                    if (output.length > 2000) {
                      log(`${output.slice(0, 2000)}\n... (truncated, ${output.length} chars total)`)
                    } else {
                      log(output)
                    }
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
                  const verb = tool === "write" ? "Creating" : "Editing"
                  UI.println(UI.Style.TEXT_SUCCESS + `${verb}: ` + UI.Style.TEXT_DIM + shortPath)
                } else if (tool === "read") {
                  // skip noisy read logs
                } else if (tool === "todowrite") {
                  const todos = input.todos || []
                  const inProgress = todos.filter((t: any) => t.status === "in_progress")
                  const completed = todos.filter((t: any) => t.status === "completed")
                  if (inProgress.length > 0) {
                    UI.println(UI.Style.TEXT_INFO + "In Progress: " + inProgress.map((t: any) => t.content).join(", "))
                  }
                  if (completed.length > 0) {
                    UI.println(UI.Style.TEXT_SUCCESS + "Completed: " + completed.map((t: any) => t.content).join(", "))
                  }
                } else {
                  if (title) {
                    UI.println(UI.Style.TEXT_DIM + `[${tool}] ${title}`)
                  }
                }
              }

              if (part.type === "text") {
                const textPart = part as any
                if (textPart.state?.done && textPart.state.content?.trim()) {
                  const content = textPart.state.content
                  const phaseDetections: Array<{ match: string; phase: string; title: string }> = [
                    { match: "Phase 0", phase: "env", title: "Phase 0: Environment Setup" },
                    { match: "Environment Setup", phase: "env", title: "Phase 0: Environment Setup" },
                    { match: "Phase 1", phase: "config", title: "Phase 1: Config Parsing" },
                    { match: "Config Pars", phase: "config", title: "Phase 1: Config Parsing" },
                    { match: "Phase 2", phase: "benchmark", title: "Phase 2: Benchmark Execution" },
                    { match: "Benchmark Execution", phase: "benchmark", title: "Phase 2: Benchmark Execution" },
                    { match: "Phase 3", phase: "profile", title: "Phase 3: Profiling" },
                    { match: "Profiling", phase: "profile", title: "Phase 3: Profiling" },
                    { match: "Phase 4", phase: "analyze", title: "Phase 4: Results Analysis" },
                    { match: "Results Analysis", phase: "analyze", title: "Phase 4: Results Analysis" },
                    { match: "Phase 5", phase: "report", title: "Phase 5: Report Generation" },
                    { match: "Report Generation", phase: "report", title: "Phase 5: Report Generation" },
                  ]
                  for (const detection of phaseDetections) {
                    if (content.includes(detection.match) && currentPhase !== detection.phase) {
                      currentPhase = detection.phase
                      log("\n" + "=".repeat(50) + `\n  ${detection.title}\n` + "=".repeat(50))
                      UI.println()
                      UI.println(UI.Style.TEXT_INFO_BOLD + "=".repeat(39))
                      UI.println(UI.Style.TEXT_INFO_BOLD + `  ${detection.title}`)
                      UI.println(UI.Style.TEXT_INFO_BOLD + "=".repeat(39))
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
            if (event.type === "permission.asked") {
              const permission = event.properties as any
              if (permission.sessionID !== sessionID) continue

              const permType = permission.permission || ""
              const patterns = (permission.patterns || []).join(", ")

              const isAutoApprove = [
                "read", "external_directory", "glob", "grep", "list",
                "codesearch", "lsp", "bash", "task", "todowrite",
                "todoread", "webfetch", "websearch", "question",
              ].includes(permType)

              if (isAutoApprove) {
                UI.println(UI.Style.TEXT_DIM + `[Auto-approved: ${permType}] ${patterns}`)
                await sdk.permission.respond({
                  sessionID,
                  permissionID: permission.id,
                  response: "always",
                })
              } else {
                UI.println()
                UI.println(UI.Style.TEXT_WARNING_BOLD + "Permission required:")
                UI.println(`  Type: ${permType}`)
                UI.println(`  Patterns: ${patterns}`)
                const result = await select({
                  message: "Allow this action?",
                  options: [
                    { value: "once", label: "Allow once" },
                    { value: "always", label: `Always allow: ${(permission.always || []).join(", ")}` },
                    { value: "reject", label: "Reject" },
                  ],
                  initialValue: "once",
                }).catch(() => "reject")
                const response = (
                  result.toString().includes("cancel") ? "reject" : result
                ) as "once" | "always" | "reject"
                await sdk.permission.respond({
                  sessionID,
                  permissionID: permission.id,
                  response,
                })
              }
            }
          }
        })()

        let modelParam
        if (llmArg) {
          modelParam = Provider.parseModel(llmArg)
        } else if (gatewayKey) {
          modelParam = Provider.parseModel("amd-anthropic/claude-opus-4-5")
          UI.println("Using default AMD gateway model: amd-anthropic/claude-opus-4-5")
        }
        UI.println("Sending prompt to LLM...")
        await sdk.session.prompt({
          sessionID,
          model: modelParam,
          parts: [{ type: "text", text: prompt }],
        })
        UI.println("Prompt sent, waiting for completion...")

        await eventProcessor
        log("Pipeline completed")
        logStream.end()
        UI.println("Event processor completed")
        UI.println(UI.Style.TEXT_SUCCESS + `Full log saved to: ${logFilePath}`)
      } finally {
        try {
          fs.rmSync(opencodeDir, { recursive: true })
        } catch {
          // ignore cleanup errors
        }
        server.stop()
      }
    })

    UI.println("")
    UI.println("============================================")
    UI.println("InferenceX Pipeline Complete")
    UI.println("============================================")
    UI.println(`Output directory: ${outputDir}`)
    UI.println(`Report: ${path.join(dirs.report, "benchmark_report.md")}`)
    UI.println("============================================")
  },
})
