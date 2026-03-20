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
import { execSync } from "child_process"
import { UI } from "../../ui"
import { Provider } from "../../../provider/provider"
import { select } from "@clack/prompts"

import type { InferenceXConfig, InferenceXDirs, PipelineMode } from "./types"
import { PHASE_ORDER } from "./types"
import { buildAgentPrompt, buildAgentConfig, EMBEDDED_SCRIPTS } from "./prompt"

const DOCKER_LABEL = "inferencex-pipeline=true"

function cleanupDockerContainers() {
  try {
    const ids = execSync(
      `docker ps -q --filter label=${DOCKER_LABEL}`,
      { encoding: "utf-8", timeout: 10_000 },
    ).trim()
    if (ids) {
      UI.println("Stopping pipeline Docker containers...")
      execSync(`docker stop ${ids.split("\n").join(" ")}`, {
        encoding: "utf-8",
        timeout: 30_000,
      })
    }
  } catch {
    // best-effort cleanup
  }
}

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
        describe: "path for InferenceX repo (uses existing clone or clones here; default: <output>/repo)",
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
      .option("tp", {
        type: "number",
        describe: "filter to specific tensor parallelism level from config search-space (e.g., 1, 4, 8)",
      })
      .option("ep", {
        type: "number",
        describe: "filter to specific expert parallelism level from config search-space (e.g., 1, 8)",
      })
      .option("conc-start", {
        type: "number",
        describe: "filter to concurrency levels >= this value",
      })
      .option("conc-end", {
        type: "number",
        describe: "filter to concurrency levels <= this value",
      })
      .option("seq-len", {
        type: "string",
        describe: "filter to specific sequence length (1k1k, 1k8k, 8k1k)",
      })
      .option("gpus", {
        type: "string",
        describe: "comma-separated GPU device IDs to use (e.g., 0,1,2,3); overrides auto-selection. Also respects CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES env vars",
      })
      .option("dry-run", {
        type: "boolean",
        describe: "preview Docker commands without running them",
        default: false,
      })
      .option("benchmark", {
        type: "boolean",
        describe: "run benchmark without profiling (env + config + benchmark + benchmark-analyze)",
        default: false,
      })
      .option("profile", {
        type: "boolean",
        describe: "run profiling only (env + config + profile + profile-analyze)",
        default: false,
      })
      .option("analyze", {
        type: "boolean",
        describe: "analyze existing traces/results only (benchmark-analyze + profile-analyze), use with -o to point at existing output dir",
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
        describe: "start from specific phase (env, config, benchmark, benchmark-analyze, profile, profile-analyze)",
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

    const now = new Date()
    const timestamp = [
      now.getFullYear(),
      String(now.getMonth() + 1).padStart(2, "0"),
      String(now.getDate()).padStart(2, "0"),
      "_",
      String(now.getHours()).padStart(2, "0"),
      String(now.getMinutes()).padStart(2, "0"),
      String(now.getSeconds()).padStart(2, "0"),
    ].join("")

    const outputDir =
      (args.output as string) || path.resolve(`./inferencex_${configKeySafe}_${timestamp}`)

    const hfCache =
      (args["hf-cache"] as string) ||
      process.env.HF_HUB_CACHE ||
      path.join(process.env.HOME || "~", ".cache/huggingface")

    const repoDir = (args["repo-dir"] as string) || path.join(outputDir, "InferenceX")

    UI.println("============================================")
    UI.println("InferenceX Benchmark Pipeline")
    UI.println("============================================")
    UI.println(`Config Key:        ${configKey}`)
    UI.println(`Output Directory:  ${outputDir}`)
    UI.println(`Repo Directory:    ${repoDir}`)
    UI.println(`HF Cache:          ${hfCache}`)
    const gpus =
      (args.gpus as string | undefined) ||
      process.env.CUDA_VISIBLE_DEVICES ||
      process.env.ROCR_VISIBLE_DEVICES ||
      process.env.HIP_VISIBLE_DEVICES ||
      ""

    const benchmarkOnly = args.benchmark as boolean
    const profileOnly = args.profile as boolean
    const analyzeOnly = args.analyze as boolean
    let mode: PipelineMode = "full"
    if (analyzeOnly) mode = "analyze"
    else if (benchmarkOnly && profileOnly) mode = "benchmark+profile"
    else if (benchmarkOnly) mode = "benchmark"
    else if (profileOnly) mode = "profile"

    UI.println(`Dry Run:           ${args["dry-run"]}`)
    UI.println(`Mode:              ${mode}`)
    if (args.tp != null) UI.println(`Filter TP:         ${args.tp}`)
    if (args.ep != null) UI.println(`Filter EP:         ${args.ep}`)
    if (args["conc-start"] != null || args["conc-end"] != null) {
      const start = args["conc-start"] != null ? String(args["conc-start"]) : "1"
      const end = args["conc-end"] != null ? String(args["conc-end"]) : "∞"
      UI.println(`Filter Conc:       ${start} – ${end}`)
    }
    if (args["seq-len"]) UI.println(`Filter Seq Len:    ${args["seq-len"]}`)
    if (gpus) UI.println(`GPUs:              ${gpus}`)
    else UI.println(`GPUs:              auto (most free)`)
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
    let fromPhase = args["from-phase"] as string | undefined
    const progressFile = path.join(outputDir, "progress.json")

    if (analyzeOnly) {
      if (!fromPhase) fromPhase = "benchmark-analyze"
      if (!args.output) {
        UI.error("--analyze requires --output (-o) pointing to an existing pipeline output directory")
        UI.println("Example: opencode inferencex-optimize -k <config-key> --analyze -o ./inferencex_<config-key>_<timestamp>")
        process.exit(1)
      }
    }

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
                startPhase = phasesOrder[i + 1] || phasesOrder[phasesOrder.length - 1]
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
      mode,
      filter_tp: args.tp != null ? String(args.tp) : "",
      filter_ep: args.ep != null ? String(args.ep) : "",
      filter_conc_start: args["conc-start"] != null ? String(args["conc-start"]) : "",
      filter_conc_end: args["conc-end"] != null ? String(args["conc-end"]) : "",
      filter_seq: args["seq-len"] || "",
      repo_url: args["repo-url"],
      hf_cache: hfCache,
      gpus,
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

    const scriptsDir = path.join(outputDir, "scripts")
    fs.mkdirSync(scriptsDir, { recursive: true })
    for (const [name, content] of Object.entries(EMBEDDED_SCRIPTS)) {
      fs.writeFileSync(path.join(scriptsDir, name), content)
    }

    const onExit = () => {
      cleanupDockerContainers()
      process.exit(1)
    }
    process.on("SIGINT", onExit)
    process.on("SIGTERM", onExit)

    const pipelineConfig: InferenceXConfig = {
      configKey,
      outputDir,
      dirs,
      repoUrl: args["repo-url"] as string,
      repoDir,
      hfCache,
      filterTp: args.tp != null ? String(args.tp) : "",
      filterEp: args.ep != null ? String(args.ep) : "",
      filterConcStart: args["conc-start"] != null ? String(args["conc-start"]) : "",
      filterConcEnd: args["conc-end"] != null ? String(args["conc-end"]) : "",
      filterSeq: (args["seq-len"] as string) || "",
      gpus,
      dryRun: args["dry-run"] as boolean,
      profile: mode === "full" || mode === "profile" || mode === "benchmark+profile" || mode === "analyze",
      mode,
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
      model: `amd-anthropic/${process.env.ANTHROPIC_DEFAULT_OPUS_MODEL || "claude-opus-4-6"}`,
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
        const shownBashOutput = new Map<string, number>()
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

              if (part.type === "tool") {
                const tool = part.tool
                const state = part.state as any
                const title = state.title || ""
                const input = (state.input || {}) as Record<string, any>
                const partId = (part as any).id || ""

                if (tool === "bash" && state.status === "running") {
                  const output = (state.output as string) || ""
                  const shown = shownBashOutput.get(partId) || 0
                  if (shown === 0 && title) {
                    UI.println(UI.Style.TEXT_INFO_BOLD + "$ " + UI.Style.TEXT_DIM + title)
                    log(`\n$ ${input.command || title}`)
                  }
                  if (output.length > shown) {
                    const newContent = output.slice(shown)
                    log(newContent)
                    UI.println(newContent.trimEnd())
                    shownBashOutput.set(partId, output.length)
                  }
                }

                if (state.status === "completed") {
                  const wasStreamed = partId && shownBashOutput.has(partId)

                  if (tool === "bash") {
                    if (!wasStreamed) {
                      const shellCmd = input.command || title
                      log(`\n$ ${shellCmd}`)
                      UI.println(UI.Style.TEXT_INFO_BOLD + "$ " + UI.Style.TEXT_DIM + title)
                      if (state.output?.trim()) {
                        const output = (state.output as string).trim()
                        if (output.length > 2000) {
                          log(`${output.slice(0, 2000)}\n... (truncated, ${output.length} chars total)`)
                        } else {
                          log(output)
                        }
                        const lines = output.split("\n")
                        if (lines.length > 50) {
                          UI.println(lines.slice(0, 40).join("\n"))
                          UI.println(UI.Style.TEXT_DIM + `... (${lines.length - 40} more lines)`)
                        } else {
                          UI.println(output)
                        }
                      }
                    } else {
                      const output = (state.output as string) || ""
                      const shown = shownBashOutput.get(partId) || 0
                      if (output.length > shown) {
                        const remaining = output.slice(shown)
                        log(remaining)
                        UI.println(remaining.trimEnd())
                      }
                      shownBashOutput.delete(partId)
                    }
                    const shellCmd = input.command || title
                    const combined = shellCmd + "\n" + ((state.output as string) || "")
                    const resolvedMatch = combined.match(/(?:Trying|Found):\s*(\/[^\s]+\.sh)/)
                    const fallbackMatch = combined.match(/BENCHMARK_SCRIPT=["']?([^$\s"']+\.sh)/)
                    const scriptPath = resolvedMatch?.[1] || fallbackMatch?.[1]
                    if (scriptPath) {
                      UI.println(UI.Style.TEXT_INFO_BOLD + "Benchmark script: " + scriptPath)
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
                    if (completed.length > 0) {
                      UI.println(UI.Style.TEXT_SUCCESS + "Completed: " + completed.map((t: any) => t.content).join(", "))
                    }
                    if (inProgress.length > 0) {
                      UI.println(UI.Style.TEXT_INFO + "In Progress: " + inProgress.map((t: any) => t.content).join(", "))
                    }
                  } else {
                    if (title) {
                      UI.println(UI.Style.TEXT_DIM + `[${tool}] ${title}`)
                    }
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
                    { match: "Phase 3", phase: "benchmark-analyze", title: "Phase 3: Benchmark Analysis" },
                    { match: "Benchmark Analysis", phase: "benchmark-analyze", title: "Phase 3: Benchmark Analysis" },
                    { match: "Phase 4", phase: "profile", title: "Phase 4: Profiling" },
                    { match: "Profiling", phase: "profile", title: "Phase 4: Profiling" },
                    { match: "Phase 5", phase: "profile-analyze", title: "Phase 5: Profile Analysis" },
                    { match: "Profile Analysis", phase: "profile-analyze", title: "Phase 5: Profile Analysis" },
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
          const opusModel = process.env.ANTHROPIC_DEFAULT_OPUS_MODEL || "claude-opus-4-6"
          modelParam = Provider.parseModel(`amd-anthropic/${opusModel}`)
          UI.println(`Using default AMD gateway model: amd-anthropic/${opusModel}`)
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
        cleanupDockerContainers()
        process.removeListener("SIGINT", onExit)
        process.removeListener("SIGTERM", onExit)
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

    const profilingReportPath = path.join(dirs.report, "profiling_report.md")
    const benchmarkReportPath = path.join(dirs.report, "benchmark_report.md")
    const reportPath = fs.existsSync(profilingReportPath) ? profilingReportPath : benchmarkReportPath
    UI.println(`Report: ${reportPath}`)
    if (fs.existsSync(reportPath)) {
      UI.println("============================================")
      UI.println("")
      const reportContent = fs.readFileSync(reportPath, "utf-8")
      UI.println(reportContent)
    }
    UI.println("============================================")
  },
})
