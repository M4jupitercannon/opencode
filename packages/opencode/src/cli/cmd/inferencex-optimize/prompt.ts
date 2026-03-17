/**
 * Prompt composer for inferencex-optimize pipeline.
 *
 * Reads pure markdown skill files (.md) and substitutes {{VAR}} placeholders.
 * All skill logic lives in .md files -- this file is minimal TS glue.
 */
import type { InferenceXConfig, PipelineMode } from "./types"
import { PHASE_ORDER } from "./types"

// @ts-ignore - Bun text import
import skill00 from "./skills/00-env-setup.md" with { type: "text" }
// @ts-ignore - Bun text import
import skill01 from "./skills/01-config-parse.md" with { type: "text" }
// @ts-ignore - Bun text import
import skill02 from "./skills/02-benchmark.md" with { type: "text" }
// @ts-ignore - Bun text import
import skill03 from "./skills/03-benchmark-analyze.md" with { type: "text" }
// @ts-ignore - Bun text import
import skill04 from "./skills/04-profile.md" with { type: "text" }
// @ts-ignore - Bun text import
import skill05 from "./skills/05-profile-analyze.md" with { type: "text" }
// @ts-ignore - Bun text import
import skillAgentConfig from "./skills/agent-config.md" with { type: "text" }

// @ts-ignore - Bun text import
import scriptTraceAnalyzer from "./scripts/trace_analyzer.py" with { type: "text" }
// @ts-ignore - Bun text import
import scriptSelectGpus from "./scripts/select_gpus.py" with { type: "text" }

const EMBEDDED_SKILLS: Record<string, string> = {
  "00-env-setup.md": skill00,
  "01-config-parse.md": skill01,
  "02-benchmark.md": skill02,
  "03-benchmark-analyze.md": skill03,
  "04-profile.md": skill04,
  "05-profile-analyze.md": skill05,
  "agent-config.md": skillAgentConfig,
}

export const EMBEDDED_SCRIPTS: Record<string, string> = {
  "trace_analyzer.py": scriptTraceAnalyzer,
  "select_gpus.py": scriptSelectGpus,
}

const SKILL_FILES = [
  "00-env-setup.md",
  "01-config-parse.md",
  "02-benchmark.md",
  "03-benchmark-analyze.md",
  "04-profile.md",
  "05-profile-analyze.md",
]

const MODE_PHASES: Record<PipelineMode, readonly string[]> = {
  full: ["env", "config", "benchmark", "benchmark-analyze", "profile", "profile-analyze"],
  benchmark: ["env", "config", "benchmark", "benchmark-analyze"],
  profile: ["env", "config", "profile", "profile-analyze"],
  "benchmark+profile": ["env", "config", "benchmark", "benchmark-analyze", "profile", "profile-analyze"],
  analyze: ["benchmark-analyze", "profile-analyze"],
}

function computeSkipLabels(startPhase: string, mode: PipelineMode): Record<string, string> {
  const phaseIndex: Record<string, number> = {}
  PHASE_ORDER.forEach((p, i) => (phaseIndex[p] = i))

  const startIdx = phaseIndex[startPhase] ?? 0
  const activePhases = MODE_PHASES[mode]
  const filePhaseMap: Record<string, string> = {
    "00": "env",
    "01": "config",
    "02": "benchmark",
    "03": "benchmark-analyze",
    "04": "profile",
    "05": "profile-analyze",
  }

  const labels: Record<string, string> = {}
  for (const [idx, phase] of Object.entries(filePhaseMap)) {
    const pIdx = phaseIndex[phase] ?? 0
    if (!activePhases.includes(phase)) {
      labels[idx] = "[SKIP - NOT IN SELECTED MODE]"
    } else if (pIdx < startIdx) {
      labels[idx] = "[SKIP - ALREADY DONE]"
    } else {
      labels[idx] = ""
    }
  }
  return labels
}

function substitute(template: string, vars: Record<string, string>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (match, key) => vars[key] ?? match)
}

function readSkill(filename: string): string {
  const content = EMBEDDED_SKILLS[filename]
  if (!content) throw new Error(`Unknown skill file: ${filename}`)
  return content
}

export function buildAgentConfig(config: InferenceXConfig): string {
  const template = readSkill("agent-config.md")
  return substitute(template, { CONFIG_KEY: config.configKey })
}

export function buildAgentPrompt(config: InferenceXConfig): string {
  const {
    configKey,
    outputDir,
    dirs,
    repoUrl,
    repoDir,
    hfCache,
    filterTp,
    filterConcStart,
    filterConcEnd,
    filterSeq,
    gpus,
    dryRun,
    profile,
    mode,
    startPhase,
    existingProgress,
  } = config

  const skipLabels = computeSkipLabels(startPhase, mode)
  const activePhases = MODE_PHASES[mode]

  const vars: Record<string, string> = {
    CONFIG_KEY: configKey,
    OUTPUT_DIR: outputDir,
    REPO_DIR: repoDir,
    REPO_URL: repoUrl,
    HF_CACHE: hfCache,
    RESULTS_DIR: dirs.results,
    PROFILE_DIR: dirs.profiles,
    REPORT_DIR: dirs.report,
    SCRIPTS_DIR: outputDir + "/scripts",
    FILTER_TP: filterTp,
    FILTER_CONC_START: filterConcStart,
    FILTER_CONC_END: filterConcEnd,
    FILTER_SEQ: filterSeq,
    GPUS: gpus,
    DRY_RUN: String(dryRun),
    PROFILE: String(profile),
    DRY_RUN_NOTE: dryRun
      ? "**DRY RUN MODE**: Only print Docker commands, do not execute them."
      : "",
    PROFILE_SKIP_NOTE: activePhases.includes("profile")
      ? ""
      : "**NOTE**: Profiling was not requested. Skip this phase entirely.",
    PROFILE_ANALYSIS_NOTE: activePhases.includes("profile")
      ? ""
      : "**NOTE**: Profiling may not have been run. If no profile traces exist in `" + dirs.profiles + "`, skip this phase entirely and proceed to the next phase.",
    PROGRESS_FILE: outputDir + "/progress.json",
    START_PHASE: startPhase,
  }

  for (const [idx, label] of Object.entries(skipLabels)) {
    vars["SKIP_LABEL_" + idx] = label
  }

  const filePhaseMap: Record<string, string> = {
    "00": "env",
    "01": "config",
    "02": "benchmark",
    "03": "benchmark-analyze",
    "04": "profile",
    "05": "profile-analyze",
  }

  const filteredSkillFiles = SKILL_FILES.filter((file) => {
    const idx = file.slice(0, 2)
    const phase = filePhaseMap[idx]
    return phase ? activePhases.includes(phase) : true
  })

  const phasePrompts = filteredSkillFiles.map((file) => {
    const template = readSkill(file)
    const idx = file.slice(0, 2)
    const phaseVars = { ...vars, SKIP_LABEL: skipLabels[idx] || "" }
    return substitute(template, phaseVars)
  })

  const header = buildHeader(vars, startPhase, existingProgress, mode)
  const footer = buildExecutionInstructions(vars, startPhase, mode)

  return [header, ...phasePrompts, footer].join("\n\n---\n\n")
}

function buildHeader(
  vars: Record<string, string>,
  startPhase: string,
  existingProgress: any,
  mode: PipelineMode,
): string {
  let resumeContext = ""
  if (startPhase !== "env") {
    let prevProgress = ""
    if (existingProgress) {
      const completed = existingProgress.phases_completed?.join(", ") || "none"
      prevProgress = "\n### Previous Progress\n- Phases completed: " + completed + "\n"
    }
    resumeContext = "\n## RESUME MODE ACTIVE\n" +
      "**Starting from Phase: " + startPhase + "**\n" +
      prevProgress +
      "\n**IMPORTANT**: Skip phases before \"" + startPhase + "\" — their artifacts already exist.\n" +
      "Review *input* artifacts from previous phases to understand current state, " +
      "but **always re-run the starting phase fully** even if its output artifacts already exist " +
      "(the user is explicitly requesting a re-run of this phase).\n\n---\n"
  }

  const activePhases = MODE_PHASES[mode]
  const modeLabel = mode === "full" ? "Full Pipeline" :
    mode === "benchmark" ? "Benchmark Only" :
    mode === "profile" ? "Profile Only" :
    mode === "analyze" ? "Analyze Only" :
    "Benchmark + Profile"

  return "# InferenceX Benchmark & Profiling Pipeline\n\n" +
    "## Target Configuration\n" +
    "- **Config Key**: " + vars.CONFIG_KEY + "\n" +
    "- **InferenceX Repo**: " + vars.REPO_DIR + "\n" +
    "- **Mode**: " + modeLabel + " (" + activePhases.join(" → ") + ")\n" +
    resumeContext +
    "\n## Output Directory Structure\n" +
    "```\n" +
    vars.OUTPUT_DIR + "/\n" +
    "  InferenceX/     # InferenceX repository clone\n" +
    "  results/        # Benchmark results and analysis\n" +
    "  profiles/       # Profiling trace files\n" +
    "  report/         # Final benchmark report\n" +
    "  scripts/        # Pipeline utility scripts (e.g. trace_analyzer.py)\n" +
    "  config.json     # Pipeline configuration\n" +
    "  progress.json   # Progress tracking\n" +
    "```\n\n" +
    "## Key Parameters\n" +
    "- **Mode**: " + modeLabel + "\n" +
    "- **Filter TP**: " + (vars.FILTER_TP || "all") + "\n" +
    "- **Filter Concurrency**: " + (vars.FILTER_CONC_START || vars.FILTER_CONC_END ? (vars.FILTER_CONC_START || "1") + " – " + (vars.FILTER_CONC_END || "∞") : "all") + "\n" +
    "- **Filter Sequence Length**: " + (vars.FILTER_SEQ || "all") + "\n" +
    "- **Dry Run**: " + vars.DRY_RUN + "\n\n" +
    "## IMPORTANT FILES\n" +
    "- **Config**: " + vars.OUTPUT_DIR + "/config.json\n" +
    "- **Progress**: " + vars.PROGRESS_FILE + "\n\n" +
    "Update progress.json after completing each phase!\n\n" +
    "## YOUR TASK: Run " + modeLabel + " — phases: " + activePhases.join(", ")
}

function buildExecutionInstructions(
  vars: Record<string, string>,
  startPhase: string,
  mode: PipelineMode,
): string {
  const activePhases = MODE_PHASES[mode]
  const modeLabel = mode === "full" ? "Full Pipeline" :
    mode === "benchmark" ? "Benchmark Only" :
    mode === "profile" ? "Profile Only" :
    mode === "analyze" ? "Analyze Only" :
    "Benchmark + Profile"

  let modeInstructions: string
  if (startPhase === "env") {
    modeInstructions =
      "\n## " + modeLabel + " Mode\n" +
      "1. **Execute these phases in order**: " + activePhases.join(" → ") + "\n" +
      "2. **ONLY run the phases listed above** — skip all other phases\n" +
      "3. Begin with Phase 0: Environment Setup\n"
  } else {
    modeInstructions =
      '\n## ' + modeLabel + ' Mode — Starting from "' + startPhase + '"\n' +
      '1. **Skip phases before "' + startPhase + '"** — their artifacts already exist\n' +
      "2. **ONLY run these phases**: " + activePhases.join(" → ") + "\n" +
      "3. **Review input artifacts from previous phases** to understand current state\n" +
      "4. **Re-run Phase " + startPhase + " fully** — clean up any old output from this phase and re-run from scratch\n\n" +
      "### Quick Start Checklist\n" +
      "- [ ] Read existing progress.json\n" +
      "- [ ] Verify input artifacts from previous phases exist\n" +
      "- [ ] Clean up old output from Phase: " + startPhase + " (if any)\n" +
      "- [ ] Run Phase: " + startPhase + " from scratch\n"
  }

  return "# EXECUTION INSTRUCTIONS\n" +
    modeInstructions + "\n" +
    "## General Rules\n" +
    "1. **Update progress.json after each phase**\n" +
    "2. **If a benchmark fails, log the error and continue with the next one**\n" +
    "3. **Never modify the InferenceX repository source code**\n" +
    "4. **Save all outputs to the designated output directory**\n" +
    "5. **If a Docker container hangs for more than 30 minutes, kill it and move on**\n\n" +
    "## Start Now\n" +
    "Begin with Phase: " + startPhase.charAt(0).toUpperCase() + startPhase.slice(1)
}
