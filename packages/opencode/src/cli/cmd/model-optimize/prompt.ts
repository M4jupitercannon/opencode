/**
 * Prompt composer for model-optimize pipeline.
 *
 * Reads pure markdown skill files (.md) and substitutes {{VAR}} placeholders.
 * All skill logic lives in .md files — this file is minimal TS glue.
 */
import * as fs from "fs"
import * as path from "path"
import type { ModelOptConfig } from "./types"
import { PHASE_ORDER } from "./types"

/** Directory containing the .md skill files */
const SKILLS_DIR = path.join(import.meta.dir, "skills")

/** Ordered list of skill files to compose */
const SKILL_FILES = [
  "00-env-setup.md",
  "01-model-download.md",
  "02-demo-generate.md",
  "03-compat-fix.md",
  "04-profiling.md",
  "05-problem-generate.md",
  "06-kernel-optimize.md",
  "07-integration.md",
  "08-report-generate.md",
]

/** Compute skip labels for each phase based on startPhase */
function computeSkipLabels(startPhase: string, skipDownload: boolean): Record<string, string> {
  // Phase skip conditions (preserved from original behavior)
  const skipConditions: Record<string, (sp: string) => boolean> = {
    "00": (sp) => sp !== "download" && sp !== "env",
    "01": (sp) => sp !== "download",
    "02": (sp) => sp !== "download" && sp !== "env" && sp !== "demo",
    "03": (sp) => !["env", "download", "demo"].includes(sp) && sp !== "compatibility",
    "04": (sp) => !["env", "download", "demo", "compatibility"].includes(sp) && sp !== "profile",
    "05": (sp) => !["env", "download", "demo", "compatibility", "profile"].includes(sp),
    "06": (sp) => !["env", "download", "demo", "compatibility", "profile", "problems"].includes(sp),
    "07": (sp) => !["env", "download", "demo", "compatibility", "profile", "problems", "optimize"].includes(sp),
    "08": (sp) => sp !== "report" && !["env", "download", "demo", "compatibility", "profile", "problems", "optimize", "integrate"].includes(sp),
  }

  const labels: Record<string, string> = {}
  for (const [phase, shouldSkip] of Object.entries(skipConditions)) {
    labels[phase] = shouldSkip(startPhase) ? "[SKIP - ALREADY DONE]" : ""
  }
  return labels
}

/** Replace all {{VAR}} placeholders in a template string */
function substitute(template: string, vars: Record<string, string>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (match, key) => vars[key] ?? match)
}

/** Read an .md file from the skills directory */
function readSkill(filename: string): string {
  return fs.readFileSync(path.join(SKILLS_DIR, filename), "utf-8")
}

/** Read the agent config .md file and substitute variables */
export function buildAgentConfig(config: ModelOptConfig): string {
  const template = readSkill("agent-config.md")
  return substitute(template, { HF_MODEL: config.hfModel })
}

/** Build the complete agent prompt by composing all skill .md files */
export function buildAgentPrompt(config: ModelOptConfig): string {
  const { hfModel, modelName, outputDir, dirs, skipDownload, startPhase, existingProgress } = config

  const skipLabels = computeSkipLabels(startPhase, skipDownload)

  // Build substitution variables
  const vars: Record<string, string> = {
    HF_MODEL: hfModel,
    MODEL_NAME: modelName,
    OUTPUT_DIR: outputDir,
    MODEL_DIR: dirs.model,
    DEMO_DIR: dirs.demo,
    PROFILE_DIR: dirs.profile,
    PROBLEMS_DIR: dirs.problems,
    OPTIMIZED_DIR: dirs.optimized,
    REPORT_DIR: dirs.report,
    CONFIG_FILE: path.join(outputDir, "config.json"),
    PROGRESS_FILE: path.join(outputDir, "progress.json"),
    START_PHASE: startPhase,
    SKIP_DOWNLOAD_LABEL: skipDownload ? "(SKIP if exists)" : "",
    // Benchmark parameters (configurable via ModelOptConfig)
    CONCURRENCY: String((config as any).concurrency ?? 16),
    INPUT_LEN: String((config as any).inputLen ?? 1024),
    OUTPUT_LEN: String((config as any).outputLen ?? 1024),
    NUM_PROMPTS: String((config as any).numPrompts ?? 100),
  }

  // Add per-phase skip labels
  for (const [idx, label] of Object.entries(skipLabels)) {
    vars[`SKIP_LABEL_${idx}`] = label
  }

  // Read and compose all skill prompts
  const phasePrompts = SKILL_FILES.map((file, i) => {
    const template = readSkill(file)
    const idx = file.slice(0, 2) // "00", "01", etc.
    // Set the generic SKIP_LABEL for this phase
    const phaseVars = { ...vars, SKIP_LABEL: skipLabels[idx] || "" }
    return substitute(template, phaseVars)
  })

  // Build header and footer
  const header = buildHeader(vars, startPhase, existingProgress)
  const footer = buildExecutionInstructions(vars, startPhase)

  return [header, ...phasePrompts, footer].join("\n\n---\n\n")
}

function buildHeader(vars: Record<string, string>, startPhase: string, existingProgress: any): string {
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
- **HuggingFace Model**: ${vars.HF_MODEL}
- **Model Name**: ${vars.MODEL_NAME}
${resumeContext}
## Output Directory Structure
\`\`\`
${vars.OUTPUT_DIR}/
├── venv/           # Project-specific Python virtual environment
├── model/          # Downloaded model files
├── demo/           # Demo scripts for running the model
├── profile/        # Profiling results
├── problems/       # Kernel problems for optimization
├── optimized/      # Optimized kernels
├── report/         # Final optimization report
├── scripts/        # Reusable Python utilities
├── config.json     # Configuration
└── progress.json   # Progress tracking
\`\`\`

## ⚠️ CRITICAL: Use Project venv for ALL Python Operations
- **ALWAYS activate venv before running Python**: \`source ${vars.OUTPUT_DIR}/venv/bin/activate\`
- **NEVER modify system Python packages** in /opt/, /usr/, or site-packages/

## 🎯 CRITICAL PRINCIPLE: Data-Driven Decisions Only
- ✅ **DO**: Read shapes from \`inference_shapes.json\`, analyze \`bottlenecks.json\` and \`kernel_shape_analysis.json\`
- ✅ **DO**: Use per-shape time breakdown from \`kernel_shape_analysis.json\` to prioritize which (operator, shape) to optimize
- ✅ **DO**: Benchmark each optimization at actual inference shapes before applying
- ✅ **DO**: Skip optimizations that don't show improvement
- ❌ **DON'T**: Use hardcoded shape values or assume any optimization will help

## IMPORTANT FILES
- **Config**: ${vars.CONFIG_FILE}
- **Progress**: ${vars.PROGRESS_FILE}

Update progress.json after completing each phase!

## YOUR TASK: Complete phases starting from "${startPhase}"`
}

function buildExecutionInstructions(vars: Record<string, string>, startPhase: string): string {
  return `# EXECUTION INSTRUCTIONS

${startPhase === "download" ? `
## Fresh Start Mode
1. **Execute phases in order**: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8
2. Begin with Phase 0: Environment Setup
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
3. **For kernel-optimize, use the existing opencode command**
4. **All monkey patches go in ${vars.DEMO_DIR}/patches/ or ${vars.OPTIMIZED_DIR}/**
5. **NEVER modify system libraries - only use monkey patching**

## Optimization Priority (Phase 5-6)
1. **FIRST**: Create and optimize FUSED kernels (residual+norm, swiglu)
2. **THEN**: Optimize remaining individual kernels
3. **SKIP**: Operators already optimized by vendor libs (rocBLAS GEMM)
4. **SKIP**: Simple elementwise ops covered by fused kernels

## Start Now
Begin with Phase: ${startPhase.charAt(0).toUpperCase() + startPhase.slice(1)}`
}
