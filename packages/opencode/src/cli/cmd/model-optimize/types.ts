/**
 * Shared types for the model-optimize pipeline.
 * Each skill module uses these types for configuration and interop.
 */

/** Directory layout for a model optimization project */
export interface ModelOptDirs {
  model: string
  demo: string
  profile: string
  problems: string
  optimized: string
  report: string
}

/** Full configuration for the model optimization pipeline */
export interface ModelOptConfig {
  /** HuggingFace model identifier (e.g., "Qwen/Qwen3-8B") */
  hfModel: string
  /** Short model name extracted from hfModel */
  modelName: string
  /** Root output directory for all artifacts */
  outputDir: string
  /** Subdirectory paths */
  dirs: ModelOptDirs
  /** Whether to skip model download if already exists */
  skipDownload: boolean
  /** Which phase to start from (for resume/from-phase) */
  startPhase: string
  /** Existing progress data when resuming */
  existingProgress: any
  /** Max concurrent requests for benchmarking (default: 16) */
  concurrency: number
  /** Input sequence length for benchmarking (default: 1024) */
  inputLen: number
  /** Output sequence length for benchmarking (default: 1024) */
  outputLen: number
  /** Number of prompts for benchmarking (default: 100) */
  numPrompts: number
}

/** Ordered list of pipeline phases */
export const PHASE_ORDER = [
  "env",
  "download",
  "demo",
  "compatibility",
  "profile",
  "problems",
  "optimize",
  "integrate",
  "report",
] as const

export type PhaseName = (typeof PHASE_ORDER)[number]
