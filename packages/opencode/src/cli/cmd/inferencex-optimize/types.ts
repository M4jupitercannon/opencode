/**
 * Shared types for the inferencex-optimize pipeline.
 */

/** Directory layout for an InferenceX benchmark project */
export interface InferenceXDirs {
  repo: string
  results: string
  profiles: string
  report: string
}

/** Pipeline execution mode */
export type PipelineMode = "full" | "benchmark" | "profile" | "benchmark+profile"

/** Full configuration for the InferenceX optimization pipeline */
export interface InferenceXConfig {
  /** Config key from the master YAML (e.g., "kimik2.5-int4-mi355x-vllm") */
  configKey: string
  /** Root output directory for all artifacts */
  outputDir: string
  /** Subdirectory paths */
  dirs: InferenceXDirs
  /** InferenceX repo URL */
  repoUrl: string
  /** Path to existing InferenceX repo clone (optional) */
  repoDir: string
  /** HuggingFace cache directory */
  hfCache: string
  /** Filter to a specific tensor parallelism level from config search-space */
  filterTp: string
  /** Filter to concurrency levels >= this value */
  filterConcStart: string
  /** Filter to concurrency levels <= this value */
  filterConcEnd: string
  /** Filter to a specific sequence length (e.g., "1k1k", "1k8k", "8k1k") */
  filterSeq: string
  /** Whether this is a dry run (preview commands only) */
  dryRun: boolean
  /** Whether to run profiling */
  profile: boolean
  /** Pipeline execution mode: which phases to run */
  mode: PipelineMode
  /** Which phase to start from */
  startPhase: string
  /** Existing progress data when resuming */
  existingProgress: any
}

/** Ordered list of pipeline phases */
export const PHASE_ORDER = [
  "env",
  "config",
  "benchmark",
  "profile",
  "analyze",
  "report",
] as const

export type PhaseName = (typeof PHASE_ORDER)[number]
