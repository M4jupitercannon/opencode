---
description: "Model analysis pipeline: environment setup, serving, profiling, and bottleneck analysis. Usage: /model-analyze <model_name> [output_dir]"
agent: model-opt
---

# Model Analysis Pipeline (Phases 0–4)

## Target

- **HuggingFace Model**: $1
- **Output Directory**: $2 (if not specified, use `/tmp/model_opt_<model_short_name>`)

## First Steps

1. Parse model name from `$1`
2. Determine output directory: `$2` if provided, else `/tmp/model_opt_<short_name>` (**MUST be outside the working directory**)
3. Create directory structure + `.gitignore` (exclude venv/, model/, \*.safetensors, etc.)
4. Copy helper scripts from `~/.config/opencode/scripts/` to `<output_dir>/scripts/`

## CRITICAL RULES

- **Docker mode** (preferred): If `env_info.json` has `env_type: "docker"`, search available docker images on dockerhub(rocm/vllm-dev:nightly preferred). prefix commands with `docker exec $CONTAINER_NAME bash -c "..."`. Set `HIP_VISIBLE_DEVICES=$BEST_GPU`.
- **venv mode** (fallback): If `env_type: "venv"`, activate venv: `source <output_dir>/venv/bin/activate`
- **ALL vLLM commands MUST redirect output to log files** (`&> logfile`) — NEVER dump vLLM logs into bash output
- **ALL decisions MUST be data-driven** — read shapes from TraceLens `analysis_summary.json` / `unified_perf_summary.csv`, not hardcoded
- **Serving benchmarks MUST use `vllm bench serve --save-result`**

---

# Phase 0: Environment Setup

Follow `skills/00-env-setup.md`.

Detect host platform, search DockerHub for a compatible `rocm/vllm-dev` nightly image, create container (or fall back to venv). Patches vLLM BlockSize for hybrid architectures. Saves `env_info.json`.

CRITICAL for all subsequent phases: If `env_type` is `docker` in `env_info.json`, prefix all commands with `docker exec $CONTAINER_NAME bash -c "..."` and use `/workspace/output` as the output directory inside the container. Set `HIP_VISIBLE_DEVICES=$BEST_GPU` to target the GPU with the most free memory.

---

# Phase 1: Model Serving with vLLM

Follow `skills/01-model-download.md`.

Start the model using `vllm serve` and verify inference works. Record model config (including hybrid architecture fields like `layer_types` and `architectures`). In vLLM mode, this covers download + demo + compatibility in one step.

---

# Phase 2: (Covered by Phase 1 in vLLM mode)

> In vLLM mode, demo generation is handled by Phase 1 (`vllm serve`). Skip this phase.

---

# Phase 3: (Covered by Phase 1 in vLLM mode)

> In vLLM mode, compatibility fixes are handled by vLLM itself. Skip this phase.

If vLLM serve failed in Phase 1, debug using vLLM logs (check `--dtype`, `--tensor-parallel-size`, `--max-model-len`).

---

# Phase 4: Performance Profiling

Follow `skills/04-profiling.md`.

Benchmark vLLM serving throughput AND collect GPU kernel trace for bottleneck analysis. Steps:

1. **Baseline Throughput Benchmark** — `vllm bench serve` with 100 prompts
2. **Collect Kernel Trace** — `--enforce-eager` + `--profiler-config` with `record_shapes: true` + `/start_profile` API. Uses adaptive trace sizing (5 prompts default, reduced token lengths for hybrid architectures).
3. **Split Trace & TraceLens Analysis** — `analyze_kernels.py` splits into prefill-decode and decode-only phases, runs TraceLens roofline analysis on each.
4. **Generate Per-Phase Bottlenecks** — produces `{decode,prefilldecode,full}_bottlenecks.json` and `{decode,prefilldecode,full}_analysis.json` with roofline data.
5. **Save Model Shapes** — `model_shapes.json` with standard + linear attention fields.

---

## TraceLens Roofline Analysis (Decode Phase)
Include the top ops from `decode_report/unified_perf_summary.csv` with roofline metrics.
This data was collected with `--enforce-eager` and `torch_profiler_record_shapes: true`.

| Op | Input Dims | % GPU Time | TFLOPS/s | TB/s | FLOPS/Byte | Bound |
|----|-----------|-----------|---------|------|-----------|-------|
| aten::mm | (16,4096)x(4096,24576) | 48.2% | 5.5 | 0.34 | 15.9 | Memory |
| ...      | ...   | ...       | ...     | ...  | ...       | ...   |

## TraceLens Roofline Analysis (Prefill-Decode Phase)
| Op | Input Dims | % GPU Time | TFLOPS/s | TB/s | FLOPS/Byte | Bound |
|----|-----------|-----------|---------|------|-----------|-------|
| aten::mm | (2048,4096)x(4096,24576) | 38.2% | 120.9 | 0.09 | 1293.5 | Compute |
| ...      | ...   | ...       | ...     | ...  | ...       | ...   |

## Files Generated
- profile/decode_bottlenecks.json - Bottlenecks from decode-only phase
- profile/decode_analysis.json - Analysis report for decode phase (categories, GEMM roofline, top bottlenecks)
- profile/prefilldecode_bottlenecks.json - Bottlenecks from prefill-decode phase
- profile/prefilldecode_analysis.json - Analysis report for prefill-decode phase
- profile/analysis_summary.json - TraceLens analysis summary (all phases)
- profile/phase_category_summary.json - Per-phase category breakdown
- profile/phase_traces/ - Split trace files (steady-state, prefill-decode, decode-only)
- profile/prefilldecode_report/ - TraceLens CSVs for prefill-decode phase
- profile/decode_report/ - TraceLens CSVs for decode-only phase

# EXECUTION INSTRUCTIONS
Execute phases: 0 → 1 → 4 (Phases 2-3 handled by vLLM).
Begin with Phase 0.
