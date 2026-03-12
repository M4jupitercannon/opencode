---
description: "vLLM kernel optimization and integration pipeline (Phases 5-8). Usage: /model-optimize <model_name> [output_dir]"
agent: model-opt
---

# vLLM Kernel Optimization & Integration Pipeline (Phases 5–8)

## Target

- **HuggingFace Model**: $1
- **Output Directory**: $2 (if not specified, use `/tmp/model_opt_<model_short_name>`)

## Prerequisites

Phases 0–4 (environment setup, model serving, profiling, bottleneck analysis) must be completed first.
See `/model-analyze` for those phases. The following artifacts from Phase 4 are required:

- `<output_dir>/profile/{decode,prefilldecode}_bottlenecks.json`
- `<output_dir>/profile/{decode,prefilldecode}_analysis.json`
- `<output_dir>/profile/analysis_summary.json`
- `<output_dir>/profile/model_shapes.json`
- `<output_dir>/profile/decode_report/unified_perf_summary.csv`
- `<output_dir>/profile/prefilldecode_report/unified_perf_summary.csv`

**For Phase 6 (GEAK optimization):**
- `AMD_LLM_API_KEY` environment variable must be set (used by GEAK to call the LLM orchestrator)
- GEAK must be installed in the Docker container (checked during Phase 0)

## Variable Definitions

Skills use `{{VAR}}` placeholders. When reading skills, substitute:

| Placeholder | Value |
|---|---|
| `{{HF_MODEL}}` | `$1` |
| `{{OUTPUT_DIR}}` | `$2` or `/tmp/model_opt_<short_name>` |
| `{{PROFILE_DIR}}` | `<output_dir>/profile` |
| `{{PROBLEMS_DIR}}` | `<output_dir>/problems` |
| `{{OPTIMIZED_DIR}}` | `<output_dir>/optimized` |
| `{{REPORT_DIR}}` | `<output_dir>/report` |
| `{{INPUT_LEN}}` | `1024` |
| `{{OUTPUT_LEN}}` | `1024` |
| `{{NUM_PROMPTS}}` | `100` |
| `{{CONCURRENCY}}` | `16` |
| `{{SKIP_LABEL}}` | (empty — execute all phases) |

## CRITICAL RULES

- **Docker mode** (preferred): If `env_info.json` has `env_type: "docker"`, prefix commands with `docker exec $CONTAINER_NAME bash -c "..."`. Set `HIP_VISIBLE_DEVICES=$BEST_GPU`.
- **venv mode** (fallback): If `env_type: "venv"`, activate venv: `source <output_dir>/venv/bin/activate`
- **ALL vLLM commands MUST redirect output to log files** (`&> logfile`) — NEVER dump vLLM logs into bash output
- **ALL decisions MUST be data-driven** — read shapes from TraceLens `analysis_summary.json` / `unified_perf_summary.csv`, not hardcoded
- **Optimized kernels MUST use @triton.jit** — torch rewrites are FORBIDDEN
- **Integrate via vLLM CustomOp.register_oot()** — NEVER modify installed packages
- **Serving benchmarks MUST use `vllm bench serve --save-result`**

## MANDATORY VALIDATION

After Phase 6 and Phase 7, run:

```bash
python <output_dir>/scripts/validate_pipeline.py --project-dir <output_dir> --phase all
```

---

# Phase 5: Generate Problem Files for Kernel Optimization

Read and follow `~/.config/opencode/skills/05-problem-generate.md`.

Convert bottleneck operators into Problem files. Classify each kernel type to determine the right GEAK optimization mode for Phase 6. Steps:

1. **Review TraceLens Analysis** — examine `analysis_summary.json` and `unified_perf_summary.csv` for hottest (category, shape) pairs
2. **Operator Fusion Analysis** — run `analyze_fusion.py` to detect fusable patterns (ResidualNorm, SwiGLU, RoPE, etc.)
2.5. **Kernel Type Classification** — trace each bottleneck op to source, classify as `triton`/`hip`/`ck`/`asm`/`aten_gemm`/`aten_elementwise`/`triton_composite`. Record `source_file` for C++ kernels.
3. **Generate Problem Files** — `generate_problems.py` (fusion + GEMM roofline + attention) + HIP/composite/individual problem files
4. **Generate Optimization Manifest** — `optimization_manifest.json` with `kernel_type` metadata, all enabled by default

---

# Phase 6: Kernel Optimization via GEAK

Read and follow `~/.config/opencode/skills/06-kernel-optimize.md`.

Use GEAK to optimize each bottleneck kernel using the appropriate mode based on `kernel_type`. Requires `AMD_LLM_API_KEY`, GEAK, and `geak-oe` (for C++ kernels) installed in the container.

Steps:
1. **Verify GEAK + geak-oe** — check `geak --help` and `/opt/geak-oe` in the container
2. **Read manifest, detect GPU architecture**
3a. **HIP/CK/ASM/composite kernels** — launch `geak --kernel-url` on C++ source (see `hip-kernel-optimize-geak.md`)
3b. **Triton/ATen kernels** — launch `geak -m claude-opus-4.6 -t "Optimize ..." --yolo` with kernel-type-aware task descriptions
3.5. **Collect patches** — extract optimized kernels from GEAK `optimization_logs/` and `geak_output/results/`
4. **Verify results** — check correctness + benchmark each optimized kernel against baseline
5. **Copy winning kernels** — speedup > 1.0x go to `<output_dir>/optimized/` (note: kernel speedup may not equal E2E speedup)

If GEAK is unavailable, fall back to manual Triton kernel writing using `kernel_test_runner.py` and `kernel_finalize.py`.

---

# Phase 7: Integration & End-to-End Testing

Read and follow `~/.config/opencode/skills/07-integration.md`.

Apply optimized kernels to vLLM via CustomOp and measure ACTUAL serving throughput.

**This phase is NOT complete until:**

1. A patched vLLM server has ACTUALLY been started and served requests
2. `vllm bench serve` has been run against the patched server
3. `optimized_serving.json` has `"label": "optimized"` (NOT "baseline")
4. The validation script passes

**FORBIDDEN**: Estimating speedup with Amdahl's law, copying baseline numbers, reporting "estimated" speedup, skipping the patched server benchmark.

Steps: generate vLLM plugin (`generate_vllm_plugin.py`), test plugin registration, benchmark baseline, benchmark patched server, validate results.

---

# Phase 8: Generate Final Report

Read and follow `~/.config/opencode/skills/08-report-generate.md`.

Create `<output_dir>/report/optimization_report.md` with ACTUAL MEASURED end-to-end speedup from `comparison_results.json`. Include: model info, summary, bottleneck analysis with TraceLens roofline data, performance results table (baseline vs optimized), comparison outputs, files generated, recommendations.

---

# EXECUTION INSTRUCTIONS

Execute phases: 5 → 6 → 7 → 8.
**ALL vLLM output to log files. Run validate_pipeline.py after Phase 6 and 7.**
Begin with Phase 5.
