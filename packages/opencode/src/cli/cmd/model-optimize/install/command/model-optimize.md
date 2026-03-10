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

- `<output_dir>/profile/bottlenecks.json`
- `<output_dir>/profile/analysis_summary.json`
- `<output_dir>/profile/model_shapes.json`
- `<output_dir>/profile/decode_report/unified_perf_summary.csv`
- `<output_dir>/profile/prefilldecode_report/unified_perf_summary.csv`

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

Convert bottleneck operators into Problem files. Analyze operators for fusion opportunities BEFORE creating individual problem files. Steps:

1. **Review TraceLens Analysis** — examine `analysis_summary.json` and `unified_perf_summary.csv` for hottest (category, shape) pairs
2. **Operator Fusion Analysis** — run `analyze_fusion.py` to detect fusable patterns (ResidualNorm, SwiGLU, RoPE, etc.)
3. **Create Fused Problem Files** (HIGH priority) — using actual shapes from TraceLens reports
4. **Create Individual Problem Files** (MEDIUM/LOW) — only for unfused ops taking > 5% time
5. **Generate Optimization Manifest** — `optimization_manifest.json` controlling which optimizations to apply

---

# Phase 6: Kernel Optimization

Read and follow `~/.config/opencode/skills/06-kernel-optimize.md`.

Write optimized Triton kernels for each problem file and verify speedup. Optimize kernels DIRECTLY in this session using the provided test scripts (`kernel_test_runner.py`, `kernel_finalize.py`). For each problem file: read source, write `ModelNew` with `@triton.jit` kernels + `@triton.autotune`, test accuracy + benchmark, iterate, finalize. Copy kernels with speedup > 1.0x to `<output_dir>/optimized/`.

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
