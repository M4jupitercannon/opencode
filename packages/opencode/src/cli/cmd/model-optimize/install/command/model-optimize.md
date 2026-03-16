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
- An LLM API key (`AMD_LLM_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY`) must be configured (prompted during Phase 0 Step 6)
- GEAK (`mini` CLI) must be installed in the Docker container (installed from `main` branch during Phase 0 Step 7)
- Check `env_info.json` for `geak_available: true` — if `false`, Phase 6 uses manual Triton fallback

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
- **Integrate via vLLM CustomOp.register_oot()** for ops with CustomOp mappings (RMSNorm, SiluAndMul, etc.)
- **Integrate GEMM kernels via torch.mm override** for decode shapes that don't map to CustomOps
- **NEVER modify installed packages**
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

Use GEAK (`mini` CLI) to optimize each bottleneck kernel using the appropriate mode based on `kernel_type`. Requires an LLM API key, GEAK (`mini`), and `geak-oe` (for C++ kernels) installed in the container.

Steps:
1. **Verify GEAK + API key** — check `mini --help`, API key in `.env`, `geak_available` in `env_info.json`. If missing, ask user for API key.
2. **Read manifest, detect GPU architecture**
3a. **HIP/CK/ASM/composite kernels** — launch `mini --config mini_kernel.yaml` on C++ source (see `hip-kernel-optimize-geak.md`)
3b. **Triton/ATen kernels** — launch `mini -m claude-opus-4.6 --config geak.yaml -t "Optimize ..." --yolo` with kernel-type-aware task descriptions, parallel across GPUs
3.5. **Collect and recover patches** — extract optimized kernels from `optimization_logs/`. If `[SelectPatch]` fails to apply, recover the best kernel directly from the patch diff and re-verify with `kernel_test_runner.py`
4. **Verify results** — check correctness + benchmark each optimized kernel against baseline
5. **Copy winning kernels** — speedup > 1.0x go to `<output_dir>/optimized/` (note: kernel speedup may not equal E2E speedup)

If GEAK is unavailable or the user cannot provide an API key, fall back to manual Triton kernel writing using `kernel_test_runner.py` and `kernel_finalize.py`.

---

# Phase 7: Integration & End-to-End Testing

Read and follow `~/.config/opencode/skills/07-integration.md`.

Apply optimized kernels to vLLM via CustomOp + torch.mm override and measure ACTUAL serving throughput.

**This phase is NOT complete until:**

1. `baseline_serving.json` exists (reused from Phase 4 `baseline_benchmark.json` — do NOT re-run)
2. A patched vLLM server has ACTUALLY been started and served requests
3. `vllm bench serve` has been run in **both compiled and eager modes** for the optimized server
4. `optimized_serving.json` (compiled) and `optimized_eager_serving.json` (eager) exist with correct labels
5. The validation script passes with `comparison_results.json`

**FORBIDDEN**: Estimating speedup with Amdahl's law, copying baseline numbers, reporting "estimated" speedup, skipping the patched server benchmark.

Steps: generate vLLM plugin (`generate_vllm_plugin.py` — creates CustomOp registrations + torch.mm GEMM override), test plugin registration, reuse Phase 4 baseline, benchmark patched server (compiled + eager), validate results.

---

# Phase 8: Generate Final Report

Read and follow `~/.config/opencode/skills/08-report-generate.md`.

Create `<output_dir>/report/optimization_report.md` with ACTUAL MEASURED end-to-end speedup from `comparison_results.json`. Include: model info, summary, bottleneck analysis with TraceLens roofline data, performance results table (baseline vs optimized), comparison outputs, files generated, recommendations.

---

# EXECUTION INSTRUCTIONS

Execute phases: 5 → 6 → 7 → 8.
**ALL vLLM output to log files. Run validate_pipeline.py after Phase 6 and 7.**
Begin with Phase 5.
