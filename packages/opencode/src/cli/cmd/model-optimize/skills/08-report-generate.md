# Phase 8: Generate Final Report {{SKIP_LABEL}}

## Goal
Create a comprehensive optimization report.

## Create Report: `{{REPORT_DIR}}/optimization_report.md`

**⚠️ CRITICAL**: Include **ACTUAL MEASURED** end-to-end speedup from comparison_results.json.

```markdown
# Model Optimization Report

## Model Information
- **Model**: {{HF_MODEL}}
- **Optimization Date**: [DATE]

## Summary
- **ACTUAL End-to-End Speedup**: X.Xx (measured, NOT estimated)
- **Kernels Optimized**: N
- **Baseline Inference Time**: X.Xs
- **Optimized Inference Time**: X.Xs

## Bottleneck Analysis
| Operator | Original Time (ms) | % of Total | Optimized | Speedup |
|----------|-------------------|------------|-----------|---------|
| ...      | ...               | ...        | ...       | ...     |

## TraceLens Roofline Analysis
Include the top hottest (operator, shape) combinations from `analysis_summary.json` and TraceLens `unified_perf_summary.csv`.
This data was collected with `--enforce-eager` and `torch_profiler_record_shapes: true` in `--profiler-config`.

| Category | Shape | % of GPU Time | Time (ms) | Count |
|----------|-------|--------------|-----------|-------|
| GEMM     | [4,4096]x[4096,24576] | 30.5% | 5023.1 | 9072 |
| ...      | ...   | ...          | ...       | ...   |

## Performance Results (ACTUAL MEASURED)
| Metric | Original | Optimized | Speedup |
|--------|----------|-----------|---------|
| End-to-End Inference Time | X.Xs | X.Xs | **X.Xx** |

## Comparison Outputs (Seed=42)
Outputs generated with fixed random seed for verification.

### Text Models:
| Original | Optimized |
|----------|-----------|
| [text]   | [text]    |

### Image Models (if applicable):
> Include this section only for vision/multimodal models that produce image outputs.

| Original | Optimized |
|:--------:|:---------:|
| ![Original](comparison_outputs/original_output.png) | ![Optimized](comparison_outputs/optimized_output.png) |

## Files Generated
- profile/bottlenecks.json - Kernel bottleneck ranking with roofline efficiency
- profile/analysis_summary.json - TraceLens analysis summary with per-phase roofline data
- profile/prefilldecode_report/ - TraceLens CSVs for prefill-decode phase
- profile/decode_report/ - TraceLens CSVs for decode-only phase
- problems/ - Problem files + optimized kernels
- optimized/vllm_plugin/ - vLLM CustomOp integration plugin
- report/baseline_serving.json - Baseline benchmark results
- report/optimized_serving.json - Optimized benchmark results
- report/optimization_report.md - This report

## Recommendations
1. ...
```

## Steps
1. Gather all results from previous phases
2. Generate the comprehensive report
3. Update progress.json: phase="complete", phases_completed.append("report")
