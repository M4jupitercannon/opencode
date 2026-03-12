# Phase 5: Generate Problem Files for Kernel Optimization {{SKIP_LABEL}}

## Goal
Convert bottleneck operators into problem files for kernel optimization (Phase 6).
**IMPORTANT**: Analyze operators for fusion opportunities BEFORE creating individual problem files.

## STEP 0: Review Per-Shape Kernel Analysis (from Phase 4)

```bash
python3 {{OUTPUT_DIR}}/scripts/generate_problems.py --review --profile-dir {{PROFILE_DIR}}
```

Review which operator categories dominate GPU time and which specific shapes are hottest.

## STEP 1: Operator Fusion Analysis

Run `analyze_fusion.py` to detect fusable operator patterns (ResidualNorm, SwiGLU, RoPE, QKV, MLP):

```bash
cp {{OUTPUT_DIR}}/scripts/analyze_fusion.py {{PROFILE_DIR}}/
cd {{PROFILE_DIR}}
python3 analyze_fusion.py --bottlenecks decode_bottlenecks.json
cat fusion_opportunities.json
```

## STEP 2: Generate All Problem Files

Run `generate_problems.py` to auto-generate problem files from Phase 4 analysis:

```bash
python3 {{OUTPUT_DIR}}/scripts/generate_problems.py \
  --profile-dir {{PROFILE_DIR}} \
  --problems-dir {{PROBLEMS_DIR}}
```

This generates three categories of problem files:

- **Fusion problems** (from `fusion_opportunities.json`) -- fused_residual_rmsnorm, fused_swiglu, etc.
- **GEMM problems** (from `{phase}_analysis.json`) -- shapes with roofline efficiency < 80%, ranked worst-first
- **Attention problems** (from `{phase}_bottlenecks.json`) -- all attention kernel types (SDPA, flash, linear, GDN) >= 1% GPU time

Each problem file contains `class Model(nn.Module)` (PyTorch baseline), `get_inputs()`, and `get_init_inputs()` using actual shapes from profiling.

## STEP 3: Create Individual Problem Files (if needed)

Only for operators: cannot be fused, take > 5% time, not already covered by STEP 2
(RMSNorm, SiLU/GELU, RoPE, standalone attention).

## STEP 4: Generate Optimization Manifest

Create `{{PROBLEMS_DIR}}/optimization_manifest.json`:

```json
{
  "model": "{{HF_MODEL}}",
  "description": "Edit 'enabled' to control which optimizations to apply",
  "optimizations": [
    {"name": "fused_residual_rmsnorm", "file": "problem_fused_residual_rmsnorm.py",
     "type": "fused", "priority": "HIGH", "enabled": true},
    {"name": "linear_gemm", "file": "problem_linear.py",
     "type": "individual", "priority": "LOW", "enabled": false,
     "notes": "rocBLAS usually optimal"}
  ]
}
```

## Steps Summary
1. Review analysis_summary.json for shape priorities
2. Run fusion analysis (`analyze_fusion.py`)
3. Generate all problem files (`generate_problems.py` -- fusion + GEMM roofline + attention)
4. Create individual problem files if needed (agent-driven)
5. Generate optimization_manifest.json
6. Update progress.json
